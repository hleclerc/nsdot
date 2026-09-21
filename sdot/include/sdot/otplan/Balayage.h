#pragma once

// =====================================================================================
// LES DIAGRAMMES DU SOLVEUR : ce que Newton demande a un diagramme de puissance, et rien d'autre.
//
//     set_weights( w )              poser des poids ( ordre utilisateur ), refaire les majorants
//     mesures( a, &fa )             UN balayage : la mesure de chaque cellule ET ses facettes
//                                   `c_ij = int_{facette} rho / ( 2 |p_i - p_j| )` ( `Laplacien.h` )
//
// Tout tourne ici sur la file CPU ( `CpuQueue::run_threads` : des tranches contigues de cellules,
// dans l'ordre du stockage, ou deux germes consecutifs sont voisins dans l'espace ). Le scratch
// des cellules est GERE ICI, pas par loom : un solveur enchaine cent diagrammes dans un seul
// appel, et un debordement ne doit relancer que le balayage en cours, pas l'appel entier. Une
// ligne de mots par fil, doublee tant qu'une cellule n'y tient pas.
//
// La cellule d'un germe est construite comme partout ( `diagram::make_cell` : le domaine, puis les
// plans que le fournisseur du stockage propose ), sa masse integree comme partout
// ( `diagram::integrate_into` ), et ses facettes lues comme `diagram::hessian_row` les lit -- ce
// fichier n'ajoute pas de geometrie, il enchaine.
//
// UNE DENSITE NON CONSTANTE sur une facette : la distribution doit savoir integrer sa densite sur
// une facette ( `facet_mass( piece, cut )` -- les gaussiennes 2D le font en forme close ) ; a
// defaut le laplacien n'est pas assemblable et la compilation le dit.
// =====================================================================================

#include <loom/support/kernels/CpuQueue.h>
#include "../diagram/Ops.h"
#include "../bsp_build_level.h"
#include "Laplacien.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <vector>

namespace sdot {
namespace otplan {

inline double now() {
    using namespace std::chrono;
    return duration<double>( steady_clock::now().time_since_epoch() ).count();
}

/// la masse d'une facette pour la densite d'un morceau : `rho * mesure` a densite constante, ce que
/// la distribution sait en dire sinon
template<class Dens,class Pc>
double masse_facette( const Dens &dens, const Pc &pc, int cut, double mes ) {
    if constexpr ( Dens::is_constant )
        return double( dens.value ) * mes;
    else if constexpr ( requires { dens.facet_mass( pc, cut ); } )
        return double( dens.facet_mass( pc, cut ) );
    else
        static_assert( Dens::is_constant, "otplan : cette distribution ne sait pas integrer sa densite sur une facette ( `facet_mass` )" );
    return 0;
}

/// UNE CELLULE du germe de rang `k` : sa masse dans `masse`, et `facette( j, c_kj )` pour chaque voisin
/// `j` ( en rangs ). Rend `false` si le scratch n'a pas suffi.
template<class PD,class Local>
bool mesure_et_facettes( const PD &pd, SI k, Local &c, Local &piece, const auto &dom, const auto &dist,
                         double &masse, auto &&facette, bool avec_facettes ) {
    using TF = typename PD::TF;
    constexpr int D = PD::ct_dim;
    if ( ! diagram::make_cell( pd, c, k, dom ) )
        return false;
    TF m = 0;
    if ( ! diagram::integrate_into<TF>( m, c, piece, dist ) )
        return false;
    masse = double( m );
    if ( ! avec_facettes )
        return true;

    // les voisins de la cellule sont ses coupes vivantes : une case par coupe, accumulee morceau
    // par morceau ( un morceau porte les memes identifiants, plus ceux de son pave )
    c.tidy();
    const int nc = c.nb_cuts();
    double vals[ 512 ];
    if ( nc > 512 )
        return false;
    for ( int q = 0; q < nc; ++q )
        vals[ q ] = 0;
    PieceWorkspace<Local> ws{ piece };
    dist.for_each_piece( c, ws, [&]( const auto &pc, const auto &dens ) {
        pc.template for_each_facet<TF>( [&]( int cut, TF mes ) {
            const int id = pc.cid[ cut ];
            if ( id < 0 )
                return;                                  // le domaine, ou un bord de pave : immobile
            const double val = masse_facette( dens, pc, cut, double( mes ) );
            for ( int q = 0; q < nc; ++q )
                if ( c.cid[ q ] == id ) {
                    vals[ q ] += val;
                    break;
                }
        } );
    } );
    if ( ws.overflow )
        return false;

    const auto p0 = pd.point( k );
    for ( int q = 0; q < nc; ++q ) {
        const int id = c.cid[ q ];
        if ( id < 0 )
            continue;
        const auto pj = pd.point( id );
        double d2 = 0;
        for ( int d = 0; d < D; ++d )
            d2 += double( pj[ d ] - p0[ d ] ) * double( pj[ d ] - p0[ d ] );
        if ( d2 > 0 )
            facette( id, vals[ q ] / ( 2 * std::sqrt( d2 ) ) );
    }
    return true;
}

/// LE DIAGRAMME VU PAR NEWTON. `pd` est le stockage ( `PowerDiagram_Bsp` / `_Plain` ) dont les
/// poids -- et les majorants de l'arbre -- sont des vues INSCRIPTIBLES ( `with_weights` ) : c'est
/// ici qu'on les ecrit.
template<class PD,class Dom,class Dist,class TK>
struct Balayage {
    using TF = typename PD::TF;
    using Local = typename Dom::template Local<TK>;
    static constexpr int D = PD::ct_dim;
    static constexpr int nbc = diagram::nb_work_cells<Dist>();

    const CpuQueue &queue;
    PD             &pd;                                  ///< les poids y sont ecrits ( vues de sortie )
    const Dom      &dom;
    const Dist     *dist;                                ///< la densite courante ( elle change d'une etape a l'autre )
    int             nt;                                  ///< fils virtuels ( tranches contigues )
    SI              cap;                                 ///< sommets par cellule locale
    SI              words = 0;
    std::vector<std::vector<std::int32_t>> scratch;      ///< une ligne par fil
    std::vector<std::vector<Facette>>      fa_th;        ///< les facettes, par fil
    std::vector<SI> rang_de;                             ///< le rang du germe `i` ( ordre utilisateur )

    SI     nb_deborde = 0;                               ///< doublements du scratch, en tout
    int    nb_diag    = 0;
    double t_maj = 0, t_diag = 0;

    Balayage( const CpuQueue &queue, PD &pd, const Dom &dom, const Dist &dist, SI cap0 )
        : queue( queue ), pd( pd ), dom( dom ), dist( &dist ), nt( std::max( queue.nb_workers(), 1 ) ), cap( std::max<SI>( cap0, 8 ) ) {
        const SI n = pd.nb_seeds();
        scratch.resize( nt );
        fa_th.resize( nt );
        rang_de.resize( n );
        for ( SI k = 0; k < n; ++k )
            rang_de[ pd.user_id( k ) ] = k;
        redimensionne();
    }

    SI n() const { return pd.nb_seeds(); }

    void redimensionne() {
        words = diagram::words_for<Local,TF>( cap, nbc, false );
        for ( auto &s : scratch )
            s.assign( size_t( words ) + 16, 0 );
    }

    /// une tranche contigue de `[ 0, n )` par fil
    static void tranche( SI n, int t, int nt, SI &b, SI &e ) {
        b = SI( ( long long ) t * n / nt );
        e = SI( ( long long ) ( t + 1 ) * n / nt );
    }

    /// LES POIDS `W` ( ordre utilisateur ) poses sur le stockage, et les majorants de l'arbre refaits
    void set_weights( const std::vector<double> &W ) {
        const double t0 = now();
        const SI n = this->n();
        if constexpr ( requires { pd.sorted_weights; } ) {
            queue.run_threads( nt, [&]( int t ) {
                SI b, e;
                tranche( n, t, nt, b, e );
                for ( SI k = b; k < e; ++k )
                    pd.sorted_weights( k ) = TF( W[ pd.user_id( k ) ] );
            } );
            const SI nb_nodes = SI( pd.tree.node_begin.shape( 0 ) );
            queue.run_threads( nt, [&]( int t ) {
                SI b, e;
                tranche( nb_nodes, t, nt, b, e );
                for ( SI m = b; m < e; ++m )
                    bsp_refresh_majorant( pd.sorted_cloud(), pd.tree.node_begin( m ), pd.tree.node_end( m ),
                                          pd.tree.node_wa( m ), pd.tree.node_wb( m ) );
            } );
        } else {
            queue.run_threads( nt, [&]( int t ) {
                SI b, e;
                tranche( n, t, nt, b, e );
                for ( SI k = b; k < e; ++k )
                    pd.weights( k ) = TF( W[ k ] );
            } );
        }
        t_maj += now() - t0;
    }

    /// LES MESURES `a` ( ordre utilisateur ) aux poids poses, et si `fa` n'est pas nul les facettes
    /// `c_ij` ( identifiants utilisateur ). Un balayage, relance sur debordement du scratch.
    void mesures( std::vector<double> &a, std::vector<Facette> *fa = nullptr ) {
        const double t0 = now();
        const SI n = this->n();
        a.resize( n );
        for ( ;; ) {
            std::atomic<bool> deborde{ false };
            queue.run_threads( nt, [&]( int t ) {
                Carver cv{ scratch[ t ].data(), words };
                Local c, piece;
                c.attach( cv, cap );
                if constexpr ( nbc > 1 ) piece.attach( cv, cap );
                else                     piece = c;
                auto &fv = fa_th[ t ];
                fv.clear();
                SI b, e;
                tranche( n, t, nt, b, e );
                for ( SI k = b; k < e; ++k ) {
                    const SI i = pd.user_id( k );
                    if ( ! mesure_et_facettes( pd, k, c, piece, dom, *dist, a[ i ],
                                               [&]( SI j, double cij ) { fv.push_back( Facette{ i, pd.user_id( j ), cij } ); },
                                               fa != nullptr ) ) {
                        deborde = true;
                        return;
                    }
                }
            } );
            if ( ! deborde )
                break;
            cap *= 2;
            redimensionne();
            ++nb_deborde;
        }
        if ( fa ) {
            fa->clear();
            for ( auto &v : fa_th )
                fa->insert( fa->end(), v.begin(), v.end() );
        }
        t_diag += now() - t0;
        ++nb_diag;
    }
};

} // namespace otplan
} // namespace sdot

#pragma once

// =====================================================================================
// LE DIAGRAMME DE PUISSANCE : ce que le solveur voit, en 2D comme en 3D.
//
// Un `PowerDiagram<D,TK,MaxNv>` possede l'arbre BSP et une copie des germes DANS L'ORDRE DE
// L'ARBRE, dans le flottant du noyau `TK`. Il offre trois choses :
//
//   `build( P, W, n, leaf )`       les germes, dans l'ordre de l'appelant ; `W == nullptr` pour
//                                  Voronoi ( le cas euclidien ne paie alors pas les majorants ) ;
//   `set_weights( W, par )`        des poids neufs, meme arbre : les majorants et la copie
//                                  refaits, rien d'autre ;
//   `measures( res, par )`         la mesure de chaque cellule, INDEXEE PAR L'IDENTIFIANT de
//   `measures_and_facets( ... )`   l'appelant, et pour Newton les facettes `( i, j, mesure )`.
//
// Le choix 2D / 3D est fait ici et nulle part ailleurs : `cellule( k, c )` deroule le moteur qui
// convient, `mesure( c, facette )` lit ce qui en sort. Le reste du fichier est commun.
//
// Les mesures sont accumulees en `TF` ( double ) depuis les sommets en `TK` : avec `TK = float`
// le plancher relatif d'une aire est ~1e-7, ce qui est le plancher du residu de Newton. `double`
// leve cette limite au prix mesure par `diagramme --kernel float|double`.
// =====================================================================================

#include "accel/AaBsp.h"
#include "cell/FournisseurBsp2D.h"
#include "cell/FournisseurBsp3D.h"
#include "cell/Noyau2D.h"
#include "cell/Noyau3D.h"
#include "util/parallel.h"

#include <atomic>
#include <cmath>
#include <type_traits>
#include <vector>

namespace sf {

template<int D, class TK, int MaxNv>
struct PowerDiagram {
    static constexpr int dim = D;
    static constexpr int max_nv = MaxNv;
    using TKernel = TK;
    using Arbre   = AaBspT<D>;
    /// la cellule telle que le moteur la rend : l'atelier en 2D, le polyedre en 3D
    using Cell    = std::conditional_t<D == 2, d2::Atelier<TK,MaxNv>, d3::Cellule3<TK,MaxNv,8>>;

    Arbre            arbre;
    std::vector<TK>  c[ D ];        ///< les positions, ordre de l'arbre, flottant du noyau
    std::vector<TK>  w;             ///< les poids, idem ( vide = Voronoi )
    std::vector<d2::SI32> ids;      ///< rang dans l'arbre -> identifiant de l'appelant
    SI               n = 0;

    // ------------------------------------------------------------------ construction, poids
    void build( const TF *const *P, const TF *W, SI nb, SI leaf ) {
        n = nb;
        arbre.build( P, W, n, leaf );
        for ( int d = 0; d < D; ++d ) c[ d ].resize( n );
        ids.resize( n );
        for ( SI k = 0; k < n; ++k ) {
            for ( int d = 0; d < D; ++d ) c[ d ][ k ] = TK( arbre.seed_c( k, d ) );
            ids[ k ] = d2::SI32( arbre.seed_id( k ) );
        }
        w.clear();
        if ( W )
            copie_poids();
    }

    /// `W` dans l'ordre de l'appelant. L'arbre passe en Laguerre s'il ne l'etait pas.
    void set_weights( const TF *W, const Parallel &par ) {
        arbre.refresh_weights( W, par );
        copie_poids();
    }

    bool laguerre() const { return ! w.empty(); }

    // ------------------------------------------------------------------ une cellule
    /// La cellule du germe de rang `k` dans l'arbre. Rend `false` si elle a DEBORDE `MaxNv`.
    bool cellule( SI k, Cell &cel ) const {
        return laguerre() ? cellule_<true>( k, cel ) : cellule_<false>( k, cel );
    }

    /// La meme cellule avec un AUTRE poids pour le germe `k` seul ( les autres inchanges, l'arbre
    /// aussi : les majorants ne parlent que des autres ). Laguerre seulement.
    bool cellule_avec_poids( SI k, TF wk, Cell &cel ) const {
        if constexpr ( D == 2 ) {
            d2::FournisseurBsp<TK,true> f( &arbre, c[ 0 ][ k ], c[ 1 ][ k ], TK( wk ), ids[ k ] );
            d2::moteur<TK>( &f, &cel );
            return cel.nb >= 0;
        } else {
            d3::FournisseurBsp3<TK,true> f( &arbre, c[ 0 ][ k ], c[ 1 ][ k ], c[ 2 ][ k ], TK( wk ), ids[ k ] );
            return d3::moteur( &f, &cel ) == 0;
        }
    }

    /// La mesure de la cellule, et ses facettes contre d'autres germes : `facette( j, mes )`
    /// avec `j` l'identifiant du voisin -- les parois du domaine ne sont pas rendues.
    template<class Facette>
    static TF mesure( const Cell &cel, Facette &&facette ) {
        if constexpr ( D == 2 ) {
            if ( cel.nb <= 0 ) return 0;
            TF a = 0;
            for ( int i = 0, j = cel.nb - 1; i < cel.nb; j = i++ ) {
                a += TF( cel.vx[ j ] ) * TF( cel.vy[ i ] ) - TF( cel.vx[ i ] ) * TF( cel.vy[ j ] );
                if ( cel.cid[ j ] >= 0 ) {               // la coupe `j` porte l'arete [ v_j, v_i ]
                    const TF ex = TF( cel.vx[ i ] ) - TF( cel.vx[ j ] );
                    const TF ey = TF( cel.vy[ i ] ) - TF( cel.vy[ j ] );
                    facette( cel.cid[ j ], std::sqrt( ex * ex + ey * ey ) );
                }
            }
            return TF( 0.5 ) * std::fabs( a );
        } else {
            return cel.volume_et_faces( [ & ]( d3::SI32 id, double aire ) {
                if ( id >= 0 ) facette( id, TF( aire ) );
            } );
        }
    }
    static TF mesure( const Cell &cel ) { return mesure( cel, []( d2::SI32, TF ) {} ); }

    // ------------------------------------------------------------------ tout le diagramme
    /// `res[ id ]` pour chaque germe. Rend le nombre de cellules qui ont DEBORDE ( leur mesure est
    /// alors fausse -- relancer avec un `MaxNv` plus grand ).
    SI measures( std::vector<TF> &res, const Parallel &par ) const {
        return measures_and_facets( res, par, []( int, d2::SI32, d2::SI32, TF ) {} );
    }

    /// Le meme, avec `facette( t, i, j, mes )` pour chaque facette de la cellule `i` contre `j`,
    /// depuis le thread `t` -- chaque paire est vue DEUX fois, une par cellule.
    template<class Facette>
    SI measures_and_facets( std::vector<TF> &res, const Parallel &par, Facette &&facette ) const {
        res.assign( n, TF( 0 ) );
        std::atomic<SI> deborde{ 0 };
        parallel_for( n, par, [ & ]( SI k, int t ) {
            Cell cel;
            const d2::SI32 i = ids[ k ];
            if ( ! cellule( k, cel ) ) { ++deborde; return; }
            res[ i ] = mesure( cel, [ & ]( d2::SI32 j, TF mes ) { facette( t, i, j, mes ); } );
        } );
        return deborde.load();
    }

private:
    void copie_poids() {
        w.resize( n );
        for ( SI k = 0; k < n; ++k ) w[ k ] = TK( arbre.seed_w( k ) );
    }

    template<bool POIDS>
    bool cellule_( SI k, Cell &cel ) const {
        const TK w0 = POIDS ? w[ k ] : TK( 0 );
        if constexpr ( D == 2 ) {
            d2::FournisseurBsp<TK,POIDS> f( &arbre, c[ 0 ][ k ], c[ 1 ][ k ], w0, ids[ k ] );
            d2::moteur<TK>( &f, &cel );
            return cel.nb >= 0;
        } else {
            d3::FournisseurBsp3<TK,POIDS> f( &arbre, c[ 0 ][ k ], c[ 1 ][ k ], c[ 2 ][ k ], w0, ids[ k ] );
            return d3::moteur( &f, &cel ) == 0;
        }
    }
};

} // namespace sf

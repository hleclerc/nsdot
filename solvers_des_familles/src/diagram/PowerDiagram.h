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

    /// LA MEMOIRE ( 3D, § 11 ) : par rang, les rangs des voisins du DERNIER diagramme, proposes en
    /// premier par `measures_and_facets*` tant qu'elle est la ( `oublie()` pour l'eteindre ).
    std::vector<SI>       rang;     ///< identifiant -> rang, bati au premier `memorise`
    std::vector<SI>       memo_beg;
    std::vector<d2::SI32> memo_pre;
    mutable std::vector<std::vector<unsigned char>> memo_saute;   ///< par fil : « deja propose »
    mutable SI            memo_coupees = 0;                         ///< coupes effectives, en tout, au dernier diagramme

    /// les paires ( identifiant, identifiant ) des facettes d'un diagramme deviennent la memoire
    void memorise( const d2::SI32 *ii, const d2::SI32 *jj, SI nb ) {
        if ( SI( rang.size() ) != n ) { rang.resize( n ); for ( SI k = 0; k < n; ++k ) rang[ ids[ k ] ] = k; }
        memo_beg.assign( n + 1, 0 );
        for ( SI q = 0; q < nb; ++q ) ++memo_beg[ rang[ ii[ q ] ] + 1 ];
        for ( SI k = 0; k < n; ++k ) memo_beg[ k + 1 ] += memo_beg[ k ];
        memo_pre.resize( nb );
        std::vector<SI> at( memo_beg.begin(), memo_beg.end() - 1 );
        for ( SI q = 0; q < nb; ++q ) memo_pre[ at[ rang[ ii[ q ] ] ]++ ] = d2::SI32( rang[ jj[ q ] ] );
    }
    void oublie() { memo_beg.clear(); memo_pre.clear(); }
    bool memoire() const { return ! memo_beg.empty(); }

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

    /// LA CELLULE AVEC MEMOIRE ( 3D, `main_memo.cpp` ) : les rangs `pre[ 0 .. npre )` proposes en
    /// premier, ceux marques dans `saute` ( par rang ) sautes au parcours. `prop` et `boites` rendent
    /// ce que le fournisseur a propose et teste. Rend `false` sur debordement.
    bool cellule_memo( SI k, Cell &cel, const d2::SI32 *pre, int npre, const unsigned char *saute, int &prop, int &boites, int &coupees, bool parcours = true ) const {
        static_assert( D == 3, "la memoire n'est ecrite qu'en 3D" );
        return laguerre() ? cellule_memo_<true>( k, cel, pre, npre, saute, prop, boites, coupees, parcours )
                          : cellule_memo_<false>( k, cel, pre, npre, saute, prop, boites, coupees, parcours );
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
        return measures_and_facets_avec( res, par, facette, []( const Cell &cel, auto &&fac, SI ) { return mesure( cel, fac ); } );
    }

    /// Le meme, avec UNE AUTRE MESURE que Lebesgue : `mes( cel, facette( j, mes ), i )` rend la mesure
    /// de la cellule `i` et appelle `facette` pour chacune de ses facettes ( `Densite.h` ).
    template<class Facette, class Mesure>
    SI measures_and_facets_avec( std::vector<TF> &res, const Parallel &par, Facette &&facette, Mesure &&mes ) const {
        res.assign( n, TF( 0 ) );
        std::atomic<SI> deborde{ 0 };
        if constexpr ( D == 3 ) {
            if ( memoire() ) {
                const int nth = std::max( par.threads, 1 );
                if ( SI( memo_saute.size() ) < nth ) memo_saute.resize( nth );
                for ( auto &v : memo_saute ) if ( SI( v.size() ) != n ) v.assign( n, 0 );
                std::atomic<SI> coupees{ 0 };
                parallel_for( n, par, [ & ]( SI k, int t ) {
                    Cell cel;
                    const d2::SI32 i = ids[ k ];
                    const d2::SI32 *p = memo_pre.data() + memo_beg[ k ];
                    const int np = int( memo_beg[ k + 1 ] - memo_beg[ k ] );
                    unsigned char *sa = memo_saute[ t ].data();
                    for ( int q = 0; q < np; ++q ) sa[ p[ q ] ] = 1;
                    int prop = 0, boites = 0, coup = 0;
                    const bool ok = cellule_memo( k, cel, p, np, sa, prop, boites, coup );
                    for ( int q = 0; q < np; ++q ) sa[ p[ q ] ] = 0;
                    coupees += coup;
                    if ( ! ok ) { ++deborde; return; }
                    res[ i ] = mes( cel, [ & ]( d2::SI32 j, TF m ) { facette( t, i, j, m ); }, SI( i ) );
                } );
                memo_coupees = coupees.load();
                return deborde.load();
            }
        }
        parallel_for( n, par, [ & ]( SI k, int t ) {
            Cell cel;
            const d2::SI32 i = ids[ k ];
            if ( ! cellule( k, cel ) ) { ++deborde; return; }
            res[ i ] = mes( cel, [ & ]( d2::SI32 j, TF m ) { facette( t, i, j, m ); }, SI( i ) );
        } );
        return deborde.load();
    }

private:
    void copie_poids() {
        w.resize( n );
        for ( SI k = 0; k < n; ++k ) w[ k ] = TK( arbre.seed_w( k ) );
    }

    template<bool POIDS>
    bool cellule_memo_( SI k, Cell &cel, const d2::SI32 *pre, int npre, const unsigned char *saute, int &prop, int &boites, int &coupees, bool parcours ) const {
        if constexpr ( D == 3 ) {
            using F = d3::FournisseurBsp3<TK,POIDS,8,true>;
            F f( &arbre, c[ 0 ][ k ], c[ 1 ][ k ], c[ 2 ][ k ], POIDS ? w[ k ] : TK( 0 ), ids[ k ] );
            f.pre = pre; f.npre = npre; f.saute = saute; f.parcours = parcours;
            typename F::Local loc;
            const int r = d3::moteur( &f, &cel, &loc );
            prop = loc.nb_prop; boites = loc.nb_boites; coupees = loc.nb_coupees;
            return r == 0;
        } else
            return false;
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

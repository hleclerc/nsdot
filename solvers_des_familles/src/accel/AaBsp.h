#pragma once

// =====================================================================================
// LE BSP ALIGNE SUR LES AXES : coupes MEDIANES sur l'axe le plus long, une poignee de germes par
// feuille. La reference du banc, en 2D comme en 3D.
//
// CE QU'IL EST : une PERMUTATION des germes ( `order` ), leurs positions et poids RANGES dans cet
// ordre ( une feuille se lit d'un seul tenant ), et des noeuds en PREORDRE -- le fils gauche est
// en `n + 1`, juste a cote, le droit est stocke. Chaque noeud porte sa boite et le majorant affine
// des poids de son sous-arbre.
//
// CE QU'IL N'EST PAS : un parcours. Le parcours est chez le fournisseur ( `cell/FournisseurBsp*.h` ),
// qui le SUSPEND entre deux demandes de la cellule.
//
// LE REGIME DE NEWTON : les poids changent, pas les positions. `refresh_weights` refait les
// majorants sans toucher a la permutation ni aux boites -- pas de `nth_element` a repayer.
// =====================================================================================

#include "accel/WeightMajorant.h"
#include "util/parallel.h"
#include <algorithm>
#include <vector>

namespace sf {

template<int D>
struct AaBspT {
    static constexpr int dim = D;

    struct Node {
        TF       lo[ D ], hi[ D ];  ///< la boite englobante du sous-arbre
        WMajT<D> wm;                ///< `w( y ) <= wm.a . y + wm.b` sur le sous-arbre
        SI       beg, end;          ///< sa tranche dans `order` / `p`
        SI       right;             ///< le fils droit ; `< 0` dit FEUILLE ( le gauche est `n + 1` )
    };
    static_assert( D != 2 || sizeof( Node ) <= 64, "en 2D un noeud tient dans une ligne de cache" );

    std::vector<Node> nodes;    ///< en PREORDRE
    std::vector<SI>   order;    ///< rang dans l'arbre -> identifiant du germe
    std::vector<TF>   p[ D ];   ///< les positions PERMUTEES
    std::vector<TF>   pw;       ///< les poids permutes ( vide = cas euclidien )
    SI                leaf_size = 10;

    TF seed_c( SI k, int d ) const { return p[ d ][ k ]; }
    TF seed_w( SI k ) const { return pw.empty() ? TF( 0 ) : pw[ k ]; }
    SI seed_id( SI k ) const { return order[ k ]; }
    SI nb_seeds() const { return SI( order.size() ); }
    bool has_weights() const { return ! pw.empty(); }

    /// `P[ d ][ i ]` les positions, `W` les poids ou `nullptr` ( euclidien ), dans l'ordre de
    /// l'appelant.
    void build( const TF *const *P, const TF *W, SI n, SI leaf ) {
        leaf_size = leaf;
        order.resize( n );
        for ( SI i = 0; i < n; ++i )
            order[ i ] = i;
        for ( int d = 0; d < D; ++d )
            p[ d ].resize( n );
        pw.clear();
        if ( W )
            pw.resize( n );

        nodes.clear();
        nodes.reserve( 2 * ( n / std::max<SI>( leaf, 1 ) + 1 ) );
        _build( P, W, 0, n );

        for ( SI k = 0; k < n; ++k ) {
            for ( int d = 0; d < D; ++d )
                p[ d ][ k ] = P[ d ][ order[ k ] ];
            if ( W )
                pw[ k ] = W[ order[ k ] ];
        }
    }

    /// DES POIDS NEUFS sur le meme arbre : `W` dans l'ordre de l'appelant. Seuls les majorants
    /// bougent, un par noeud, en parallele.
    void refresh_weights( const TF *W, const Parallel &par ) {
        const SI n = nb_seeds();
        pw.resize( n );
        for ( SI k = 0; k < n; ++k )
            pw[ k ] = W[ order[ k ] ];
        parallel_for( SI( nodes.size() ), par, [ & ]( SI i, int ) {
            Node &nd = nodes[ i ];
            nd.wm = weight_majorant<D>( nd.beg, nd.end, [ & ]( SI k, Vec<D> &q, TF &w ) {
                for ( int d = 0; d < D; ++d ) q[ d ] = p[ d ][ k ];
                w = pw[ k ];
            } );
        } );
    }

private:
    /// Recursif, donc PREORDRE par construction.
    SI _build( const TF *const *P, const TF *W, SI beg, SI end ) {
        const SI me = SI( nodes.size() );
        nodes.push_back( Node{} );

        TF lo[ D ], hi[ D ];
        for ( int d = 0; d < D; ++d )
            lo[ d ] = hi[ d ] = P[ d ][ order[ beg ] ];
        for ( SI k = beg + 1; k < end; ++k ) {
            const SI i = order[ k ];
            for ( int d = 0; d < D; ++d ) {
                const TF v = P[ d ][ i ];
                lo[ d ] = v < lo[ d ] ? v : lo[ d ];
                hi[ d ] = v > hi[ d ] ? v : hi[ d ];
            }
        }
        for ( int d = 0; d < D; ++d ) { nodes[ me ].lo[ d ] = lo[ d ]; nodes[ me ].hi[ d ] = hi[ d ]; }
        nodes[ me ].beg = beg;
        nodes[ me ].end = end;

        if ( W )
            nodes[ me ].wm = weight_majorant<D>( beg, end, [ & ]( SI k, Vec<D> &y, TF &w ) {
                const SI i = order[ k ];
                for ( int d = 0; d < D; ++d ) y[ d ] = P[ d ][ i ];
                w = W[ i ];
            } );

        int ax = 0;
        for ( int d = 1; d < D; ++d )
            if ( hi[ d ] - lo[ d ] > hi[ ax ] - lo[ ax ] ) ax = d;
        const TF span = hi[ ax ] - lo[ ax ];

        // `span <= 0` : tous les germes au meme endroit, aucune coupe ne les separerait.
        if ( end - beg <= leaf_size || ! ( span > 0 ) ) {
            nodes[ me ].right = -1;
            return me;
        }

        // la MEDIANE et non le milieu de la boite : la profondeur est bornee par `log2( n / leaf )`
        // quelle que soit la distribution.
        const SI mid = beg + ( end - beg ) / 2;
        const TF *C = P[ ax ];
        std::nth_element( order.begin() + beg, order.begin() + mid, order.begin() + end,
                          [ & ]( SI a, SI b ) { return C[ a ] < C[ b ]; } );

        _build( P, W, beg, mid );                       // == me + 1
        nodes[ me ].right = _build( P, W, mid, end );
        return me;
    }
};

} // namespace sf

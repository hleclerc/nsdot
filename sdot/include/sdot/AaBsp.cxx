#pragma once

#include <loom/support/common_macros.h>
#include <loom/support/containers/Vector.h>
#include "AaBsp.h"

#define UTP SDOT_TEMPLATE_DECL_FOR_AaBsp
#define DTP AaBsp<SDOT_TEMPLATE_ARGS_FOR_AaBsp>

namespace sdot {

UTP auto DTP::nearness( const auto &from, SI n ) const {
    TF res = 0;
    for ( PI d = 0; d < ct_dim; ++d ) {
        const TF lo = TF( node_box( n, 0, d ) );
        const TF hi = TF( node_box( n, 1, d ) );
        const TF p  = from[ d ];
        const TF e  = p < lo ? lo - p : ( p > hi ? p - hi : TF( 0 ) );
        res += e * e;
    }
    return res;
}

UTP void DTP::for_each_candidate( const auto &from, SI i0, auto &&scratch, auto &&may_cut, auto &&cut_with ) const {
    // Les noeuds sont ranges en PREORDRE (voir `AaBsp.py::_preorder_of_heap`) : le fils gauche est
    // JUSTE A COTE (`n+1`) et le droit a `2^(h-1)` noeuds, ou `h` est la hauteur du sous-arbre.
    // Les sauts retrecissent donc en descendant, la ou la numerotation en tas les faisait doubler
    // -- et c'est le bas de l'arbre qui est le plus visite et le plus gros.
    //
    // `h` voyage SUR LA PILE, empaquete avec l'indice : la pile est deja la, elle est en L1, et
    // c'est six bits (la profondeur est un `ceil( log2( n / leaf ) ) + 1`, jamais 64). La lire
    // ailleurs aurait voulu dire un tableau de plus par noeud -- exactement ce qu'on cherche a
    // supprimer.
    const SI nb_nodes = SI( node_begin.shape( 0 ) );
    SI depth = 0;
    for ( SI m = nb_nodes; m; m >>= 1 )         // `nb_nodes == 2^depth - 1`
        ++depth;

    SI top = 0;
    scratch( top++ ) = depth;                   // la racine : indice 0, hauteur `depth`

    while ( top > 0 ) {
        const SI e = SI( scratch( --top ) );
        const SI n = e >> 6;
        const SI h = e & 63;

        // an EMPTY slot: the right child of a node that had nothing left to split and passed its
        // whole slice to the left one (see `AaBsp.py::_build`). Two integer loads answer it, where
        // `may_cut` would have swept the cell's vertices against a box that means nothing.
        const SI beg = SI( node_begin( n ) );
        const SI end = SI( node_end( n ) );
        if ( beg >= end )
            continue;

        // une seule lecture CONTIGUE pour la boite : `lo` et `hi` sont voisins en memoire (voir
        // `AaBsp.py::node_box`), donc une ligne de cache la porte entiere en 2D.
        const auto lo = Vector<TF,ct_dim>::with_func( [&]( PI d ) { return TF( node_box( n, 0, d ) ); } );
        const auto hi = Vector<TF,ct_dim>::with_func( [&]( PI d ) { return TF( node_box( n, 1, d ) ); } );

        // no weights at all -> no majorant to read: the two tensors are `NoneTensor`, the branch
        // goes at COMPILE time, and the caller's test degenerates to the plain distance one.
        auto wa = Vector<TF,ct_dim>::zeros();
        TF wb = 0;
        if constexpr ( CT_VALUE( node_wa.is_valid() ) ) {
            wa = Vector<TF,ct_dim>::with_func( [&]( PI d ) { return TF( node_wa( n, d ) ); } );
            wb = TF( node_wb( n ) );
        }

        // the whole point of the tree: a subtree that cannot reach the cell is not descended into.
        // Tested on POP and not on push, so it is answered against the cell as it is NOW -- every
        // cut made since this node was pushed has made the answer more likely to be "no".
        if ( ! may_cut( lo, hi, wa, wb ) )
            continue;

        // les fils sont DEDUITS, pas lus : l'arbre est binaire PARFAIT, donc leur place se calcule
        // (voir plus haut), et « suis-je une feuille » se lit sur la hauteur. Deux tableaux de
        // moins a lire par noeud visite.
        const SI l = n + 1;
        if ( h <= 1 ) {                             // une feuille : plus rien en dessous
            // Le candidat est rendu par son indice ET par son RANG dans le regroupement. Le rang
            // est ce qui permet a l'appelant de lire la position dans une copie triee CONTIGUE
            // (voir `PowerDiagram::sorted_positions`) : les germes d'une feuille sont voisins ici,
            // alors qu'ils sont epars dans le tableau d'origine. Ce n'est PAS a nous de fournir la
            // position : elle appartient au diagramme, et un accelerateur qui la fournirait
            // rendrait les positions du diagramme non determinantes, donc ses derivees fausses.
            for ( SI k = beg; k < end; ++k ) {
                const SI i1 = SI( seed_indices( k ) );
                if ( i1 != i0 && ! cut_with( i1, k ) )
                    return;
            }
            continue;
        }

        // the nearer child is pushed LAST, so it is popped FIRST.
        const SI r = n + ( SI( 1 ) << ( h - 1 ) );
        const SI hc = h - 1;
        if ( nearness( from, l ) <= nearness( from, r ) ) {
            scratch( top++ ) = ( r << 6 ) | hc;
            scratch( top++ ) = ( l << 6 ) | hc;
        } else {
            scratch( top++ ) = ( l << 6 ) | hc;
            scratch( top++ ) = ( r << 6 ) | hc;
        }
    }
}

#undef UTP
#undef DTP

}

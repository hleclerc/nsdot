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
        const TF lo = TF( node_lo( n, d ) );
        const TF hi = TF( node_hi( n, d ) );
        const TF p  = from[ d ];
        const TF e  = p < lo ? lo - p : ( p > hi ? p - hi : TF( 0 ) );
        res += e * e;
    }
    return res;
}

UTP void DTP::for_each_candidate( const auto &from, SI i0, auto &&scratch, auto &&may_cut, auto &&cut_with ) const {
    SI top = 0;
    scratch( top++ ) = 0;                       // the root -- the nodes are numbered in a HEAP

    while ( top > 0 ) {
        const SI n = SI( scratch( --top ) );

        // an EMPTY slot: the right child of a node that had nothing left to split and passed its
        // whole slice to the left one (see `AaBsp.py::_build`). Two integer loads answer it, where
        // `may_cut` would have swept the cell's vertices against a box that means nothing.
        const SI beg = SI( node_begin( n ) );
        const SI end = SI( node_end( n ) );
        if ( beg >= end )
            continue;

        const auto lo = Vector<TF,ct_dim>::with_func( [&]( PI d ) { return TF( node_lo( n, d ) ); } );
        const auto hi = Vector<TF,ct_dim>::with_func( [&]( PI d ) { return TF( node_hi( n, d ) ); } );

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

        const SI l = SI( node_left( n ) );
        if ( l < 0 ) {                          // a leaf: `node_left < 0` says so (see `AaBsp.py`)
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
        const SI r = SI( node_right( n ) );
        if ( nearness( from, l ) <= nearness( from, r ) ) {
            scratch( top++ ) = r;
            scratch( top++ ) = l;
        } else {
            scratch( top++ ) = l;
            scratch( top++ ) = r;
        }
    }
}

#undef UTP
#undef DTP

}

#pragma once

#include "Cell.h"

#define UTP SDOT_TEMPLATE_DECL_FOR_Cell
#define DTP Cell<SDOT_TEMPLATE_ARGS_FOR_Cell>

namespace sdot {

UTP void DTP::init_as_aligned_simplex( SI cut_id ) {
    // `nb_dims` is a compile-time `Ct` member, but a member is read through `this`, and `this` is
    // never a constant expression -- so bare `if constexpr ( nb_dims > 2 )` cannot compile (it is
    // `this->nb_dims`). Shadow it with its value, reached through the TYPE (`decltype` does not
    // touch `this`): from here `nb_dims` is a plain `constexpr` int, usable in `if constexpr`.
    constexpr auto nb_dims = decltype( this->nb_dims )::value;

    is_fully_bounded = cut_id != CellBoundary::INFINITE;
    nb_vertices = nb_dims + 1;
    if constexpr ( nb_dims > 2 )
        nb_edges = ( nb_dims + 1 ) * nb_dims / 2;
    nb_cuts = nb_dims + 1;

    // vertex_positions
    for( PI n = 0; n < nb_vertices; ++n )
        for( PI d = 0; d < nb_dims; ++d )
            vertex_positions( num_vertex = n, dim = d ) = ( d + 1 == n );

    // vertex_inds
    if constexpr ( nb_dims > 2 )
        for( PI num_vertex = 0; num_vertex < nb_vertices; ++num_vertex )
            for( PI d = 0; d < nb_dims; ++d )
                vertex_indices( num_vertex, d ) = d + ( d >= num_vertex );

    // edge_indices
    if constexpr ( nb_dims > 2 ) {
        for ( PI a = 0, o = 0; a < nb_vertices; ++a ) {
            for ( PI b = a + 1; b < nb_vertices; ++b ) {
                if ( a != b ) {
                    edge_indices( num_edge = o, ein = 0 ) = a;
                    edge_indices( num_edge = o, ein = 1 ) = b;
                    for( PI d = 0; d < nb_dims - 1; ++d )
                        edge_indices( o, ein = 2 + d ) = d + ( d >= a ) + ( d >= b - 1 );
                    ++o;
                }
            }
        }
    }

    // cut_planes
    if constexpr ( nb_dims > 2 ) {
        for( PI n = 0; n < nb_dims; ++n ) {
            for( PI d = 0; d < nb_dims; ++d )
                cut_vectors( num_cut = n, dim = d ) = - ( d == n );
            cut_offsets( num_cut = n ) = 0;
        }
        for( PI d = 0; d < nb_dims; ++d )
            cut_vectors( num_cut = d, dim = d ) = 1;
        cut_offsets( num_cut = dim ) = 1;
    } else {
        cut_vectors( num_cut = 0, dim = 0 ) =  0; cut_vectors( num_cut = 0, dim = 1 ) = -1; cut_offsets( num_cut = 0 ) = 0;
        cut_vectors( num_cut = 1, dim = 0 ) = +1; cut_vectors( num_cut = 1, dim = 1 ) = +1; cut_offsets( num_cut = 1 ) = 1;
        cut_vectors( num_cut = 2, dim = 0 ) = -1; cut_vectors( num_cut = 2, dim = 1 ) =  0; cut_offsets( num_cut = 2 ) = 0;
    }

    // cut_ids
    for( PI n = 0; n < nb_cuts; ++n )
        cut_ids( num_cut = n ) = cut_id;
}

UTP void DTP::init_as_hypercube( auto &&origin, auto &&axes, SI cut_id ) {
    INFO( origin );
    INFO( axes );
}

UTP void DTP::init_as_unbounded() {
    init_as_aligned_simplex( CellBoundary::INFINITE );
}

}

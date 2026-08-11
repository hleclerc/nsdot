#pragma once

// the axes this body names, as declared symbols (autocompletion, standalone compile) instead of
// globals the generated source happens to define around us. Written to the build include tree.
#include <sdot/generated/aggregates/Cell.h>
#include "support/common_macros.h"
#include "Cell/CellBoundary.h"

namespace sdot {

SDOT_TEMPLATE_DECL_FOR_Cell
struct Cell {
    SDOT_ATTRIBUTES_OF_Cell

    static constexpr int ct_dim = DECAYED_TYPE_OF( nb_dims )::value;
    using TF = DECAYED_TYPE_OF( vertex_positions )::TF;

    void init_as_aligned_simplex( SI cut_id );

    void init_as_hypercube_bwd  ( auto &&origin, auto &&axes, auto &&grad_cell, auto &&grad_for_origin, auto &&grad_for_axes ) const;
    void init_as_hypercube      ( auto &&origin, auto &&axes, SI cut_id = CellBoundary::BOUNDARY );

    void init_as_unbounded      ();

    void measure_bwd            ( auto &&res, auto &&item_map, auto &&nb_map_items, auto &&grad_res, auto &&grad_vertex_positions ) const;
    void measure                ( auto &&res, auto &&item_map, auto &&nb_map_items ) const;
};

}

#include "Cell.cxx"

#pragma once

// the axes this body names, as declared symbols (autocompletion, standalone compile) instead of
// globals the generated source happens to define around us. Written to the build include tree.
#include "sdot/generated/aggregates/Cell.h"
#include "sdot/support/common_macros.h"

namespace sdot {

SDOT_TEMPLATE_DECL_FOR_Cell
struct Cell {
    SDOT_ATTRIBUTES_OF_Cell

    static constexpr int ct_dim = DECAYED_TYPE_OF( nb_dims )::value;

    void init_as_aligned_simplex( SI cut_id );
    void init_unbounded         ();
};

}

#include "Cell.cxx"

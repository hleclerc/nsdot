#pragma once

// the axes this body names, as declared symbols (autocompletion, standalone compile) instead of
// globals the generated source happens to define around us. Written to the build include tree.
#include <sdot/generated/aggregates/Voronoi.h>
#include <loom/support/common_macros.h>
#include "Cell/CellBoundary.h"

namespace sdot {

// Euclidean Voronoi diagram of `positions`, clipped to the convex domain `bnd_directions . x <=
// bnd_offsets` (absent -> the cells that reach infinity stay infinite, and measure as such).
//
// There is NO diagram here: not a vertex, not a facet, not an adjacency. The class holds its
// SEEDS and rebuilds whatever is asked of it, cell by cell, inside the kernel -- so what a query
// costs in memory is what a WORK-ITEM needs to hold one cell, times the number of work-items,
// and never a function of the number of seeds. A cell exists between the moment it is built and
// the moment the answer is read off it (`measures`: a volume written straight into its slot);
// nothing of it outlives that.
//
// Hence the two cells per work-item rather than one: `Cell::cut` writes its result into a
// SEPARATE cell (see `Cell.h`), so a sequence of cuts PING-PONGS between the two, and which one
// holds the cell at the end is a parity the builder reports (`make_cell`).
SDOT_TEMPLATE_DECL_FOR_Voronoi
struct Voronoi {
    SDOT_ATTRIBUTES_OF_Voronoi

    static constexpr int ct_dim = DECAYED_TYPE_OF( nb_dims )::value;
    using TF = DECAYED_TYPE_OF( positions )::TF;

    auto point         ( SI i ) const;                          ///< seed `i`, as a point

    // One cut, applied to whichever of the two work cells currently holds the geometry, the
    // result landing in the other -- `in_0` is that parity, and this is what flips it. Answers
    // whether the result fitted (see `Cell::cut`).
    bool cut_by        ( bool &in_0, auto &&ws_0, auto &&ws_1, auto &&corr, const auto &dir, TF off, SI cut_id ) const;
    void make_empty    ( auto &&cell ) const;                   ///< a well-defined nothing, for the bail-outs

    // The cell of seed `i0`, built from scratch into `ws_0`/`ws_1`; returns `true` when it ends up
    // in `ws_0`. The domain is cut FIRST on purpose: until the cell is bounded every cut pays for
    // pushing the stand-in simplex out (`Cell::growth_for_cut`), and a bounded one pays nothing.
    bool make_cell     ( SI i0, auto &&ws_0, auto &&ws_1, auto &&corr ) const;

    void measure_into  ( auto &&res, auto &&cell, auto &&facet_apex ) const;

    // The cell of seed `i`, KEPT: built in the two work cells like any other, then copied out into
    // `res`. This is the one query that does not collapse a cell to a number, so it is also the one
    // whose memory is a function of the number of seeds -- which is what DISPLAY is: every cell has
    // to exist at once for a picture to be drawn of it. One work-item per seed here (no strided
    // loop), so the work cells cost twice the output rather than a fixed per-work-item budget.
    void build_cell    ( SI i, auto &&res, auto &&ws_0, auto &&ws_1, auto &&corr ) const;

    // `res( i )` = the measure of cell `i`, for the seeds this work-item is in charge of. The
    // strided loop is the point: `nb_threads` work-items share `nb_points` cells, so the two work
    // cells (and `corr` / `facet_apex`) are reused from one seed to the next and memory is sized
    // on CONCURRENCY, not on the number of seeds.
    void measures      ( auto &&res, auto &&ws_0, auto &&ws_1, auto &&corr, auto &&facet_apex,
                         SI thread_index, SI nb_threads ) const;
};

}

#include "Voronoi.cxx"

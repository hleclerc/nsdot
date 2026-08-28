#pragma once

// the axes this body names, as declared symbols (autocompletion, standalone compile) instead of
// globals the generated source happens to define around us. Written to the build include tree.
#include <sdot/generated/aggregates/PowerDiagram.h>
#include <loom/support/common_macros.h>
#include "Cell/CellBoundary.h"
#include "EverySeed.h"

namespace sdot {

// Power (Laguerre) diagram of the weighted seeds `positions` / `weights`, clipped to the convex
// domain `bnd_directions . x <= bnd_offsets` (absent -> the cells that reach infinity stay
// infinite, and measure as such).
//
// The cell of seed `i` is where its POWER DISTANCE wins: `|x - d_i|^2 - w_i <= |x - d_j|^2 - w_j`
// for every other `j`. Expanded, the quadratic term cancels and what is left is a half-space,
// which is why a power diagram costs exactly what a Voronoi one costs -- one plane per rival, same
// clip. `weights` ABSENT is the Euclidean case: only DIFFERENCES of weights reach the planes, so
// "all equal" and "none at all" are the same diagram, and the absent one is a `NoneTensor` whose
// term the compiler removes rather than a buffer of zeros to read (see `make_cell`).
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
SDOT_TEMPLATE_DECL_FOR_PowerDiagram
struct PowerDiagram {
    SDOT_ATTRIBUTES_OF_PowerDiagram

    static constexpr int ct_dim = DECAYED_TYPE_OF( nb_dims )::value;
    using TF = DECAYED_TYPE_OF( positions )::TF;

    auto point         ( SI i ) const;                          ///< seed `i`, as a point
    auto weight        ( SI i ) const;                          ///< seed `i`'s weight, or 0 when there are none

    // ---- WHICH seeds are worth trying ------------------------------------------------------------
    // `make_cell` does not enumerate the seeds itself: it asks an ACCELERATOR to, and answers the
    // one question an accelerator cannot ("could anything in this region still cut what is left of
    // the cell?"). That split is the whole interface -- the accelerator owns the WALK, we own the
    // GEOMETRY -- and it is what lets a tree, a grid, or nothing at all be dropped in without a
    // second cell builder. The concept it has to implement is written out in `SpatialAccelerator.py`.

    /// The accelerator that accelerates nothing: every seed, in index order. What `make_cell` gets
    /// when the caller supplied none, so that "no acceleration" is an ordinary case of the same
    /// code and not a second implementation of it.
    EverySeed every_seed() const { return EverySeed{ SI( nb_points ) }; }

    /// The prune test, and the ONLY thing the accelerator asks of us: could a seed sitting anywhere
    /// in the box `[ lo, hi ]`, of weight at most `wa . y + wb`, still cut `cell`?
    ///
    /// Exact, given those two bounds. A seed `y` of weight `w` cuts the cell iff SOME point of it
    /// is closer to `y` than to `p0` in power distance; that comparison loses its `|x|^2` (the very
    /// reason a power diagram is made of half-spaces), so it is LINEAR in the point and its extreme
    /// sits at a VERTEX. Sweeping the vertices is therefore not a heuristic -- it answers the
    /// question -- and what makes the answer conservative rather than exact is only the two bounds
    /// the node hands us.
    ///
    /// Relaxing the seed to the whole box is where the affine majorant pays for itself:
    /// `min_y ( |p - y|^2 - wa . y )` is SEPARABLE per axis, its free minimum is at `y = p + wa/2`,
    /// so one clamp per axis gives it exactly. A constant majorant is the same code with `wa = 0`.
    bool cell_may_be_cut( const auto &cell, const auto &p0, TF w0,
                          const auto &lo, const auto &hi, const auto &wa, TF wb ) const;

    // One cut, applied to whichever of the two work cells currently holds the geometry, the
    // result landing in the other -- `in_0` is that parity, and this is what flips it. Answers
    // whether the result fitted (see `Cell::cut`).
    bool cut_by        ( bool &in_0, auto &&ws_0, auto &&ws_1, auto &&corr, const auto &dir, TF off, SI cut_id ) const;
    void make_empty    ( auto &&cell ) const;                   ///< a well-defined nothing, for the bail-outs

    // The cell of seed `i0`, built from scratch into `ws_0`/`ws_1`; returns `true` when it ends up
    // in `ws_0`. The domain is cut FIRST on purpose: until the cell is bounded every cut pays for
    // pushing the stand-in simplex out (`Cell::growth_for_cut`), and a bounded one pays nothing.
    bool make_cell     ( SI i0, auto &&ws_0, auto &&ws_1, auto &&corr, const auto &acc, auto &&acc_ws ) const;

    void measure_into  ( auto &&res, auto &&cell, auto &&facet_apex ) const;

    // The cell of seed `i`, KEPT: built in the two work cells like any other, then copied out into
    // `res`. This is the one query that does not collapse a cell to a number, so it is also the one
    // whose memory is a function of the number of seeds -- which is what DISPLAY is: every cell has
    // to exist at once for a picture to be drawn of it. One work-item per seed here (no strided
    // loop), so the work cells cost twice the output rather than a fixed per-work-item budget.
    void build_cell    ( SI i, auto &&res, auto &&ws_0, auto &&ws_1, auto &&corr, const auto &acc, auto &&acc_ws ) const;

    // `res( i )` = the measure of cell `i`, for the seeds this work-item is in charge of. The
    // strided loop is the point: `nb_threads` work-items share `nb_points` cells, so the two work
    // cells (and `corr` / `facet_apex`) are reused from one seed to the next and memory is sized
    // on CONCURRENCY, not on the number of seeds.
    void measures      ( auto &&res, auto &&ws_0, auto &&ws_1, auto &&corr, auto &&facet_apex,
                         const auto &acc, auto &&acc_ws, SI thread_index, SI nb_threads ) const;

    // ---- the adjoint of `measures` ---------------------------------------------------------------
    // The chain is `m_i <- vertices <- planes <- seeds`, and each arrow is a closed form:
    //
    //   * `Cell::measure_bwd` already answers the first (shoelace in 2D, the simplex fan above);
    //   * a vertex is the CORNER of its `ct_dim` cuts, so it solves `A x = b` with the cut
    //     directions as rows -- one small solve per vertex turns its cotangent into cotangents on
    //     those planes (`scatter_cell_grad`);
    //   * a plane is the weighted bisector of two seeds, which differentiates in two lines.
    //
    // Nothing of the forward is kept: the cell is REBUILT here (`make_cell` again) rather than
    // stored, exactly as the forward refuses to store the diagram -- the backward costs one more
    // build, and still nothing that grows with the number of seeds.
    //
    // What it does NOT answer: the DOMAIN. A cut coming from `bnd_directions` / `bnd_offsets`
    // carries `cut_id == BOUNDARY`, which says "not a seed" and not WHICH boundary, so its share is
    // dropped -- the domain is a constant here. The vertices standing on it are still handled
    // exactly: the solve accounts for all `ct_dim` rows, only the scatter of the boundary rows is
    // skipped, so what does flow to the seeds is right.
    void measures_bwd     ( auto &&res, auto &&grad_res, auto &&grad_positions, auto &&grad_weights,
                            auto &&ws_0, auto &&ws_1, auto &&corr, auto &&facet_apex, auto &&grad_vp,
                            const auto &acc, auto &&acc_ws, SI thread_index, SI nb_threads ) const;

    void measure_bwd_into ( auto &&res, auto &&grad_res, auto &&cell, auto &&facet_apex, auto &&grad_vp ) const;

    // `grad_vp` (a cotangent per vertex of `cell`, as `measure_bwd` left it) -> the seeds. Every
    // other seed the cell touches gets an atomic add; seed `i0`'s own share, which EVERY vertex
    // contributes to, is summed in a register and added once.
    void scatter_cell_grad( SI i0, auto &&cell, auto &&grad_vp, auto &&grad_positions, auto &&grad_weights ) const;
};

}

#include "PowerDiagram.cxx"

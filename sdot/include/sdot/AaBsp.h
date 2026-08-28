#pragma once

// the axes this body names, as declared symbols (autocompletion, standalone compile) instead of
// globals the generated source happens to define around us. Written to the build include tree.
#include <sdot/generated/aggregates/AaBsp.h>
#include <loom/support/common_macros.h>
#include <loom/support/containers/Vector.h>

namespace sdot {

// An AXIS-ALIGNED BSP over the seeds: a binary tree of boxes, a handful of seeds per leaf. See
// `AaBsp.py` for what a node carries and why the weight majorant is AFFINE rather than constant.
//
// It implements the accelerator concept (`SpatialAccelerator.py`): one method, `for_each_candidate`,
// which walks the tree and asks the caller -- who is the one holding a cell -- whether a region can
// still be reached. Nothing here knows what a cell is.
SDOT_TEMPLATE_DECL_FOR_AaBsp
struct AaBsp {
    SDOT_ATTRIBUTES_OF_AaBsp

    static constexpr int ct_dim = DECAYED_TYPE_OF( nb_dims )::value;
    using TF = DECAYED_TYPE_OF( node_lo )::TF;

    // A DEPTH-FIRST walk, nearest child first, over an explicit stack (`scratch`): no recursion in
    // a SYCL kernel, and -- the reason this beats the priority queue the same walk usually gets --
    // a bounded one. Each level pops one node and pushes two, so the stack never exceeds the depth
    // of the tree, which the build measured (`AaBsp.py::thread_scratch`). A best-first front would
    // be bounded by the number of nodes VISITED, i.e. by nothing one can allocate up front.
    //
    // Nearest child first is what makes the plain stack enough: the first leaf reached is the one
    // holding `from` itself, so the cell collapses onto its immediate neighbours before anything
    // else is looked at, and the rest of the tree is pruned against an already small cell.
    void for_each_candidate( const auto &from, SI i0, auto &&scratch, auto &&may_cut, auto &&cut_with ) const;

    // Squared distance from `from` to node `n`'s box -- 0 inside it. Only an ORDERING key, so it
    // stays this cheap on purpose: the real test (`may_cut`) sweeps the cell's vertices, and
    // paying for it on both children just to decide which to look at first would double the walk.
    auto nearness( const auto &from, SI n ) const;
};

}

#include "AaBsp.cxx"

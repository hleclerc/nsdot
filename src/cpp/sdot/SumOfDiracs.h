#pragma once

#include <sdot/generated/aggregates/SumOfDiracs.h>
#include "support/common_macros.h"

namespace sdot {

SDOT_TEMPLATE_DECL_FOR_SumOfDiracs
struct SumOfDiracs {
    SDOT_ATTRIBUTES_OF_SumOfDiracs

    SCInt ct_dim        = DECAYED_TYPE_OF( nb_dims )::value;
    using TF            = DECAYED_TYPE_OF( positions )::TF;

    // Uniform 1D-source contract consumed by OtPlan1d (polymorphic over the source type -- a
    // ProjectedSumOfDiracs answers the same two methods but computes the position on the fly and
    // scatters the gradient back to its 2D points; see [[per-thread-scratch-facility]] cousin work).
    // Here the position is a plain buffer read of dim 0, and the gradient a per-angle buffer write.

    /// This dirac's 1D coordinate (the value OtPlan1d sorts and sweeps on).
    TF position( SI i ) const { return positions( ::num_dirac = i, dim = 0 ); }

    /// Compile-time: is the position gradient wanted for this source? (lets OtPlan1d skip the whole
    /// re-sort + re-sweep of the positions-grad block when it is not). Returns the `is_valid()`
    /// compile-time flag of the member `add_position_grad` would write.
    auto position_grad_wanted( auto &&grad_src ) const { return grad_src.positions.is_valid(); }

    /// Accumulate d cost / d position(i) into the gradient twin `grad_src`. Guarded at compile time:
    /// an unperturbed input reaches us as a NoneTensor (no `operator=`), so the write must vanish.
    /// A per-angle buffer, so each angle writes its OWN slot -- a plain `=`, no cross-angle race.
    void add_position_grad( auto &&grad_src, SI i, TF grad_s ) const {
        if constexpr ( CT_VALUE( grad_src.positions.is_valid() ) )
            grad_src.positions( ::num_dirac = i, dim = 0 ) = grad_s;
    }
};

}

#include "SumOfDiracs.cxx"

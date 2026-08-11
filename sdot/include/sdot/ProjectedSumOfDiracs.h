#pragma once

#include <sdot/generated/aggregates/ProjectedSumOfDiracs.h>
#include "support/common_macros.h"
#include "support/atomic_add.h"

namespace sdot {

SDOT_TEMPLATE_DECL_FOR_ProjectedSumOfDiracs
struct ProjectedSumOfDiracs {
    SDOT_ATTRIBUTES_OF_ProjectedSumOfDiracs

    SCInt ct_dim  = DECAYED_TYPE_OF( nb_dims )::value;    // OT (projected) space: 1
    SCInt ct_proj = DECAYED_TYPE_OF( proj_dims )::value;  // ambient space: 2
    using TF      = DECAYED_TYPE_OF( points )::TF;

    // Same 1D-source contract as SumOfDiracs, consumed polymorphically by OtPlan1d -- but the position
    // is computed ON THE FLY (no materialized [nb_angles, n] projection) and the gradient scatters
    // back to the shared ambient points.

    /// This dirac's 1D coordinate: its ambient point projected onto THIS item's normal.
    TF position( SI i ) const {
        TF s = 0;
        for ( SI d = 0; d < ct_proj; ++d )
            s += TF( points( ::num_dirac = i, proj_dim = d ) ) * TF( normal( proj_dim = d ) );
        return s;
    }

    /// Compile-time: is the position gradient wanted? (here it flows to the ambient `points`).
    auto position_grad_wanted( auto &&grad_src ) const { return grad_src.points.is_valid(); }

    /// d cost / d position(i) flows to the ambient point: grad_points(i,d) += grad_s * normal(d). The
    /// points gradient is SHARED across the batch (one point, every angle contributes) -> ATOMIC add.
    /// Compile-time no-op when the points gradient is not wanted (a NoneTensor twin, no `ref()`).
    void add_position_grad( auto &&grad_src, SI i, TF grad_s ) const {
        if constexpr ( CT_VALUE( grad_src.points.is_valid() ) )
            for ( SI d = 0; d < ct_proj; ++d )
                atomic_add( grad_src.points( ::num_dirac = i, proj_dim = d ).ref(),
                            TF( grad_s * TF( normal( proj_dim = d ) ) ) );
    }

    /// Clear the whole `grad_points` buffer -- called ONCE, as a pre-pass BEFORE `add_position_grad`'s
    /// atomic adds (see `OtPlan1d.py`'s `bwd_setup_code`). Needed because the points gradient
    /// ACCUMULATES ACROSS EVERY ANGLE sharing this buffer (unlike a per-angle value): a fresh Jax/XLA
    /// FFI output buffer is NOT guaranteed to start zeroed, so without this the atomic adds land on
    /// whatever device memory happened to be there before. Routed through `Tensor::fill_with( queue,
    /// ... )` (-> `run_parallel`) rather than a hand-rolled kernel submission, so it goes through the
    /// same, already-proven device-kernel launch path as every other kernel here. Compile-time
    /// no-op when the points gradient is not wanted.
    void zero_position_grad( auto &&queue, auto &&grad_src ) const {
        if constexpr ( CT_VALUE( grad_src.points.is_valid() ) )
            grad_src.points.fill_with( queue, TF( 0 ) );
    }
};

}

#include "ProjectedSumOfDiracs.cxx"

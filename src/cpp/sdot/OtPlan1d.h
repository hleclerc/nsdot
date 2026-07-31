#pragma once

#include <sdot/generated/aggregates/OtPlan1d.h>
#include "support/common_macros.h"

namespace sdot {

SDOT_TEMPLATE_DECL_FOR_OtPlan1d
struct OtPlan1d {
    SDOT_ATTRIBUTES_OF_OtPlan1d

    SCInt ct_dim        = DECAYED_TYPE_OF( nb_dims )::value;
    using TF            = DECAYED_TYPE_OF( cost )::TF;

    // `local_index`/`local_size`/`group`/`local_scratch`: the work-group cooperating on ONE angle
    // (see run_parallel.h/FfiCodeParallel's `group_size` docstring) -- `group_size == 1` degenerates
    // exactly to the single-work-item algorithm these generalize.
    void  sort_diracs( auto &&sorted_indices, auto &&radix_tmp, auto &&sorted_pos,
                        int local_index, int local_size, auto &&group, auto &&local_scratch ) const;
    void  update_outputs( auto &&sorted_indices, auto &&radix_tmp, auto &&sorted_pos,
                           int local_index, int local_size, auto &&group, auto &&local_scratch );
    void  update_outputs_bwd( auto &&grad_plan, auto &&sorted_indices, auto &&radix_tmp, auto &&sorted_pos,
                               int local_index, int local_size, auto &&group, auto &&local_scratch ) const;
};

}

#include "OtPlan1d.cxx"

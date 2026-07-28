#pragma once

#include <sdot/generated/aggregates/OtPlan1d.h>
#include "support/common_macros.h"

namespace sdot {

SDOT_TEMPLATE_DECL_FOR_OtPlan1d
struct OtPlan1d {
    SDOT_ATTRIBUTES_OF_OtPlan1d

    SCInt ct_dim        = DECAYED_TYPE_OF( nb_dims )::value;
    using TF            = DECAYED_TYPE_OF( cost )::TF;

    void  update_outputs( auto &&sorted_indices, auto &&radix_tmp );
    void  update_outputs_bwd( auto &&grad_plan, auto &&sorted_indices ) const;
};

}

#include "OtPlan1d.cxx"

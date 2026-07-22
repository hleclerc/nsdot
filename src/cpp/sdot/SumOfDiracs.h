#pragma once

#include <sdot/generated/aggregates/SumOfDiracs.h>
#include "support/common_macros.h"

namespace sdot {

SDOT_TEMPLATE_DECL_FOR_SumOfDiracs
struct SumOfDiracs {
    SDOT_ATTRIBUTES_OF_SumOfDiracs

    SCInt ct_dim        = DECAYED_TYPE_OF( nb_dims )::value;
    using TF            = DECAYED_TYPE_OF( positions )::TF;
};

}

#include "SumOfDiracs.cxx"

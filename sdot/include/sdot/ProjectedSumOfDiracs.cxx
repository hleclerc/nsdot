#pragma once

#include "ProjectedSumOfDiracs.h"

#define UTP SDOT_TEMPLATE_DECL_FOR_ProjectedSumOfDiracs
#define DTP ProjectedSumOfDiracs<SDOT_TEMPLATE_ARGS_FOR_ProjectedSumOfDiracs>

namespace sdot {

// all methods are inline in the header (small, kernel-side); nothing out-of-line here yet.

}

#undef UTP
#undef DTP

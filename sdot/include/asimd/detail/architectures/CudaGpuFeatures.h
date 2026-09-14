#pragma once

#include <cstddef>
#include <string>
#include <array>

namespace asimd {

namespace features {

struct CudaGpuInfoFeature {
    using                            A2  = std::array<int,2>;
    using                            A3  = std::array<int,3>;

    static std::string               name() { return "CudaProcInfoFeature"; }

    std::size_t                      num = 0;                    /**< num gpu board */

    #define NGIF( TYPE, NAME, INFO ) TYPE NAME;
    #include                         "CudaGpuInfoFeaturesDecl.h"
    #undef NGIF
};


} // namespace features

} // namespace asimd

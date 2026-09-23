#pragma once

#include "GenericFeatures.h"
#include "FeatureSet.h"

namespace asimd {

/**
*/
template<int ptr_size = 8 * sizeof( void * ),class... Features>
struct CudaGpu : FeatureSet<Features...> {
    using                 size_type = typename std::conditional<ptr_size==64,std::uint64_t,std::uint32_t>::type;
    static constexpr bool cuda      = true;

    static std::string    name      () { return "CudaProc<" + std::to_string( ptr_size ) + FeatureSet<Features...>::feature_names() + ">"; }
};

} // namespace asimd

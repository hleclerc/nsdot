#pragma once

#include "GenericFeatures.h"
#include "ArmCpuFeatures.h"
#include "FeatureSet.h"

namespace asimd {

/**
*/
template<int ptr_size_in_bits,class... Features>
struct ArmCpu : FeatureSet<Features...> {
    using                 size_type = typename std::conditional<ptr_size_in_bits==64,std::uint64_t,std::uint32_t>::type;
    static constexpr bool cpu       = true;

    static std::string    name      () { return "Arm<" + std::to_string( ptr_size_in_bits ) + FeatureSet<Features...>::feature_names() + ">"; }
};

} // namespace asimd

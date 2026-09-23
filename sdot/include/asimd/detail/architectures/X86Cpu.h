#pragma once

#include "GenericFeatures.h"
#include "X86CpuFeatures.h"
#include "FeatureSet.h"

namespace asimd {

/**
*/
template<int ptr_size_in_bits,class... Features>
struct X86Cpu : FeatureSet<Features...> {
    using                 size_type = typename std::conditional<ptr_size_in_bits==64,std::uint64_t,std::uint32_t>::type;
    static constexpr bool cpu       = true;

    static std::string    name      () { return "X86<" + std::to_string( ptr_size_in_bits ) + FeatureSet<Features...>::feature_names() + ">"; }
};

} // namespace asimd

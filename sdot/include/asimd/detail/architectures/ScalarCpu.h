#pragma once

#include "GenericFeatures.h"
#include "FeatureSet.h"

namespace asimd {

/**
  THE ARCHITECTURE WITH NO VECTORS. One lane, every operation through the generic forms.

  This is what `NativeCpu` becomes on a target nobody has written a backend for. It is not a
  placeholder: it is what makes a new backend ADDITIVE. Without it the library did not compile
  at all outside x86 and Apple ARM, so any port had to start by making everything build again.
*/
template<int ptr_size_in_bits>
struct ScalarCpu : FeatureSet<> {
    using                 size_type = typename std::conditional<ptr_size_in_bits==64,std::uint64_t,std::uint32_t>::type;
    static constexpr bool cpu       = true;

    static std::string    name      () { return "Scalar<" + std::to_string( ptr_size_in_bits ) + ">"; }
};

} // namespace asimd

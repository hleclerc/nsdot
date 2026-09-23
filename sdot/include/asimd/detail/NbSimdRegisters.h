#pragma once

#include "architectures/NativeCpu.h"
#include "SimdSize.h"

namespace asimd {

template<class T,int simd_size=SimdSize<T>::value,class Arch=NativeCpu>
struct NbSimdRegisters {
    /// default behavior (if NbSimdRegisters is not surdefined): we take sets of 2 smallers registers to make a virtual one
    static constexpr int value = Arch::template NbSimdRegisters<T,simd_size>::value; // <T,simd_size/2,Arch>::value / 2;
};

} // namespace asimd

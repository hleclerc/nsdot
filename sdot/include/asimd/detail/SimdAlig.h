#pragma once

#include "architectures/NativeCpu.h"
#include "SimdSize.h"

namespace asimd {

/**
  gives needed alignment in bits for aligned load/store instructions

  simd_size = nb of `T` item
*/
template<class T,int simd_size=SimdSize<T>::value,class Arch=NativeCpu>
struct SimdAlig {
    static constexpr int value = Arch::template SimdAlig<T,simd_size>::value; ///< in bits
};

} // namespace asimd

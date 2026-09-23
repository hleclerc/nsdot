#pragma once

#include "architectures/NativeCpu.h"

namespace asimd {

template<class T,class Arch=NativeCpu>
struct SimdSize {
    static constexpr int value = Arch::template SimdSize<T>::value;
};

} // namespace asimd

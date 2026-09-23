#pragma once

#include "SimdSize.h"

#include <algorithm>

namespace asimd {

/**
  get max simd size for each type in `Types`

  Arch = architectures (ex: Native, X86<...>)
*/
template<class Arch,class... Types>
struct MaxSimdSize;

template<class Arch,class Head,class... Tail>
struct MaxSimdSize<Arch,Head,Tail...> {
    static constexpr int value = std::max( SimdSize<Head,Arch>::value, MaxSimdSize<Arch,Tail...>::value );
};

template<class Arch>
struct MaxSimdSize<Arch> {
    static constexpr int value = 0;
};

} // namespace asimd

#pragma once

#include "FeatureSet.h"

#include <algorithm>

namespace asimd {

namespace features {

/**
*/
template<int size_in_bits,int nb_registers,class... Types>
struct SimdFeatureOn {
    template<class T> struct SimdSize { static constexpr int value = FeatureSet<Types...>::template Has<T>::value ? size_in_bits / ( 8 * sizeof( T ) ) : 1; };
    template<class T,int simd_size> struct NbSimdRegisters { static constexpr int value = nb_registers / std::max( 1, simd_size / SimdSize<T>::value ); };
    template<class T> struct SimdAlig { static constexpr int value = size_in_bits; };
};

} // namespace Features

} // namespace asimd

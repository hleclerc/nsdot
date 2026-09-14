#pragma once

#include "x86_intrin.h"

#ifdef ASIMD_X86_HAS_AVX512F

#include "../architectures/X86CpuFeatures.h"
#include "SimdBoolImpl_Generic.h"

namespace asimd {
namespace internal {


// struct SimdBoolImpl<...>
// 2 and 4 lanes ride in a __mmask8 as well. Without them, `SimdVec<double,4>` and every 128-bit
// type had no bit-flavoured mask at all, so the AVX-512 forms below could not be reached there.
SIMD_BOOL_IMPL_REG_BITS_UNSPLITABLE( AVX512,  2, __mmask8  )
SIMD_BOOL_IMPL_REG_BITS_UNSPLITABLE( AVX512,  4, __mmask8  )
SIMD_BOOL_IMPL_REG_BITS_UNSPLITABLE( AVX512,  8, __mmask8  )
SIMD_BOOL_IMPL_REG_BITS_SPLITABLE  ( AVX512, 16, __mmask16 )
SIMD_BOOL_IMPL_REG_BITS_SPLITABLE  ( AVX512, 32, __mmask32 )
SIMD_BOOL_IMPL_REG_BITS_SPLITABLE  ( AVX512, 64, __mmask64 )

// `all` MUST COMPLEMENT IN THE MASK TYPE. `~mask.data.reg == 0` integer-PROMOTES __mmask8 to int
// first, so a full mask gives `~255 == -256`: `all` was always false for every width but 64.
// gcc had been saying so on this very line -- "promoted bitwise complement of an unsigned value is
// always nonzero [-Wsign-compare]" -- and it was the only warning the library emitted.
SIMD_BOOL_IMPL_REG_REDUCTION( AVX512,  2, 1, all, ( mask.data.reg & 0x3 ) == 0x3 )
SIMD_BOOL_IMPL_REG_REDUCTION( AVX512,  4, 1, all, ( mask.data.reg & 0xf ) == 0xf )
SIMD_BOOL_IMPL_REG_REDUCTION( AVX512,  8, 1, all, __mmask8 ( ~mask.data.reg ) == 0 )
SIMD_BOOL_IMPL_REG_REDUCTION( AVX512, 16, 1, all, __mmask16( ~mask.data.reg ) == 0 )
SIMD_BOOL_IMPL_REG_REDUCTION( AVX512, 32, 1, all, __mmask32( ~mask.data.reg ) == 0 )
SIMD_BOOL_IMPL_REG_REDUCTION( AVX512, 64, 1, all, __mmask64( ~mask.data.reg ) == 0 )

SIMD_BOOL_IMPL_REG_REDUCTION( AVX512,  2, 1, any, ( mask.data.reg & 0x3 ) != 0 )
SIMD_BOOL_IMPL_REG_REDUCTION( AVX512,  4, 1, any, ( mask.data.reg & 0xf ) != 0 )
SIMD_BOOL_IMPL_REG_REDUCTION( AVX512,  8, 1, any, mask.data.reg != 0 ) 
SIMD_BOOL_IMPL_REG_REDUCTION( AVX512, 16, 1, any, mask.data.reg != 0 ) 
SIMD_BOOL_IMPL_REG_REDUCTION( AVX512, 32, 1, any, mask.data.reg != 0 ) 
SIMD_BOOL_IMPL_REG_REDUCTION( AVX512, 64, 1, any, mask.data.reg != 0 ) 


} // namespace internal
} // namespace asimd

#endif // ASIMD_X86_HAS_AVX512F

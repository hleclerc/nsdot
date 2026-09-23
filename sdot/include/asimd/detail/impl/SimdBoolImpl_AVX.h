#pragma once

#include "x86_intrin.h"

#ifdef ASIMD_X86_HAS_AVX

#include "../architectures/X86CpuFeatures.h"
#include "SimdBoolImpl_Generic.h"

namespace asimd {
namespace internal {


// struct Impl<...>
SIMD_BOOL_IMPL_REG_LARGE( AVX, 4, 64, __m256i )
SIMD_BOOL_IMPL_REG_LARGE( AVX, 8, 32, __m256i )

// `_mm256_xor_si256` is AVX2, in a file guarded by __AVX__. It is not needed at all: VPTEST's
// CF output -- `_mm256_testc_si256` -- already answers "are all those bits set", so `all` is one
// instruction rather than two, and stays inside AVX.
SIMD_BOOL_IMPL_REG_REDUCTION( AVX, 4, 64, all, _mm256_testc_si256( mask.data.reg, _mm256_set1_epi32( -1 ) ) != 0 )
SIMD_BOOL_IMPL_REG_REDUCTION( AVX, 8, 32, all, _mm256_testc_si256( mask.data.reg, _mm256_set1_epi32( -1 ) ) != 0 )

SIMD_BOOL_IMPL_REG_REDUCTION( AVX, 4, 64, any, _mm256_testz_si256( mask.data.reg, _mm256_set1_epi32( -1 ) ) == 0 )
SIMD_BOOL_IMPL_REG_REDUCTION( AVX, 8, 32, any, _mm256_testz_si256( mask.data.reg, _mm256_set1_epi32( -1 ) ) == 0 )

} // namespace internal
} // namespace asimd

#endif // ASIMD_X86_HAS_AVX

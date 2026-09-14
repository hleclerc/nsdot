#pragma once

#include "x86_intrin.h"

#ifdef ASIMD_X86_HAS_SSE2

#include "../architectures/X86CpuFeatures.h"
#include "SimdBoolImpl_Generic.h"

namespace asimd {
namespace internal {

// struct SimdBoolImpl<...>
SIMD_BOOL_IMPL_REG_LARGE( SSE2, 2, 64, __m128i )
SIMD_BOOL_IMPL_REG_LARGE( SSE2, 4, 32, __m128i )

// `ptest` is SSE4.1, not SSE2. And `all` is `testc`, not an xor followed by `testz`: TESTC asks
// exactly "are all the bits of the second operand set in the first", which is the question --
// one instruction instead of two, and no need for the complement.
// Below SSE4.1 the generic reductions in SimdBoolImpl_Generic.h take over.
#ifdef ASIMD_X86_HAS_SSE4_1
SIMD_BOOL_IMPL_REG_REDUCTION( SSE4_1, 2, 64, all, _mm_testc_si128( mask.data.reg, _mm_set1_epi32( -1 ) ) != 0 )
SIMD_BOOL_IMPL_REG_REDUCTION( SSE4_1, 4, 32, all, _mm_testc_si128( mask.data.reg, _mm_set1_epi32( -1 ) ) != 0 )

SIMD_BOOL_IMPL_REG_REDUCTION( SSE4_1, 2, 64, any, _mm_testz_si128( mask.data.reg, _mm_set1_epi32( -1 ) ) == 0 )
SIMD_BOOL_IMPL_REG_REDUCTION( SSE4_1, 4, 32, any, _mm_testz_si128( mask.data.reg, _mm_set1_epi32( -1 ) ) == 0 )
#endif


} // namespace internal
} // namespace asimd

#endif // ASIMD_X86_HAS_SSE2

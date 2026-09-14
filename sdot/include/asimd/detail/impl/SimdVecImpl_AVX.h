#pragma once

#include "x86_intrin.h"

#ifdef ASIMD_X86_HAS_AVX

#include "../architectures/X86CpuFeatures.h"
#include "SimdVecImpl_Generic.h"

namespace asimd {
namespace internal {


// struct Impl<...>
SIMD_VEC_IMPL_REG( AVX, PI64, 4, __m256i ); 
SIMD_VEC_IMPL_REG( AVX, SI64, 4, __m256i );
SIMD_VEC_IMPL_REG( AVX, FP64, 4, __m256d );
SIMD_VEC_IMPL_REG( AVX, PI32, 8, __m256i );
SIMD_VEC_IMPL_REG( AVX, SI32, 8, __m256i );
SIMD_VEC_IMPL_REG( AVX, FP32, 8, __m256  );

// init ----------------------------------------------------------------------
SIMD_VEC_IMPL_REG_INIT_1( AVX, PI64, 4, _mm256_set1_epi64x( a ) );
SIMD_VEC_IMPL_REG_INIT_1( AVX, SI64, 4, _mm256_set1_epi64x( a ) );
SIMD_VEC_IMPL_REG_INIT_1( AVX, FP64, 4, _mm256_set1_pd( a ) );
SIMD_VEC_IMPL_REG_INIT_1( AVX, PI32, 8, _mm256_set1_epi32( a ) );
SIMD_VEC_IMPL_REG_INIT_1( AVX, SI32, 8, _mm256_set1_epi32( a ) );
SIMD_VEC_IMPL_REG_INIT_1( AVX, FP32, 8, _mm256_set1_ps( a ) );

SIMD_VEC_IMPL_REG_INIT_4( AVX, PI64, 4, _mm256_set_epi64x( d, c, b, a ) );
SIMD_VEC_IMPL_REG_INIT_4( AVX, SI64, 4, _mm256_set_epi64x( d, c, b, a ) );
SIMD_VEC_IMPL_REG_INIT_4( AVX, FP64, 4, _mm256_set_pd( d, c, b, a ) );
SIMD_VEC_IMPL_REG_INIT_8( AVX, PI32, 8, _mm256_set_epi32( h, g, f, e, d, c, b, a ) );
SIMD_VEC_IMPL_REG_INIT_8( AVX, SI32, 8, _mm256_set_epi32( h, g, f, e, d, c, b, a ) );
SIMD_VEC_IMPL_REG_INIT_8( AVX, FP32, 8, _mm256_set_ps( h, g, f, e, d, c, b, a ) );

// load_aligned -----------------------------------------------------------------------
SIMD_VEC_IMPL_REG_LOAD_ALIGNED( AVX, PI64, 4, 256, _mm256_load_si256( (const __m256i *)data ) );  
SIMD_VEC_IMPL_REG_LOAD_ALIGNED( AVX, SI64, 4, 256, _mm256_load_si256( (const __m256i *)data ) );
SIMD_VEC_IMPL_REG_LOAD_ALIGNED( AVX, FP64, 4, 256, _mm256_load_pd   (                  data ) );
SIMD_VEC_IMPL_REG_LOAD_ALIGNED( AVX, PI32, 8, 256, _mm256_load_si256( (const __m256i *)data ) );
SIMD_VEC_IMPL_REG_LOAD_ALIGNED( AVX, SI32, 8, 256, _mm256_load_si256( (const __m256i *)data ) );
SIMD_VEC_IMPL_REG_LOAD_ALIGNED( AVX, FP32, 8, 256, _mm256_load_ps   (                  data ) );

// `vmovntdqa` at 256 bits is AVX2, and the FP forms cast to `__m256` -- the wrong type for FP64.
#ifdef ASIMD_X86_HAS_AVX2
SIMD_VEC_IMPL_REG_LOAD_ALIGNED_STREAM( AVX2, PI64, 4, 256,                     _mm256_stream_load_si256( (const __m256i *)data )   );
SIMD_VEC_IMPL_REG_LOAD_ALIGNED_STREAM( AVX2, SI64, 4, 256,                     _mm256_stream_load_si256( (const __m256i *)data )   );
SIMD_VEC_IMPL_REG_LOAD_ALIGNED_STREAM( AVX2, FP64, 4, 256, _mm256_castsi256_pd( _mm256_stream_load_si256( (const __m256i *)data ) ) );
SIMD_VEC_IMPL_REG_LOAD_ALIGNED_STREAM( AVX2, PI32, 8, 256,                     _mm256_stream_load_si256( (const __m256i *)data )   );
SIMD_VEC_IMPL_REG_LOAD_ALIGNED_STREAM( AVX2, SI32, 8, 256,                     _mm256_stream_load_si256( (const __m256i *)data )   );
SIMD_VEC_IMPL_REG_LOAD_ALIGNED_STREAM( AVX2, FP32, 8, 256, _mm256_castsi256_ps( _mm256_stream_load_si256( (const __m256i *)data ) ) );
#endif

// load unaligned ----------------------------------------------------------------------
SIMD_VEC_IMPL_REG_LOAD_UNALIGNED( AVX, PI64, 4, _mm256_loadu_si256( (const __m256i *)data ) ); 
SIMD_VEC_IMPL_REG_LOAD_UNALIGNED( AVX, SI64, 4, _mm256_loadu_si256( (const __m256i *)data ) );
SIMD_VEC_IMPL_REG_LOAD_UNALIGNED( AVX, FP64, 4, _mm256_loadu_pd   (                  data ) );
SIMD_VEC_IMPL_REG_LOAD_UNALIGNED( AVX, PI32, 8, _mm256_loadu_si256( (const __m256i *)data ) );
SIMD_VEC_IMPL_REG_LOAD_UNALIGNED( AVX, SI32, 8, _mm256_loadu_si256( (const __m256i *)data ) ); 
SIMD_VEC_IMPL_REG_LOAD_UNALIGNED( AVX, FP32, 8, _mm256_loadu_ps   (                  data ) );

// store_aligned -----------------------------------------------------------------------
SIMD_VEC_IMPL_REG_STORE_ALIGNED( AVX, PI64, 4, 256, _mm256_store_si256( (__m256i *)data, impl.data.reg ) );
SIMD_VEC_IMPL_REG_STORE_ALIGNED( AVX, SI64, 4, 256, _mm256_store_si256( (__m256i *)data, impl.data.reg ) );
SIMD_VEC_IMPL_REG_STORE_ALIGNED( AVX, FP64, 4, 256, _mm256_store_pd   (            data, impl.data.reg ) );
SIMD_VEC_IMPL_REG_STORE_ALIGNED( AVX, PI32, 8, 256, _mm256_store_si256( (__m256i *)data, impl.data.reg ) );
SIMD_VEC_IMPL_REG_STORE_ALIGNED( AVX, SI32, 8, 256, _mm256_store_si256( (__m256i *)data, impl.data.reg ) );
SIMD_VEC_IMPL_REG_STORE_ALIGNED( AVX, FP32, 8, 256, _mm256_store_ps   (            data, impl.data.reg ) );

SIMD_VEC_IMPL_REG_STORE_ALIGNED_STREAM( AVX, PI64, 4, 256, _mm256_stream_si256( (__m256i *)data, impl.data.reg ) );
SIMD_VEC_IMPL_REG_STORE_ALIGNED_STREAM( AVX, SI64, 4, 256, _mm256_stream_si256( (__m256i *)data, impl.data.reg ) );
SIMD_VEC_IMPL_REG_STORE_ALIGNED_STREAM( AVX, FP64, 4, 256, _mm256_stream_pd   (            data, impl.data.reg ) );
SIMD_VEC_IMPL_REG_STORE_ALIGNED_STREAM( AVX, PI32, 8, 256, _mm256_stream_si256( (__m256i *)data, impl.data.reg ) );
SIMD_VEC_IMPL_REG_STORE_ALIGNED_STREAM( AVX, SI32, 8, 256, _mm256_stream_si256( (__m256i *)data, impl.data.reg ) );
SIMD_VEC_IMPL_REG_STORE_ALIGNED_STREAM( AVX, FP32, 8, 256, _mm256_stream_ps   (            data, impl.data.reg ) );

// store unaligned ---------------------------------------------------------------------
SIMD_VEC_IMPL_REG_STORE_UNALIGNED( AVX, PI64, 4, _mm256_storeu_si256( (__m256i *)data, impl.data.reg ) );
SIMD_VEC_IMPL_REG_STORE_UNALIGNED( AVX, SI64, 4, _mm256_storeu_si256( (__m256i *)data, impl.data.reg ) );
SIMD_VEC_IMPL_REG_STORE_UNALIGNED( AVX, FP64, 4, _mm256_storeu_pd   (            data, impl.data.reg ) );
SIMD_VEC_IMPL_REG_STORE_UNALIGNED( AVX, PI32, 8, _mm256_storeu_si256( (__m256i *)data, impl.data.reg ) );
SIMD_VEC_IMPL_REG_STORE_UNALIGNED( AVX, SI32, 8, _mm256_storeu_si256( (__m256i *)data, impl.data.reg ) );
SIMD_VEC_IMPL_REG_STORE_UNALIGNED( AVX, FP32, 8, _mm256_storeu_ps   (            data, impl.data.reg ) );

//// arithmetic -------------------------------------------------------------------------
//
// AVX HAS NO 256-BIT INTEGER ALU. `vpaddd`, `vpminsd` &c. at 256 bits are AVX2 -- what AVX adds
// is the 256-bit FLOATING POINT unit, plus the wider register file. `add`, `sub`, `min`, `max`
// and `and` were registered here for the integer types too, so on a genuine AVX target (Sandy
// Bridge, Ivy Bridge) the ENTIRE integer column failed to compile. They now live in
// SimdVecImpl_AVX2.h. The register LAYOUTS above are right at this level and stay.
#define SIMD_VEC_IMPL_REG_ARITHMETIC_OP_AVX_F( NAME ) \
    SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX, FP64, 4, NAME, _mm256_##NAME##_pd ); \
    SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX, FP32, 8, NAME, _mm256_##NAME##_ps );

SIMD_VEC_IMPL_REG_ARITHMETIC_OP_AVX_F( add );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP_AVX_F( sub );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP_AVX_F( mul );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP_AVX_F( div );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP_AVX_F( min );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP_AVX_F( max );

#undef SIMD_VEC_IMPL_REG_ARITHMETIC_OP_AVX_F

SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX , FP64, 4, anb, _mm256_and_pd );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX , FP32, 8, anb, _mm256_and_ps );

//// cmp operations ------------------------------------------------------------------
//
// The integer rows here called `_mm256_cmp_epi64` and `_mm_cmp_epi64`, WHICH DO NOT EXIST as
// intrinsics -- and applied them to 32-bit data, and in signed form to the unsigned types. Three
// errors on one line. There is no VEX integer compare with a predicate operand before AVX-512;
// the AVX2 file registers the real `vpcmpeqd` / `vpcmpgtd` instead.
#define SIMD_VEC_IMPL_CMP_OP_SIMDVEC_AVX( NAME, CMP ) \
    SIMD_VEC_IMPL_CMP_OP_SIMDVEC( AVX, FP64, 4, 64, NAME, _mm256_castpd_si256( _mm256_cmp_pd( a.data.reg, b.data.reg, CMP ) ) ); \
    SIMD_VEC_IMPL_CMP_OP_SIMDVEC( AVX, FP32, 8, 32, NAME, _mm256_castps_si256( _mm256_cmp_ps( a.data.reg, b.data.reg, CMP ) ) ); \
    SIMD_VEC_IMPL_CMP_OP_SIMDVEC( AVX, FP64, 2, 64, NAME, _mm_castpd_si128   ( _mm_cmp_pd   ( a.data.reg, b.data.reg, CMP ) ) ); \
    SIMD_VEC_IMPL_CMP_OP_SIMDVEC( AVX, FP32, 4, 32, NAME, _mm_castps_si128   ( _mm_cmp_ps   ( a.data.reg, b.data.reg, CMP ) ) );

SIMD_VEC_IMPL_CMP_OP_SIMDVEC_AVX( lt, _CMP_LT_OQ )
SIMD_VEC_IMPL_CMP_OP_SIMDVEC_AVX( gt, _CMP_GT_OQ )

#undef SIMD_VEC_IMPL_CMP_OP_SIMDVEC_AVX

//// horizontal sum: fold the two 128-bit halves together, then fall into the SSE2 ladder above.
SIMD_VEC_IMPL_REG_HSUM( AVX, FP32, 8, horizontal_sum( impl_from_reg<SimdVecImpl<FP32,4,Arch>>( _mm_add_ps(
    _mm256_castps256_ps128( impl.data.reg ), _mm256_extractf128_ps( impl.data.reg, 1 ) ) ) ) );
SIMD_VEC_IMPL_REG_HSUM( AVX, FP64, 4, horizontal_sum( impl_from_reg<SimdVecImpl<FP64,2,Arch>>( _mm_add_pd(
    _mm256_castpd256_pd128( impl.data.reg ), _mm256_extractf128_pd( impl.data.reg, 1 ) ) ) ) );

// iota, register form (see the comment in SimdVecImpl_AVX2.h)
template<class Arch> requires ( Arch::template Has<features::AVX>::value ) HaD
SimdVecImpl<FP32,8,Arch> iota( FP32 beg, S<SimdVecImpl<FP32,8,Arch>> ) {
    SimdVecImpl<FP32,8,Arch> res;
    res.data.reg = _mm256_add_ps( _mm256_setr_ps( 0, 1, 2, 3, 4, 5, 6, 7 ), _mm256_set1_ps( beg ) );
    return res;
}

} // namespace internal
} // namespace asimd

#endif // ASIMD_X86_HAS_AVX

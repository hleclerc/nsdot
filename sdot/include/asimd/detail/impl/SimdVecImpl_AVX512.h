#pragma once
 
#include "x86_intrin.h"

#ifdef ASIMD_X86_HAS_AVX512F

#include "../architectures/X86CpuFeatures.h"
#include "SimdVecImpl_Generic.h"

namespace asimd {
namespace internal {
 

// struct Impl<...>
SIMD_VEC_IMPL_REG( AVX512, PI64,  8, __m512i );
SIMD_VEC_IMPL_REG( AVX512, SI64,  8, __m512i );
SIMD_VEC_IMPL_REG( AVX512, FP64,  8, __m512d );
SIMD_VEC_IMPL_REG( AVX512, PI32, 16, __m512i );
SIMD_VEC_IMPL_REG( AVX512, SI32, 16, __m512i );
SIMD_VEC_IMPL_REG( AVX512, FP32, 16, __m512  );

// init ----------------------------------------------------------------------
SIMD_VEC_IMPL_REG_INIT_1( AVX512, PI64,  8, _mm512_set1_epi64( a ) );
SIMD_VEC_IMPL_REG_INIT_1( AVX512, SI64,  8, _mm512_set1_epi64( a ) );
SIMD_VEC_IMPL_REG_INIT_1( AVX512, FP64,  8, _mm512_set1_pd   ( a ) );
SIMD_VEC_IMPL_REG_INIT_1( AVX512, PI32, 16, _mm512_set1_epi32( a ) );
SIMD_VEC_IMPL_REG_INIT_1( AVX512, SI32, 16, _mm512_set1_epi32( a ) );
SIMD_VEC_IMPL_REG_INIT_1( AVX512, FP32, 16, _mm512_set1_ps   ( a ) );

SIMD_VEC_IMPL_REG_INIT_8 ( AVX512, PI64,  8, _mm512_set_epi64( h, g, f, e, d, c, b, a ) );
SIMD_VEC_IMPL_REG_INIT_8 ( AVX512, SI64,  8, _mm512_set_epi64( h, g, f, e, d, c, b, a ) );
SIMD_VEC_IMPL_REG_INIT_8 ( AVX512, FP64,  8, _mm512_set_pd   ( h, g, f, e, d, c, b, a ) );
SIMD_VEC_IMPL_REG_INIT_16( AVX512, PI32, 16, _mm512_set_epi32( p, o, n, m, l, k, j, i, h, g, f, e, d, c, b, a ) );
SIMD_VEC_IMPL_REG_INIT_16( AVX512, SI32, 16, _mm512_set_epi32( p, o, n, m, l, k, j, i, h, g, f, e, d, c, b, a ) );
SIMD_VEC_IMPL_REG_INIT_16( AVX512, FP32, 16, _mm512_set_ps   ( p, o, n, m, l, k, j, i, h, g, f, e, d, c, b, a ) );

// load_aligned -----------------------------------------------------------------------
SIMD_VEC_IMPL_REG_LOAD_ALIGNED( AVX512, PI64,  8, 512, _mm512_load_si512( (const __m512i *)data ) );
SIMD_VEC_IMPL_REG_LOAD_ALIGNED( AVX512, SI64,  8, 512, _mm512_load_si512( (const __m512i *)data ) );
SIMD_VEC_IMPL_REG_LOAD_ALIGNED( AVX512, FP64,  8, 512, _mm512_load_pd   (                  data ) );
SIMD_VEC_IMPL_REG_LOAD_ALIGNED( AVX512, PI32, 16, 512, _mm512_load_si512( (const __m512i *)data ) );
SIMD_VEC_IMPL_REG_LOAD_ALIGNED( AVX512, SI32, 16, 512, _mm512_load_si512( (const __m512i *)data ) );
SIMD_VEC_IMPL_REG_LOAD_ALIGNED( AVX512, FP32, 16, 512, _mm512_load_ps   (                  data ) );

SIMD_VEC_IMPL_REG_LOAD_ALIGNED_STREAM( AVX512, PI64,  8, 512,                     _mm512_stream_load_si512( (void *)data )   );
SIMD_VEC_IMPL_REG_LOAD_ALIGNED_STREAM( AVX512, SI64,  8, 512,                     _mm512_stream_load_si512( (void *)data )   );
SIMD_VEC_IMPL_REG_LOAD_ALIGNED_STREAM( AVX512, FP64,  8, 512, _mm512_castsi512_pd( _mm512_stream_load_si512( (void *)data ) ) ); // was (__m512): the wrong type
SIMD_VEC_IMPL_REG_LOAD_ALIGNED_STREAM( AVX512, PI32, 16, 512,                     _mm512_stream_load_si512( (void *)data )   );
SIMD_VEC_IMPL_REG_LOAD_ALIGNED_STREAM( AVX512, SI32, 16, 512,                     _mm512_stream_load_si512( (void *)data )   );
SIMD_VEC_IMPL_REG_LOAD_ALIGNED_STREAM( AVX512, FP32, 16, 512, _mm512_castsi512_ps( _mm512_stream_load_si512( (void *)data ) ) );

// load -------------------------------------------------------------------------------
SIMD_VEC_IMPL_REG_LOAD_UNALIGNED( AVX512, PI64,  8, _mm512_loadu_si512( (const __m512i *)data ) );
SIMD_VEC_IMPL_REG_LOAD_UNALIGNED( AVX512, SI64,  8, _mm512_loadu_si512( (const __m512i *)data ) );
SIMD_VEC_IMPL_REG_LOAD_UNALIGNED( AVX512, FP64,  8, _mm512_loadu_pd   (                  data ) );
SIMD_VEC_IMPL_REG_LOAD_UNALIGNED( AVX512, PI32, 16, _mm512_loadu_si512( (const __m512i *)data ) );
SIMD_VEC_IMPL_REG_LOAD_UNALIGNED( AVX512, SI32, 16, _mm512_loadu_si512( (const __m512i *)data ) );
SIMD_VEC_IMPL_REG_LOAD_UNALIGNED( AVX512, FP32, 16, _mm512_loadu_ps   (                  data ) );

// store_aligned -----------------------------------------------------------------------
SIMD_VEC_IMPL_REG_STORE_ALIGNED( AVX512, PI64,  8, 512, _mm512_store_si512( (__m512i *)data, impl.data.reg ) );
SIMD_VEC_IMPL_REG_STORE_ALIGNED( AVX512, SI64,  8, 512, _mm512_store_si512( (__m512i *)data, impl.data.reg ) );
SIMD_VEC_IMPL_REG_STORE_ALIGNED( AVX512, FP64,  8, 512, _mm512_store_pd   (            data, impl.data.reg ) );
SIMD_VEC_IMPL_REG_STORE_ALIGNED( AVX512, PI32, 16, 512, _mm512_store_si512( (__m512i *)data, impl.data.reg ) );
SIMD_VEC_IMPL_REG_STORE_ALIGNED( AVX512, SI32, 16, 512, _mm512_store_si512( (__m512i *)data, impl.data.reg ) );
SIMD_VEC_IMPL_REG_STORE_ALIGNED( AVX512, FP32, 16, 512, _mm512_store_ps   (            data, impl.data.reg ) );

SIMD_VEC_IMPL_REG_STORE_ALIGNED_STREAM( AVX512, PI64,  8, 512, _mm512_stream_si512( (__m512i *)data, impl.data.reg ) );
SIMD_VEC_IMPL_REG_STORE_ALIGNED_STREAM( AVX512, SI64,  8, 512, _mm512_stream_si512( (__m512i *)data, impl.data.reg ) );
SIMD_VEC_IMPL_REG_STORE_ALIGNED_STREAM( AVX512, FP64,  8, 512, _mm512_stream_pd   (            data, impl.data.reg ) );
SIMD_VEC_IMPL_REG_STORE_ALIGNED_STREAM( AVX512, PI32, 16, 512, _mm512_stream_si512( (__m512i *)data, impl.data.reg ) );
SIMD_VEC_IMPL_REG_STORE_ALIGNED_STREAM( AVX512, SI32, 16, 512, _mm512_stream_si512( (__m512i *)data, impl.data.reg ) );
SIMD_VEC_IMPL_REG_STORE_ALIGNED_STREAM( AVX512, FP32, 16, 512, _mm512_stream_ps   (            data, impl.data.reg ) );

// store -----------------------------------------------------------------------
SIMD_VEC_IMPL_REG_STORE_UNALIGNED( AVX512, PI64,  8, _mm512_storeu_si512( (__m512i *)data, impl.data.reg ) );
SIMD_VEC_IMPL_REG_STORE_UNALIGNED( AVX512, SI64,  8, _mm512_storeu_si512( (__m512i *)data, impl.data.reg ) );
SIMD_VEC_IMPL_REG_STORE_UNALIGNED( AVX512, FP64,  8, _mm512_storeu_pd   (            data, impl.data.reg ) );
SIMD_VEC_IMPL_REG_STORE_UNALIGNED( AVX512, PI32, 16, _mm512_storeu_si512( (__m512i *)data, impl.data.reg ) );
SIMD_VEC_IMPL_REG_STORE_UNALIGNED( AVX512, SI32, 16, _mm512_storeu_si512( (__m512i *)data, impl.data.reg ) );
SIMD_VEC_IMPL_REG_STORE_UNALIGNED( AVX512, FP32, 16, _mm512_storeu_ps   (            data, impl.data.reg ) );

//// arithmetic operations that work on all types ------------------------------------------------
#define SIMD_VEC_IMPL_REG_ARITHMETIC_OP_SSE2_A( NAME ) \
    SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX512, PI64,  8, NAME, _mm512_##NAME##_epi64 ); \
    SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX512, SI64,  8, NAME, _mm512_##NAME##_epi64 ); \
    SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX512, FP64,  8, NAME, _mm512_##NAME##_pd    ); \
    SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX512, PI32, 16, NAME, _mm512_##NAME##_epi32 ); \
    SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX512, SI32, 16, NAME, _mm512_##NAME##_epi32 ); \
    SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX512, FP32, 16, NAME, _mm512_##NAME##_ps    );

    SIMD_VEC_IMPL_REG_ARITHMETIC_OP_SSE2_A( add );
    SIMD_VEC_IMPL_REG_ARITHMETIC_OP_SSE2_A( sub );

#undef SIMD_VEC_IMPL_REG_ARITHMETIC_OP_SSE2_A

//// min / max: the unsigned types went through the SIGNED instruction here too.
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX512, FP64,  8, min, _mm512_min_pd    );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX512, FP32, 16, min, _mm512_min_ps    );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX512, SI64,  8, min, _mm512_min_epi64 );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX512, PI64,  8, min, _mm512_min_epu64 );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX512, SI32, 16, min, _mm512_min_epi32 );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX512, PI32, 16, min, _mm512_min_epu32 );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX512, FP64,  8, max, _mm512_max_pd    );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX512, FP32, 16, max, _mm512_max_ps    );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX512, SI64,  8, max, _mm512_max_epi64 );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX512, PI64,  8, max, _mm512_max_epu64 );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX512, SI32, 16, max, _mm512_max_epi32 );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX512, PI32, 16, max, _mm512_max_epu32 );

//// 32-bit multiply; the 64-bit one (`vpmullq`) is AVX-512DQ.
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX512, SI32, 16, mul, _mm512_mullo_epi32 );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX512, PI32, 16, mul, _mm512_mullo_epi32 );
#ifdef ASIMD_X86_HAS_AVX512DQ
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX512DQ, SI64, 8, mul, _mm512_mullo_epi64 );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX512DQ, PI64, 8, mul, _mm512_mullo_epi64 );
#ifdef ASIMD_X86_HAS_AVX512VL
// ... and at 128 and 256 bits, which needs VL on top of DQ. Without these, a 64-bit multiply at
// four lanes fell back to a lane loop: 23 instructions against one `vpmullq`.
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX512DQ, SI64, 4, mul, _mm256_mullo_epi64 );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX512DQ, PI64, 4, mul, _mm256_mullo_epi64 );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX512DQ, SI64, 2, mul, _mm_mullo_epi64    );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX512DQ, PI64, 2, mul, _mm_mullo_epi64    );
#endif
#endif

//// 64-bit integer min/max at 128 and 256 bits: AVX-512VL. These fill the holes SSE4.1 and AVX2
//// leave -- there simply is no `pminsq` before AVX-512.
#ifdef ASIMD_X86_HAS_AVX512VL
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX512VL, SI64, 4, min, _mm256_min_epi64 );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX512VL, PI64, 4, min, _mm256_min_epu64 );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX512VL, SI64, 4, max, _mm256_max_epi64 );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX512VL, PI64, 4, max, _mm256_max_epu64 );
#endif

//// arithmetic operations that work only on float types ------------------------------------------------
#define SIMD_VEC_IMPL_REG_ARITHMETIC_OP_SSE2_F( NAME ) \
    SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX512, double,  8, NAME, _mm512_##NAME##_pd ); \
    SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX512, float , 16, NAME, _mm512_##NAME##_ps );

    SIMD_VEC_IMPL_REG_ARITHMETIC_OP_SSE2_F( mul );
    SIMD_VEC_IMPL_REG_ARITHMETIC_OP_SSE2_F( div );

#undef SIMD_VEC_IMPL_REG_ARITHMETIC_OP_SSE2_F

//// arithmetic operations with != func and name -------------------------------------------------------
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX512, PI64,  8, anb, _mm512_and_si512 );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX512, SI64,  8, anb, _mm512_and_si512 );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX512, FP64,  8, anb, _mm512_and_pd    );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX512, PI32, 16, anb, _mm512_and_si512 );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX512, SI32, 16, anb, _mm512_and_si512 );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX512, FP32, 16, anb, _mm512_and_ps    );

//// arithmetic operations that work only on int types ------------------------------------------------
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX512, PI64,  8, sll, _mm512_sllv_epi64 );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX512, SI64,  8, sll, _mm512_sllv_epi64 );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX512, PI32, 16, sll, _mm512_sllv_epi32 );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX512, SI32, 16, sll, _mm512_sllv_epi32 );

//// horizontal sum: AVX-512 spells it out as an intrinsic, and gcc/clang expand it into the same
//// fold ladder -- no reason to write it by hand here.
SIMD_VEC_IMPL_REG_HSUM( AVX512, FP32, 16, _mm512_reduce_add_ps   ( impl.data.reg ) );
SIMD_VEC_IMPL_REG_HSUM( AVX512, FP64,  8, _mm512_reduce_add_pd   ( impl.data.reg ) );
SIMD_VEC_IMPL_REG_HSUM( AVX512, SI32, 16, _mm512_reduce_add_epi32( impl.data.reg ) );
SIMD_VEC_IMPL_REG_HSUM( AVX512, PI32, 16, PI32( _mm512_reduce_add_epi32( impl.data.reg ) ) );
SIMD_VEC_IMPL_REG_HSUM( AVX512, SI64,  8, _mm512_reduce_add_epi64( impl.data.reg ) );
SIMD_VEC_IMPL_REG_HSUM( AVX512, PI64,  8, PI64( _mm512_reduce_add_epi64( impl.data.reg ) ) );

// iota ---------------------------------------------------------------------------------------
// A constant load plus one add. Without a register form gcc materialises the sequence lane by
// lane -- 26 instructions at sixteen lanes, and no way for the compiler to see it is a constant.
#define ASIMD_AVX512_IOTA( T, N, W, SET, ADD ) \
    template<class Arch> requires ( Arch::template Has<features::AVX512>::value ) HaD \
    SimdVecImpl<T,N,Arch> iota( T beg, S<SimdVecImpl<T,N,Arch>> ) { \
        SimdVecImpl<T,N,Arch> res; \
        res.data.reg = ADD( SET, _mm512_set1_##W( beg ) ); return res; \
    }

ASIMD_AVX512_IOTA( FP32, 16, ps, _mm512_setr_ps( 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 ), _mm512_add_ps );
ASIMD_AVX512_IOTA( FP64,  8, pd, _mm512_setr_pd( 0,1,2,3,4,5,6,7 ), _mm512_add_pd );
ASIMD_AVX512_IOTA( SI32, 16, epi32, _mm512_setr_epi32( 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 ), _mm512_add_epi32 );
ASIMD_AVX512_IOTA( PI32, 16, epi32, _mm512_setr_epi32( 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 ), _mm512_add_epi32 );
ASIMD_AVX512_IOTA( SI64,  8, epi64, _mm512_setr_epi64( 0,1,2,3,4,5,6,7 ), _mm512_add_epi64 );
ASIMD_AVX512_IOTA( PI64,  8, epi64, _mm512_setr_epi64( 0,1,2,3,4,5,6,7 ), _mm512_add_epi64 );

#undef ASIMD_AVX512_IOTA

// gather -----------------------------------------------------------------------------------------------
SIMD_VEC_IMPL_REG_GATHER( AVX512, PI64, PI32,  8, _mm512_i32gather_epi64( ind.data.reg, data, 8 ) );
SIMD_VEC_IMPL_REG_GATHER( AVX512, SI64, PI32,  8, _mm512_i32gather_epi64( ind.data.reg, data, 8 ) );
SIMD_VEC_IMPL_REG_GATHER( AVX512, FP64, PI32,  8, _mm512_i32gather_pd   ( ind.data.reg, data, 8 ) );
SIMD_VEC_IMPL_REG_GATHER( AVX512, PI32, PI32, 16, _mm512_i32gather_epi32( ind.data.reg, data, 4 ) );
SIMD_VEC_IMPL_REG_GATHER( AVX512, SI32, PI32, 16, _mm512_i32gather_epi32( ind.data.reg, data, 4 ) );
SIMD_VEC_IMPL_REG_GATHER( AVX512, FP32, PI32, 16, _mm512_i32gather_ps   ( ind.data.reg, data, 4 ) );

SIMD_VEC_IMPL_REG_GATHER( AVX512, PI64, SI32,  8, _mm512_i32gather_epi64( ind.data.reg, data, 8 ) );
SIMD_VEC_IMPL_REG_GATHER( AVX512, SI64, SI32,  8, _mm512_i32gather_epi64( ind.data.reg, data, 8 ) );
SIMD_VEC_IMPL_REG_GATHER( AVX512, FP64, SI32,  8, _mm512_i32gather_pd   ( ind.data.reg, data, 8 ) );
SIMD_VEC_IMPL_REG_GATHER( AVX512, PI32, SI32, 16, _mm512_i32gather_epi32( ind.data.reg, data, 4 ) );
SIMD_VEC_IMPL_REG_GATHER( AVX512, SI32, SI32, 16, _mm512_i32gather_epi32( ind.data.reg, data, 4 ) );
SIMD_VEC_IMPL_REG_GATHER( AVX512, FP32, SI32, 16, _mm512_i32gather_ps   ( ind.data.reg, data, 4 ) );

// scatter -----------------------------------------------------------------------------------------------
SIMD_VEC_IMPL_REG_SCATTER( AVX512, PI64, PI32,  8, _mm512_i32scatter_epi64( data, ind.data.reg, vec.data.reg, 8 ) );
SIMD_VEC_IMPL_REG_SCATTER( AVX512, SI64, PI32,  8, _mm512_i32scatter_epi64( data, ind.data.reg, vec.data.reg, 8 ) );
SIMD_VEC_IMPL_REG_SCATTER( AVX512, FP64, PI32,  8, _mm512_i32scatter_pd   ( data, ind.data.reg, vec.data.reg, 8 ) );
SIMD_VEC_IMPL_REG_SCATTER( AVX512, PI32, PI32, 16, _mm512_i32scatter_epi32( data, ind.data.reg, vec.data.reg, 4 ) );
SIMD_VEC_IMPL_REG_SCATTER( AVX512, SI32, PI32, 16, _mm512_i32scatter_epi32( data, ind.data.reg, vec.data.reg, 4 ) );
SIMD_VEC_IMPL_REG_SCATTER( AVX512, FP32, PI32, 16, _mm512_i32scatter_ps   ( data, ind.data.reg, vec.data.reg, 4 ) );

SIMD_VEC_IMPL_REG_SCATTER( AVX512, PI64, SI32,  8, _mm512_i32scatter_epi64( data, ind.data.reg, vec.data.reg, 8 ) );
SIMD_VEC_IMPL_REG_SCATTER( AVX512, SI64, SI32,  8, _mm512_i32scatter_epi64( data, ind.data.reg, vec.data.reg, 8 ) );
SIMD_VEC_IMPL_REG_SCATTER( AVX512, FP64, SI32,  8, _mm512_i32scatter_pd   ( data, ind.data.reg, vec.data.reg, 8 ) );
SIMD_VEC_IMPL_REG_SCATTER( AVX512, PI32, SI32, 16, _mm512_i32scatter_epi32( data, ind.data.reg, vec.data.reg, 4 ) );
SIMD_VEC_IMPL_REG_SCATTER( AVX512, SI32, SI32, 16, _mm512_i32scatter_epi32( data, ind.data.reg, vec.data.reg, 4 ) );
SIMD_VEC_IMPL_REG_SCATTER( AVX512, FP32, SI32, 16, _mm512_i32scatter_ps   ( data, ind.data.reg, vec.data.reg, 4 ) );

// 128- and 256-bit scatters are AVX-512VL, not AVX-512F.
#ifdef ASIMD_X86_HAS_AVX512VL
// AVX2 sizes
SIMD_VEC_IMPL_REG_SCATTER( AVX512VL, PI64, PI32, 4, _mm256_i32scatter_epi64( data, ind.data.reg, vec.data.reg, 8 ) );
SIMD_VEC_IMPL_REG_SCATTER( AVX512VL, SI64, PI32, 4, _mm256_i32scatter_epi64( data, ind.data.reg, vec.data.reg, 8 ) );
SIMD_VEC_IMPL_REG_SCATTER( AVX512VL, FP64, PI32, 4, _mm256_i32scatter_pd   ( data, ind.data.reg, vec.data.reg, 8 ) );
SIMD_VEC_IMPL_REG_SCATTER( AVX512VL, PI32, PI32, 8, _mm256_i32scatter_epi32( data, ind.data.reg, vec.data.reg, 4 ) );
SIMD_VEC_IMPL_REG_SCATTER( AVX512VL, SI32, PI32, 8, _mm256_i32scatter_epi32( data, ind.data.reg, vec.data.reg, 4 ) );
SIMD_VEC_IMPL_REG_SCATTER( AVX512VL, FP32, PI32, 8, _mm256_i32scatter_ps   ( data, ind.data.reg, vec.data.reg, 4 ) );

SIMD_VEC_IMPL_REG_SCATTER( AVX512VL, PI64, SI32, 4, _mm256_i32scatter_epi64( data, ind.data.reg, vec.data.reg, 8 ) );
SIMD_VEC_IMPL_REG_SCATTER( AVX512VL, SI64, SI32, 4, _mm256_i32scatter_epi64( data, ind.data.reg, vec.data.reg, 8 ) );
SIMD_VEC_IMPL_REG_SCATTER( AVX512VL, FP64, SI32, 4, _mm256_i32scatter_pd   ( data, ind.data.reg, vec.data.reg, 8 ) );
SIMD_VEC_IMPL_REG_SCATTER( AVX512VL, PI32, SI32, 8, _mm256_i32scatter_epi32( data, ind.data.reg, vec.data.reg, 4 ) );
SIMD_VEC_IMPL_REG_SCATTER( AVX512VL, SI32, SI32, 8, _mm256_i32scatter_epi32( data, ind.data.reg, vec.data.reg, 4 ) );
SIMD_VEC_IMPL_REG_SCATTER( AVX512VL, FP32, SI32, 8, _mm256_i32scatter_ps   ( data, ind.data.reg, vec.data.reg, 4 ) );

// SSE2 sizes
SIMD_VEC_IMPL_REG_SCATTER( AVX512VL, PI64, PI32, 2, _mm_i32scatter_epi64( data, ind.data.reg, vec.data.reg, 8 ) );
SIMD_VEC_IMPL_REG_SCATTER( AVX512VL, SI64, PI32, 2, _mm_i32scatter_epi64( data, ind.data.reg, vec.data.reg, 8 ) );
SIMD_VEC_IMPL_REG_SCATTER( AVX512VL, FP64, PI32, 2, _mm_i32scatter_pd   ( data, ind.data.reg, vec.data.reg, 8 ) );
SIMD_VEC_IMPL_REG_SCATTER( AVX512VL, PI32, PI32, 4, _mm_i32scatter_epi32( data, ind.data.reg, vec.data.reg, 4 ) );
SIMD_VEC_IMPL_REG_SCATTER( AVX512VL, SI32, PI32, 4, _mm_i32scatter_epi32( data, ind.data.reg, vec.data.reg, 4 ) );
SIMD_VEC_IMPL_REG_SCATTER( AVX512VL, FP32, PI32, 4, _mm_i32scatter_ps   ( data, ind.data.reg, vec.data.reg, 4 ) );

SIMD_VEC_IMPL_REG_SCATTER( AVX512VL, PI64, SI32, 2, _mm_i32scatter_epi64( data, ind.data.reg, vec.data.reg, 8 ) );
SIMD_VEC_IMPL_REG_SCATTER( AVX512VL, SI64, SI32, 2, _mm_i32scatter_epi64( data, ind.data.reg, vec.data.reg, 8 ) );
SIMD_VEC_IMPL_REG_SCATTER( AVX512VL, FP64, SI32, 2, _mm_i32scatter_pd   ( data, ind.data.reg, vec.data.reg, 8 ) );
SIMD_VEC_IMPL_REG_SCATTER( AVX512VL, PI32, SI32, 4, _mm_i32scatter_epi32( data, ind.data.reg, vec.data.reg, 4 ) );
SIMD_VEC_IMPL_REG_SCATTER( AVX512VL, SI32, SI32, 4, _mm_i32scatter_epi32( data, ind.data.reg, vec.data.reg, 4 ) );
SIMD_VEC_IMPL_REG_SCATTER( AVX512VL, FP32, SI32, 4, _mm_i32scatter_ps   ( data, ind.data.reg, vec.data.reg, 4 ) );
#endif // ASIMD_X86_HAS_AVX512VL

// cmp simdvec ---------------------------------------------------------------------------------------------
#define SIMD_VEC_IMPL_CMP_OP_SIMDVEC_AVX512( NAME, FLAG_F, FLAG_I ) \
    SIMD_VEC_IMPL_CMP_OP_SIMDVEC( AVX512, PI64,  8, 1, NAME, _mm512_cmp_epu64_mask( a.data.reg, b.data.reg, FLAG_I ) ); \
    SIMD_VEC_IMPL_CMP_OP_SIMDVEC( AVX512, SI64,  8, 1, NAME, _mm512_cmp_epi64_mask( a.data.reg, b.data.reg, FLAG_I ) ); \
    SIMD_VEC_IMPL_CMP_OP_SIMDVEC( AVX512, FP64,  8, 1, NAME, _mm512_cmp_pd_mask   ( a.data.reg, b.data.reg, FLAG_F ) ); \
    SIMD_VEC_IMPL_CMP_OP_SIMDVEC( AVX512, PI32, 16, 1, NAME, _mm512_cmp_epu32_mask( a.data.reg, b.data.reg, FLAG_I ) ); \
    SIMD_VEC_IMPL_CMP_OP_SIMDVEC( AVX512, SI32, 16, 1, NAME, _mm512_cmp_epi32_mask( a.data.reg, b.data.reg, FLAG_I ) ); \
    SIMD_VEC_IMPL_CMP_OP_SIMDVEC( AVX512, FP32, 16, 1, NAME, _mm512_cmp_ps_mask   ( a.data.reg, b.data.reg, FLAG_F ) );

SIMD_VEC_IMPL_CMP_OP_SIMDVEC_AVX512( lt, _CMP_LT_OQ, _MM_CMPINT_LT )
SIMD_VEC_IMPL_CMP_OP_SIMDVEC_AVX512( gt, _CMP_GT_OQ, _MM_CMPINT_GT )

#undef SIMD_VEC_IMPL_CMP_OP_SIMDVEC_AVX512

// 8- and 16-bit lanes -----------------------------------------------------------------------
//
// These comparisons WERE ALREADY HERE, guarded by __AVX512F__ although they are AVX-512BW, and
// -- the reason nobody noticed -- registered against `SimdVecImpl<PI16,32,Arch>` and
// `SimdVecImpl<PI8,64,Arch>`, FOR WHICH NO REGISTER IMPL EXISTED. There was no `.data.reg` to
// write into, so the lines could never instantiate: dead code that also would not have compiled.
//
// `X86CpuFeatures.h` advertises these types (`SimdSize<SI16,AVX512>` is 32), so the honest fix
// is to give them the register impl the comparisons were already assuming, plus the load, store
// and arithmetic that make it usable. Below AVX-512BW they stay on the generic path.
#ifdef ASIMD_X86_HAS_AVX512BW

SIMD_VEC_IMPL_REG( AVX512BW, PI16, 32, __m512i );
SIMD_VEC_IMPL_REG( AVX512BW, SI16, 32, __m512i );
SIMD_VEC_IMPL_REG( AVX512BW, PI8 , 64, __m512i );
SIMD_VEC_IMPL_REG( AVX512BW, SI8 , 64, __m512i );

SIMD_VEC_IMPL_REG_INIT_1( AVX512BW, PI16, 32, _mm512_set1_epi16( a ) );
SIMD_VEC_IMPL_REG_INIT_1( AVX512BW, SI16, 32, _mm512_set1_epi16( a ) );
SIMD_VEC_IMPL_REG_INIT_1( AVX512BW, PI8 , 64, _mm512_set1_epi8 ( a ) );
SIMD_VEC_IMPL_REG_INIT_1( AVX512BW, SI8 , 64, _mm512_set1_epi8 ( a ) );

#define ASIMD_AVX512BW_MEM( T, N ) \
    SIMD_VEC_IMPL_REG_LOAD_ALIGNED   ( AVX512BW, T, N, 512, _mm512_load_si512 ( (const void *)data ) ); \
    SIMD_VEC_IMPL_REG_LOAD_UNALIGNED ( AVX512BW, T, N,      _mm512_loadu_si512( (const void *)data ) ); \
    SIMD_VEC_IMPL_REG_STORE_ALIGNED  ( AVX512BW, T, N, 512, _mm512_store_si512 ( (void *)data, impl.data.reg ) ); \
    SIMD_VEC_IMPL_REG_STORE_UNALIGNED( AVX512BW, T, N,      _mm512_storeu_si512( (void *)data, impl.data.reg ) );

ASIMD_AVX512BW_MEM( PI16, 32 )
ASIMD_AVX512BW_MEM( SI16, 32 )
ASIMD_AVX512BW_MEM( PI8 , 64 )
ASIMD_AVX512BW_MEM( SI8 , 64 )

#undef ASIMD_AVX512BW_MEM

#define ASIMD_AVX512BW_ARITH( T, N, W, SU ) \
    SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX512BW, T, N, add, _mm512_add_epi##W ); \
    SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX512BW, T, N, sub, _mm512_sub_epi##W ); \
    SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX512BW, T, N, min, _mm512_min_ep##SU##W ); \
    SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX512BW, T, N, max, _mm512_max_ep##SU##W ); \
    SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX512BW, T, N, anb, _mm512_and_si512 );

ASIMD_AVX512BW_ARITH( PI16, 32, 16, u )
ASIMD_AVX512BW_ARITH( SI16, 32, 16, i )
ASIMD_AVX512BW_ARITH( PI8 , 64,  8, u )
ASIMD_AVX512BW_ARITH( SI8 , 64,  8, i )

#undef ASIMD_AVX512BW_ARITH

SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX512BW, PI16, 32, mul, _mm512_mullo_epi16 );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX512BW, SI16, 32, mul, _mm512_mullo_epi16 );

#define SIMD_VEC_IMPL_CMP_OP_SIMDVEC_AVX512BW( NAME, FLAG_I ) \
    SIMD_VEC_IMPL_CMP_OP_SIMDVEC( AVX512BW, PI16, 32, 1, NAME, _mm512_cmp_epu16_mask( a.data.reg, b.data.reg, FLAG_I ) ); \
    SIMD_VEC_IMPL_CMP_OP_SIMDVEC( AVX512BW, SI16, 32, 1, NAME, _mm512_cmp_epi16_mask( a.data.reg, b.data.reg, FLAG_I ) ); \
    SIMD_VEC_IMPL_CMP_OP_SIMDVEC( AVX512BW, PI8 , 64, 1, NAME, _mm512_cmp_epu8_mask ( a.data.reg, b.data.reg, FLAG_I ) ); \
    SIMD_VEC_IMPL_CMP_OP_SIMDVEC( AVX512BW, SI8 , 64, 1, NAME, _mm512_cmp_epi8_mask ( a.data.reg, b.data.reg, FLAG_I ) );

SIMD_VEC_IMPL_CMP_OP_SIMDVEC_AVX512BW( lt, _MM_CMPINT_LT )
SIMD_VEC_IMPL_CMP_OP_SIMDVEC_AVX512BW( gt, _MM_CMPINT_GT )

#undef SIMD_VEC_IMPL_CMP_OP_SIMDVEC_AVX512BW

#endif // ASIMD_X86_HAS_AVX512BW

} // namespace internal
} // namespace asimd

#endif // ASIMD_X86_HAS_AVX512F

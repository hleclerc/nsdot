#pragma once

#include "x86_intrin.h"

#ifdef ASIMD_X86_HAS_SSE2

#include "../architectures/X86CpuFeatures.h"
#include "SimdVecImpl_Generic.h"

namespace asimd {
namespace internal {

// struct Impl<...>
SIMD_VEC_IMPL_REG( SSE2, PI64, 2, __m128i );
SIMD_VEC_IMPL_REG( SSE2, SI64, 2, __m128i );
SIMD_VEC_IMPL_REG( SSE2, FP64, 2, __m128d );
SIMD_VEC_IMPL_REG( SSE2, PI32, 4, __m128i );
SIMD_VEC_IMPL_REG( SSE2, SI32, 4, __m128i );
SIMD_VEC_IMPL_REG( SSE2, FP32, 4, __m128  );

// init -------------------------------------------------------------------------------
SIMD_VEC_IMPL_REG_INIT_1( SSE2, PI64, 2, _mm_set1_epi64x( a ) );
SIMD_VEC_IMPL_REG_INIT_1( SSE2, SI64, 2, _mm_set1_epi64x( a ) );
SIMD_VEC_IMPL_REG_INIT_1( SSE2, FP64, 2, _mm_set1_pd    ( a ) );
SIMD_VEC_IMPL_REG_INIT_1( SSE2, PI32, 4, _mm_set1_epi32 ( a ) );
SIMD_VEC_IMPL_REG_INIT_1( SSE2, SI32, 4, _mm_set1_epi32 ( a ) );
SIMD_VEC_IMPL_REG_INIT_1( SSE2, FP32, 4, _mm_set1_ps    ( a ) );

SIMD_VEC_IMPL_REG_INIT_2( SSE2, PI64, 2, _mm_set_epi64x( b, a ) );
SIMD_VEC_IMPL_REG_INIT_2( SSE2, SI64, 2, _mm_set_epi64x( b, a ) );
SIMD_VEC_IMPL_REG_INIT_2( SSE2, FP64, 2, _mm_set_pd    ( b, a ) );
SIMD_VEC_IMPL_REG_INIT_4( SSE2, PI32, 4, _mm_set_epi32 ( d, c, b, a ) );
SIMD_VEC_IMPL_REG_INIT_4( SSE2, SI32, 4, _mm_set_epi32 ( d, c, b, a ) );
SIMD_VEC_IMPL_REG_INIT_4( SSE2, FP32, 4, _mm_set_ps    ( d, c, b, a ) );

// prefetch ----------------------------------------------------------------------
template<int len,class Arch>
auto prefetch( const void *beg, N<len>, S<Arch> ) -> typename std::enable_if<(len <= 64) && Arch::template Has<features::SSE>::value>::type {
    ASIMD_DEBUG_ON_OP("prefetch","SSE","_mm_prefetch")
    // T0, NOT ET0. `beg` is a `const void *`: this is a prefetch for READING. ET0 emits
    // `prefetchw`, which needs the PREFETCHW feature and takes the line in EXCLUSIVE state --
    // on a buffer several threads read, that turns a shared line into a ping-pong.
    _mm_prefetch( reinterpret_cast<const char *>( beg ), _MM_HINT_T0 );
}

template<int len,class Arch>
auto prefetch( const void *beg, N<len>, S<Arch> ) -> typename std::enable_if<(len > 64) && Arch::template Has<features::SSE>::value>::type {
    prefetch( reinterpret_cast<const char *>( beg ) + 0 * len / 2, N<len / 2>(), S<Arch>() );
    prefetch( reinterpret_cast<const char *>( beg ) + 1 * len / 2, N<len / 2>(), S<Arch>() );
}

// load_aligned -----------------------------------------------------------------------
SIMD_VEC_IMPL_REG_LOAD_ALIGNED( SSE2, PI64, 2, 128, _mm_load_si128( (__m128i *)data ) );
SIMD_VEC_IMPL_REG_LOAD_ALIGNED( SSE2, SI64, 2, 128, _mm_load_si128( (__m128i *)data ) );
SIMD_VEC_IMPL_REG_LOAD_ALIGNED( SSE2, FP64, 2, 128, _mm_load_pd   (            data ) );
SIMD_VEC_IMPL_REG_LOAD_ALIGNED( SSE2, PI32, 4, 128, _mm_load_si128( (__m128i *)data ) );
SIMD_VEC_IMPL_REG_LOAD_ALIGNED( SSE2, SI32, 4, 128, _mm_load_si128( (__m128i *)data ) );
SIMD_VEC_IMPL_REG_LOAD_ALIGNED( SSE2, FP32, 4, 128, _mm_load_ps   (            data ) );

// `movntdqa` is SSE4.1, not SSE2 -- guarded by __SSE2__ this file did not compile on a real
// SSE2 target. And the two floating point forms cast to `__m128`: for FP64 that is the wrong
// type outright ("cannot convert __m128 to __m128d"), and for FP32 a C cast between vector
// types where `_mm_castsi128_ps` is the portable spelling (MSVC rejects the C cast).
#ifdef ASIMD_X86_HAS_SSE4_1
SIMD_VEC_IMPL_REG_LOAD_ALIGNED_STREAM( SSE4_1, PI64, 2, 128,                  _mm_stream_load_si128( (__m128i *)data )   );
SIMD_VEC_IMPL_REG_LOAD_ALIGNED_STREAM( SSE4_1, SI64, 2, 128,                  _mm_stream_load_si128( (__m128i *)data )   );
SIMD_VEC_IMPL_REG_LOAD_ALIGNED_STREAM( SSE4_1, FP64, 2, 128, _mm_castsi128_pd( _mm_stream_load_si128( (__m128i *)data ) ) );
SIMD_VEC_IMPL_REG_LOAD_ALIGNED_STREAM( SSE4_1, PI32, 4, 128,                  _mm_stream_load_si128( (__m128i *)data )   );
SIMD_VEC_IMPL_REG_LOAD_ALIGNED_STREAM( SSE4_1, SI32, 4, 128,                  _mm_stream_load_si128( (__m128i *)data )   );
SIMD_VEC_IMPL_REG_LOAD_ALIGNED_STREAM( SSE4_1, FP32, 4, 128, _mm_castsi128_ps( _mm_stream_load_si128( (__m128i *)data ) ) );
#endif

// load ------------------------------------------------------------------------------
SIMD_VEC_IMPL_REG_LOAD_UNALIGNED( SSE2, PI64, 2, _mm_loadu_si128( (__m128i *)data ) );
SIMD_VEC_IMPL_REG_LOAD_UNALIGNED( SSE2, SI64, 2, _mm_loadu_si128( (__m128i *)data ) );
SIMD_VEC_IMPL_REG_LOAD_UNALIGNED( SSE2, FP64, 2, _mm_loadu_pd   (            data ) );
SIMD_VEC_IMPL_REG_LOAD_UNALIGNED( SSE2, PI32, 4, _mm_loadu_si128( (__m128i *)data ) );
SIMD_VEC_IMPL_REG_LOAD_UNALIGNED( SSE2, SI32, 4, _mm_loadu_si128( (__m128i *)data ) );
SIMD_VEC_IMPL_REG_LOAD_UNALIGNED( SSE2, FP32, 4, _mm_loadu_ps   (            data ) );

// store_aligned -----------------------------------------------------------------------
SIMD_VEC_IMPL_REG_STORE_ALIGNED( SSE2, PI64, 2, 128, _mm_store_si128( (__m128i *)data, impl.data.reg ) );
SIMD_VEC_IMPL_REG_STORE_ALIGNED( SSE2, SI64, 2, 128, _mm_store_si128( (__m128i *)data, impl.data.reg ) );
SIMD_VEC_IMPL_REG_STORE_ALIGNED( SSE2, FP64, 2, 128, _mm_store_pd   (            data, impl.data.reg ) );
SIMD_VEC_IMPL_REG_STORE_ALIGNED( SSE2, PI32, 4, 128, _mm_store_si128( (__m128i *)data, impl.data.reg ) );
SIMD_VEC_IMPL_REG_STORE_ALIGNED( SSE2, SI32, 4, 128, _mm_store_si128( (__m128i *)data, impl.data.reg ) );
SIMD_VEC_IMPL_REG_STORE_ALIGNED( SSE2, FP32, 4, 128, _mm_store_ps   (            data, impl.data.reg ) );

SIMD_VEC_IMPL_REG_STORE_ALIGNED_STREAM( SSE2, PI64, 2, 128, _mm_stream_si128( (__m128i *)data, impl.data.reg ) );
SIMD_VEC_IMPL_REG_STORE_ALIGNED_STREAM( SSE2, SI64, 2, 128, _mm_stream_si128( (__m128i *)data, impl.data.reg ) );
SIMD_VEC_IMPL_REG_STORE_ALIGNED_STREAM( SSE2, FP64, 2, 128, _mm_stream_pd   (            data, impl.data.reg ) );
SIMD_VEC_IMPL_REG_STORE_ALIGNED_STREAM( SSE2, PI32, 4, 128, _mm_stream_si128( (__m128i *)data, impl.data.reg ) );
SIMD_VEC_IMPL_REG_STORE_ALIGNED_STREAM( SSE2, SI32, 4, 128, _mm_stream_si128( (__m128i *)data, impl.data.reg ) );
SIMD_VEC_IMPL_REG_STORE_ALIGNED_STREAM( SSE2, FP32, 4, 128, _mm_stream_ps   (            data, impl.data.reg ) );

// store -------------------------------------------------------------------------------
SIMD_VEC_IMPL_REG_STORE_UNALIGNED( SSE2, PI64, 2, _mm_storeu_si128( (__m128i *)data, impl.data.reg ) );
SIMD_VEC_IMPL_REG_STORE_UNALIGNED( SSE2, SI64, 2, _mm_storeu_si128( (__m128i *)data, impl.data.reg ) );
SIMD_VEC_IMPL_REG_STORE_UNALIGNED( SSE2, FP64, 2, _mm_storeu_pd   (            data, impl.data.reg ) );
SIMD_VEC_IMPL_REG_STORE_UNALIGNED( SSE2, PI32, 4, _mm_storeu_si128( (__m128i *)data, impl.data.reg ) );
SIMD_VEC_IMPL_REG_STORE_UNALIGNED( SSE2, SI32, 4, _mm_storeu_si128( (__m128i *)data, impl.data.reg ) );
SIMD_VEC_IMPL_REG_STORE_UNALIGNED( SSE2, FP32, 4, _mm_storeu_ps   (            data, impl.data.reg ) );

//// add / sub: these really are SSE2, on every type ------------------------------------
#define SIMD_VEC_IMPL_REG_ARITHMETIC_OP_SSE2_A( NAME ) \
    SIMD_VEC_IMPL_REG_ARITHMETIC_OP( SSE2, PI64, 2, NAME, _mm_##NAME##_epi64 ); \
    SIMD_VEC_IMPL_REG_ARITHMETIC_OP( SSE2, SI64, 2, NAME, _mm_##NAME##_epi64 ); \
    SIMD_VEC_IMPL_REG_ARITHMETIC_OP( SSE2, FP64, 2, NAME, _mm_##NAME##_pd    ); \
    SIMD_VEC_IMPL_REG_ARITHMETIC_OP( SSE2, PI32, 4, NAME, _mm_##NAME##_epi32 ); \
    SIMD_VEC_IMPL_REG_ARITHMETIC_OP( SSE2, SI32, 4, NAME, _mm_##NAME##_epi32 ); \
    SIMD_VEC_IMPL_REG_ARITHMETIC_OP( SSE2, FP32, 4, NAME, _mm_##NAME##_ps    );

    SIMD_VEC_IMPL_REG_ARITHMETIC_OP_SSE2_A( add );
    SIMD_VEC_IMPL_REG_ARITHMETIC_OP_SSE2_A( sub );

#undef SIMD_VEC_IMPL_REG_ARITHMETIC_OP_SSE2_A

//// min / max ---------------------------------------------------------------------------
//
// TWO THINGS WERE WRONG HERE. `_mm_min_epi32` is SSE4.1 and `_mm_min_epi64` is AVX-512VL, both
// registered under SSE2; and the UNSIGNED types were routed to the SIGNED instruction, so
// `min( PI32( 0xFFFFFFFF ), 9 )` returned 0xFFFFFFFF -- 0xFFFFFFFF read as -1.
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( SSE2, FP64, 2, min, _mm_min_pd );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( SSE2, FP32, 4, min, _mm_min_ps );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( SSE2, FP64, 2, max, _mm_max_pd );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( SSE2, FP32, 4, max, _mm_max_ps );

#ifdef ASIMD_X86_HAS_SSE4_1
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( SSE4_1, SI32, 4, min, _mm_min_epi32 );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( SSE4_1, PI32, 4, min, _mm_min_epu32 );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( SSE4_1, SI32, 4, max, _mm_max_epi32 );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( SSE4_1, PI32, 4, max, _mm_max_epu32 );

//// 32-bit integer multiply is SSE4.1 as well (`pmulld`); below that the generic form applies
//// the vector operator to `values` and gcc/clang synthesise it.
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( SSE4_1, SI32, 4, mul, _mm_mullo_epi32 );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( SSE4_1, PI32, 4, mul, _mm_mullo_epi32 );
#endif

//// 64-bit integer min/max only exist as instructions from AVX-512VL on.
#ifdef ASIMD_X86_HAS_AVX512VL
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX512VL, SI64, 2, min, _mm_min_epi64 );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX512VL, PI64, 2, min, _mm_min_epu64 );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX512VL, SI64, 2, max, _mm_max_epi64 );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX512VL, PI64, 2, max, _mm_max_epu64 );
#endif

//// arithmetic operations that work only on float types ------------------------------------------------
#define SIMD_VEC_IMPL_REG_ARITHMETIC_OP_SSE2_F( NAME ) \
    SIMD_VEC_IMPL_REG_ARITHMETIC_OP( SSE2, double, 2, NAME, _mm_##NAME##_pd ); \
    SIMD_VEC_IMPL_REG_ARITHMETIC_OP( SSE2, float , 4, NAME, _mm_##NAME##_ps );

    SIMD_VEC_IMPL_REG_ARITHMETIC_OP_SSE2_F( mul );
    SIMD_VEC_IMPL_REG_ARITHMETIC_OP_SSE2_F( div );

#undef SIMD_VEC_IMPL_REG_ARITHMETIC_OP_SSE2_F

//// arithmetic operations with != func and name -------------------------------------------------------
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( SSE2, PI64, 2, anb, _mm_and_si128 );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( SSE2, SI64, 2, anb, _mm_and_si128 );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( SSE2, FP64, 2, anb, _mm_and_pd    );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( SSE2, PI32, 4, anb, _mm_and_si128 );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( SSE2, SI32, 4, anb, _mm_and_si128 );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( SSE2, FP32, 4, anb, _mm_and_ps    );

//// INTEGER COMPARISONS. `pcmpgtd` has been there since SSE2 and nothing was registered against
//// it: every integer comparison at 128 bits went lane by lane. Same shape as the AVX2 forms --
//// signed only in hardware, unsigned obtained by flipping the sign bit of both operands.
SIMD_VEC_IMPL_CMP_OP_SIMDVEC( SSE2, SI32, 4, 32, gt, _mm_cmpgt_epi32( a.data.reg, b.data.reg ) );
SIMD_VEC_IMPL_CMP_OP_SIMDVEC( SSE2, SI32, 4, 32, lt, _mm_cmpgt_epi32( b.data.reg, a.data.reg ) );

#define ASIMD_SSE2_UNSIGNED_CMP( T, N, IS, W, SIGN ) \
    SIMD_VEC_IMPL_CMP_OP_SIMDVEC( SSE2, T, N, IS, gt, _mm_cmpgt_epi##W( \
        _mm_xor_si128( a.data.reg, SIGN ), _mm_xor_si128( b.data.reg, SIGN ) ) ); \
    SIMD_VEC_IMPL_CMP_OP_SIMDVEC( SSE2, T, N, IS, lt, _mm_cmpgt_epi##W( \
        _mm_xor_si128( b.data.reg, SIGN ), _mm_xor_si128( a.data.reg, SIGN ) ) );

ASIMD_SSE2_UNSIGNED_CMP( PI32, 4, 32, 32, _mm_set1_epi32( int( 0x80000000u ) ) )

#undef ASIMD_SSE2_UNSIGNED_CMP

//// `pcmpgtq` on 64-bit lanes is SSE4.2, not SSE2.
#ifdef ASIMD_X86_HAS_SSE4_2
SIMD_VEC_IMPL_CMP_OP_SIMDVEC( SSE4_2, SI64, 2, 64, gt, _mm_cmpgt_epi64( a.data.reg, b.data.reg ) );
SIMD_VEC_IMPL_CMP_OP_SIMDVEC( SSE4_2, SI64, 2, 64, lt, _mm_cmpgt_epi64( b.data.reg, a.data.reg ) );
SIMD_VEC_IMPL_CMP_OP_SIMDVEC( SSE4_2, PI64, 2, 64, gt, _mm_cmpgt_epi64(
    _mm_xor_si128( a.data.reg, _mm_set1_epi64x( SI64( 0x8000000000000000ull ) ) ),
    _mm_xor_si128( b.data.reg, _mm_set1_epi64x( SI64( 0x8000000000000000ull ) ) ) ) );
SIMD_VEC_IMPL_CMP_OP_SIMDVEC( SSE4_2, PI64, 2, 64, lt, _mm_cmpgt_epi64(
    _mm_xor_si128( b.data.reg, _mm_set1_epi64x( SI64( 0x8000000000000000ull ) ) ),
    _mm_xor_si128( a.data.reg, _mm_set1_epi64x( SI64( 0x8000000000000000ull ) ) ) ) );
#endif

// iota, 128 bits -----------------------------------------------------------------------------
#define ASIMD_SSE2_IOTA( T, N, SET, ADD, SET1 ) \
    template<class Arch> requires ( Arch::template Has<features::SSE2>::value ) HaD \
    SimdVecImpl<T,N,Arch> iota( T beg, S<SimdVecImpl<T,N,Arch>> ) { \
        SimdVecImpl<T,N,Arch> res; res.data.reg = ADD( SET, SET1( beg ) ); return res; \
    }

ASIMD_SSE2_IOTA( FP32, 4, _mm_setr_ps( 0, 1, 2, 3 ), _mm_add_ps, _mm_set1_ps );
ASIMD_SSE2_IOTA( FP64, 2, _mm_setr_pd( 0, 1 ), _mm_add_pd, _mm_set1_pd );
ASIMD_SSE2_IOTA( SI32, 4, _mm_setr_epi32( 0, 1, 2, 3 ), _mm_add_epi32, _mm_set1_epi32 );
ASIMD_SSE2_IOTA( PI32, 4, _mm_setr_epi32( 0, 1, 2, 3 ), _mm_add_epi32, _mm_set1_epi32 );
ASIMD_SSE2_IOTA( SI64, 2, _mm_set_epi64x( 1, 0 ), _mm_add_epi64, _mm_set1_epi64x );
ASIMD_SSE2_IOTA( PI64, 2, _mm_set_epi64x( 1, 0 ), _mm_add_epi64, _mm_set1_epi64x );

#undef ASIMD_SSE2_IOTA

//// horizontal sum -----------------------------------------------------------------------
//// The classic ladder: fold the high half onto the low half, twice, then read lane 0.
SIMD_VEC_IMPL_REG_HSUM( SSE2, FP32, 4, ( [ & ] {
    __m128 v = impl.data.reg;
    v = _mm_add_ps( v, _mm_movehl_ps( v, v ) );
    v = _mm_add_ss( v, _mm_shuffle_ps( v, v, 1 ) );
    return _mm_cvtss_f32( v ); }() ) );
SIMD_VEC_IMPL_REG_HSUM( SSE2, FP64, 2, ( [ & ] {
    __m128d v = impl.data.reg;
    return _mm_cvtsd_f64( _mm_add_sd( v, _mm_unpackhi_pd( v, v ) ) ); }() ) );

#define ASIMD_SSE2_HSUM_I32( T ) \
    SIMD_VEC_IMPL_REG_HSUM( SSE2, T, 4, ( [ & ] { \
        __m128i v = impl.data.reg; \
        v = _mm_add_epi32( v, _mm_shuffle_epi32( v, _MM_SHUFFLE( 1, 0, 3, 2 ) ) ); \
        v = _mm_add_epi32( v, _mm_shuffle_epi32( v, _MM_SHUFFLE( 2, 3, 0, 1 ) ) ); \
        return T( _mm_cvtsi128_si32( v ) ); }() ) )
ASIMD_SSE2_HSUM_I32( SI32 );
ASIMD_SSE2_HSUM_I32( PI32 );
#undef ASIMD_SSE2_HSUM_I32

// `_mm_cvtsi128_si64` moves a 64-bit lane to a general register, which only exists on x86-64:
// on 32-bit there is no 64-bit general register to move it to. Read the two halves instead.
#if defined( __x86_64__ ) || defined( _M_X64 )
    #define ASIMD_SSE2_LANE0_64( v ) _mm_cvtsi128_si64( v )
#else
    #define ASIMD_SSE2_LANE0_64( v ) ( ( PI64( PI32( _mm_cvtsi128_si32( _mm_shuffle_epi32( v, 1 ) ) ) ) << 32 ) \
                                     |   PI64( PI32( _mm_cvtsi128_si32( v ) ) ) )
#endif

#define ASIMD_SSE2_HSUM_I64( T ) \
    SIMD_VEC_IMPL_REG_HSUM( SSE2, T, 2, ( [ & ] { \
        __m128i v = impl.data.reg; \
        v = _mm_add_epi64( v, _mm_unpackhi_epi64( v, v ) ); \
        return T( ASIMD_SSE2_LANE0_64( v ) ); }() ) )
ASIMD_SSE2_HSUM_I64( SI64 );
ASIMD_SSE2_HSUM_I64( PI64 );
#undef ASIMD_SSE2_HSUM_I64

//// FP comparisons at 128 bits exist in SSE2 too (`cmpps` / `cmppd`, non-VEX).
////
//// AND NOT ON AVX, which is not an optimisation but a correctness fix. `SimdVecImpl_AVX.h`
//// registers the SAME two cells -- `FP32 x 4` and `FP64 x 2` -- with the VEX three-operand form
//// and a predicate operand. These are ORDINARY FUNCTION OVERLOADS, not ranked variants, so on a
//// target with both features both constraints held, neither subsumed the other, and the call
//// was ambiguous: `any( a > b )` on four floats did not compile under `-mavx`. That is exactly
//// the situation `Selection.h`'s KNOWN LIMITS note describes, and the remedy it prescribes --
//// state that the older form is the FALLBACK, not a rival.
SIMD_VEC_IMPL_CMP_OP_SIMDVEC_EXCL( SSE2, AVX, FP32, 4, 32, gt, _mm_castps_si128( _mm_cmpgt_ps( a.data.reg, b.data.reg ) ) );
SIMD_VEC_IMPL_CMP_OP_SIMDVEC_EXCL( SSE2, AVX, FP32, 4, 32, lt, _mm_castps_si128( _mm_cmplt_ps( a.data.reg, b.data.reg ) ) );
SIMD_VEC_IMPL_CMP_OP_SIMDVEC_EXCL( SSE2, AVX, FP64, 2, 64, gt, _mm_castpd_si128( _mm_cmpgt_pd( a.data.reg, b.data.reg ) ) );
SIMD_VEC_IMPL_CMP_OP_SIMDVEC_EXCL( SSE2, AVX, FP64, 2, 64, lt, _mm_castpd_si128( _mm_cmplt_pd( a.data.reg, b.data.reg ) ) );

} // namespace internal
} // namespace asimd

#endif // ASIMD_X86_HAS_SSE2

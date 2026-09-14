#pragma once

#include "x86_intrin.h"

#ifdef ASIMD_X86_HAS_AVX2

#include "../architectures/X86CpuFeatures.h"
#include "SimdVecImpl_Generic.h"
#include "SimdVecImpl_AVX.h"

namespace asimd {
namespace internal {

//// THE 256-BIT INTEGER ALU. This is what AVX2 adds over AVX -- `vpaddd`, `vpsubq`, `vpminsd`,
//// `vpmulld`, `vpand` and the integer compares. They used to be registered under `features::AVX`
//// one file over, which meant a genuine AVX target (Sandy Bridge, Ivy Bridge) could not compile
//// a single integer vector operation.
#define SIMD_VEC_IMPL_REG_ARITHMETIC_OP_AVX2_I( NAME ) \
    SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX2, PI64, 4, NAME, _mm256_##NAME##_epi64 ); \
    SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX2, SI64, 4, NAME, _mm256_##NAME##_epi64 ); \
    SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX2, PI32, 8, NAME, _mm256_##NAME##_epi32 ); \
    SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX2, SI32, 8, NAME, _mm256_##NAME##_epi32 );

SIMD_VEC_IMPL_REG_ARITHMETIC_OP_AVX2_I( add );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP_AVX2_I( sub );

#undef SIMD_VEC_IMPL_REG_ARITHMETIC_OP_AVX2_I

//// min / max: SIGNEDNESS IS NOT OPTIONAL. `_mm256_min_epi32` on a PI32 reads 0xFFFFFFFF as -1.
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX2, SI32, 8, min, _mm256_min_epi32 );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX2, PI32, 8, min, _mm256_min_epu32 );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX2, SI32, 8, max, _mm256_max_epi32 );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX2, PI32, 8, max, _mm256_max_epu32 );
//// 64-bit integer min/max are AVX-512VL; see SimdVecImpl_AVX512.h.

//// 32-bit multiply. Nothing registered one anywhere before, at any width.
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX2, SI32, 8, mul, _mm256_mullo_epi32 );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX2, PI32, 8, mul, _mm256_mullo_epi32 );

//// arithmetic operations with != func and name -------------------------------------------------------
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX2, PI64, 4, anb, _mm256_and_si256 );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX2, SI64, 4, anb, _mm256_and_si256 );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX2, PI32, 8, anb, _mm256_and_si256 );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX2, SI32, 8, anb, _mm256_and_si256 );

//// INTEGER COMPARISONS. There is no VEX integer compare with a predicate operand before
//// AVX-512: only `vpcmpgtd` / `vpcmpgtq`, and only SIGNED. An unsigned compare is the signed one
//// on operands with their sign bit flipped, which is one extra `vpxor` per operand and still far
//// cheaper than the lane-by-lane fallback. `lt( a, b )` is `gt( b, a )` -- the operands swap.
#define ASIMD_AVX2_SIGNED_CMP( T, N, IS, W, A, B ) \
    SIMD_VEC_IMPL_CMP_OP_SIMDVEC( AVX2, T, N, IS, gt, _mm256_cmpgt_epi##W( A.data.reg, B.data.reg ) ); \
    SIMD_VEC_IMPL_CMP_OP_SIMDVEC( AVX2, T, N, IS, lt, _mm256_cmpgt_epi##W( B.data.reg, A.data.reg ) );

ASIMD_AVX2_SIGNED_CMP( SI32, 8, 32, 32, a, b )
ASIMD_AVX2_SIGNED_CMP( SI64, 4, 64, 64, a, b )

#undef ASIMD_AVX2_SIGNED_CMP

#define ASIMD_AVX2_UNSIGNED_CMP( T, N, IS, W, SIGN ) \
    SIMD_VEC_IMPL_CMP_OP_SIMDVEC( AVX2, T, N, IS, gt, _mm256_cmpgt_epi##W( \
        _mm256_xor_si256( a.data.reg, SIGN ), _mm256_xor_si256( b.data.reg, SIGN ) ) ); \
    SIMD_VEC_IMPL_CMP_OP_SIMDVEC( AVX2, T, N, IS, lt, _mm256_cmpgt_epi##W( \
        _mm256_xor_si256( b.data.reg, SIGN ), _mm256_xor_si256( a.data.reg, SIGN ) ) );

ASIMD_AVX2_UNSIGNED_CMP( PI32, 8, 32, 32, _mm256_set1_epi32( int( 0x80000000u ) ) )
ASIMD_AVX2_UNSIGNED_CMP( PI64, 4, 64, 64, _mm256_set1_epi64x( SI64( 0x8000000000000000ull ) ) )

#undef ASIMD_AVX2_UNSIGNED_CMP


//// arithmetic operations that work only on int types -------------------------------------------------
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX2, PI64, 4, sll, _mm256_sllv_epi64 );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX2, SI64, 4, sll, _mm256_sllv_epi64 );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX2, PI32, 8, sll, _mm256_sllv_epi32 );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX2, SI32, 8, sll, _mm256_sllv_epi32 );

// => SSE2 size
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX2, PI64, 2, sll, _mm_sllv_epi64 );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX2, SI64, 2, sll, _mm_sllv_epi64 );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX2, PI32, 4, sll, _mm_sllv_epi32 );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( AVX2, SI32, 4, sll, _mm_sllv_epi32 );

// iota ------------------------------------------------------------------------------------------------
// WITHOUT THIS, `iota` falls back to the generic form -- a loop writing lane by lane -- and gcc
// materializes it as a chain of `vpinsrd` rather than a constant load. Measured: six `vpinsrd`
// per call, inside a hot loop.
template<class Arch> requires ( Arch::template Has<features::AVX2>::value ) HaD
SimdVecImpl<SI32,8,Arch> iota( SI32 beg, S<SimdVecImpl<SI32,8,Arch>> ) {
    SimdVecImpl<SI32,8,Arch> res;
    res.data.reg = _mm256_add_epi32( _mm256_setr_epi32( 0, 1, 2, 3, 4, 5, 6, 7 ), _mm256_set1_epi32( beg ) );
    return res;
}

template<class Arch> requires ( Arch::template Has<features::AVX2>::value ) HaD
SimdVecImpl<PI32,8,Arch> iota( PI32 beg, S<SimdVecImpl<PI32,8,Arch>> ) {
    SimdVecImpl<PI32,8,Arch> res;
    res.data.reg = _mm256_add_epi32( _mm256_setr_epi32( 0, 1, 2, 3, 4, 5, 6, 7 ), _mm256_set1_epi32( beg ) );
    return res;
}

// iota, the rest of the 256-bit set ----------------------------------------------------------
#define ASIMD_AVX2_IOTA( T, N, SET, ADD, SET1 ) \
    template<class Arch> requires ( Arch::template Has<features::AVX2>::value ) HaD \
    SimdVecImpl<T,N,Arch> iota( T beg, S<SimdVecImpl<T,N,Arch>> ) { \
        SimdVecImpl<T,N,Arch> res; res.data.reg = ADD( SET, SET1( beg ) ); return res; \
    }

ASIMD_AVX2_IOTA( FP64, 4, _mm256_setr_pd( 0, 1, 2, 3 ), _mm256_add_pd, _mm256_set1_pd );
ASIMD_AVX2_IOTA( SI64, 4, _mm256_setr_epi64x( 0, 1, 2, 3 ), _mm256_add_epi64, _mm256_set1_epi64x );
ASIMD_AVX2_IOTA( PI64, 4, _mm256_setr_epi64x( 0, 1, 2, 3 ), _mm256_add_epi64, _mm256_set1_epi64x );

#undef ASIMD_AVX2_IOTA

//// horizontal sum on the integer types: the 256-bit fold is `vpaddd`, hence AVX2.
#define ASIMD_AVX2_HSUM( T, N, HALF, ADD ) \
    SIMD_VEC_IMPL_REG_HSUM( AVX2, T, N, horizontal_sum( impl_from_reg<SimdVecImpl<T,HALF,Arch>>( ADD( \
        _mm256_castsi256_si128( impl.data.reg ), _mm256_extracti128_si256( impl.data.reg, 1 ) ) ) ) )
ASIMD_AVX2_HSUM( SI32, 8, 4, _mm_add_epi32 );
ASIMD_AVX2_HSUM( PI32, 8, 4, _mm_add_epi32 );
ASIMD_AVX2_HSUM( SI64, 4, 2, _mm_add_epi64 );
ASIMD_AVX2_HSUM( PI64, 4, 2, _mm_add_epi64 );
#undef ASIMD_AVX2_HSUM

// gather -----------------------------------------------------------------------------------------------
SIMD_VEC_IMPL_REG_GATHER( AVX2, PI64, PI32, 4, _mm256_i32gather_epi64( (const long long *)data, ind.data.reg, 8 ) );
SIMD_VEC_IMPL_REG_GATHER( AVX2, SI64, PI32, 4, _mm256_i32gather_epi64( (const long long *)data, ind.data.reg, 8 ) );
SIMD_VEC_IMPL_REG_GATHER( AVX2, FP64, PI32, 4, _mm256_i32gather_pd   ( data, ind.data.reg, 8 ) );
SIMD_VEC_IMPL_REG_GATHER( AVX2, PI32, PI32, 8, _mm256_i32gather_epi32( (const int *)data, ind.data.reg, 4 ) );
SIMD_VEC_IMPL_REG_GATHER( AVX2, SI32, PI32, 8, _mm256_i32gather_epi32( (const int *)data, ind.data.reg, 4 ) );
SIMD_VEC_IMPL_REG_GATHER( AVX2, FP32, PI32, 8, _mm256_i32gather_ps   ( data, ind.data.reg, 4 ) );

SIMD_VEC_IMPL_REG_GATHER( AVX2, PI64, SI32, 4, _mm256_i32gather_epi64( (const long long *)data, ind.data.reg, 8 ) );
SIMD_VEC_IMPL_REG_GATHER( AVX2, SI64, SI32, 4, _mm256_i32gather_epi64( (const long long *)data, ind.data.reg, 8 ) );
SIMD_VEC_IMPL_REG_GATHER( AVX2, FP64, SI32, 4, _mm256_i32gather_pd   ( data, ind.data.reg, 8 ) );
SIMD_VEC_IMPL_REG_GATHER( AVX2, PI32, SI32, 8, _mm256_i32gather_epi32( (const int *)data, ind.data.reg, 4 ) );
SIMD_VEC_IMPL_REG_GATHER( AVX2, SI32, SI32, 8, _mm256_i32gather_epi32( (const int *)data, ind.data.reg, 4 ) );
SIMD_VEC_IMPL_REG_GATHER( AVX2, FP32, SI32, 8, _mm256_i32gather_ps   ( data, ind.data.reg, 4 ) );

// SSE2 sizes
SIMD_VEC_IMPL_REG_GATHER( AVX2, PI64, PI32, 2, _mm_i32gather_epi64   ( (const long long *)data, ind.data.reg, 8 ) );
SIMD_VEC_IMPL_REG_GATHER( AVX2, SI64, PI32, 2, _mm_i32gather_epi64   ( data, ind.data.reg, 8 ) );
SIMD_VEC_IMPL_REG_GATHER( AVX2, FP64, PI32, 2, _mm_i32gather_pd      ( data, ind.data.reg, 8 ) );
SIMD_VEC_IMPL_REG_GATHER( AVX2, PI32, PI32, 4, _mm_i32gather_epi32   ( (const int *)data, ind.data.reg, 4 ) );
SIMD_VEC_IMPL_REG_GATHER( AVX2, SI32, PI32, 4, _mm_i32gather_epi32   ( (const int *)data, ind.data.reg, 4 ) );
SIMD_VEC_IMPL_REG_GATHER( AVX2, FP32, PI32, 4, _mm_i32gather_ps      ( data, ind.data.reg, 4 ) );
   
SIMD_VEC_IMPL_REG_GATHER( AVX2, PI64, SI32, 2, _mm_i32gather_epi64   ( (const long long *)data, ind.data.reg, 8 ) );
SIMD_VEC_IMPL_REG_GATHER( AVX2, SI64, SI32, 2, _mm_i32gather_epi64   ( data, ind.data.reg, 8 ) );
SIMD_VEC_IMPL_REG_GATHER( AVX2, FP64, SI32, 2, _mm_i32gather_pd      ( data, ind.data.reg, 8 ) );
SIMD_VEC_IMPL_REG_GATHER( AVX2, PI32, SI32, 4, _mm_i32gather_epi32   ( (const int *)data, ind.data.reg, 4 ) );
SIMD_VEC_IMPL_REG_GATHER( AVX2, SI32, SI32, 4, _mm_i32gather_epi32   ( (const int *)data, ind.data.reg, 4 ) );
SIMD_VEC_IMPL_REG_GATHER( AVX2, FP32, SI32, 4, _mm_i32gather_ps      ( data, ind.data.reg, 4 ) );


} // namespace internal
} // namespace asimd

#endif // ASIMD_X86_HAS_AVX2
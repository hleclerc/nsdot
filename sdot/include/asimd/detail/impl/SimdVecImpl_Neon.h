#pragma once

#include "arm_intrin.h"

#ifdef ASIMD_ARM_HAS_NEON

#include "../architectures/ArmCpuFeatures.h"
#include "SimdVecImpl_Generic.h"

// =============================================================================================
// THE 128-BIT ARM BACKEND.
//
// Laid out like `SimdVecImpl_SSE2.h`, which is its opposite number: same width, same six main
// lane types, same macros from `SimdVecImpl_Generic.h`. Four things differ, and all four are
// visible below.
//
//   TWO FEATURE LEVELS, NOT ONE. Everything registered against `NEON` is an instruction an
//   ARMv7-A part really has; everything against `ASIMD` is A64 only. `float64x2_t`, `vdivq_*`,
//   the horizontal reductions and the 64-bit compares are all on the A64 side. See
//   `ArmCpuFeatures.h`.
//
//   THE 64-BIT INTEGER TYPES ARE NOT WHAT THE INTRINSICS EXPECT. `asimd::SI64` is `long` on an
//   LP64 target, `int64_t` is `long long` on Darwin -- the same width and a DIFFERENT TYPE, so
//   `vld1q_s64( data )` does not compile there. Every load and store therefore casts its pointer
//   to the ACLE scalar type explicitly, which is also the only spelling that is right on both
//   Linux and Darwin.
//
//   NEON IS UNIFORM WHERE SSE IS NOT, and that is worth using rather than mirroring. The
//   unsigned compares are real instructions (`vcgtq_u32`), where x86 needs both operands' sign
//   bits flipped; the per-lane variable shift (`vshlq_*`) is one instruction at every width,
//   where x86 has none below AVX2. Both are registered here, so the ARM side of the grid is
//   FULLER than the x86 one at 128 bits, not thinner.
//
//   THERE IS NO ALIGNED/UNALIGNED DISTINCTION. `vld1q_f32` handles both; ARMv7's `:128` address
//   qualifier is not exposed by ACLE and A64 dropped it. So the aligned and unaligned forms are
//   the same instruction, registered twice on purpose -- `SimdVec::load( Ptr )` dispatches on
//   `P::alignment` and needs both to exist.
// =============================================================================================

namespace asimd {
namespace internal {

// =============================================================================================
// WHAT EVERY TYPE GETS, at 128 bits: the impl struct, the broadcast, load/store in all four
// flavours, and add/sub -- the operations whose intrinsic name follows `v<op>q_<suffix>` with no
// exception anywhere in the type list.
//
// ELT is the ACLE scalar type, which is what the pointer must be cast to; see the note above.
// =============================================================================================
#define ASIMD_NEON_COMMON( COND, T, N, REG, SUF, ELT ) \
    SIMD_VEC_IMPL_REG( COND, T, N, REG ); \
    SIMD_VEC_IMPL_REG_INIT_1( COND, T, N, vdupq_n_##SUF( ELT( a ) ) ); \
    SIMD_VEC_IMPL_REG_LOAD_UNALIGNED ( COND, T, N,      vld1q_##SUF( (const ELT *) data ) ); \
    SIMD_VEC_IMPL_REG_LOAD_ALIGNED   ( COND, T, N, 128, vld1q_##SUF( (const ELT *) data ) ); \
    SIMD_VEC_IMPL_REG_STORE_UNALIGNED( COND, T, N,      vst1q_##SUF( (ELT *) data, impl.data.reg ) ); \
    SIMD_VEC_IMPL_REG_STORE_ALIGNED  ( COND, T, N, 128, vst1q_##SUF( (ELT *) data, impl.data.reg ) ); \
    /* NO NON-TEMPORAL SIMD LOAD OR STORE ON ARM. A64 has `LDNP`/`STNP`, which ACLE does not */ \
    /* expose, and ARMv7 has nothing at all. `stream` is a HINT, so the honest mapping is the */ \
    /* ordinary instruction -- and it is the one that matters: without these the generic form */ \
    /* walks the lanes one at a time, which is not a missing hint but a missing vector store. */ \
    SIMD_VEC_IMPL_REG_LOAD_ALIGNED_STREAM ( COND, T, N, 128, vld1q_##SUF( (const ELT *) data ) ); \
    SIMD_VEC_IMPL_REG_STORE_ALIGNED_STREAM( COND, T, N, 128, vst1q_##SUF( (ELT *) data, impl.data.reg ) ); \
    SIMD_VEC_IMPL_REG_ARITHMETIC_OP( COND, T, N, add, vaddq_##SUF ); \
    SIMD_VEC_IMPL_REG_ARITHMETIC_OP( COND, T, N, sub, vsubq_##SUF )

// ---------------------------------------------------------------------------------------------
// `iota`, at 128 bits. The register form exists for the same reason as on x86: written through
// `values` the constant is materialised one lane at a time -- and here there is no
// `vector_size` arithmetic to save it on the MSVC path.
// ---------------------------------------------------------------------------------------------
#define ASIMD_NEON_IOTA( COND, T, N, SUF, ELT, ... ) \
    template<class Arch> requires ( Arch::template Has<features::COND>::value ) HaD \
    SimdVecImpl<T,N,Arch> iota( T beg, S<SimdVecImpl<T,N,Arch>> ) { \
        ASIMD_DEBUG_ON_OP("iota",#COND,"vaddq") \
        const ELT k[ N ] = { __VA_ARGS__ }; \
        SimdVecImpl<T,N,Arch> res; \
        res.data.reg = vaddq_##SUF( vld1q_##SUF( k ), vdupq_n_##SUF( ELT( beg ) ) ); \
        return res; \
    }

// ---------------------------------------------------------------------------------------------
// the comparisons that feed `SimdVec`'s lazy `a > b`. One instruction per relation, and the
// UNSIGNED ones are instructions too -- `ASIMD_SSE2_UNSIGNED_CMP` needs four.
// ---------------------------------------------------------------------------------------------
#define ASIMD_NEON_CMP( COND, T, N, IS, SUF ) \
    SIMD_VEC_IMPL_CMP_OP_SIMDVEC( COND, T, N, IS, gt, vcgtq_##SUF( a.data.reg, b.data.reg ) ); \
    SIMD_VEC_IMPL_CMP_OP_SIMDVEC( COND, T, N, IS, lt, vcltq_##SUF( a.data.reg, b.data.reg ) )

// =============================================================================================
// 1. THE ARMv7-A FLOOR -- everything a 32-bit NEON part has
// =============================================================================================

// ---- the impl structs, the broadcast, load/store, add/sub -----------------------------------
ASIMD_NEON_COMMON( NEON, FP32,  4, float32x4_t, f32, float          );
ASIMD_NEON_COMMON( NEON, SI32,  4, int32x4_t  , s32, std::int32_t   );
ASIMD_NEON_COMMON( NEON, PI32,  4, uint32x4_t , u32, std::uint32_t  );
ASIMD_NEON_COMMON( NEON, SI64,  2, int64x2_t  , s64, std::int64_t   );
ASIMD_NEON_COMMON( NEON, PI64,  2, uint64x2_t , u64, std::uint64_t  );
ASIMD_NEON_COMMON( NEON, SI16,  8, int16x8_t  , s16, std::int16_t   );
ASIMD_NEON_COMMON( NEON, PI16,  8, uint16x8_t , u16, std::uint16_t  );
ASIMD_NEON_COMMON( NEON, SI8 , 16, int8x16_t  , s8 , std::int8_t    );
ASIMD_NEON_COMMON( NEON, PI8 , 16, uint8x16_t , u8 , std::uint8_t   );

// ---- prefetch -------------------------------------------------------------------------------
//
// `PRFM PLDL1KEEP`, i.e. a prefetch for READING that keeps the line in a shared state -- the
// same choice as the x86 side, and for the same reason: the write-intent form takes the line
// exclusive, which turns a buffer several threads read into a ping-pong.
template<int len,class Arch>
auto prefetch( const void *beg, N<len>, S<Arch> ) -> typename std::enable_if<(len <= 64) && Arch::template Has<features::NEON>::value>::type {
    ASIMD_DEBUG_ON_OP("prefetch","NEON","prfm")
#if defined( _MSC_VER ) && ! defined( __clang__ )
    __prefetch( beg );
#else
    __builtin_prefetch( beg, 0, 3 );
#endif
}

template<int len,class Arch>
auto prefetch( const void *beg, N<len>, S<Arch> ) -> typename std::enable_if<(len > 64) && Arch::template Has<features::NEON>::value>::type {
    prefetch( reinterpret_cast<const char *>( beg ) + 0 * len / 2, N<len / 2>(), S<Arch>() );
    prefetch( reinterpret_cast<const char *>( beg ) + 1 * len / 2, N<len / 2>(), S<Arch>() );
}

// ---- multiply -------------------------------------------------------------------------------
//
// `vmulq_s64` DOES NOT EXIST, at any ARM level -- there is no 64-bit SIMD integer multiply, as
// on x86 below AVX-512DQ. Those two cells keep the generic form, which applies `*` to the whole
// `values` vector and lets the compiler synthesise it.
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( NEON, FP32,  4, mul, vmulq_f32 );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( NEON, SI32,  4, mul, vmulq_s32 );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( NEON, PI32,  4, mul, vmulq_u32 );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( NEON, SI16,  8, mul, vmulq_s16 );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( NEON, PI16,  8, mul, vmulq_u16 );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( NEON, SI8 , 16, mul, vmulq_s8  );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( NEON, PI8 , 16, mul, vmulq_u8  );

// ---- min / max ------------------------------------------------------------------------------
//
// SIGNEDNESS IS PART OF THE INSTRUCTION NAME HERE, which is what makes this table hard to get
// wrong -- the x86 one had every unsigned type routed to the signed instruction, so
// `min( PI32( 0xFFFFFFFF ), 9 )` returned 0xFFFFFFFF.
//
// `vminq_f32` is FMIN, not minNum: it PROPAGATES a NaN where `_mm_min_ps` returns its second
// operand. `vminnmq_f32` is the IEEE-754 `minNum` and needs `__ARM_FEATURE_NUMERIC_MAXMIN`.
// FMIN is the closer match to `std::min` on ordered inputs and is what every other portable
// layer picks, so that is what is registered; the difference only shows on a NaN.
//
// 64-bit integer min/max, again, have no instruction -- exactly as on x86 before AVX-512VL.
#define ASIMD_NEON_MINMAX( COND, T, N, SUF ) \
    SIMD_VEC_IMPL_REG_ARITHMETIC_OP( COND, T, N, min, vminq_##SUF ); \
    SIMD_VEC_IMPL_REG_ARITHMETIC_OP( COND, T, N, max, vmaxq_##SUF )

ASIMD_NEON_MINMAX( NEON, FP32,  4, f32 );
ASIMD_NEON_MINMAX( NEON, SI32,  4, s32 );
ASIMD_NEON_MINMAX( NEON, PI32,  4, u32 );
ASIMD_NEON_MINMAX( NEON, SI16,  8, s16 );
ASIMD_NEON_MINMAX( NEON, PI16,  8, u16 );
ASIMD_NEON_MINMAX( NEON, SI8 , 16, s8  );
ASIMD_NEON_MINMAX( NEON, PI8 , 16, u8  );

// ---- bitwise and ----------------------------------------------------------------------------
//
// On the floating point types this is a reinterpret sandwich rather than a `vandq_f32`, because
// there is no such intrinsic: NEON's logical operations are typed integer-only. That is not a
// cost -- `AND` does not care what the bits mean, and the reinterprets emit nothing.
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( NEON, SI32,  4, anb, vandq_s32 );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( NEON, PI32,  4, anb, vandq_u32 );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( NEON, SI64,  2, anb, vandq_s64 );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( NEON, PI64,  2, anb, vandq_u64 );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( NEON, SI16,  8, anb, vandq_s16 );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( NEON, PI16,  8, anb, vandq_u16 );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( NEON, SI8 , 16, anb, vandq_s8  );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( NEON, PI8 , 16, anb, vandq_u8  );

#define ASIMD_NEON_ANB_FP( COND, T, N, SUF ) \
    SIMD_VEC_IMPL_REG_ARITHMETIC_OP( COND, T, N, anb, ASIMD_NEON_ANB_FP_##SUF )
#define ASIMD_NEON_ANB_FP_f32( ra, rb ) vreinterpretq_f32_u32( vandq_u32( vreinterpretq_u32_f32( ra ), vreinterpretq_u32_f32( rb ) ) )
#define ASIMD_NEON_ANB_FP_f64( ra, rb ) vreinterpretq_f64_u64( vandq_u64( vreinterpretq_u64_f64( ra ), vreinterpretq_u64_f64( rb ) ) )

ASIMD_NEON_ANB_FP( NEON, FP32, 4, f32 );

// ---- per-lane VARIABLE shift ----------------------------------------------------------------
//
// `vshlq_*` shifts each lane by its own amount, at every width, on every ARM part. x86 has no
// such instruction below AVX2 (`vpsllvd`), and none at all on the 8- and 16-bit types -- so
// `SimdVec<T,N> << SimdVec<T,N>`, which the generic form can only express as a lane loop when
// the compiler has no vector extensions, is a single instruction here.
//
// The SHIFT COUNT IS SIGNED even for an unsigned value -- a negative count shifts right -- which
// is why the unsigned forms reinterpret `b` rather than passing it through.
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( NEON, SI32,  4, sll, vshlq_s32 );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( NEON, SI64,  2, sll, vshlq_s64 );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( NEON, SI16,  8, sll, vshlq_s16 );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( NEON, SI8 , 16, sll, vshlq_s8  );

#define ASIMD_NEON_SLL_U( COND, T, N, SUF ) \
    SIMD_VEC_IMPL_REG_ARITHMETIC_OP( COND, T, N, sll, ASIMD_NEON_SLL_U_##SUF )
#define ASIMD_NEON_SLL_U_u32( ra, rb ) vshlq_u32( ra, vreinterpretq_s32_u32( rb ) )
#define ASIMD_NEON_SLL_U_u64( ra, rb ) vshlq_u64( ra, vreinterpretq_s64_u64( rb ) )
#define ASIMD_NEON_SLL_U_u16( ra, rb ) vshlq_u16( ra, vreinterpretq_s16_u16( rb ) )
#define ASIMD_NEON_SLL_U_u8(  ra, rb ) vshlq_u8 ( ra, vreinterpretq_s8_u8 ( rb ) )

ASIMD_NEON_SLL_U( NEON, PI32,  4, u32 );
ASIMD_NEON_SLL_U( NEON, PI64,  2, u64 );
ASIMD_NEON_SLL_U( NEON, PI16,  8, u16 );
ASIMD_NEON_SLL_U( NEON, PI8 , 16, u8  );

// ---- comparisons ----------------------------------------------------------------------------
//
// The 64-bit ones are missing on purpose: `vcgtq_s64` is A64, and lands below.
ASIMD_NEON_CMP( NEON, FP32,  4, 32, f32 );
ASIMD_NEON_CMP( NEON, SI32,  4, 32, s32 );
ASIMD_NEON_CMP( NEON, PI32,  4, 32, u32 );
ASIMD_NEON_CMP( NEON, SI16,  8, 16, s16 );
ASIMD_NEON_CMP( NEON, PI16,  8, 16, u16 );
ASIMD_NEON_CMP( NEON, SI8 , 16,  8, s8  );
ASIMD_NEON_CMP( NEON, PI8 , 16,  8, u8  );

// ---- iota -----------------------------------------------------------------------------------
ASIMD_NEON_IOTA( NEON, FP32,  4, f32, float        , 0, 1, 2, 3 );
ASIMD_NEON_IOTA( NEON, SI32,  4, s32, std::int32_t , 0, 1, 2, 3 );
ASIMD_NEON_IOTA( NEON, PI32,  4, u32, std::uint32_t, 0, 1, 2, 3 );
ASIMD_NEON_IOTA( NEON, SI64,  2, s64, std::int64_t , 0, 1 );
ASIMD_NEON_IOTA( NEON, PI64,  2, u64, std::uint64_t, 0, 1 );
ASIMD_NEON_IOTA( NEON, SI16,  8, s16, std::int16_t , 0, 1, 2, 3, 4, 5, 6, 7 );
ASIMD_NEON_IOTA( NEON, PI16,  8, u16, std::uint16_t, 0, 1, 2, 3, 4, 5, 6, 7 );
ASIMD_NEON_IOTA( NEON, SI8 , 16, s8 , std::int8_t  , 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 );
ASIMD_NEON_IOTA( NEON, PI8 , 16, u8 , std::uint8_t , 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 );

// =============================================================================================
// 2. AArch64 -- what ARMv7 does not have
// =============================================================================================
#ifdef ASIMD_ARM_HAS_ASIMD

// ---- double precision. A whole lane type, not an operation: ARMv7 NEON has no `float64x2_t`,
// which is why `features::NEON` does not list `double` among its types at all.
ASIMD_NEON_COMMON( ASIMD, FP64, 2, float64x2_t, f64, double );
ASIMD_NEON_IOTA  ( ASIMD, FP64, 2, f64, double, 0, 1 );
ASIMD_NEON_MINMAX( ASIMD, FP64, 2, f64 );
ASIMD_NEON_CMP   ( ASIMD, FP64, 2, 64, f64 );
ASIMD_NEON_ANB_FP( ASIMD, FP64, 2, f64 );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( ASIMD, FP64, 2, mul, vmulq_f64 );

// ---- floating point DIVIDE. There is no vector `fdiv` on ARMv7 at all -- the usual answer
// there is a Newton-Raphson refinement of `vrecpeq_f32`, which is an approximation and so not
// something a `/` operator may quietly become. A64 has the real instruction.
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( ASIMD, FP32, 4, div, vdivq_f32 );
SIMD_VEC_IMPL_REG_ARITHMETIC_OP( ASIMD, FP64, 2, div, vdivq_f64 );

// ---- the 64-bit integer comparisons ---------------------------------------------------------
ASIMD_NEON_CMP( ASIMD, SI64, 2, 64, s64 );
ASIMD_NEON_CMP( ASIMD, PI64, 2, 64, u64 );

// ---- 64-bit integer min / max, SYNTHESISED, because there is no instruction -------------------
//
// `SMIN`/`UMIN` stop at 32-bit lanes on ARM, exactly as x86's stop below AVX-512VL. A compare and
// a bitwise select do it in two instructions, which is what the compiler emits from the generic
// whole-vector form anyway -- so why register it? BECAUSE THE COMPILER IS NOT ALWAYS THERE TO DO
// IT. The generic form only reaches for vector arithmetic when `values` is a `vector_size` type;
// on MSVC, which has no such thing, it walks the lanes. `tests/no_vecext.sh` measured exactly
// that: `min` on `SI64 x 4` went from 8 instructions to 15 with the compiler's vectors switched
// off, and on `SI64 x 8` from 15 to 29. Registering the two instructions makes the cell
// independent of the compiler, which is the whole argument of that script.
//
// Registered at TWO lanes only: the 4- and 8-lane widths are splits, and `min`/`max` recurse
// through the split on their own.
#define ASIMD_NEON_MINMAX_64( T, SUF, CMP ) \
    SIMD_VEC_IMPL_REG_ARITHMETIC_OP( ASIMD, T, 2, min, ASIMD_NEON_MIN64_##SUF ); \
    SIMD_VEC_IMPL_REG_ARITHMETIC_OP( ASIMD, T, 2, max, ASIMD_NEON_MAX64_##SUF )

#define ASIMD_NEON_MIN64_s64( ra, rb ) vbslq_s64( vcgtq_s64( rb, ra ), ra, rb )
#define ASIMD_NEON_MAX64_s64( ra, rb ) vbslq_s64( vcgtq_s64( ra, rb ), ra, rb )
#define ASIMD_NEON_MIN64_u64( ra, rb ) vbslq_u64( vcgtq_u64( rb, ra ), ra, rb )
#define ASIMD_NEON_MAX64_u64( ra, rb ) vbslq_u64( vcgtq_u64( ra, rb ), ra, rb )

ASIMD_NEON_MINMAX_64( SI64, s64, cgt );
ASIMD_NEON_MINMAX_64( PI64, u64, cgt );

#undef ASIMD_NEON_MAX64_u64
#undef ASIMD_NEON_MIN64_u64
#undef ASIMD_NEON_MAX64_s64
#undef ASIMD_NEON_MIN64_s64
#undef ASIMD_NEON_MINMAX_64

// ---- the horizontal sum ---------------------------------------------------------------------
//
// ONE INSTRUCTION, which is the thing to know about this backend. `ADDV` reduces a whole
// register to a scalar, so `.sum()` is a single `addv` plus the move to a general register --
// against the four-instruction shuffle ladder the SSE2 side needs, and against the 44
// instructions the generic pairwise tree came out at before any register form existed.
//
// `vaddvq_s16` returns an `int16_t` and `vaddvq_s8` an `int8_t`, so a sum that overflows wraps
// in the LANE type. That is what the generic form does too -- it accumulates in `T` -- so the
// two agree, which is what the grid test checks.
#define ASIMD_NEON_HSUM( T, N, SUF ) \
    SIMD_VEC_IMPL_REG_HSUM( ASIMD, T, N, T( vaddvq_##SUF( impl.data.reg ) ) )

ASIMD_NEON_HSUM( FP32,  4, f32 );
ASIMD_NEON_HSUM( FP64,  2, f64 );
ASIMD_NEON_HSUM( SI32,  4, s32 );
ASIMD_NEON_HSUM( PI32,  4, u32 );
ASIMD_NEON_HSUM( SI64,  2, s64 );
ASIMD_NEON_HSUM( PI64,  2, u64 );
ASIMD_NEON_HSUM( SI16,  8, s16 );
ASIMD_NEON_HSUM( PI16,  8, u16 );
ASIMD_NEON_HSUM( SI8 , 16, s8  );
ASIMD_NEON_HSUM( PI8 , 16, u8  );

#undef ASIMD_NEON_HSUM

#endif // ASIMD_ARM_HAS_ASIMD

// NO `gather`, NO `scatter`, at any ARM level. Neither exists as an instruction outside SVE
// (`LD1W`/`ST1W` with a vector of offsets, under a predicate), so both keep the generic form --
// a lane loop, which is what the hardware would have to do anyway. This is the honest
// counterpart of x86's `vpgatherdd`: a real gap, not an unregistered one.

#undef ASIMD_NEON_SLL_U_u8
#undef ASIMD_NEON_SLL_U_u16
#undef ASIMD_NEON_SLL_U_u64
#undef ASIMD_NEON_SLL_U_u32
#undef ASIMD_NEON_SLL_U
#undef ASIMD_NEON_ANB_FP_f64
#undef ASIMD_NEON_ANB_FP_f32
#undef ASIMD_NEON_ANB_FP
#undef ASIMD_NEON_MINMAX
#undef ASIMD_NEON_CMP
#undef ASIMD_NEON_IOTA
#undef ASIMD_NEON_COMMON

} // namespace internal
} // namespace asimd

#endif // ASIMD_ARM_HAS_NEON

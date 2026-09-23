#pragma once

// =============================================================================================
// REGISTER-LEVEL VARIANTS FOR ARM.
//
// The table; the shapes are in `Shapes.h`, shared with x86. Laid out by feature
// level, like its x86 counterpart:
//
//   1. 128 bits -- NEON, i.e. what an ARMv7-A part has
//   2. 128 bits -- ASIMD, i.e. what only A64 has
//
// THERE IS NO THIRD SECTION, and that is the shape of this architecture rather than an omission.
// x86 grows a section per width -- 128, 256, 512 -- because each generation widened the register.
// ARM never did: Advanced SIMD has been 128 bits since 2005 and still is. The width above 128
// comes from SVE, whose vector length is not a compile-time constant at all, so it cannot be a
// `SimdVecImpl` register (see `features::SVE`). Which means: ON ARM, `SimdVec<float,8>` IS THE
// SPLIT CASE, always. The one thing asimd exists for is the ordinary situation here, not the
// exception -- so `Split.h` is load-bearing on this target in a way it is not on x86.
//
// WHAT IS BETTER HERE THAN ON x86, and it is worth naming because a port written by mirroring
// would have missed all four:
//
//   `vcgtq_u32` &c.   the UNSIGNED compares are instructions. x86 has none below AVX-512: it
//                     flips the sign bit of both operands and uses the signed one, four
//                     instructions where this is one
//   `vcgeq_*`         all four relations exist for every type. On x86, `cmp_ge` on an integer
//                     is `not( a < b )`, hence two instructions and a materialised all-ones
//   `vmlaq_s32`       an INTEGER multiply-accumulate, one instruction. x86 has no integer fma at
//                     any feature level -- `ops::fma` on `SI32` is one of the four cells the x86
//                     README lists as honestly generic. Here it is a register form
//   `ADDV` / `UMAXV`  a horizontal reduction in ONE instruction, where SSE2 needs a shuffle
//                     ladder. This is what makes `to_bits` cheap despite the next point
//
// AND THE ONE THING THAT IS WORSE. There is no `movemask`, and no mask register either: a
// comparison yields a LANE mask, always. `to_bits` is therefore a reduction -- AND with
// `{1,2,4,8}`, then `ADDV` -- which is two instructions plus a constant, against one `movmskps`.
// The README predicted "three or four"; two is as good as it gets, and it is still the only
// operation in the library where ARM needs more instructions than x86 for the same answer.
// =============================================================================================

#include "../impl/arm_intrin.h"

#ifdef ASIMD_ARM_HAS_NEON

#include "../architectures/ArmCpuFeatures.h"
#include "Shapes.h"

namespace asimd {

// =============================================================================================
// 1. THE ARMv7-A FLOOR
// =============================================================================================

// ---- the four comparisons -------------------------------------------------------------------
//
// ONE INSTRUCTION EACH, FOR EVERY TYPE, SIGNED OR NOT. Compare this table with the x86 one: no
// predicate operand to pick, no `not` for `ge`, no sign-bit flip for the unsigned types, and no
// feature guard per relation. `CMGT`/`CMGE`/`CMEQ` come in signed, unsigned and floating point
// forms and have since ARMv7.
#define ASIMD_NEON_PLUS_CMP( COND, T, N, IS, SUF ) \
    ASIMD_OPS_CMP( COND, cmp_gt, T, N, IS, REGISTER, vcgtq_##SUF( a.data.reg, b.data.reg ) ); \
    ASIMD_OPS_CMP( COND, cmp_lt, T, N, IS, REGISTER, vcltq_##SUF( a.data.reg, b.data.reg ) ); \
    ASIMD_OPS_CMP( COND, cmp_eq, T, N, IS, REGISTER, vceqq_##SUF( a.data.reg, b.data.reg ) ); \
    ASIMD_OPS_CMP( COND, cmp_ge, T, N, IS, REGISTER, vcgeq_##SUF( a.data.reg, b.data.reg ) )

ASIMD_NEON_PLUS_CMP( NEON, FP32,  4, 32, f32 );
ASIMD_NEON_PLUS_CMP( NEON, SI32,  4, 32, s32 );
ASIMD_NEON_PLUS_CMP( NEON, PI32,  4, 32, u32 );
ASIMD_NEON_PLUS_CMP( NEON, SI16,  8, 16, s16 );
ASIMD_NEON_PLUS_CMP( NEON, PI16,  8, 16, u16 );
ASIMD_NEON_PLUS_CMP( NEON, SI8 , 16,  8, s8  );
ASIMD_NEON_PLUS_CMP( NEON, PI8 , 16,  8, u8  );

// ---- select ---------------------------------------------------------------------------------
//
// `BSL` -- bitwise select -- takes the mask in the DESTINATION register and needs no feature
// beyond NEON, where x86's 128-bit `blendv` waited for SSE4.1 and the 256-bit one for AVX. It
// also reads every bit of the mask rather than just the sign bit, which is why it works
// unchanged on a mask built by `mask_from_bits` as well as on one from a comparison.
#define ASIMD_NEON_PLUS_SELECT( COND, T, N, IS, SUF ) \
    ASIMD_OPS_SELECT( COND, T, N, IS, REGISTER, vbslq_##SUF( m.data.reg, a.data.reg, b.data.reg ) )

ASIMD_NEON_PLUS_SELECT( NEON, FP32,  4, 32, f32 );
ASIMD_NEON_PLUS_SELECT( NEON, SI32,  4, 32, s32 );
ASIMD_NEON_PLUS_SELECT( NEON, PI32,  4, 32, u32 );
ASIMD_NEON_PLUS_SELECT( NEON, SI64,  2, 64, s64 );
ASIMD_NEON_PLUS_SELECT( NEON, PI64,  2, 64, u64 );
ASIMD_NEON_PLUS_SELECT( NEON, SI16,  8, 16, s16 );
ASIMD_NEON_PLUS_SELECT( NEON, PI16,  8, 16, u16 );
ASIMD_NEON_PLUS_SELECT( NEON, SI8 , 16,  8, s8  );
ASIMD_NEON_PLUS_SELECT( NEON, PI8 , 16,  8, u8  );

// ---- fma ------------------------------------------------------------------------------------
//
// THE ACCUMULATOR COMES FIRST: `vfmaq_f32( acc, x, y )` is `acc + x * y`, where
// `_mm_fmadd_ps( x, y, acc )` puts it last. Hence `ASIMD_OPS_FMA_EXPR` rather than
// `ASIMD_OPS_FMA` -- the operand order is part of the instruction, not of the operation.
ASIMD_OPS_FMA_EXPR( NEON, FMA, FP32, 4, vfmaq_f32( c.data.reg, a.data.reg, b.data.reg ) );

// AN INTEGER MULTIPLY-ACCUMULATE, which x86 does not have at any feature level: `ops::fma` on
// `SI32` is one of the four cells the x86 side lists as legitimately generic ("x86 has no integer
// fused multiply-add"). `MLA` is one instruction and, being integer, exact -- there is no
// rounding to fuse, so the word "fused" simply does not apply and the result is identical to a
// multiply followed by an add. Guarded by NEON alone, not by FMA: `features::FMA` is about
// FUSING FLOATING POINT, and MLA needs nothing beyond base Advanced SIMD.
//
// No 64-bit form: there is no `vmlaq_s64`, for the same reason there is no `vmulq_s64`.
#define ASIMD_NEON_PLUS_MLA( T, N, SUF ) \
    ASIMD_OPS_FMA_EXPR( NEON, NEON, T, N, vmlaq_##SUF( c.data.reg, a.data.reg, b.data.reg ) )

ASIMD_NEON_PLUS_MLA( SI32,  4, s32 );
ASIMD_NEON_PLUS_MLA( PI32,  4, u32 );
ASIMD_NEON_PLUS_MLA( SI16,  8, s16 );
ASIMD_NEON_PLUS_MLA( PI16,  8, u16 );
ASIMD_NEON_PLUS_MLA( SI8 , 16, s8  );
ASIMD_NEON_PLUS_MLA( PI8 , 16, u8  );

// ---- mask_from_bits, lane flavour -----------------------------------------------------------
//
// Broadcast the integer, keep one bit per lane, compare it back to itself -- the same three
// instructions as the SSE2 form, and the same reason for existing: the generic form writes the
// lanes one at a time.
//
// REGISTERED AT 4 AND 2 LANES ONLY, on purpose. `Key<void,N,Arch>` carries no item size, so ONE
// registration per width decides the mask flavour for that width, and every later `select` has
// to accept it. Registering a 16-bit form at N = 8 would make `mask_from_bits<8>` return a
// `uint16x8_t` -- correct, and it would then drive `select` on `SimdVec<float,8>` (two 32-bit
// registers) down to the generic lane loop, which is the flagship width on this architecture.
// So the 8- and 16-lane cases go to the SPLIT variant instead, which builds two 32-bit halves
// and keeps the flavour the vector types actually want. See `Split.h`.
ASIMD_OPS_MASK_FROM_BITS( NEON, 4, 32, REGISTER, ( [ & ] {
    const std::uint32_t s[ 4 ] = { 1, 2, 4, 8 };
    const uint32x4_t sel = vld1q_u32( s );
    return vceqq_u32( vandq_u32( vdupq_n_u32( std::uint32_t( b ) ), sel ), sel ); }() ) );

// =============================================================================================
// 2. AArch64 -- the reductions, the permutation, double precision
// =============================================================================================
#ifdef ASIMD_ARM_HAS_ASIMD

// ---- double precision, and the 64-bit integer comparisons -----------------------------------
ASIMD_NEON_PLUS_CMP   ( ASIMD, FP64, 2, 64, f64 );
ASIMD_NEON_PLUS_CMP   ( ASIMD, SI64, 2, 64, s64 );
ASIMD_NEON_PLUS_CMP   ( ASIMD, PI64, 2, 64, u64 );
ASIMD_NEON_PLUS_SELECT( ASIMD, FP64, 2, 64, f64 );
ASIMD_OPS_FMA_EXPR   ( ASIMD, FMA, FP64, 2, vfmaq_f64( c.data.reg, a.data.reg, b.data.reg ) );

ASIMD_OPS_MASK_FROM_BITS( ASIMD, 2, 64, REGISTER, ( [ & ] {
    const std::uint64_t s[ 2 ] = { 1, 2 };
    const uint64x2_t sel = vld1q_u64( s );
    return vceqq_u64( vandq_u64( vdupq_n_u64( std::uint64_t( b ) ), sel ), sel ); }() ) );

// ---- to_bits --------------------------------------------------------------------------------
//
// THE OPERATION WITH NO ARM INSTRUCTION. There is no `movemask`: the lane mask has to be reduced
// to an integer by hand. Two instructions and a constant, and the shape is worth understanding
// because it is the same one every ARM SIMD library converges on --
//
//     AND the mask with { 1, 2, 4, 8 }        each set lane keeps exactly its own bit
//     ADDV                                    sum the lanes; they are disjoint bits, so the
//                                             sum IS the bitwise or
//
// The `or` is spelled as a sum because `ADDV` exists and there is no horizontal `or`. It is
// exact: a lane is all-ones or all-zeros, so after the AND each lane holds either 0 or a
// distinct power of two, and no carry can occur.
//
// `vaddvq_u16` returns a `uint16_t` and eight bits fit; `vaddvq_u8` returns a `uint8_t` and
// SIXTEEN DO NOT, which is why the 16-lane form reduces the two halves separately. Getting that
// wrong would have lost the top eight lanes silently -- the mask would still have been "a
// plausible integer".
ASIMD_OPS_TO_BITS( ASIMD, 4, 32, REGISTER, ( [ & ] {
    const std::uint32_t s[ 4 ] = { 1, 2, 4, 8 };
    return PI64( vaddvq_u32( vandq_u32( m.data.reg, vld1q_u32( s ) ) ) ); }() ) );

ASIMD_OPS_TO_BITS( ASIMD, 2, 64, REGISTER, ( [ & ] {
    const std::uint64_t s[ 2 ] = { 1, 2 };
    return PI64( vaddvq_u64( vandq_u64( m.data.reg, vld1q_u64( s ) ) ) ); }() ) );

ASIMD_OPS_TO_BITS( ASIMD, 8, 16, REGISTER, ( [ & ] {
    const std::uint16_t s[ 8 ] = { 1, 2, 4, 8, 16, 32, 64, 128 };
    return PI64( vaddvq_u16( vandq_u16( m.data.reg, vld1q_u16( s ) ) ) ); }() ) );

ASIMD_OPS_TO_BITS( ASIMD, 16, 8, REGISTER, ( [ & ] {
    const std::uint8_t s[ 16 ] = { 1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128 };
    const uint8x16_t x = vandq_u8( m.data.reg, vld1q_u8( s ) );
    return PI64( vaddv_u8( vget_low_u8( x ) ) ) | ( PI64( vaddv_u8( vget_high_u8( x ) ) ) << 8 ); }() ) );

// ---- bcast_lane -----------------------------------------------------------------------------
//
// `DUP Vd.4S, Vn.S[i]` -- one instruction, from any lane of a 128-bit register. ARMv7 has only
// the 64-bit-source form (`vdupq_lane_f32` takes a `float32x2_t`), so reaching lanes 2 and 3
// there means moving through a general-purpose register; that is left on the generic path rather
// than registered as an improvement it would not be.
//
// `LANE & ( N - 1 )` because the intrinsic's lane argument must be in range at COMPILE time --
// an out-of-range `bcast_lane<9>` on four lanes would be a hard error rather than the wrapped
// answer `vpermps` gives. The mask makes it wrap, which is the x86 behaviour.
#define ASIMD_NEON_PLUS_BCAST( T, N, SUF ) \
    ASIMD_OPS_BCAST( ASIMD, T, N, vdupq_laneq_##SUF( v.data.reg, LANE & ( N - 1 ) ) )

ASIMD_NEON_PLUS_BCAST( FP32,  4, f32 );
ASIMD_NEON_PLUS_BCAST( FP64,  2, f64 );
ASIMD_NEON_PLUS_BCAST( SI32,  4, s32 );
ASIMD_NEON_PLUS_BCAST( PI32,  4, u32 );
ASIMD_NEON_PLUS_BCAST( SI64,  2, s64 );
ASIMD_NEON_PLUS_BCAST( PI64,  2, u64 );
ASIMD_NEON_PLUS_BCAST( SI16,  8, s16 );
ASIMD_NEON_PLUS_BCAST( PI16,  8, u16 );
ASIMD_NEON_PLUS_BCAST( SI8 , 16, s8  );
ASIMD_NEON_PLUS_BCAST( PI8 , 16, u8  );

// ---- permute --------------------------------------------------------------------------------
//
// `TBL` IS A BYTE LOOKUP, which the README flagged as the primitive to watch, and it is: a
// 32-bit lane index has to become four consecutive byte indices. The same problem as `pshufb` on
// x86, and the arithmetic that solves it is nicer here --
//
//     j = idx & 3                     wrap, because that is what a register permutation does
//     c = j * 0x04040404              ALL FOUR BYTES of the lane become 4*j at once. 4*j is at
//                                     most 12, so nothing carries between bytes
//     c += { 0,1,2,3, 0,1,2,3, ... }  the byte within the lane
//     TBL                             one lookup over the whole 128-bit register
//
// Four instructions and one constant, against `pshufb`'s five (it needs a second constant and a
// shuffle to broadcast the low byte, where one multiply does it here). One `vpermilps` still
// beats both, but that is AVX -- the honest comparison is with SSSE3, and this wins it.
//
// TBL RETURNS ZERO for an index past the end of the table, where `vpermilps` wraps. That is why
// the `& 3` is not optional: `Split.h` hands a sub-permutation indices that point
// into the OTHER half, expects them wrapped rather than zeroed, and discards those lanes in the
// blend. Zeroing would give the same final answer here -- the blend throws the lane away either
// way -- but it would make this form disagree with the generic one on a bare out-of-range index,
// and those two have to agree: the grid test runs both.
#define ASIMD_NEON_PLUS_PERMUTE_32( T, SUF ) \
    ASIMD_OPS_PERMUTE( ASIMD, T, 4, ( [ & ] { \
        const std::uint8_t p[ 16 ] = { 0,1,2,3, 0,1,2,3, 0,1,2,3, 0,1,2,3 }; \
        const uint32x4_t j = vandq_u32( vreinterpretq_u32_s32( idx.data.reg ), vdupq_n_u32( 3 ) ); \
        const uint8x16_t c = vaddq_u8( vreinterpretq_u8_u32( vmulq_n_u32( j, 0x04040404u ) ), \
                                       vld1q_u8( p ) ); \
        return vreinterpretq_##SUF##_u8( vqtbl1q_u8( vreinterpretq_u8_##SUF( v.data.reg ), c ) ); }() ) )

ASIMD_NEON_PLUS_PERMUTE_32( FP32, f32 );
ASIMD_NEON_PLUS_PERMUTE_32( SI32, s32 );
ASIMD_NEON_PLUS_PERMUTE_32( PI32, u32 );

// The byte view of a register, and the way back. `vreinterpretq_u8_u8` DOES NOT EXIST -- ACLE
// has no identity reinterpret -- so the `PI8` rows need a no-op where the others need a cast.
#define ASIMD_NEON_AS_U8_s8(  x ) vreinterpretq_u8_s8 ( x )
#define ASIMD_NEON_AS_U8_u8(  x ) ( x )
#define ASIMD_NEON_AS_U8_s16( x ) vreinterpretq_u8_s16( x )
#define ASIMD_NEON_AS_U8_u16( x ) vreinterpretq_u8_u16( x )
#define ASIMD_NEON_OF_U8_s8(  x ) vreinterpretq_s8_u8 ( x )
#define ASIMD_NEON_OF_U8_u8(  x ) ( x )
#define ASIMD_NEON_OF_U8_s16( x ) vreinterpretq_s16_u8( x )
#define ASIMD_NEON_OF_U8_u16( x ) vreinterpretq_u16_u8( x )

// ---- permute on the NARROW types, where `TBL` is not a workaround but the native operation --
//
// A byte shuffle IS what `TBL` does, so `SI8 x 16` ought to be its best case. It was its worst:
// 78 instructions, against 30 before this backend existed. Adding a register impl had made the
// operation slower, because the generic form reads and writes `values` ONE LANE AT A TIME and
// `values` became a `vector_size` type -- thirty-two lane moves that a plain array does not need.
// A regression an ARM port introduces in ARM's signature instruction is not one to document.
//
// THE WORK IS ALL IN THE INDEX. `permute`'s index vector is always `SimdVec<SI32,N>`, so at
// sixteen lanes it is FOUR registers of 32-bit indices that have to become one register of bytes.
// `vmovn` halves a lane width, twice: 32 -> 16 -> 8, four narrows and three combines, and the
// combines are free (they are register naming). The index arrives through `idx.data.values`
// because a four-register impl has no single `.reg` to read.
#define ASIMD_NEON_PLUS_PERMUTE_8( T, SUF ) \
    ASIMD_OPS_PERMUTE( ASIMD, T, 16, ( [ & ] { \
        const std::int32_t *p = idx.data.values; \
        const uint8x16_t c = vandq_u8( vcombine_u8( \
            vmovn_u16( vcombine_u16( vmovn_u32( vreinterpretq_u32_s32( vld1q_s32( p +  0 ) ) ), \
                                     vmovn_u32( vreinterpretq_u32_s32( vld1q_s32( p +  4 ) ) ) ) ), \
            vmovn_u16( vcombine_u16( vmovn_u32( vreinterpretq_u32_s32( vld1q_s32( p +  8 ) ) ), \
                                     vmovn_u32( vreinterpretq_u32_s32( vld1q_s32( p + 12 ) ) ) ) ) ), \
            vdupq_n_u8( 15 ) ); \
        return ASIMD_NEON_OF_U8_##SUF( vqtbl1q_u8( ASIMD_NEON_AS_U8_##SUF( v.data.reg ), c ) ); }() ) )

// 16-bit lanes: two index registers instead of four, one narrow instead of two, and then the
// same trick as the 32-bit form -- multiply by 0x0202 to put `2*j` in BOTH bytes of the lane at
// once (2*j is at most 14, so nothing carries), then add the byte-within-lane pattern.
#define ASIMD_NEON_PLUS_PERMUTE_16( T, SUF ) \
    ASIMD_OPS_PERMUTE( ASIMD, T, 8, ( [ & ] { \
        const std::int32_t *p = idx.data.values; \
        const std::uint8_t pat[ 16 ] = { 0,1, 0,1, 0,1, 0,1, 0,1, 0,1, 0,1, 0,1 }; \
        const uint16x8_t j = vandq_u16( \
            vcombine_u16( vmovn_u32( vreinterpretq_u32_s32( vld1q_s32( p + 0 ) ) ), \
                          vmovn_u32( vreinterpretq_u32_s32( vld1q_s32( p + 4 ) ) ) ), \
            vdupq_n_u16( 7 ) ); \
        const uint8x16_t c = vaddq_u8( vreinterpretq_u8_u16( vmulq_n_u16( j, 0x0202u ) ), \
                                       vld1q_u8( pat ) ); \
        return ASIMD_NEON_OF_U8_##SUF( vqtbl1q_u8( ASIMD_NEON_AS_U8_##SUF( v.data.reg ), c ) ); }() ) )

ASIMD_NEON_PLUS_PERMUTE_8 ( SI8 , s8  );
ASIMD_NEON_PLUS_PERMUTE_8 ( PI8 , u8  );
ASIMD_NEON_PLUS_PERMUTE_16( SI16, s16 );
ASIMD_NEON_PLUS_PERMUTE_16( PI16, u16 );

#undef ASIMD_NEON_PLUS_PERMUTE_16
#undef ASIMD_NEON_PLUS_PERMUTE_8
#undef ASIMD_NEON_OF_U8_u16
#undef ASIMD_NEON_OF_U8_s16
#undef ASIMD_NEON_OF_U8_u8
#undef ASIMD_NEON_OF_U8_s8
#undef ASIMD_NEON_AS_U8_u16
#undef ASIMD_NEON_AS_U8_s16
#undef ASIMD_NEON_AS_U8_u8
#undef ASIMD_NEON_AS_U8_s8
#undef ASIMD_NEON_PLUS_PERMUTE_32
#undef ASIMD_NEON_PLUS_BCAST

// A TWO-LANE PERMUTATION OF 64-BIT ELEMENTS IS NOT REGISTERED, and it is the same cell the x86
// side leaves generic, for the same reason: the index vector is `SimdVec<SI32,2>`, a width with
// no register form of its own -- 128 bits hold four `SI32`, so at two lanes the index impl is a
// split of two single lanes and there is no `idx.data.reg` to read. Two lanes are two moves
// whatever one does.

#endif // ASIMD_ARM_HAS_ASIMD

// =============================================================================================
// 3. LANE ROTATIONS -- where this architecture has the better instruction.
//
// `EXT Vd.16B, Vn.16B, Vm.16B, #b` extracts sixteen bytes from the concatenation `Vn:Vm` at
// byte `b`: with `Vn = Vm` it is a rotation of the whole register by any number of lanes of any
// width, in ONE instruction with an immediate and no constant to load. It is ARMv7 (`vext`), so
// it is registered under NEON and covers the ARMv7 floor. x86 has no equivalent below AVX-512
// (`palignr` is per 128-bit lane), which is the one place in this file where ARM is ahead on a
// data-movement primitive rather than on an arithmetic one.
//
// A rotation of a PREFIX of the register (`n < N`) is not an `EXT`: the lanes past `n` stay
// put. That is a `TBL` with a constant byte table, one instruction plus a load, and `TBL` on a
// 128-bit table is A64 only -- so the prefix rows are ASIMD, the whole-register rows NEON.
// =============================================================================================
#define ASIMD_NEON_AS_U8_f32( x ) vreinterpretq_u8_f32( x )
#define ASIMD_NEON_AS_U8_f64( x ) vreinterpretq_u8_f64( x )
#define ASIMD_NEON_AS_U8_s32( x ) vreinterpretq_u8_s32( x )
#define ASIMD_NEON_AS_U8_u32( x ) vreinterpretq_u8_u32( x )
#define ASIMD_NEON_AS_U8_s64( x ) vreinterpretq_u8_s64( x )
#define ASIMD_NEON_AS_U8_u64( x ) vreinterpretq_u8_u64( x )
#define ASIMD_NEON_AS_U8_s16( x ) vreinterpretq_u8_s16( x )
#define ASIMD_NEON_AS_U8_u16( x ) vreinterpretq_u8_u16( x )
#define ASIMD_NEON_AS_U8_s8(  x ) vreinterpretq_u8_s8 ( x )
#define ASIMD_NEON_AS_U8_u8(  x ) ( x )
#define ASIMD_NEON_OF_U8_f32( x ) vreinterpretq_f32_u8( x )
#define ASIMD_NEON_OF_U8_f64( x ) vreinterpretq_f64_u8( x )
#define ASIMD_NEON_OF_U8_s32( x ) vreinterpretq_s32_u8( x )
#define ASIMD_NEON_OF_U8_u32( x ) vreinterpretq_u32_u8( x )
#define ASIMD_NEON_OF_U8_s64( x ) vreinterpretq_s64_u8( x )
#define ASIMD_NEON_OF_U8_u64( x ) vreinterpretq_u64_u8( x )
#define ASIMD_NEON_OF_U8_s16( x ) vreinterpretq_s16_u8( x )
#define ASIMD_NEON_OF_U8_u16( x ) vreinterpretq_u16_u8( x )
#define ASIMD_NEON_OF_U8_s8(  x ) vreinterpretq_s8_u8 ( x )
#define ASIMD_NEON_OF_U8_u8(  x ) ( x )

/// `ext_lanes` and the whole-register `rotate_lanes` are the same `EXT`.
#define ASIMD_NEON_EXT( COND, T, N, SUF ) \
    ASIMD_OPS_EXT( COND, T, N, ASIMD_NEON_OF_U8_##SUF( vextq_u8( \
        ASIMD_NEON_AS_U8_##SUF( a.data.reg ), ASIMD_NEON_AS_U8_##SUF( b.data.reg ), K * int( sizeof( T ) ) ) ) ); \
    ASIMD_OPS_ROTATE_IF( COND, n == N, T, N, ASIMD_NEON_OF_U8_##SUF( vextq_u8( \
        ASIMD_NEON_AS_U8_##SUF( v.data.reg ), ASIMD_NEON_AS_U8_##SUF( v.data.reg ), K * int( sizeof( T ) ) ) ) )

/// the prefix rotation: a `TBL` over `rot::Bytes`.
#define ASIMD_NEON_ROTATE_PREFIX( T, N, SUF ) \
    ASIMD_OPS_ROTATE_IF( ASIMD, n < N, T, N, ASIMD_NEON_OF_U8_##SUF( vqtbl1q_u8( \
        ASIMD_NEON_AS_U8_##SUF( v.data.reg ), vld1q_u8( ( rot::Bytes<K,n,N,sizeof( T )>::v.data() ) ) ) ) )

ASIMD_NEON_EXT( NEON, FP32,  4, f32 );
ASIMD_NEON_EXT( NEON, SI32,  4, s32 );
ASIMD_NEON_EXT( NEON, PI32,  4, u32 );
ASIMD_NEON_EXT( NEON, SI64,  2, s64 );
ASIMD_NEON_EXT( NEON, PI64,  2, u64 );
ASIMD_NEON_EXT( NEON, SI16,  8, s16 );
ASIMD_NEON_EXT( NEON, PI16,  8, u16 );
ASIMD_NEON_EXT( NEON, SI8 , 16, s8  );
ASIMD_NEON_EXT( NEON, PI8 , 16, u8  );

#ifdef ASIMD_ARM_HAS_ASIMD
ASIMD_NEON_EXT( ASIMD, FP64,  2, f64 );

ASIMD_NEON_ROTATE_PREFIX( FP32,  4, f32 );
ASIMD_NEON_ROTATE_PREFIX( FP64,  2, f64 );
ASIMD_NEON_ROTATE_PREFIX( SI32,  4, s32 );
ASIMD_NEON_ROTATE_PREFIX( PI32,  4, u32 );
ASIMD_NEON_ROTATE_PREFIX( SI64,  2, s64 );
ASIMD_NEON_ROTATE_PREFIX( PI64,  2, u64 );
ASIMD_NEON_ROTATE_PREFIX( SI16,  8, s16 );
ASIMD_NEON_ROTATE_PREFIX( PI16,  8, u16 );
ASIMD_NEON_ROTATE_PREFIX( SI8 , 16, s8  );
ASIMD_NEON_ROTATE_PREFIX( PI8 , 16, u8  );
#endif // ASIMD_ARM_HAS_ASIMD

#undef ASIMD_NEON_ROTATE_PREFIX
#undef ASIMD_NEON_EXT
#undef ASIMD_NEON_OF_U8_u8
#undef ASIMD_NEON_OF_U8_s8
#undef ASIMD_NEON_OF_U8_u16
#undef ASIMD_NEON_OF_U8_s16
#undef ASIMD_NEON_OF_U8_u64
#undef ASIMD_NEON_OF_U8_s64
#undef ASIMD_NEON_OF_U8_u32
#undef ASIMD_NEON_OF_U8_s32
#undef ASIMD_NEON_OF_U8_f64
#undef ASIMD_NEON_OF_U8_f32
#undef ASIMD_NEON_AS_U8_u8
#undef ASIMD_NEON_AS_U8_s8
#undef ASIMD_NEON_AS_U8_u16
#undef ASIMD_NEON_AS_U8_s16
#undef ASIMD_NEON_AS_U8_u64
#undef ASIMD_NEON_AS_U8_s64
#undef ASIMD_NEON_AS_U8_u32
#undef ASIMD_NEON_AS_U8_s32
#undef ASIMD_NEON_AS_U8_f64
#undef ASIMD_NEON_AS_U8_f32

#undef ASIMD_NEON_PLUS_MLA
#undef ASIMD_NEON_PLUS_SELECT
#undef ASIMD_NEON_PLUS_CMP

} // namespace asimd

#endif // ASIMD_ARM_HAS_NEON

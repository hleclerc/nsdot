#pragma once

#include "arm_intrin.h"

#ifdef ASIMD_ARM_HAS_NEON

#include "../architectures/ArmCpuFeatures.h"
#include "SimdBoolImpl_Generic.h"

// =============================================================================================
// ARM MASKS, and the one structural difference from x86.
//
// THERE ARE NO MASK REGISTERS ON ARM -- not on NEON, not on ASIMD. A comparison yields a LANE
// mask, a full register of all-ones or all-zeros words, exactly as on x86 below AVX-512, and
// `select` is a `BSL`. So the `MASK_REGISTER` rank is simply never reached on this target, and
// `to_bits` is a REDUCTION rather than a register move: see `ops/Neon.h`. SVE is where
// that changes -- it has real predicate registers, `p0`-`p15` -- which is one more reason the
// SVE feature in `ArmCpuFeatures.h` is a declaration and not yet a backend.
//
// WHAT IS HERE IS STORAGE PLUS TWO REDUCTIONS. The impl structs, under `NEON` because a
// `uint32x4_t` is a type on every ARM part; and `any`/`all`, under `ASIMD` because they are
// `UMAXV`/`UMINV`, which are A64.
//
// The ABI rule of README section 3 applies to these unchanged: `SIMD_BOOL_IMPL_REG_LARGE` gives
// `values` a `vector_size` type and keeps `Split` out of the union, so a mask crosses a call in
// its register. `tests/abi_probe.cpp` measures it on ARM too.
// =============================================================================================

namespace asimd {
namespace internal {

// struct SimdBoolImpl<...> -- one per lane width a comparison can produce at 128 bits.
SIMD_BOOL_IMPL_REG_LARGE( NEON,  2, 64, uint64x2_t )
SIMD_BOOL_IMPL_REG_LARGE( NEON,  4, 32, uint32x4_t )
SIMD_BOOL_IMPL_REG_LARGE( NEON,  8, 16, uint16x8_t )
SIMD_BOOL_IMPL_REG_LARGE( NEON, 16,  8, uint8x16_t )

#ifdef ASIMD_ARM_HAS_ASIMD

// ---- any / all ------------------------------------------------------------------------------
//
// ONE INSTRUCTION EACH, and the pair is exactly the right shape for the question. `UMAXV`
// reduces to the largest lane, so "is any lane set" is `vmaxvq != 0`; `UMINV` reduces to the
// smallest, so "are all lanes set" is `vminvq == all ones`. No complement, no second pass --
// the same economy as `_mm_testc_si128` on the x86 side, where `all` had been written as an xor
// followed by a `testz`.
//
// THE 64-BIT CASE GOES THROUGH `u32`, and that is not a cast to a smaller type: there is no
// `vmaxvq_u64` on A64. A lane mask is all-ones or all-zeros, so reading a `uint64x2_t` as four
// 32-bit lanes cannot change either answer -- both halves of a set lane are 0xFFFFFFFF.
SIMD_BOOL_IMPL_REG_REDUCTION( ASIMD,  2, 64, any, vmaxvq_u32( vreinterpretq_u32_u64( mask.data.reg ) ) != 0 )
SIMD_BOOL_IMPL_REG_REDUCTION( ASIMD,  4, 32, any, vmaxvq_u32( mask.data.reg ) != 0 )
SIMD_BOOL_IMPL_REG_REDUCTION( ASIMD,  8, 16, any, vmaxvq_u16( mask.data.reg ) != 0 )
SIMD_BOOL_IMPL_REG_REDUCTION( ASIMD, 16,  8, any, vmaxvq_u8 ( mask.data.reg ) != 0 )

// NOT `== ~std::uint8_t( 0 )`, and this is the AVX-512 `all` bug of FINDINGS.md waiting to
// happen again: `~` INTEGER-PROMOTES its operand, so `~std::uint8_t( 0 )` is the `int`
// 0xFFFFFFFF, while `vminvq_u8` returns a `std::uint8_t` that promotes to at most 255. The
// comparison would be false at every width but 64 -- i.e. `all` would always be false. The
// all-ones value is therefore written out, at the lane type's own width.
SIMD_BOOL_IMPL_REG_REDUCTION( ASIMD,  2, 64, all, vminvq_u32( vreinterpretq_u32_u64( mask.data.reg ) ) == std::uint32_t( 0xFFFFFFFFu ) )
SIMD_BOOL_IMPL_REG_REDUCTION( ASIMD,  4, 32, all, vminvq_u32( mask.data.reg ) == std::uint32_t( 0xFFFFFFFFu ) )
SIMD_BOOL_IMPL_REG_REDUCTION( ASIMD,  8, 16, all, vminvq_u16( mask.data.reg ) == std::uint16_t( 0xFFFFu ) )
SIMD_BOOL_IMPL_REG_REDUCTION( ASIMD, 16,  8, all, vminvq_u8 ( mask.data.reg ) == std::uint8_t ( 0xFFu ) )

#endif // ASIMD_ARM_HAS_ASIMD

} // namespace internal
} // namespace asimd

#endif // ASIMD_ARM_HAS_NEON

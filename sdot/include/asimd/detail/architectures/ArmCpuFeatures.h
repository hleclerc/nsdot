#pragma once

#include "SimdFeatureOn.h"
#include "GenericFeatures.h"

namespace asimd {
namespace features {

// =============================================================================================
// THE LATTICE HAS TO MATCH THE INSTRUCTIONS, not the marketing generations -- the same rule as
// `X86CpuFeatures.h`, and on ARM it bites in a different place.
//
// THERE IS NO SINGLE "NEON". `__ARM_NEON` is defined on ARMv7-A and on AArch64 alike, and a good
// third of what a 128-bit backend wants to call does not exist on the first one:
//
//   `vdivq_f32`        no vector floating point divide on ARMv7 -- it is A64 only
//   `float64x2_t`      no double-precision LANES at all on ARMv7 NEON
//   `vaddvq_*`         the horizontal reductions (ADDV, UMAXV, UMINV) are A64 only. On ARMv7 a
//                      sum is a ladder of `vpadd`, which is why `to_bits` costs what it costs
//   `vqtbl1q_u8`       TBL over a whole 128-bit register is A64. ARMv7 has `vtbl1_u8`, which
//                      reads a 64-bit table, so a full-register permutation is two lookups
//   `vcgtq_s64`        64-bit integer compares are A64
//   16 vs 32 registers ARMv7-A has 16 q-registers, AArch64 has 32
//
// Writing all of that under one feature would repeat exactly the mistake the x86 lattice was
// fixed for: guarded by `__SSE2__`, half the SSE2 backend was really SSE4.1 and a genuine SSE2
// target did not build. So there are TWO width-carrying features here, and the split is where
// the instruction set actually splits -- at A64.
//
// TWO KINDS OF FEATURE, as on x86. The ones that carry a WIDTH derive from `SimdFeatureOn` and
// set `SimdSize`; the ones that only unlock instructions at a width someone else already
// provides are plain markers. `FeatureSet::SimdSize` takes the max over the set, and a marker
// contributes 1, so a marker can never lower a width.
// =============================================================================================

// The types a 128-bit ARM register can hold. `double` is NOT in this list: on ARMv7 NEON there
// are no double-precision lanes, so `SimdSize<double,ArmCpu<32,NEON>>` must be 1 -- one scalar
// lane through the generic forms -- and not 2. AArch64 adds it back through ASIMD below.
#define ASIMD_ARM_NEON_TYPES float,std::int8_t,std::int16_t,std::int32_t,std::int64_t,std::uint8_t,std::uint16_t,std::uint32_t,std::uint64_t
#define ASIMD_ARM_A64_TYPES  float,double,std::int8_t,std::int16_t,std::int32_t,std::int64_t,std::uint8_t,std::uint16_t,std::uint32_t,std::uint64_t

// ---- 128 bits ----

/// AArch64 Advanced SIMD. 128 bits, **32** registers, double-precision lanes, `fdiv`, the
/// horizontal reductions and TBL over a full register. This is what any AArch64 target has,
/// architecturally -- there is no `-mneon` to forget.
struct ASIMD : SimdFeatureOn<128,32,ASIMD_ARM_A64_TYPES>  { static std::string name() { return "ASIMD"  ; } };

/// ARMv7-A NEON. 128 bits, 16 registers, single precision only. Kept as its own feature rather
/// than folded into ASIMD because it is the honest floor: everything registered against `NEON`
/// alone is an instruction a 32-bit ARM part really has.
struct NEON  : SimdFeatureOn<128,16,ASIMD_ARM_NEON_TYPES> { static std::string name() { return "NEON"   ; } };

#undef ASIMD_ARM_A64_TYPES
#undef ASIMD_ARM_NEON_TYPES

// ---- markers: instructions, at a width something else already gives ----

// `FMA` -- FMLA / FMLS -- is declared in `GenericFeatures.h`, shared with x86: the two
// architectures need the same feature and a type declared twice collides. Architectural on
// AArch64; on ARMv7 it rides on VFPv4, which is why it is a feature and not an assumption.

/// ARMv8.2-A +fp16: arithmetic on `float16x8_t`, i.e. EIGHT half-precision lanes in 128 bits.
/// Declared, not yet dispatched on: `asimd`'s type list has no 16-bit float, so there is no
/// `SimdVec<FP16,8>` for a variant to key on. That is a `common_types.h` change, not a backend
/// one, and it is the next honest step for this file.
struct FP16    { static std::string name() { return "FP16"   ; } };

/// ARMv8.2-A +bf16: BFDOT / BFMMLA. Same situation as FP16 -- the lane type does not exist in
/// the library yet -- with the extra wrinkle that bfloat16 arithmetic is only ever a dot
/// product, never a plain per-lane multiply.
struct BF16    { static std::string name() { return "BF16"   ; } };

/// ARMv8.2-A +dotprod: SDOT / UDOT, four 8-bit products accumulated into each 32-bit lane.
/// Declared because it is the extension that matters most for integer kernels on ARM, and
/// because it is a REDUCING operation -- it has no x86 equivalent below AVX-512-VNNI -- so it
/// wants an operation of its own (`dot_4x8`) rather than a variant of one that exists.
struct DOTPROD { static std::string name() { return "DOTPROD"; } };

/// ARMv8.6-A +i8mm: SMMLA / UMMLA, an 8-bit integer matrix multiply. Same shape of gap as
/// DOTPROD: no operation in the library's surface maps onto it yet.
struct I8MM    { static std::string name() { return "I8MM"   ; } };

/// ARMv8.1-A: SQRDMLAH / SQRDMLSH, the rounding saturating doubling multiply-accumulate. Waiting
/// on saturating arithmetic being expressible at all -- `asimd`'s `mul` wraps.
struct RDM     { static std::string name() { return "RDM"    ; } };

/// ARMv8.3-A: FCMLA / FCADD, complex multiply-accumulate on interleaved pairs.
struct FCMA    { static std::string name() { return "FCMA"   ; } };

/// SVE, and SVE2.
///
/// DELIBERATELY NOT WIDTH-CARRYING, and that is the interesting one. An SVE vector is between 128
/// and 2048 bits and the length is not known until the program runs, so `svfloat32_t` is a
/// SIZELESS type: it cannot be a class member, hence cannot sit in the union
/// `SIMD_VEC_IMPL_REG` builds, hence cannot be a `SimdVecImpl` register. Declaring a width for
/// SVE here would make `SimdSize<float>` a number the hardware may contradict.
///
/// There IS a door: `-msve-vector-bits=N` sets `__ARM_FEATURE_SVE_BITS` and makes the SVE types
/// sized, at which point they become union members like any other and an `SVE<N>` feature
/// carrying a width would be exactly right. That is a backend, not a declaration, so what is
/// here is the declaration -- an SVE machine is described correctly, and runs the NEON forms.
struct SVE     { static std::string name() { return "SVE"    ; } };
struct SVE2    { static std::string name() { return "SVE2"   ; } };

// AES / SHA / CRC are deliberately absent. They are instructions on a SIMD register, not
// data-parallel arithmetic on lanes: nothing in `SimdVec`'s surface could ever dispatch to them,
// so declaring them would only pad the feature list.

} // namespace features
} // namespace asimd

#pragma once

#include "SimdFeatureOn.h"
#include "GenericFeatures.h"

namespace asimd {
namespace features {

#define ASIMD_CMON_TYPES float,double,std::int8_t,std::int16_t,std::int32_t,std::int64_t,std::uint8_t,std::uint16_t,std::uint32_t,std::uint64_t

// =============================================================================================
// THE LATTICE HAS TO MATCH THE INSTRUCTIONS, not the marketing generations.
//
// It used to go SSE2 -> AVX -> AVX2 -> AVX512, and a good part of what the SSE2 backend called
// lives in SSE4.1: `pminsd`, `pmulld`, `ptest`, `movntdqa`, `blendvps`. Those were guarded by
// `__SSE2__`, so a genuine SSE2 build did not compile at all. The AVX512 backend had the same
// problem one level up: `vpscatterdd` at 128/256 bits is AVX-512VL, the 8- and 16-bit compares
// are AVX-512BW, and both were guarded by `__AVX512F__`.
//
// TWO KINDS OF FEATURE below. The ones that carry a WIDTH derive from `SimdFeatureOn` and set
// `SimdSize`; the ones that only unlock instructions at a width someone else already provides
// -- FMA, AVX-512VL/BW/DQ -- are plain markers. `FeatureSet::SimdSize` takes the max over the
// set, and a marker contributes 1, so it cannot lower anything.
// =============================================================================================

// ---- 512-bit ----
struct AVX512   : SimdFeatureOn<512,32,ASIMD_CMON_TYPES> { static std::string name() { return "AVX512"  ; } };
struct AVX512VL {                                          static std::string name() { return "AVX512VL"; } }; ///< the AVX-512 encodings at 128 and 256 bits
struct AVX512BW {                                          static std::string name() { return "AVX512BW"; } }; ///< 8- and 16-bit lanes
struct AVX512DQ {                                          static std::string name() { return "AVX512DQ"; } }; ///< 64-bit integer multiply, and more

// ---- 256-bit ----
struct AVX2     : SimdFeatureOn<256,16,ASIMD_CMON_TYPES> { static std::string name() { return "AVX2"    ; } };
struct AVX      : SimdFeatureOn<256,16,ASIMD_CMON_TYPES> { static std::string name() { return "AVX"     ; } };
// `FMA` lives in `GenericFeatures.h`: ARM needs the same feature, and a type declared in both
// feature headers collides as soon as `NativeCpu.h` includes them together. It is still
// orthogonal to AVX/AVX2 here -- on paper, and on the first AMD parts to carry it.

// ---- 128-bit ----
struct SSE4_2   : SimdFeatureOn<128,8,ASIMD_CMON_TYPES>  { static std::string name() { return "SSE4.2"  ; } };
struct SSE4_1   : SimdFeatureOn<128,8,ASIMD_CMON_TYPES>  { static std::string name() { return "SSE4.1"  ; } }; ///< pminsd/pmaxsd, pmulld, ptest, movntdqa, blendv
struct SSSE3    : SimdFeatureOn<128,8,ASIMD_CMON_TYPES>  { static std::string name() { return "SSSE3"   ; } };
struct SSE3     : SimdFeatureOn<128,8,ASIMD_CMON_TYPES>  { static std::string name() { return "SSE3"    ; } };
struct SSE2     : SimdFeatureOn<128,8,ASIMD_CMON_TYPES>  { static std::string name() { return "SSE2"    ; } };
struct SSE      : SimdFeatureOn<128,8,ASIMD_CMON_TYPES>  { static std::string name() { return "SSE"     ; } };

#undef ASIMD_CMON_TYPES

} // namespace features
} // namespace asimd

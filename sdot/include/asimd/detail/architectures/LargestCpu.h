#pragma once

#include "X86Cpu.h"
#include "ArmCpu.h"
#include "NativeCpu.h"

namespace asimd {


// -------------------------- Native --------------------------
#if ( defined(_M_IX86) || defined(__i386__) || defined(_M_X64) || defined(__x86_64__) )
using LargestCpu = X86Cpu< 8 * sizeof( void * ),
    features::AVX512, features::AVX512VL, features::AVX512BW, features::AVX512DQ,
    features::AVX2, features::AVX, features::FMA,
    features::SSE4_2, features::SSE4_1, features::SSSE3, features::SSE3,
    features::SSE2, features::SSE
>;
// -------------------------- ARM --------------------------
// EVERY FEATURE, INCLUDING THE ONES NOTHING DISPATCHES ON YET. This type is "the most capable
// machine of this family", used to size compile-time decisions rather than to emit code, so a
// declared-but-unused feature belongs in it: leaving `DOTPROD` out would make `LargestCpu` claim
// less than the architecture offers the day a backend for it appears.
//
// `SVE` is here too, and it carries no width -- deliberately, because an SVE vector's length is
// not a compile-time constant (see `ArmCpuFeatures.h`). So `LargestCpu` on ARM is 128 bits wide,
// which is the truth about Advanced SIMD and not a placeholder.
#elif defined( __aarch64__ ) || defined( __arm64__ ) || defined( _M_ARM64 ) || defined( _M_ARM64EC ) \
   || defined( __arm__ ) || defined( _M_ARM ) || defined( __ARM_ARCH )
using LargestCpu = ArmCpu< 8 * sizeof( void * ),
    features::ASIMD, features::NEON, features::FMA,
    features::FP16, features::BF16, features::DOTPROD, features::I8MM,
    features::RDM, features::FCMA,
    features::SVE, features::SVE2
>;

#else
using LargestCpu = NativeCpu; ///< no "largest" is known for this target; the native one will do.
#endif


} // namespace asimd

#pragma once

#include "GenericFeatures.h"
#include "X86Cpu.h"
#include "ArmCpu.h"
#include "ScalarCpu.h"

namespace asimd {

// =============================================================================================
// THE NATIVE ARCHITECTURE, deduced from what the compiler says it is targeting.
//
// TWO THINGS THIS FILE HAS TO GET RIGHT, and neither is obvious:
//
//   MSVC DOES NOT DEFINE `__SSE2__`, EVER. It defines `_M_X64` / `_M_IX86_FP`, and defines
//   `__AVX__` / `__AVX2__` / `__AVX512F__` only under the matching `/arch:`. Deducing the
//   feature set from `__SSE2__` alone therefore gave MSVC x64 an EMPTY feature set -- every
//   operation silently scalar, which is the exact failure `Selection.h` exists to prevent, on
//   one of the three compilers this has to build with. On x86-64, SSE2 is architectural.
//
//   AN UNKNOWN TARGET MUST STILL COMPILE. There used to be no `#else`: on anything that was
//   neither x86 nor Apple's `__arm64__`, `NativeCpu` was simply not declared, and every
//   translation unit that included the library failed. `ScalarCpu` is the honest answer -- one
//   lane, everything through the generic forms -- and it is what makes each new backend
//   ADDITIVE rather than a precondition.
// =============================================================================================

// -------------------------- x86 / x86-64 --------------------------
#if defined( _M_X64 ) || defined( _M_IX86 ) || defined( __i386__ ) || defined( __x86_64__ )

    // SSE / SSE2 are architectural on x86-64; on 32-bit they need /arch: or -msse2.
    #if defined( __x86_64__ ) || defined( _M_X64 ) || defined( __SSE2__ ) || ( defined( _M_IX86_FP ) && _M_IX86_FP >= 2 )
        #define ASIMD_HAS_SSE2 1
    #endif

using NativeCpu = X86Cpu< 8 * sizeof( void * )
    #ifdef __AVX512F__
        , features::AVX512
    #endif
    #ifdef __AVX512VL__
        , features::AVX512VL
    #endif
    #ifdef __AVX512BW__
        , features::AVX512BW
    #endif
    #ifdef __AVX512DQ__
        , features::AVX512DQ
    #endif
    #ifdef __AVX2__
        , features::AVX2
    #endif
    #ifdef __AVX__
        , features::AVX
    #endif
    #if defined( __FMA__ ) || ( defined( _MSC_VER ) && defined( __AVX2__ ) ) // MSVC folds FMA into /arch:AVX2
        , features::FMA
    #endif
    #if defined( __SSE4_2__ ) || ( defined( _MSC_VER ) && defined( __AVX__ ) )
        , features::SSE4_2
    #endif
    #if defined( __SSE4_1__ ) || ( defined( _MSC_VER ) && defined( __AVX__ ) )
        , features::SSE4_1
    #endif
    #if defined( __SSSE3__ ) || ( defined( _MSC_VER ) && defined( __AVX__ ) )
        , features::SSSE3
    #endif
    #if defined( __SSE3__ ) || ( defined( _MSC_VER ) && defined( __AVX__ ) )
        , features::SSE3
    #endif
    #ifdef ASIMD_HAS_SSE2
        , features::SSE2
        , features::SSE
    #endif
>;

// -------------------------- ARM / AArch64 --------------------------
// `__arm64__` alone was Apple's spelling: gcc and clang on Linux use `__aarch64__`, MSVC uses
// `_M_ARM64`, so the branch was unreachable anywhere else.
//
// ONE BRANCH FOR BOTH ARMs, where there used to be two. The 32-bit and 64-bit lists had drifted
// apart -- the AArch64 one hard-coded `features::NEON` and the ARMv7 one guarded it -- and
// neither named anything but NEON. What actually distinguishes the two is a FEATURE, `ASIMD`,
// so it belongs in the list rather than in the preprocessor: everything below is one
// `#ifdef` per feature, exactly like the x86 list above, and the two targets differ only by
// which of them fire.
#elif defined( __aarch64__ ) || defined( __arm64__ ) || defined( _M_ARM64 ) || defined( _M_ARM64EC ) \
   || defined( __arm__ ) || defined( _M_ARM ) || defined( __ARM_ARCH )

    // AArch64: Advanced SIMD is architectural, there is no flag to forget. MSVC's ARM64 targets
    // it unconditionally too, and defines none of the `__ARM_FEATURE_*` macros.
    #if defined( __aarch64__ ) || defined( __arm64__ ) || defined( _M_ARM64 ) || defined( _M_ARM64EC ) || defined( __ARM_ARCH_ISA_A64 )
        #define ASIMD_HAS_A64 1
    #endif

    // ARMv7-A NEON. `__ARM_NEON` is the portable spelling and covers AArch64 as well; MSVC's
    // 32-bit ARM target is ARMv7 with NEON and defines neither.
    #if defined( __ARM_NEON ) || defined( __ARM_NEON__ ) || defined( ASIMD_HAS_A64 ) || defined( _M_ARM )
        #define ASIMD_HAS_NEON 1
    #endif

using NativeCpu = ArmCpu< 8 * sizeof( void * )
    #ifdef ASIMD_HAS_A64
        , features::ASIMD
    #endif
    #ifdef ASIMD_HAS_NEON
        , features::NEON
    #endif
    // FMLA is architectural on AArch64; on ARMv7 it rides on VFPv4.
    #if defined( __ARM_FEATURE_FMA ) || defined( ASIMD_HAS_A64 )
        , features::FMA
    #endif
    #ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        , features::FP16
    #endif
    #if defined( __ARM_FEATURE_BF16_VECTOR_ARITHMETIC ) || defined( __ARM_FEATURE_BF16 )
        , features::BF16
    #endif
    #ifdef __ARM_FEATURE_DOTPROD
        , features::DOTPROD
    #endif
    #ifdef __ARM_FEATURE_MATMUL_INT8
        , features::I8MM
    #endif
    // ARMv8.1-A's rounding saturating multiply-accumulate. The macro is spelled QRDMX, not RDM.
    #ifdef __ARM_FEATURE_QRDMX
        , features::RDM
    #endif
    #ifdef __ARM_FEATURE_COMPLEX
        , features::FCMA
    #endif
    #ifdef __ARM_FEATURE_SVE
        , features::SVE
    #endif
    #ifdef __ARM_FEATURE_SVE2
        , features::SVE2
    #endif
>;

// -------------------------- anything else --------------------------
#else

using NativeCpu = ScalarCpu< 8 * sizeof( void * ) >;

#endif

} // namespace asimd

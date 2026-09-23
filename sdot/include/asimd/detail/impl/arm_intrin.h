#pragma once

// =============================================================================================
// ONE PLACE THAT KNOWS WHAT THE COMPILER IS OFFERING ON ARM, for all three of them -- the exact
// counterpart of `x86_intrin.h`, and it has to answer two questions that file does not.
//
//   THE HEADER IS NOT ALWAYS THERE. `<arm_neon.h>` exists only when the target actually has
//   NEON: on ARMv7 it needs `-mfpu=neon`, and on a Cortex-M there is no such header at all.
//   Including it unconditionally is how a port breaks every embedded ARM build. Hence the guard
//   AROUND the include, not just around the intrinsics. MSVC spells it `<arm64_neon.h>` on
//   ARM64 and `<arm_neon.h>` on ARM32.
//
//   `__ARM_NEON` DOES NOT MEAN WHAT IT LOOKS LIKE. It is defined on ARMv7-A *and* on AArch64,
//   and roughly a third of a 128-bit backend -- `vdivq_f32`, `float64x2_t`, `vaddvq_*`,
//   `vqtbl1q_u8`, the 64-bit compares -- is A64 only. So there are two macros below, and the
//   second one is the one most registrations are guarded by. See `ArmCpuFeatures.h` for the
//   full list of what falls on which side.
//
// The ASIMD_ARM_HAS_* macros are the single source of truth for "may this translation unit emit
// that instruction". They must agree with the feature list `NativeCpu` builds, and the two files
// are meant to be read together.
// =============================================================================================

#if defined( __aarch64__ ) || defined( __arm64__ ) || defined( _M_ARM64 ) || defined( _M_ARM64EC ) \
 || defined( __arm__ ) || defined( _M_ARM ) || defined( __ARM_ARCH )

// ---- A64, i.e. AArch64. `__ARM_ARCH_ISA_A64` is the portable spelling; the rest are what the
// individual compilers happen to define.
#if defined( __aarch64__ ) || defined( __arm64__ ) || defined( _M_ARM64 ) || defined( _M_ARM64EC ) || defined( __ARM_ARCH_ISA_A64 )
    #define ASIMD_ARM_HAS_ASIMD 1
#endif

// ---- 128-bit Advanced SIMD in any of its forms.
#if defined( __ARM_NEON ) || defined( __ARM_NEON__ ) || defined( ASIMD_ARM_HAS_ASIMD ) || defined( _M_ARM )
    #define ASIMD_ARM_HAS_NEON 1
#endif

#ifdef ASIMD_ARM_HAS_NEON
    #if defined( _MSC_VER ) && ! defined( __clang__ )
        #if defined( _M_ARM64 ) || defined( _M_ARM64EC )
            #include <arm64_neon.h>
        #else
            #include <arm_neon.h>
        #endif
    #else
        #include <arm_neon.h>
    #endif
#endif

// ---- the markers. MSVC defines none of the `__ARM_FEATURE_*` macros, so on ARM64 we take what
// the architecture guarantees (FMLA) and nothing more -- the same conservative reading the x86
// file gives MSVC below /arch:AVX.
#if defined( __ARM_FEATURE_FMA ) || defined( ASIMD_ARM_HAS_ASIMD )
    #define ASIMD_ARM_HAS_FMA 1
#endif
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    #define ASIMD_ARM_HAS_FP16 1
#endif
#if defined( __ARM_FEATURE_BF16_VECTOR_ARITHMETIC ) || defined( __ARM_FEATURE_BF16 )
    #define ASIMD_ARM_HAS_BF16 1
#endif
#ifdef __ARM_FEATURE_DOTPROD
    #define ASIMD_ARM_HAS_DOTPROD 1
#endif
#ifdef __ARM_FEATURE_MATMUL_INT8
    #define ASIMD_ARM_HAS_I8MM 1
#endif
#ifdef __ARM_FEATURE_QRDMX
    #define ASIMD_ARM_HAS_RDM 1
#endif
#ifdef __ARM_FEATURE_COMPLEX
    #define ASIMD_ARM_HAS_FCMA 1
#endif
#ifdef __ARM_FEATURE_SVE
    #define ASIMD_ARM_HAS_SVE 1
#endif
#ifdef __ARM_FEATURE_SVE2
    #define ASIMD_ARM_HAS_SVE2 1
#endif

#endif // ARM

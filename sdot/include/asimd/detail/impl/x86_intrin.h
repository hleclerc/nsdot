#pragma once

// =============================================================================================
// ONE PLACE THAT KNOWS WHAT THE COMPILER IS OFFERING, for all three of them.
//
//   THE HEADER. `<x86intrin.h>` is a gcc/clang spelling; MSVC has never had it. `<immintrin.h>`
//   covers SSE through AVX-512 on gcc, clang and MSVC alike, so that is what we include.
//
//   THE MACROS. gcc and clang define `__SSE4_1__`, `__AVX2__`, `__FMA__` &c. from the `-m` flags.
//   MSVC defines almost none of them: it has `__AVX__` / `__AVX2__` / `__AVX512F__` under the
//   matching `/arch:` and nothing else -- no `__SSE2__`, no `__SSE4_1__`, no `__FMA__`. Every
//   backend here used to be guarded by the gcc spelling, so on MSVC not one of them compiled in
//   and the whole library silently ran scalar.
//
// The ASIMD_X86_HAS_* macros below are the single source of truth for "may this translation unit
// emit that instruction". They must agree with the feature list `NativeCpu` builds, and the two
// files are meant to be read together.
// =============================================================================================

#if defined( _M_X64 ) || defined( _M_IX86 ) || defined( __i386__ ) || defined( __x86_64__ )

#include <immintrin.h>

#if defined( _MSC_VER ) && ! defined( __clang__ )
    // MSVC. SSE2 is architectural on x64; the rest rides on /arch:, which only ever sets
    // __AVX__ / __AVX2__ / __AVX512F__. Everything below AVX is folded into /arch:AVX, which is
    // what `NativeCpu` assumes too -- the two must stay in step.
    #if defined( _M_X64 ) || ( defined( _M_IX86_FP ) && _M_IX86_FP >= 2 )
        #define ASIMD_X86_HAS_SSE2 1
    #endif
    #ifdef __AVX__
        #define ASIMD_X86_HAS_SSE3   1
        #define ASIMD_X86_HAS_SSSE3  1
        #define ASIMD_X86_HAS_SSE4_1 1
        #define ASIMD_X86_HAS_SSE4_2 1
        #define ASIMD_X86_HAS_AVX    1
    #endif
    #ifdef __AVX2__
        #define ASIMD_X86_HAS_AVX2 1
        #define ASIMD_X86_HAS_FMA  1
    #endif
    #ifdef __AVX512F__
        // /arch:AVX512 turns on F, VL, BW and DQ together.
        #define ASIMD_X86_HAS_AVX512F  1
        #define ASIMD_X86_HAS_AVX512VL 1
        #define ASIMD_X86_HAS_AVX512BW 1
        #define ASIMD_X86_HAS_AVX512DQ 1
    #endif
#else
    // gcc, clang, icx: one macro per -m flag, and they are independent.
    #ifdef __SSE2__
        #define ASIMD_X86_HAS_SSE2 1
    #endif
    #ifdef __SSE3__
        #define ASIMD_X86_HAS_SSE3 1
    #endif
    #ifdef __SSSE3__
        #define ASIMD_X86_HAS_SSSE3 1
    #endif
    #ifdef __SSE4_1__
        #define ASIMD_X86_HAS_SSE4_1 1
    #endif
    #ifdef __SSE4_2__
        #define ASIMD_X86_HAS_SSE4_2 1
    #endif
    #ifdef __AVX__
        #define ASIMD_X86_HAS_AVX 1
    #endif
    #ifdef __FMA__
        #define ASIMD_X86_HAS_FMA 1
    #endif
    #ifdef __AVX2__
        #define ASIMD_X86_HAS_AVX2 1
    #endif
    #ifdef __AVX512F__
        #define ASIMD_X86_HAS_AVX512F 1
    #endif
    #ifdef __AVX512VL__
        #define ASIMD_X86_HAS_AVX512VL 1
    #endif
    #ifdef __AVX512BW__
        #define ASIMD_X86_HAS_AVX512BW 1
    #endif
    #ifdef __AVX512DQ__
        #define ASIMD_X86_HAS_AVX512DQ 1
    #endif
#endif

#endif // x86

#pragma once

// =============================================================================================
// THE `values` VIEW OF A REGISTER IMPL, and why its TYPE matters.
//
// A register impl holds its data in a union: the register itself, and a lane-addressable view
// the generic forms use when no register form exists. The obvious spelling of that view is
// `T values[ N ]` -- and on the SysV x86-64 ABI it costs a factor of two.
//
// The rule: an aggregate larger than two eightbytes travels in registers only if the first
// eightbyte classifies SSE and every following one SSEUP. `float[ 8 ]` classifies
// SSE,SSE,SSE,SSE -- an array is not a vector -- and a union takes the WORST class among its
// members, so the whole thing goes to MEMORY. Measured on a by-value `fma`: 8 instructions and 4
// memory accesses, against 4 and 0 with a vector-typed view.
//
// gcc and clang have `vector_size`, which keeps `values[ i ]` indexing AND classifies correctly.
// It buys a second thing: whole-vector arithmetic, `values + values`, which is what lets the
// generic fallbacks produce a real instruction instead of a lane loop.
//
// MSVC HAS NO EQUIVALENT, and does not need one: its x64 convention passes vectors by reference
// for anything but `__vectorcall`, so the classification question does not arise. There we fall
// back to the array, and the `requires { a.data.values + b.data.values; }` test in
// SimdVecImpl_Generic.h quietly routes those operations lane by lane instead.
// =============================================================================================

// ---------------------------------------------------------------------------------------------
// `ASIMD_NO_COMPILER_VECTORS` forces the array form on a compiler that HAS `vector_size`.
//
// It is not a portability switch -- nobody should build with it. It is a MEASURING INSTRUMENT:
// it reproduces MSVC's constraint on a compiler we have, so `tests/no_vecext.sh` can answer the
// only question that matters here -- does the dispatch table carry its own weight, or has it
// been quietly leaning on gcc's vector arithmetic to paper over its holes?
//
// The holes it finds are real holes on gcc too. They just cost less there.
// ---------------------------------------------------------------------------------------------
#if ( defined( __GNUC__ ) || defined( __clang__ ) ) && ! defined( ASIMD_NO_COMPILER_VECTORS )
    #define ASIMD_VALUES_TYPE( NAME, T, N ) typedef T NAME __attribute__(( vector_size( sizeof( T ) * ( N ) ) ))
    #define ASIMD_HAS_COMPILER_VECTORS 1
#else
    #define ASIMD_VALUES_TYPE( NAME, T, N ) typedef T NAME[ N ]
#endif

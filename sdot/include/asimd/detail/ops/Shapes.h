#pragma once

// =============================================================================================
// THE SHAPE OF A REGISTER-LEVEL VARIANT -- one macro per (operation, mask flavour), and nothing
// architecture-specific in any of them.
//
// What a variant of `select` looks like is a property of `Selection.h` and of `SimdBoolImpl`,
// not of the instruction set, so every backend uses THE SAME shapes: a backend is a TABLE, and
// this is the table's header row. A new backend is `#include "Shapes.h"` plus its own rows.
//
// What each macro leaves to the row: `a`/`b` for a binary operation, `v` for a unary one, `m`
// for a mask, `idx` for a permutation index, `LANE` for a compile-time lane -- all named inside
// FUNC.
// =============================================================================================

#include "Key.h"

#include <cstdint>
#include <array>

namespace asimd {

// ---------------------------------------------------------------------------------------------
// CONSTANT CONTROLS FOR THE LANE ROTATIONS. A compile-time rotation is a constant shuffle, and
// every instruction set spells its constant differently: an immediate for `shufps` and
// `vpermq`, a vector of lane indices for `vpermps`, a vector of BYTE indices for `TBL` and
// `pshufb`. All three are computed here, once, from the same lane map, so that a row is one
// intrinsic call and the map is written in exactly one place.
// ---------------------------------------------------------------------------------------------
namespace rot {

/// where lane `j` of `rotate_lanes<K,n>( v )` comes from.
constexpr int src( int j, int K, int n ) { return j < n ? ( j + K ) % n : j; }

/// a 2-bit-per-lane immediate over four lanes, as `_MM_SHUFFLE` reads it (lane 0 in bits 0-1).
constexpr int imm4( int K, int n ) {
    return src( 0, K, n ) | src( 1, K, n ) << 2 | src( 2, K, n ) << 4 | src( 3, K, n ) << 6;
}

/// the same, for TWO 64-bit lanes seen as four 32-bit ones -- `shuffle_epi32` on a `__m128i`
/// holding two doubles or two 64-bit integers.
constexpr int imm4_of_pairs( int K, int n ) {
    return ( 2 * src( 0, K, n ) ) | ( 2 * src( 0, K, n ) + 1 ) << 2
         | ( 2 * src( 1, K, n ) ) << 4 | ( 2 * src( 1, K, n ) + 1 ) << 6;
}

/// the lane map as a vector of 32-bit indices (`vpermps`, `vpermd`, `vpermw`).
template<int K,int n,int N,class I = std::int32_t>
struct Idx {
    static constexpr std::array<I,N> make() {
        std::array<I,N> r{};
        for ( int j = 0; j < N; ++j ) r[ j ] = I( src( j, K, n ) );
        return r;
    }
    alignas( 64 ) static constexpr std::array<I,N> v = make();
};

/// the lane map as a vector of BYTE indices, `S` bytes per lane (`TBL`, `pshufb`).
template<int K,int n,int N,int S>
struct Bytes {
    static constexpr std::array<std::uint8_t,N*S> make() {
        std::array<std::uint8_t,N*S> r{};
        for ( int i = 0; i < N * S; ++i ) r[ i ] = std::uint8_t( src( i / S, K, n ) * S + i % S );
        return r;
    }
    alignas( 64 ) static constexpr std::array<std::uint8_t,N*S> v = make();
};

/// `ext_lanes<K>` as a two-source index map: `j + K`, which a two-table permutation
/// (`vpermi2w`) reads as "lane `j+K-N` of the second table" once it passes `N`.
template<int K,int N,class I = std::int32_t>
struct ExtIdx {
    static constexpr std::array<I,N> make() {
        std::array<I,N> r{};
        for ( int j = 0; j < N; ++j ) r[ j ] = I( j + K );
        return r;
    }
    alignas( 64 ) static constexpr std::array<I,N> v = make();
};

} // namespace rot


/// "this target has that feature", and the two-feature conjunction. A vector width and the
/// instruction that uses it are often separate features -- FMA on x86, FMA on ARMv7 -- and the
/// conjunction is how a variant says it needs both.
#define ASIMD_OPS_REQ1( C1 )     Arch::template Has<features::C1>::value
#define ASIMD_OPS_REQ2( C1, C2 ) ( Arch::template Has<features::C1>::value && Arch::template Has<features::C2>::value )

/// "this feature, and NOT that one". Two backends at the same rank are ambiguous -- the rank
/// orders the levels, not the variants inside one (see the KNOWN LIMITS note in Selection.h).
/// Where an older instruction is a strictly worse fallback for a newer one, saying so in the
/// constraint is more honest than inventing a rank between REGISTER and REGISTER.
#define ASIMD_OPS_REQ_EXCL( C1, CNOT ) ( Arch::template Has<features::C1>::value && ! Arch::template Has<features::CNOT>::value )

/// fma. Two features: the one that gives the width, and FMA itself -- they are orthogonal on
/// paper and were orthogonal in practice on the first AMD parts to carry FMA, and on ARMv7,
/// where fusing needs VFPv4.
#define ASIMD_OPS_FMA( C1, C2, T, N, FUNC ) \
    template<class Arch> requires ( ASIMD_OPS_REQ2( C1, C2 ) ) \
    struct sel::Variant<ops::fma,Key<T,N,Arch>,sel::REGISTER> { \
        static constexpr bool available = true; \
        using V = internal::SimdVecImpl<T,N,Arch>; \
        static V run( const V &a, const V &b, const V &c ) { \
            V res; res.data.reg = FUNC( a.data.reg, b.data.reg, c.data.reg ); return res; } \
    }

/// idem, but with the operands passed to FUNC in whatever order the instruction wants. ARM's
/// `vfmaq_f32( acc, x, y )` computes `acc + x * y`, so the ACCUMULATOR COMES FIRST -- the
/// opposite of `_mm_fmadd_ps( x, y, acc )`. A shape that fixed the order would force every ARM
/// row to wrap itself in a lambda.
#define ASIMD_OPS_FMA_EXPR( C1, C2, T, N, FUNC ) \
    template<class Arch> requires ( ASIMD_OPS_REQ2( C1, C2 ) ) \
    struct sel::Variant<ops::fma,Key<T,N,Arch>,sel::REGISTER> { \
        static constexpr bool available = true; \
        using V = internal::SimdVecImpl<T,N,Arch>; \
        static V run( const V &a, const V &b, const V &c ) { \
            V res; res.data.reg = FUNC; return res; } \
    }

/// variable-index permutation. `v` and the index vector may have different element types, so the
/// index impl is named separately.
#define ASIMD_OPS_PERMUTE( C1, T, N, FUNC ) \
    template<class Arch> requires ( ASIMD_OPS_REQ1( C1 ) ) \
    struct sel::Variant<ops::permute,Key<T,N,Arch>,sel::REGISTER> { \
        static constexpr bool available = true; \
        using V = internal::SimdVecImpl<T,N,Arch>; \
        using I = internal::SimdVecImpl<SI32,N,Arch>; \
        static V run( const V &v, const I &idx ) { V res; res.data.reg = FUNC; return res; } \
    }

/// `permute` with an explicit exclusion, for exactly the case ASIMD_OPS_REQ_EXCL describes.
#define ASIMD_OPS_PERMUTE_EXCL( C1, CNOT, T, N, FUNC ) \
    template<class Arch> requires ( ASIMD_OPS_REQ_EXCL( C1, CNOT ) ) \
    struct sel::Variant<ops::permute,Key<T,N,Arch>,sel::REGISTER> { \
        static constexpr bool available = true; \
        using V = internal::SimdVecImpl<T,N,Arch>; \
        using I = internal::SimdVecImpl<SI32,N,Arch>; \
        static V run( const V &v, const I &idx ) { V res; res.data.reg = FUNC; return res; } \
    }

/// broadcast of a compile-time lane. `LANE` is available inside FUNC.
#define ASIMD_OPS_BCAST( C1, T, N, FUNC ) \
    template<int LANE,class Arch> requires ( ASIMD_OPS_REQ1( C1 ) ) \
    struct sel::Variant<ops::bcast_lane<LANE>,Key<T,N,Arch>,sel::REGISTER> { \
        static constexpr bool available = true; \
        using V = internal::SimdVecImpl<T,N,Arch>; \
        static V run( const V &v ) { V res; res.data.reg = FUNC; return res; } \
    }

/// comparison yielding a LANE mask (rank REGISTER) or a BIT mask (rank MASK_REGISTER).
#define ASIMD_OPS_CMP( C1, TAG, T, N, IS, RANK, FUNC ) \
    template<class Arch> requires ( ASIMD_OPS_REQ1( C1 ) ) \
    struct sel::Variant<ops::TAG,Key<T,N,Arch>,sel::RANK> { \
        static constexpr bool available = true; \
        using V = internal::SimdVecImpl<T,N,Arch>; \
        static internal::SimdBoolImpl<N,IS,Arch> run( const V &a, const V &b ) { \
            internal::SimdBoolImpl<N,IS,Arch> res; res.data.reg = FUNC; return res; } \
    }

/// blend. `m`, `a`, `b` are available inside FUNC; the result is `m ? a : b`.
#define ASIMD_OPS_SELECT( C1, T, N, IS, RANK, FUNC ) \
    template<class Arch> requires ( ASIMD_OPS_REQ1( C1 ) ) \
    struct sel::Variant<ops::select,Key<T,N,Arch,IS>,sel::RANK> { \
        static constexpr bool available = true; \
        using V = internal::SimdVecImpl<T,N,Arch>; \
        using M = internal::SimdBoolImpl<N,IS,Arch>; \
        static V run( const M &m, const V &a, const V &b ) { \
            V res; res.data.reg = FUNC; return res; } \
    }

#define ASIMD_OPS_TO_BITS( C1, N, IS, RANK, FUNC ) \
    template<class Arch> requires ( ASIMD_OPS_REQ1( C1 ) ) \
    struct sel::Variant<ops::to_bits,Key<void,N,Arch,IS>,sel::RANK> { \
        static constexpr bool available = true; \
        using M = internal::SimdBoolImpl<N,IS,Arch>; \
        static PI64 run( const M &m ) { return FUNC; } \
    }

#define ASIMD_OPS_MASK_FROM_BITS( C1, N, IS, RANK, FUNC ) \
    template<class Arch> requires ( ASIMD_OPS_REQ1( C1 ) ) \
    struct sel::Variant<ops::mask_from_bits,Key<void,N,Arch>,sel::RANK> { \
        static constexpr bool available = true; \
        static internal::SimdBoolImpl<N,IS,Arch> run( PI64 b ) { \
            internal::SimdBoolImpl<N,IS,Arch> res; res.data.reg = FUNC; return res; } \
    }

/// compile-time rotation of the first `n` lanes by `K`. `K`, `n` and `v` are available inside
/// FUNC; `COND` is an extra constraint on `( K, n )`, for a backend whose best instruction for
/// the whole register (`EXT`, `valignd`) is not the one for a prefix of it (`TBL`, `vpermps`).
#define ASIMD_OPS_ROTATE_IF( C1, COND, T, N, FUNC ) \
    template<int K,int n,class Arch> requires ( ASIMD_OPS_REQ1( C1 ) && ( COND ) ) \
    struct sel::Variant<ops::rotate_lanes<K,n>,Key<T,N,Arch>,sel::REGISTER> { \
        static constexpr bool available = true; \
        using V = internal::SimdVecImpl<T,N,Arch>; \
        static V run( const V &v ) { V res; res.data.reg = FUNC; return res; } \
    }

/// the same row for every `( K, n )`.
#define ASIMD_OPS_ROTATE( C1, T, N, FUNC ) ASIMD_OPS_ROTATE_IF( C1, true, T, N, FUNC )

/// `ASIMD_OPS_ROTATE_IF` with an exclusion -- the AVX form that AVX2 supersedes.
#define ASIMD_OPS_ROTATE_IF_EXCL( C1, CNOT, COND, T, N, FUNC ) \
    template<int K,int n,class Arch> requires ( ASIMD_OPS_REQ_EXCL( C1, CNOT ) && ( COND ) ) \
    struct sel::Variant<ops::rotate_lanes<K,n>,Key<T,N,Arch>,sel::REGISTER> { \
        static constexpr bool available = true; \
        using V = internal::SimdVecImpl<T,N,Arch>; \
        static V run( const V &v ) { V res; res.data.reg = FUNC; return res; } \
    }

/// `a[K..N) ++ b[0..K)`. `K`, `a` and `b` are available inside FUNC.
#define ASIMD_OPS_EXT( C1, T, N, FUNC ) \
    template<int K,class Arch> requires ( ASIMD_OPS_REQ1( C1 ) ) \
    struct sel::Variant<ops::ext_lanes<K>,Key<T,N,Arch>,sel::REGISTER> { \
        static constexpr bool available = true; \
        using V = internal::SimdVecImpl<T,N,Arch>; \
        static V run( const V &a, const V &b ) { V res; res.data.reg = FUNC; return res; } \
    }

/// `ext_lanes` with an exclusion, for the SSE2 form that SSSE3's `palignr` supersedes.
#define ASIMD_OPS_EXT_EXCL( C1, CNOT, T, N, FUNC ) \
    template<int K,class Arch> requires ( ASIMD_OPS_REQ_EXCL( C1, CNOT ) ) \
    struct sel::Variant<ops::ext_lanes<K>,Key<T,N,Arch>,sel::REGISTER> { \
        static constexpr bool available = true; \
        using V = internal::SimdVecImpl<T,N,Arch>; \
        static V run( const V &a, const V &b ) { V res; res.data.reg = FUNC; return res; } \
    }

/// partial load / store. `ptr` and `set` are available inside FUNC, plus `v` for the store.
/// `REQ` is the whole constraint, so that a row can exclude the feature that supersedes it: a
/// masked move on an AVX target gives way to the masked encoding of AVX-512VL, at the same rank.
#define ASIMD_OPS_LOAD_PARTIAL( REQ, T, N, FUNC ) \
    template<class Arch> requires ( REQ ) \
    struct sel::Variant<ops::load_partial,Key<T,N,Arch>,sel::REGISTER> { \
        static constexpr bool available = true; \
        using V = internal::SimdVecImpl<T,N,Arch>; \
        template<LaneSet S> static V run( const T *ptr, const S &set ) { V res; res.data.reg = FUNC; return res; } \
    }
#define ASIMD_OPS_STORE_PARTIAL( REQ, T, N, FUNC ) \
    template<class Arch> requires ( REQ ) \
    struct sel::Variant<ops::store_partial,Key<T,N,Arch>,sel::REGISTER> { \
        static constexpr bool available = true; \
        using V = internal::SimdVecImpl<T,N,Arch>; \
        template<LaneSet S> static void run( T *ptr, const V &v, const S &set ) { FUNC; } \
    }

} // namespace asimd

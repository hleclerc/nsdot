#pragma once

// =============================================================================================
// THE OPERATIONS BEYOND LOAD/STORE/ARITHMETIC, on `SimdVec` and `SimdBool`:
//
//   `fma`             fused multiply-add
//   `permute`         VARIABLE-INDEX permutation
//   `bcast_lane<i>`   broadcast a compile-time lane
//   `select`          blend two vectors according to a mask
//   `to_bits`         a mask AS AN INTEGER -- what carries `ctz`, `popcount` and the rotations
//   `mask_from_bits`  the dual: a mask from an integer
//   `gt` `lt` `eq` `ge`  comparisons materialized as a mask, in whatever flavour the target has
//   `rotate_lanes`    lanes `[0,n)` rotated by `k` -- `k` and `n` each compile-time (`N<>`) or not
//   `ext_lanes<K>`    `a[K..N) ++ b[0..K)`: `EXT` / `alignr`, the primitive a rotation is made of
//   `load_partial` `store_partial`  the lanes of a set and not one byte outside them (`V::load_partial( p, n )`)
//   `add` `sub` `mul` `div` `min` `max` `sum`  the arithmetic as free functions, so that they can
//                     take a LANE SET (`LaneSet.h`): `add( a, b, LaneRange<0,3>() )` computes
//                     only the registers that hold lanes 0-2. `fma`, `select`, the comparisons,
//                     `sum`, `to_bits`, `any` and `all` take one too.
//
// This file holds the FACADES only: each one names an operation tag and a `Key`, and lets
// `Selection.h` pick the highest-ranked variant registered for it. The variants live in `ops/`:
//
//   ops/Key.h       the tags and the key
//   ops/Generic.h   rank GENERIC -- a lane loop, always available
//   ops/Shapes.h    the macros a backend fills its table with
//   ops/X86.h       rank REGISTER / MASK_REGISTER, the x86 table
//   ops/Neon.h      rank REGISTER, the ARM table
//   ops/Split.h     rank SPLIT -- a width above the register, delegated to the halves
//
// `permute` deserves a note. Its generic form goes through MEMORY (store, index, reload), because
// a split into two half-registers cannot move a lane from one half to the other without meeting
// somewhere. That is correct everywhere and slow; the AVX2 form (`vpermps`) and the NEON one
// (`tbl`) are a single instruction.
// =============================================================================================

#include "LaneSet.h"
#include "ops/Key.h"
#include "ops/Generic.h"

// Register-level variants. Class specializations are looked up at instantiation, so these could
// sit anywhere -- they are here for readability, not out of necessity. That is precisely what
// `Selection.h` buys: with qualified calls to overloaded functions, putting them after the
// facades below would silently cost every bit of vectorization.
#include "ops/X86.h"
#include "ops/Neon.h"

// ... and the SPLIT-rank forms, which delegate to whatever the register level above resolved to.
// They have to come after it: `available` asks the halves what rank they reached. On ARM that is
// not a refinement but the main path: the register is 128 bits and never wider, so any width
// above four floats IS a split -- see the header comment of `ops/Neon.h`.
#include "ops/Split.h"

namespace asimd {


template<class T,int N,class Arch>
SimdVec<T,N,Arch> fma( const SimdVec<T,N,Arch> &a, const SimdVec<T,N,Arch> &b,
                       const SimdVec<T,N,Arch> &c ) {
    return sel::call<ops::fma,Key<T,N,Arch>>( a.impl, b.impl, c.impl );
}

template<class T,int N,class Arch>
SimdVec<T,N,Arch> permute( const SimdVec<T,N,Arch> &v, const SimdVec<SI32,N,Arch> &idx ) {
    return sel::call<ops::permute,Key<T,N,Arch>>( v.impl, idx.impl );
}

template<int LANE,class T,int N,class Arch>
SimdVec<T,N,Arch> bcast_lane( const SimdVec<T,N,Arch> &v ) {
    return sel::call<ops::bcast_lane<LANE>,Key<T,N,Arch>>( v.impl );
}

template<class T,int N,int IS,class Arch>
SimdVec<T,N,Arch> select( const SimdBool<N,IS,Arch> &m, const SimdVec<T,N,Arch> &a,
                          const SimdVec<T,N,Arch> &b ) {
    return sel::call<ops::select,Key<T,N,Arch,IS>>( m.impl, a.impl, b.impl );
}

template<int N,int IS,class Arch>
PI64 to_bits( const SimdBool<N,IS,Arch> &m ) {
    return sel::call<ops::to_bits,Key<void,N,Arch,IS>>( m.impl );
}

/// THE DUAL OF `to_bits`, and it is worth having. Designating a lane through `eq( iota, i )`
/// takes a `vpbroadcastd` then a `vpcmpeqd`; on a target with mask registers the same mask is one
/// `kmovb` from an integer you already computed. Code that designates a few lanes per iteration
/// pays that difference -- six instructions against three -- in its hottest loop.
///
/// It is also what lets portable code express "the first n lanes", "every other lane", or any
/// computed pattern -- all things a vector comparison can only say by abusing an `iota`.
template<int N,class Arch = NativeCpu>
auto mask_from_bits( PI64 b ) {
    return simd_bool_from_simd_bool_impl( sel::call<ops::mask_from_bits,Key<void,N,Arch>>( b ) );
}

// =====================================================================================
// LANE ROTATIONS
//
//   rotate_lanes( v, k, n )[ i ] = i < n ? v[ ( i + k ) % n ] : v[ i ]
//
// lane `k` lands in lane 0 (a `k > 0` moves data DOWN, like `std::rotate`); `k` may be negative
// or larger than `n`; lanes at and past `n` keep their value; `n` defaults to the whole vector.
//
// Four spellings, because each argument may or may not be known at compile time -- `N<k>` says
// it is -- and the best code differs:
//
//   `N<k>`, `N<n>`   a constant shuffle: `EXT` / `shufps` / `vpermq` with an immediate, or
//                    `TBL` / `vpermps` with a constant table. One instruction, and across a
//                    split, two.
//   `int`,  `N<n>`   `permute` with an index built from `iota + k` -- and when `n` is the whole
//                    of a power-of-two width held in one register, nothing else: every `permute`
//                    form wraps its index (`vpermps` keeps the low bits, `TBL` is masked, the
//                    generic form takes a modulo), so the wrap is free.
//   `int`,  `int`    the same, with the wrap and the "lanes past `n`" blend computed.
//   `N<k>`, `int`    the `int, int` path with `k` folded by the compiler.
//
// `ext_lanes<K>( a, b )` is `a[K..N) ++ b[0..K)`: what a whole-vector rotation is when the two
// operands are the same, and what a rotation ACROSS registers is made of when they are not. It
// is exposed because sliding a window over two consecutive vectors is exactly this.
// =====================================================================================

template<int K,class T,int W,class Arch>
SimdVec<T,W,Arch> ext_lanes( const SimdVec<T,W,Arch> &a, const SimdVec<T,W,Arch> &b, N<K> ) {
    static_assert( K >= 0 && K <= W, "ext_lanes: K must be in [0,W]" );
    if constexpr ( K == 0 ) return a;
    else if constexpr ( K == W ) return b;
    else return sel::call<ops::ext_lanes<K>,Key<T,W,Arch>>( a.impl, b.impl );
}

template<int k,int n,class T,int W,class Arch>
SimdVec<T,W,Arch> rotate_lanes( const SimdVec<T,W,Arch> &v, N<k>, N<n> ) {
    static_assert( n >= 1 && n <= W, "rotate_lanes: n must be in [1,W]" );
    constexpr int K = ( k % n + n ) % n;
    if constexpr ( K == 0 ) return v;
    else return sel::call<ops::rotate_lanes<K,n>,Key<T,W,Arch>>( v.impl );
}

template<int k,class T,int W,class Arch>
SimdVec<T,W,Arch> rotate_lanes( const SimdVec<T,W,Arch> &v, N<k> ) {
    return rotate_lanes( v, N<k>(), N<W>() );
}

namespace internal {
    /// the runtime rotation: a `permute` whose index is `iota + k`, wrapped at `n` and blended
    /// with `iota` past it. `n_ct` is `n` when known at compile time, 0 otherwise.
    template<int n_ct,class T,int W,class Arch>
    SimdVec<T,W,Arch> rotate_lanes_rt( const SimdVec<T,W,Arch> &v, int k, int n ) {
        using I = SimdVec<SI32,W,Arch>;
        k = ( k % n + n ) % n;
        const I io  = I::iota();
        I       idx = io + I( SI32( k ) );
        // the wrap. Free when `n` is the whole of a power-of-two width held in ONE register:
        // every register `permute` form reads its index modulo the width, and so does the
        // generic one. NOT across a split -- there an index past the width selects the upper
        // half, and wraps inside it.
        constexpr bool pow2      = n_ct > 0 && ( n_ct & ( n_ct - 1 ) ) == 0;
        constexpr bool free_wrap = n_ct == W && pow2
                                && ! internal::HasSplit<internal::SimdVecImpl<T,W,Arch>>;
        if constexpr ( free_wrap ) {}
        else if constexpr ( pow2 ) idx = idx & I( SI32( n_ct - 1 ) );   // one `and`; the lanes past `n` are overwritten below
        else idx = select( ge( idx, I( SI32( n ) ) ), idx - I( SI32( n ) ), idx );
        // the lanes past `n`, left alone
        if constexpr ( n_ct == 0 ) { if ( n != W ) idx = select( lt( io, I( SI32( n ) ) ), idx, io ); }
        else if constexpr ( n_ct != W ) idx = select( lt( io, I( SI32( n ) ) ), idx, io );
        return permute( v, idx );
    }
}

template<int n,class T,int W,class Arch>
SimdVec<T,W,Arch> rotate_lanes( const SimdVec<T,W,Arch> &v, int k, N<n> ) {
    static_assert( n >= 1 && n <= W, "rotate_lanes: n must be in [1,W]" );
    return internal::rotate_lanes_rt<n>( v, k, n );
}

template<class T,int W,class Arch>
SimdVec<T,W,Arch> rotate_lanes( const SimdVec<T,W,Arch> &v, int k, int n ) {
    return internal::rotate_lanes_rt<0>( v, k, n );
}

template<class T,int W,class Arch>
SimdVec<T,W,Arch> rotate_lanes( const SimdVec<T,W,Arch> &v, int k ) {
    return internal::rotate_lanes_rt<W>( v, k, W );
}

template<int k,class T,int W,class Arch>
SimdVec<T,W,Arch> rotate_lanes( const SimdVec<T,W,Arch> &v, N<k>, int n ) {
    return internal::rotate_lanes_rt<0>( v, k, n );
}

// =====================================================================================
// LANE SETS -- the trailing argument. See `LaneSet.h` for the contract: lanes outside the set
// are UNSPECIFIED in a vector result, EXCLUDED from a reduction.
//
// Only lane-wise operations take one. A permutation reads every lane of its input whatever the
// caller wants of its output, so `permute`, `rotate_lanes` and `ext_lanes` do not -- and
// `rotate_lanes` already has `n`, which is the prefix it moves and leaves the rest untouched.
// =====================================================================================

namespace internal {
    template<class X0,class... X> struct FirstOf { using type = X0; };

    /// a lane-wise op through `sel::call`, keyed on the width of whatever impl it is handed --
    /// which is how the same lambda serves the whole vector and each of its halves.
    template<class Op,class T,class Arch>
    struct LaneWise {
        template<class... X>
        auto operator()( const X &...x ) const {
            return sel::call<Op,Key<T,width_of<typename FirstOf<X...>::type>,Arch>>( x... );
        }
    };
    template<class Op,class T,class Arch>
    struct LaneWiseSelect {
        template<class M,class X>
        auto operator()( const M &m, const X &a, const X &b ) const {
            return sel::call<Op,Key<T,width_of<X>,Arch,item_size_of<M>>>( m, a, b );
        }
    };
}

#define ASIMD_LANEWISE_ARITH( NAME, IMPL ) \
    template<class T,int W,class Arch> \
    SimdVec<T,W,Arch> NAME( const SimdVec<T,W,Arch> &a, const SimdVec<T,W,Arch> &b ) { \
        return IMPL( a.impl, b.impl ); \
    } \
    template<class T,int W,class Arch,LaneSet S> \
    SimdVec<T,W,Arch> NAME( const SimdVec<T,W,Arch> &a, const SimdVec<T,W,Arch> &b, const S &s ) { \
        return internal::prune( s, []( const auto &x, const auto &y ) { return IMPL( x, y ); }, a.impl, b.impl ); \
    }

ASIMD_LANEWISE_ARITH( add, internal::add )
ASIMD_LANEWISE_ARITH( sub, internal::sub )
ASIMD_LANEWISE_ARITH( mul, internal::mul )
ASIMD_LANEWISE_ARITH( div, internal::div )
#undef ASIMD_LANEWISE_ARITH

// `min` / `max` without a set are in `SimdVec.h`
template<class T,int W,class Arch,LaneSet S>
SimdVec<T,W,Arch> min( const SimdVec<T,W,Arch> &a, const SimdVec<T,W,Arch> &b, const S &s ) {
    return internal::prune( s, []( const auto &x, const auto &y ) { return internal::min( x, y ); }, a.impl, b.impl );
}
template<class T,int W,class Arch,LaneSet S>
SimdVec<T,W,Arch> max( const SimdVec<T,W,Arch> &a, const SimdVec<T,W,Arch> &b, const S &s ) {
    return internal::prune( s, []( const auto &x, const auto &y ) { return internal::max( x, y ); }, a.impl, b.impl );
}

template<class T,int W,class Arch,LaneSet S>
SimdVec<T,W,Arch> fma( const SimdVec<T,W,Arch> &a, const SimdVec<T,W,Arch> &b, const SimdVec<T,W,Arch> &c, const S &s ) {
    return internal::prune( s, internal::LaneWise<ops::fma,T,Arch>(), a.impl, b.impl, c.impl );
}

template<class T,int W,int IS,class Arch,LaneSet S>
SimdVec<T,W,Arch> select( const SimdBool<W,IS,Arch> &m, const SimdVec<T,W,Arch> &a, const SimdVec<T,W,Arch> &b, const S &s ) {
    return internal::prune( s, internal::LaneWiseSelect<ops::select,T,Arch>(), m.impl, a.impl, b.impl );
}

#define ASIMD_LANEWISE_CMP( NAME, TAG ) \
    template<class T,int W,class Arch,LaneSet S> \
    auto NAME( const SimdVec<T,W,Arch> &a, const SimdVec<T,W,Arch> &b, const S &s ) { \
        return simd_bool_from_simd_bool_impl( internal::prune( s, internal::LaneWise<ops::TAG,T,Arch>(), a.impl, b.impl ) ); \
    }
ASIMD_LANEWISE_CMP( gt, cmp_gt )
ASIMD_LANEWISE_CMP( lt, cmp_lt )
ASIMD_LANEWISE_CMP( eq, cmp_eq )
ASIMD_LANEWISE_CMP( ge, cmp_ge )
#undef ASIMD_LANEWISE_CMP

/// a selection driven by a lazy comparison, both pruned
template<class T,class U,int W,class Arch,LaneSet S>
SimdVec<U,W,Arch> select( const internal::Op_gt<T,W,Arch> &op, const SimdVec<U,W,Arch> &a, const SimdVec<U,W,Arch> &b, const S &s ) {
    const auto m = internal::prune( s, internal::LaneWise<ops::cmp_gt,T,Arch>(), op.a, op.b );
    return internal::prune( s, internal::LaneWiseSelect<ops::select,U,Arch>(), m, a.impl, b.impl );
}

// ---- reductions: the set is part of the answer -----------------------------------------------
namespace internal {
    template<LaneSet Set,class T,int N,class Arch>
    SimdVecImpl<T,N,Arch> load_partial( const T *ptr, const Set &set, S<SimdVecImpl<T,N,Arch>> ) {
        return sel::call<ops::load_partial,Key<T,N,Arch>>( ptr, set );
    }
    template<LaneSet Set,class T,int N,class Arch>
    void store_partial( T *ptr, const SimdVecImpl<T,N,Arch> &v, const Set &set ) {
        sel::call<ops::store_partial,Key<T,N,Arch>>( ptr, v, set );
    }

    template<LaneSet S,class T,int N,class Arch>
    T sum_pruned( const S &s, const SimdVecImpl<T,N,Arch> &v ) {
        using I = SimdVecImpl<T,N,Arch>;
        if constexpr ( hull_covers<S,0,N> ) {
            return horizontal_sum( v );
        } else if constexpr ( HasSplit<I> ) {
            constexpr int n0 = I::split_size_0;
            if constexpr ( hull_disjoint<S,n0,N> ) {
                return sum_pruned( s, v.data.split.v0 );
            } else if constexpr ( hull_disjoint<S,0,n0> ) {
                return sum_pruned( s.template shifted<n0>(), v.data.split.v1 );
            } else {
                if constexpr ( ! S::is_static && ( HasSplit<Half0<I>> || HasSplit<Half1<I>> ) ) {
                    if ( s.empty( n0, N ) ) return sum_pruned( s, v.data.split.v0 );
                    if ( s.empty( 0, n0 ) ) return sum_pruned( s.template shifted<n0>(), v.data.split.v1 );
                }
                return sum_pruned( s, v.data.split.v0 ) + sum_pruned( s.template shifted<n0>(), v.data.split.v1 );
            }
        } else if constexpr ( S::is_static ) {
            // one register, partly wanted: keep the lanes of the set, reduce the whole
            return horizontal_sum( anb( v, load_aligned( LanePattern<S,T,N>::v.data(), asimd::S<I>() ) ) );
        } else {
            T res = 0;
            for ( int i = 0; i < N; ++i )
                if ( s.has( i ) ) res += T( at( v, i ) );
            return res;
        }
    }

    template<LaneSet S,int N,int IS,class Arch>
    PI64 to_bits_pruned( const S &s, const SimdBoolImpl<N,IS,Arch> &m ) {
        using I = SimdBoolImpl<N,IS,Arch>;
        if constexpr ( HasSplit<I> ) {
            constexpr int n0 = I::split_size_0;
            if constexpr ( hull_disjoint<S,n0,N> ) {
                return to_bits_pruned( s, m.data.split.v0 );
            } else if constexpr ( hull_disjoint<S,0,n0> ) {
                return to_bits_pruned( s.template shifted<n0>(), m.data.split.v1 ) << n0;
            } else {
                if constexpr ( ! S::is_static && ( HasSplit<Half0<I>> || HasSplit<Half1<I>> ) ) {
                    if ( s.empty( n0, N ) ) return to_bits_pruned( s, m.data.split.v0 );
                    if ( s.empty( 0, n0 ) ) return to_bits_pruned( s.template shifted<n0>(), m.data.split.v1 ) << n0;
                }
                return sel::call<ops::to_bits,Key<void,N,Arch,IS>>( m ) & s.bits( N );
            }
        } else {
            return sel::call<ops::to_bits,Key<void,N,Arch,IS>>( m ) & s.bits( N );
        }
    }
}

template<class T,int W,class Arch>
T sum( const SimdVec<T,W,Arch> &v ) { return internal::horizontal_sum( v.impl ); }

template<class T,int W,class Arch,LaneSet S>
T sum( const SimdVec<T,W,Arch> &v, const S &s ) { return internal::sum_pruned( s, v.impl ); }

template<int W,int IS,class Arch,LaneSet S>
PI64 to_bits( const SimdBool<W,IS,Arch> &m, const S &s ) { return internal::to_bits_pruned( s, m.impl ); }

template<int W,int IS,class Arch,LaneSet S>
bool any( const SimdBool<W,IS,Arch> &m, const S &s ) { return internal::to_bits_pruned( s, m.impl ) != 0; }

template<int W,int IS,class Arch,LaneSet S>
bool all( const SimdBool<W,IS,Arch> &m, const S &s ) { return internal::to_bits_pruned( s, m.impl ) == s.bits( W ); }

#define ASIMD_LANEWISE_LAZY_REDUCE( NAME, TAG ) \
    template<class T,int W,class Arch,LaneSet S> \
    PI64 to_bits( const internal::Op_##NAME<T,W,Arch> &op, const S &s ) { \
        return internal::to_bits_pruned( s, internal::prune( s, internal::LaneWise<ops::TAG,T,Arch>(), op.a, op.b ) ); \
    } \
    template<class T,int W,class Arch,LaneSet S> \
    bool any( const internal::Op_##NAME<T,W,Arch> &op, const S &s ) { return to_bits( op, s ) != 0; } \
    template<class T,int W,class Arch,LaneSet S> \
    bool all( const internal::Op_##NAME<T,W,Arch> &op, const S &s ) { return to_bits( op, s ) == s.bits( W ); }
ASIMD_LANEWISE_LAZY_REDUCE( gt, cmp_gt )
ASIMD_LANEWISE_LAZY_REDUCE( lt, cmp_lt )
#undef ASIMD_LANEWISE_LAZY_REDUCE

/// COMPARISONS ARE LAZY IN ASIMD: `a > b` computes nothing, it returns an object holding both
/// operands, and the caller decides in which form to materialize it. The named forms below
/// materialize one directly as a mask; the overloads after them go straight to "bits" or to a
/// "selection" without naming the mask flavour in between.
///
/// The return type is deduced on purpose: the mask FLAVOUR depends on the target -- bits where
/// there are mask registers, full lanes elsewhere -- and pinning it here would force a choice.
/// `select` and `to_bits` accept both.
#define ASIMD_OPS_CMP_FACADE( NAME, TAG )                                                  \
    template<class T,int N,class Arch>                                                      \
    auto NAME( const SimdVec<T,N,Arch> &a, const SimdVec<T,N,Arch> &b ) {                   \
        return simd_bool_from_simd_bool_impl( sel::call<ops::TAG,Key<T,N,Arch>>( a.impl, b.impl ) ); \
    }

ASIMD_OPS_CMP_FACADE( gt, cmp_gt )
ASIMD_OPS_CMP_FACADE( lt, cmp_lt )
ASIMD_OPS_CMP_FACADE( eq, cmp_eq )
ASIMD_OPS_CMP_FACADE( ge, cmp_ge )

#undef ASIMD_OPS_CMP_FACADE

/// `a > b` straight to bits, without naming the mask flavour in between.
template<class T,int N,class Arch>
PI64 to_bits( const internal::Op_gt<T,N,Arch> &op ) {
    auto m = sel::call<ops::cmp_gt,Key<T,N,Arch>>( op.a, op.b );
    return sel::call<ops::to_bits,Key<void,N,Arch,mask_item_size<decltype( m )>::value>>( m );
}
template<class T,int N,class Arch>
PI64 to_bits( const internal::Op_lt<T,N,Arch> &op ) {
    auto m = sel::call<ops::cmp_lt,Key<T,N,Arch>>( op.a, op.b );
    return sel::call<ops::to_bits,Key<void,N,Arch,mask_item_size<decltype( m )>::value>>( m );
}

/// idem for a selection driven by a lazy comparison.
template<class T,class U,int N,class Arch>
SimdVec<U,N,Arch> select( const internal::Op_gt<T,N,Arch> &op, const SimdVec<U,N,Arch> &a,
                          const SimdVec<U,N,Arch> &b ) {
    auto m = sel::call<ops::cmp_gt,Key<T,N,Arch>>( op.a, op.b );
    return sel::call<ops::select,Key<U,N,Arch,mask_item_size<decltype( m )>::value>>( m, a.impl, b.impl );
}

} // namespace asimd

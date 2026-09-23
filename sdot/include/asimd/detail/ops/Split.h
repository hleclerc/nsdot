#pragma once

// =============================================================================================
// THE SPLIT RANK: a width above what the register holds, done as two operations on the halves.
//
// asimd's premise is that the width is the author's choice: `SimdVec<float,8>` must work on a
// target whose registers hold four. The vector impls split recursively and the arithmetic follows
// the split; the operations of `SimdOps.h` follow it through this file, since their generic forms
// walk `values` lane by lane and would ignore the split entirely.
//
// Measured, `to_bits( a > b )` on `float x 8` under SSE2:
//
//     written with plain gcc vector extensions      48 instructions
//     generic forms                                 73          <- worse than doing nothing
//     the split forms below                          9
//
// It is not per-type work: a split form is the same three lines for every type and every width,
// because it delegates to whatever the halves resolve to -- a register form, a mask register
// form, or another split.
//
// `permute` is the awkward one, and it is included anyway. A permutation moves a lane from one
// half to the OTHER, which is exactly what two half-registers cannot do directly -- so it costs
// two half-permutations and a blend per half, four and two for the whole, where a target with a
// full-width `vpermps` needs one instruction. It is still worth having: the alternative is the
// generic form, which stores the vector to the stack, indexes it there and reloads. Slow beats a
// round trip through memory, and more to the point the contract is that EVERY operation works at
// EVERY width -- an operation that opts out of the split mechanism is a hole in the premise.
// =============================================================================================

#include "Key.h"

namespace asimd {

namespace internal {

/// The two halves of an impl, or zero when it does not split. Written as a specialisation rather
/// than an `if constexpr` because `SimdVecImpl<T,1,Arch>` has no `split_size_0` to name at all.
template<class T,int N,class Arch,bool = HasSplit<SimdVecImpl<T,N,Arch>>>
struct SplitOf { static constexpr int n0 = 0, n1 = 0; };

template<class T,int N,class Arch>
struct SplitOf<T,N,Arch,true> {
    static constexpr int n0 = SimdVecImpl<T,N,Arch>::split_size_0;
    static constexpr int n1 = SimdVecImpl<T,N,Arch>::split_size_1;
};

/// idem for a mask.
template<int N,int IS,class Arch,bool = requires ( SimdBoolImpl<N,IS,Arch> m ) { m.data.split.v0; }>
struct MaskSplits { static constexpr bool value = false; };
template<int N,int IS,class Arch>
struct MaskSplits<N,IS,Arch,true> { static constexpr bool value = true; };

} // namespace internal

namespace internal {

/// The two halves of a split need not have the same width -- at N = 5 they are 4 and 1 -- so a
/// value broadcast in one half has to be RE-FORMED at the other half's width rather than copied
/// across. When the widths do match this is the identity and compiles to nothing.
///
/// A FREE function template rather than a member of the variant, deliberately: calling a member
/// template with explicit template arguments (`spread<n1>( h )`) from inside its own class is
/// exactly the shape MSVC's two-phase lookup has historically been weakest on, and MSVC is the
/// one compiler this cannot be tried on before pushing. Nothing is lost by moving it out.
template<class T,int M,class Arch,class H>
static SimdVecImpl<T,M,Arch> spread_lane_0( const H &h ) {
    if constexpr ( std::is_same<H,SimdVecImpl<T,M,Arch>>::value ) {
        return h;
    } else {
        SimdVecImpl<T,M,Arch> r;
        init_sc( r, T( at( h, 0 ) ) );
        return r;
    }
}

} // namespace internal

/// "is the half worth delegating to" -- i.e. does it resolve to something better than another
/// lane loop. Guarded by a specialisation so that `rank<..., Key<T,0,Arch>>` is never named.
template<class Op,class T,int N,class Arch,bool = ( N > 0 )>
struct HalfIsWorthIt { static constexpr bool value = false; };
template<class Op,class T,int N,class Arch>
struct HalfIsWorthIt<Op,T,N,Arch,true> {
    static constexpr bool value = sel::rank<Op,Key<T,N,Arch>> > sel::GENERIC;
};

/// what a comparison actually returns at a given width, once the selector has had its say. The
/// flavour is not fixed -- lane mask below AVX-512, bit mask above -- so it has to be read back.
template<class Op,class T,int N,class Arch>
using CmpResultOf = decltype( sel::call<Op,Key<T,N,Arch>>(
    std::declval<const internal::SimdVecImpl<T,N,Arch> &>(),
    std::declval<const internal::SimdVecImpl<T,N,Arch> &>() ) );

// ---------------------------------------------------------------------------------------------
// comparisons
// ---------------------------------------------------------------------------------------------
#define ASIMD_OPS_SPLIT_CMP( TAG )                                                              \
    template<class T,int N,class Arch>                                                           \
    struct sel::Variant<ops::TAG,Key<T,N,Arch>,sel::SPLIT> {                                     \
        using V = internal::SimdVecImpl<T,N,Arch>;                                               \
        static constexpr int n0 = internal::SplitOf<T,N,Arch>::n0;                               \
        static constexpr int n1 = internal::SplitOf<T,N,Arch>::n1;                               \
        static constexpr int is0 = ASIMD_ITEM_SIZE_OR_0( TAG, n0 );                              \
        static constexpr int is1 = ASIMD_ITEM_SIZE_OR_0( TAG, n1 );                              \
        /* both halves must agree on the mask flavour, and the whole must be able to hold two */ \
        static constexpr bool available =                                                        \
            HalfIsWorthIt<ops::TAG,T,n0,Arch>::value && is0 != 0 && is0 == is1                   \
            && internal::MaskSplits<N,is0,Arch>::value;                                          \
        static internal::SimdBoolImpl<N,is0,Arch> run( const V &a, const V &b ) {                \
            internal::SimdBoolImpl<N,is0,Arch> res;                                              \
            res.data.split.v0 = sel::call<ops::TAG,Key<T,n0,Arch>>( a.data.split.v0, b.data.split.v0 ); \
            res.data.split.v1 = sel::call<ops::TAG,Key<T,n1,Arch>>( a.data.split.v1, b.data.split.v1 ); \
            return res;                                                                          \
        }                                                                                        \
    }

/// the mask item size a comparison yields at width N, or 0 when N is not a real width.
template<class Op,class T,int N,class Arch,bool = ( N > 0 )>
struct ItemSizeOf { static constexpr int value = 0; };
template<class Op,class T,int N,class Arch>
struct ItemSizeOf<Op,T,N,Arch,true> {
    static constexpr int value = mask_item_size<CmpResultOf<Op,T,N,Arch>>::value;
};

#define ASIMD_ITEM_SIZE_OR_0( TAG, NN ) ItemSizeOf<ops::TAG,T,NN,Arch>::value

ASIMD_OPS_SPLIT_CMP( cmp_gt );
ASIMD_OPS_SPLIT_CMP( cmp_lt );
ASIMD_OPS_SPLIT_CMP( cmp_eq );
ASIMD_OPS_SPLIT_CMP( cmp_ge );

#undef ASIMD_ITEM_SIZE_OR_0
#undef ASIMD_OPS_SPLIT_CMP

// ---------------------------------------------------------------------------------------------
// to_bits -- the one that was 73 instructions. Each half yields its own bits; shift and or.
// ---------------------------------------------------------------------------------------------
template<int N,int IS,class Arch,bool = ( N > 0 )>
struct MaskHalfWorthIt { static constexpr bool value = false; };
template<int N,int IS,class Arch>
struct MaskHalfWorthIt<N,IS,Arch,true> {
    static constexpr bool value = sel::rank<ops::to_bits,Key<void,N,Arch,IS>> > sel::GENERIC;
};

template<int N,int IS,class Arch,bool = requires ( internal::SimdBoolImpl<N,IS,Arch> m ) { m.data.split.v0; }>
struct MaskSplitSizes { static constexpr int n0 = 0, n1 = 0; };
template<int N,int IS,class Arch>
struct MaskSplitSizes<N,IS,Arch,true> {
    static constexpr int n0 = internal::SimdBoolImpl<N,IS,Arch>::split_size_0;
    static constexpr int n1 = internal::SimdBoolImpl<N,IS,Arch>::split_size_1;
};

template<int N,class Arch,int IS>
struct sel::Variant<ops::to_bits,Key<void,N,Arch,IS>,sel::SPLIT> {
    static constexpr int n0 = MaskSplitSizes<N,IS,Arch>::n0;
    static constexpr int n1 = MaskSplitSizes<N,IS,Arch>::n1;
    static constexpr bool available = MaskHalfWorthIt<n0,IS,Arch>::value;
    static PI64 run( const internal::SimdBoolImpl<N,IS,Arch> &m ) {
        return sel::call<ops::to_bits,Key<void,n0,Arch,IS>>( m.data.split.v0 )
             | ( sel::call<ops::to_bits,Key<void,n1,Arch,IS>>( m.data.split.v1 ) << n0 );
    }
};

// ---------------------------------------------------------------------------------------------
// mask_from_bits -- the dual of `to_bits`.
//
// WHY IT MATTERS MORE ON ARM THAN ON x86. `Key<void,N,Arch>` carries no item size, so ONE
// registration per width decides which FLAVOUR of mask that width produces, and every later
// `select` has to live with it. On x86 the register forms cover 4, 8 and 16 lanes, so the
// question rarely arises. On ARM the register is 128 bits and never wider: `mask_from_bits<8>`
// has no width to be registered at, and the generic form writes eight lanes one at a time --
// after which `select` at eight lanes gets a mask its own split form cannot use.
//
// Hence this: build each half with whatever the half resolved to, and keep the flavour they
// agree on. Three lines, no per-type work, and the same three lines improve the x86 SSE-only
// path -- `mask_from_bits<8>` there was generic too.
//
// `available` has to check FOUR things, and the last two are the ones that bite:
//   - the half resolves to better than a lane loop, or there is nothing to gain;
//   - both halves agree on the item size, or there is no type for the result;
//   - the whole mask actually splits, at that item size;
//   - and it splits THE SAME WAY the halves were computed -- `split_size_0 == n0`. A mask of 12
//     items splits 8 + 4, so the two are not interchangeable.
template<int N,class Arch,bool = ( N > 0 )>
struct MfbItemSizeOf { static constexpr int value = 0; };
template<int N,class Arch>
struct MfbItemSizeOf<N,Arch,true> {
    static constexpr int value = mask_item_size<decltype( sel::call<ops::mask_from_bits,Key<void,N,Arch>>( PI64( 0 ) ) )>::value;
};

template<int N,class Arch,bool = ( N > 0 )>
struct MfbHalfWorthIt { static constexpr bool value = false; };
template<int N,class Arch>
struct MfbHalfWorthIt<N,Arch,true> {
    static constexpr bool value = sel::rank<ops::mask_from_bits,Key<void,N,Arch>> > sel::GENERIC;
};

/// EVERYTHING ABOUT THE SPLIT, BEHIND ONE GUARD -- and the guard is `prev_pow_2( N ) < N`, not
/// just `N > 1`.
///
/// The other split variants get this for free: they take their halves from
/// `SimdVecImpl<T,N,Arch>::split_size_0`, which only exists when the impl really splits, so at
/// N = 1 there is nothing to name and `available` is false. This one has no impl to ask -- the
/// width whose flavour it is deciding is the one being computed -- so it uses `prev_pow_2`
/// directly, and `prev_pow_2( 1 )` is 1. Written without the guard, the variant at N = 1 asked
/// what `mask_from_bits<1>` resolves to in order to decide what `mask_from_bits<1>` resolves to:
/// clang reported it as "`is0` must be initialized by a constant expression" sixty-odd
/// instantiations deep, which is what an infinite template recursion looks like from the outside.
template<int N,class Arch,bool = ( N >= 2 && prev_pow_2( N ) < N )>
struct MfbSplitOf {
    static constexpr int n0 = 0, n1 = 0, is = 0;
    static constexpr bool ok = false;
};

template<int N,class Arch>
struct MfbSplitOf<N,Arch,true> {
    static constexpr int n0  = prev_pow_2( N );
    static constexpr int n1  = N - n0;
    static constexpr int is  = MfbItemSizeOf<n0,Arch>::value;
    static constexpr int is1 = MfbItemSizeOf<n1,Arch>::value;
    static constexpr bool ok =
        MfbHalfWorthIt<n0,Arch>::value                  // the half is better than a lane loop
        && is != 0 && is == is1                         // both halves agree on the flavour
        && internal::MaskSplits<N,is,Arch>::value       // the whole mask splits, at that flavour
        && MaskSplitSizes<N,is,Arch>::n0 == n0;         // ... and it splits THE SAME WAY
};

template<int N,class Arch>
struct sel::Variant<ops::mask_from_bits,Key<void,N,Arch>,sel::SPLIT> {
    using SP = MfbSplitOf<N,Arch>;
    static constexpr bool available = SP::ok;
    static internal::SimdBoolImpl<N,SP::is,Arch> run( PI64 b ) {
        internal::SimdBoolImpl<N,SP::is,Arch> res;
        res.data.split.v0 = sel::call<ops::mask_from_bits,Key<void,SP::n0,Arch>>( b );
        res.data.split.v1 = sel::call<ops::mask_from_bits,Key<void,SP::n1,Arch>>( b >> SP::n0 );
        return res;
    }
};

// ---------------------------------------------------------------------------------------------
// select -- two half blends
// ---------------------------------------------------------------------------------------------
template<class T,int N,class Arch,int IS,bool = ( N > 0 )>
struct SelectHalfWorthIt { static constexpr bool value = false; };
template<class T,int N,class Arch,int IS>
struct SelectHalfWorthIt<T,N,Arch,IS,true> {
    static constexpr bool value = sel::rank<ops::select,Key<T,N,Arch,IS>> > sel::GENERIC;
};

template<class T,int N,class Arch,int IS>
struct sel::Variant<ops::select,Key<T,N,Arch,IS>,sel::SPLIT> {
    using V = internal::SimdVecImpl<T,N,Arch>;
    using M = internal::SimdBoolImpl<N,IS,Arch>;
    static constexpr int n0 = internal::SplitOf<T,N,Arch>::n0;
    static constexpr int n1 = internal::SplitOf<T,N,Arch>::n1;
    static constexpr bool available = SelectHalfWorthIt<T,n0,Arch,IS>::value
        && internal::MaskSplits<N,IS,Arch>::value
        && MaskSplitSizes<N,IS,Arch>::n0 == n0;
    static V run( const M &m, const V &a, const V &b ) {
        V res;
        res.data.split.v0 = sel::call<ops::select,Key<T,n0,Arch,IS>>( m.data.split.v0, a.data.split.v0, b.data.split.v0 );
        res.data.split.v1 = sel::call<ops::select,Key<T,n1,Arch,IS>>( m.data.split.v1, a.data.split.v1, b.data.split.v1 );
        return res;
    }
};

// ---------------------------------------------------------------------------------------------
// fma -- three-operand, otherwise identical
// ---------------------------------------------------------------------------------------------
template<class T,int N,class Arch>
struct sel::Variant<ops::fma,Key<T,N,Arch>,sel::SPLIT> {
    using V = internal::SimdVecImpl<T,N,Arch>;
    static constexpr int n0 = internal::SplitOf<T,N,Arch>::n0;
    static constexpr int n1 = internal::SplitOf<T,N,Arch>::n1;
    static constexpr bool available = HalfIsWorthIt<ops::fma,T,n0,Arch>::value;
    static V run( const V &a, const V &b, const V &c ) {
        V res;
        res.data.split.v0 = sel::call<ops::fma,Key<T,n0,Arch>>( a.data.split.v0, b.data.split.v0, c.data.split.v0 );
        res.data.split.v1 = sel::call<ops::fma,Key<T,n1,Arch>>( a.data.split.v1, b.data.split.v1, c.data.split.v1 );
        return res;
    }
};

// ---------------------------------------------------------------------------------------------
// bcast_lane -- and this one is EXACT, where `permute` above is a compromise.
//
// A permutation has to move lanes between halves, which is why it costs two half-permutations
// and a blend. A broadcast does not: the lane it reads lives in exactly one half, and every
// lane of the RESULT is that same value. So the whole thing is one half-broadcast, stored
// twice -- no blend, no comparison, no index arithmetic.
//
// `LANE < n0` picks which half holds it, and the lane index has to be REBASED into that half
// (`LANE - n0` for the upper one). Getting that wrong is silent: it would broadcast a
// neighbouring lane, and every value would still look plausible.
//
// This was six of the nine generic cells left in the ARM grid -- `bcast_lane` had no split form
// at all, so on a target whose register never exceeds 128 bits it was a lane loop at every width
// above four. It buys the same thing on an x86 build without AVX2.
// ---------------------------------------------------------------------------------------------
template<int LANE,class T,int N,class Arch>
struct sel::Variant<ops::bcast_lane<LANE>,Key<T,N,Arch>,sel::SPLIT> {
    using V = internal::SimdVecImpl<T,N,Arch>;
    static constexpr int n0 = internal::SplitOf<T,N,Arch>::n0;
    static constexpr int n1 = internal::SplitOf<T,N,Arch>::n1;

    /// which half holds `LANE`, and its index there.
    static constexpr int nh = LANE < n0 ? n0 : n1;
    static constexpr int lh = LANE < n0 ? LANE : LANE - n0;

    /// the half must have a register form of its own, and `LANE` must be in range -- a
    /// `bcast_lane<9>` on eight lanes is the caller's problem, but it must not be turned into a
    /// hard error here by naming `bcast_lane<1>` on a zero-width half.
    static constexpr bool available =
        n0 > 0 && LANE >= 0 && LANE < N
        && HalfIsWorthIt<ops::bcast_lane<lh>,T,nh,Arch>::value;

    static V run( const V &v ) {
        V res;
        // ONE half-broadcast, whichever half `LANE` falls in; both halves of the result get it.
        if constexpr ( LANE < n0 ) {
            const auto h = sel::call<ops::bcast_lane<lh>,Key<T,n0,Arch>>( v.data.split.v0 );
            res.data.split.v0 = h;
            res.data.split.v1 = internal::spread_lane_0<T,n1,Arch>( h );
        } else {
            const auto h = sel::call<ops::bcast_lane<lh>,Key<T,n1,Arch>>( v.data.split.v1 );
            res.data.split.v0 = internal::spread_lane_0<T,n0,Arch>( h );
            res.data.split.v1 = h;
        }
        return res;
    }
};

// ---------------------------------------------------------------------------------------------
// permute -- the cross-half one
//
//   r_k = select( i_k < n0, permute( v0, i_k ), permute( v1, i_k - n0 ) )
//
// for each half k of the index. Both half-permutations are computed and one is thrown away: a
// permutation index that points into the other half is out of range for the half we hand it to,
// which is harmless because the sub-permute wraps it (`vpermps` keeps the low bits, the generic
// form takes a modulo) and the blend discards that lane anyway.
//
// ONLY WHEN THE TWO HALVES HAVE THE SAME WIDTH. At a width that is not a power of two the halves
// differ -- 5 splits into 4 + 1 -- and there is no type in which to express "permute a one-lane
// vector with a four-lane index". Those widths keep the generic form, which is correct.
// ---------------------------------------------------------------------------------------------
template<class T,int N,class Arch>
struct sel::Variant<ops::permute,Key<T,N,Arch>,sel::SPLIT> {
    using V  = internal::SimdVecImpl<T,N,Arch>;
    using I  = internal::SimdVecImpl<SI32,N,Arch>;
    static constexpr int n0 = internal::SplitOf<T,N,Arch>::n0;
    static constexpr int n1 = internal::SplitOf<T,N,Arch>::n1;
    using Vh = internal::SimdVecImpl<T,n0,Arch>;
    using Ih = internal::SimdVecImpl<SI32,n0,Arch>;
    static constexpr int is = ItemSizeOf<ops::cmp_lt,SI32,n0,Arch>::value;

    /// ALL THREE PIECES MUST BE REGISTER FORMS, not just the permutation.
    ///
    /// A split permutation is four half-permutations plus two blends. If the BLEND is a lane
    /// loop, those two dominate and the whole thing loses to the generic form -- which stores to
    /// the stack and reloads, and is at least linear. Measured under -mssse3, where `pshufb`
    /// gives a permutation at four lanes but `blendv` needs SSE4.1: the split form came out at
    /// 75 instructions against the generic form's 51, and was selected anyway because its rank
    /// was higher.
    ///
    /// A rank says "better if available". Deciding whether it IS better is what `available` is
    /// for, and for a composite form that means asking about every piece.
    static constexpr bool available =
        n0 > 0 && n0 == n1
        && HalfIsWorthIt<ops::permute,T,n0,Arch>::value
        && SelectHalfWorthIt<T,n0,Arch,is>::value;

    /// the index vector does not necessarily split the way the value vector does: on AVX2,
    /// `SimdVec<double,8>` is two registers while `SimdVec<SI32,8>` is one. So take the halves
    /// through `values` when there is no split to take them from.
    template<int OFF>
    static Ih idx_half( const I &idx ) {
        if constexpr ( internal::HasSplit<I> && I::split_size_0 == n0 ) {
            if constexpr ( OFF == 0 ) return idx.data.split.v0; else return idx.data.split.v1;
        } else {
            Ih r;
            for ( int i = 0; i < n0; ++i ) r.data.values[ i ] = idx.data.values[ OFF + i ];
            return r;
        }
    }

    static Vh splat_n0() { Ih r; internal::init_sc( r, SI32( n0 ) ); return r; }

    static Vh half( const Vh &v0, const Vh &v1, const Ih &ih ) {
        Ih lo; internal::init_sc( lo, SI32( n0 ) );
        const Vh from_0 = sel::call<ops::permute,Key<T,n0,Arch>>( v0, ih );
        const Vh from_1 = sel::call<ops::permute,Key<T,n0,Arch>>( v1, internal::sub( ih, lo ) );
        const auto m    = sel::call<ops::cmp_lt,Key<SI32,n0,Arch>>( ih, lo );
        return sel::call<ops::select,Key<T,n0,Arch,is>>( m, from_0, from_1 );
    }

    static V run( const V &v, const I &idx ) {
        V res;
        res.data.split.v0 = half( v.data.split.v0, v.data.split.v1, idx_half<0>( idx ) );
        res.data.split.v1 = half( v.data.split.v0, v.data.split.v1, idx_half<n0>( idx ) );
        return res;
    }
};

// ---------------------------------------------------------------------------------------------
// ext_lanes -- the concatenation `a:b` shifted by K lanes, across a split. With halves of equal
// width `h`, the K lanes that cross the boundary do so one half at a time:
//
//     K < h :  r0 = ext<K>( a0, a1 )     r1 = ext<K>( a1, b0 )
//     K = h :  r0 = a1                   r1 = b0                 -- a register rename, no work
//     K > h :  r0 = ext<K-h>( a1, b0 )   r1 = ext<K-h>( b0, b1 )
//
// Two half-`EXT`s for the whole, which on a 128-bit-only architecture is the ordinary case: a
// rotation of `float x 8` on NEON is two `EXT`, not a trip through memory.
//
// ONLY WHEN THE TWO HALVES HAVE THE SAME WIDTH, for the same reason as `permute` above.
// ---------------------------------------------------------------------------------------------

/// is `ext_lanes<K>` worth delegating to at width `h` -- reads as "yes" when K is 0, since
/// that case is a copy and names no variant at all.
template<int K,class T,int N,class Arch,bool = ( N > 0 && K > 0 )>
struct ExtHalfWorthIt { static constexpr bool value = N > 0; };
template<int K,class T,int N,class Arch>
struct ExtHalfWorthIt<K,T,N,Arch,true> {
    static constexpr bool value = sel::rank<ops::ext_lanes<K>,Key<T,N,Arch>> > sel::GENERIC;
};

template<int K,class T,int N,class Arch>
struct sel::Variant<ops::ext_lanes<K>,Key<T,N,Arch>,sel::SPLIT> {
    using V  = internal::SimdVecImpl<T,N,Arch>;
    static constexpr int h  = internal::SplitOf<T,N,Arch>::n0;
    static constexpr int n1 = internal::SplitOf<T,N,Arch>::n1;
    static constexpr int k  = K < h ? K : K - h;   ///< the shift once rebased into a half
    using Vh = internal::SimdVecImpl<T,h,Arch>;

    static constexpr bool available = h > 0 && h == n1 && ExtHalfWorthIt<k,T,h,Arch>::value;

    static Vh ext( const Vh &x, const Vh &y ) {
        if constexpr ( k == 0 ) return x;
        else return sel::call<ops::ext_lanes<k>,Key<T,h,Arch>>( x, y );
    }

    static V run( const V &a, const V &b ) {
        V res;
        if constexpr ( K < h ) {
            res.data.split.v0 = ext( a.data.split.v0, a.data.split.v1 );
            res.data.split.v1 = ext( a.data.split.v1, b.data.split.v0 );
        } else {
            res.data.split.v0 = ext( a.data.split.v1, b.data.split.v0 );
            res.data.split.v1 = ext( b.data.split.v0, b.data.split.v1 );
        }
        return res;
    }
};

// ---------------------------------------------------------------------------------------------
// rotate_lanes, across a split. Two cases have a cheap answer:
//
//   the rotated prefix fits in the first half (`n <= n0`): rotate that half, copy the other.
//   the whole width rotates (`n == N`, equal halves): `rotate<K>( v ) == ext<K>( v, v )`, i.e.
//     the `ext_lanes` split above with `a = b = v`.
//
// A prefix that straddles the boundary without covering the whole (`n0 < n < N`) has no split
// form and stays generic: it would take a permutation, and the generic form IS one.
// ---------------------------------------------------------------------------------------------
template<int K,int n,class T,int N,class Arch>
struct sel::Variant<ops::rotate_lanes<K,n>,Key<T,N,Arch>,sel::SPLIT> {
    using V  = internal::SimdVecImpl<T,N,Arch>;
    static constexpr int n0 = internal::SplitOf<T,N,Arch>::n0;
    static constexpr int n1 = internal::SplitOf<T,N,Arch>::n1;
    static constexpr int k  = K < n0 ? K : K - n0;

    static constexpr bool in_first_half = n0 > 0 && n <= n0;
    static constexpr bool whole         = n0 > 0 && n == N && n0 == n1;

    static constexpr bool available =
        ( in_first_half && HalfIsWorthIt<ops::rotate_lanes<K,n>,T,n0,Arch>::value )
        || ( whole && ExtHalfWorthIt<k,T,n0,Arch>::value );

    using Vh = internal::SimdVecImpl<T,n0,Arch>;
    static Vh ext( const Vh &x, const Vh &y ) {
        if constexpr ( k == 0 ) return x;
        else return sel::call<ops::ext_lanes<k>,Key<T,n0,Arch>>( x, y );
    }

    static V run( const V &v ) {
        V res;
        if constexpr ( in_first_half ) {
            res.data.split.v0 = sel::call<ops::rotate_lanes<K,n>,Key<T,n0,Arch>>( v.data.split.v0 );
            res.data.split.v1 = v.data.split.v1;
        } else if constexpr ( K < n0 ) {
            res.data.split.v0 = ext( v.data.split.v0, v.data.split.v1 );
            res.data.split.v1 = ext( v.data.split.v1, v.data.split.v0 );
        } else {
            res.data.split.v0 = ext( v.data.split.v1, v.data.split.v0 );
            res.data.split.v1 = ext( v.data.split.v0, v.data.split.v1 );
        }
        return res;
    }
};

} // namespace asimd

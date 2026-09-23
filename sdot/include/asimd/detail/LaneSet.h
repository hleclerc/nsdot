#pragma once

// =============================================================================================
// LANE SETS -- "these are the lanes whose result I will read".
//
// A `SimdVec<float,8>` is two registers on NEON, four for sixteen lanes. When the caller only
// needs the first three lanes, the second register is work for nothing. A lane set, passed as a
// trailing argument (`add( a, b, LaneRange<0,3>() )`), says so.
//
// THE CONTRACT IS "DON'T CARE", NOT "MERGE". Lanes outside the set hold an UNSPECIFIED value in
// the result -- typically whatever was in the register or on the stack before. That is the
// whole point: it is what allows a half to be skipped rather than computed and blended. For
// "keep the other lanes" there is `select`, which costs a blend and saves nothing.
//
// Because outside lanes are unspecified, every strategy is correct, including computing all of
// them. So the library is free to choose, and it chooses:
//
//   - STATIC bounds (`LaneRange<0,3>`): a half of a split whose lanes all lie outside the set is
//     skipped at compile time, recursively. Inside one register, nothing changes -- an
//     instruction computes its whole register at the same price.
//   - DYNAMIC bounds (`LaneRange( 0, n )`, `LaneMask( bits )`): a half is skipped behind a
//     branch, but only when that half is itself made of several registers or is a lane loop;
//     never in front of a single instruction, where the branch would cost what it saves. The
//     generic lane loops are bounded by the set. This is the case that matters when the
//     operation has no register form at all.
//   - REDUCTIONS (`sum`, `to_bits`, `any`, `all`): the set is part of the answer, so lanes
//     outside it are excluded, not ignored: masked before the reduce, at the finest register
//     that still intersects the set.
//
// Two shapes:
//
//   `LaneRange<beg,end>`   an interval. Each bound is a compile-time constant or `dyn`, in which
//                          case it is a member: `LaneRange<0,3>()`, `LaneRange<0>( n )`,
//                          `LaneRange( b, e )`.
//   `LaneMask<beg,end>`    one bit per lane, dynamic, with a STATIC promise that every set bit
//                          lies in `[beg,end)` (`[0,everything)` by default). The promise is
//                          what lets halves be skipped at compile time even though the bits are
//                          not known; the bits are what the generic loops and the reductions
//                          use. It is an integer, the same one `to_bits` produces.
//
// Not to be confused with `SimdBool`, which is a VECTOR of truth values and the result of a
// comparison. A `LaneMask` masks operations; a `SimdBool` is data.
// =============================================================================================

#include "impl/SimdVecImpl_Generic.h"
#include "impl/SimdBoolImpl_Generic.h"
#include "support/common_types.h"

#include <array>
#include <bit>
#include <type_traits>
#include <utility>

namespace asimd {

/// a bound that is not known at compile time
inline constexpr int dyn = -1;

namespace internal {
    constexpr int max0( int a ) { return a > 0 ? a : 0; }
    constexpr int sub_or_dyn( int a, int off ) { return a == dyn ? dyn : max0( a - off ); }
    /// the low `n` bits, for n in [0,64]
    constexpr PI64 low_bits( int n ) { return n >= 64 ? ~PI64( 0 ) : ( PI64( 1 ) << n ) - 1; }
}

// ---------------------------------------------------------------------------------------------
// LaneRange
// ---------------------------------------------------------------------------------------------
template<int BEG = dyn,int END = dyn> struct LaneRange;

/// both bounds known
template<int BEG,int END> requires ( BEG >= 0 && END >= 0 )
struct LaneRange<BEG,END> {
    static constexpr int  hull_beg  = BEG;     ///< every lane of the set is at or after this
    static constexpr int  hull_end  = END;     ///< ... and before this (`dyn` = unknown)
    static constexpr bool is_static = true;    ///< no runtime information at all

    constexpr int  beg  () const { return BEG; }
    constexpr int  end  () const { return END; }
    constexpr bool has  ( int i ) const { return i >= BEG && i < END; }
    constexpr bool empty( int lo, int hi ) const { return END <= lo || BEG >= hi; }
    /// the same set seen from lane `OFF` (for the upper half of a split)
    template<int OFF> constexpr auto shifted() const { return LaneRange<internal::max0( BEG - OFF ),internal::max0( END - OFF )>(); }
    /// the lanes of `[0,n)` in the set, as bits
    constexpr PI64 bits( int n ) const { return internal::low_bits( END < n ? END : n ) & ~internal::low_bits( BEG ); }
};

/// the beginning known, the end not -- the loop tail
template<int BEG> requires ( BEG >= 0 )
struct LaneRange<BEG,dyn> {
    static constexpr int  hull_beg  = BEG;
    static constexpr int  hull_end  = dyn;
    static constexpr bool is_static = false;

    constexpr LaneRange( int end ) : end_( end ) {}

    constexpr int  beg  () const { return BEG; }
    constexpr int  end  () const { return end_; }
    constexpr bool has  ( int i ) const { return i >= BEG && i < end_; }
    constexpr bool empty( int lo, int hi ) const { return end_ <= lo || BEG >= hi; }
    template<int OFF> constexpr auto shifted() const { return LaneRange<internal::max0( BEG - OFF ),dyn>( internal::max0( end_ - OFF ) ); }
    constexpr PI64 bits( int n ) const { return internal::low_bits( end_ < n ? end_ : n ) & ~internal::low_bits( BEG ); }

    int end_;
};

/// nothing known
template<>
struct LaneRange<dyn,dyn> {
    static constexpr int  hull_beg  = 0;
    static constexpr int  hull_end  = dyn;
    static constexpr bool is_static = false;

    constexpr LaneRange( int beg, int end ) : beg_( beg ), end_( end ) {}

    constexpr int  beg  () const { return beg_; }
    constexpr int  end  () const { return end_; }
    constexpr bool has  ( int i ) const { return i >= beg_ && i < end_; }
    constexpr bool empty( int lo, int hi ) const { return end_ <= lo || beg_ >= hi; }
    template<int OFF> constexpr auto shifted() const { return LaneRange<dyn,dyn>( internal::max0( beg_ - OFF ), internal::max0( end_ - OFF ) ); }
    constexpr PI64 bits( int n ) const { return internal::low_bits( end_ < n ? end_ : n ) & ~internal::low_bits( beg_ ); }

    int beg_, end_;
};

LaneRange( int, int ) -> LaneRange<dyn,dyn>;

// ---------------------------------------------------------------------------------------------
// LaneMask
// ---------------------------------------------------------------------------------------------
template<int BEG = 0,int END = dyn>
struct LaneMask {
    static_assert( BEG >= 0 && ( END == dyn || END >= BEG ), "LaneMask: the hull must be a range" );
    static constexpr int  hull_beg  = BEG;
    static constexpr int  hull_end  = END;
    static constexpr bool is_static = false;

    constexpr LaneMask( PI64 bits ) : bits_( bits ) {}

    constexpr bool has  ( int i ) const { return ( bits_ >> i ) & 1; }
    constexpr bool empty( int lo, int hi ) const { return ( ( bits_ >> lo ) & internal::low_bits( hi - lo ) ) == 0; }
    template<int OFF> constexpr auto shifted() const { return LaneMask<internal::max0( BEG - OFF ),internal::sub_or_dyn( END, OFF )>( bits_ >> OFF ); }
    constexpr PI64 bits( int n ) const { return bits_ & internal::low_bits( n ); }

    PI64 bits_;
};

/// what the pruning needs from a set, and nothing more
template<class S>
concept LaneSet = requires ( const S &s, int i ) {
    { S::hull_beg } -> std::convertible_to<int>;
    { S::hull_end } -> std::convertible_to<int>;
    { S::is_static } -> std::convertible_to<bool>;
    { s.has( i ) } -> std::convertible_to<bool>;
    { s.empty( i, i ) } -> std::convertible_to<bool>;
    { s.template shifted<1>() };
    { s.bits( i ) } -> std::convertible_to<PI64>;
};

namespace internal {

/// is `[lo,hi)` KNOWN to hold no lane of the set
template<LaneSet S,int lo,int hi>
inline constexpr bool hull_disjoint = ( S::hull_end != dyn && S::hull_end <= lo ) || S::hull_beg >= hi;

/// is `[lo,hi)` KNOWN to be entirely in the set -- only a static range can promise that
template<LaneSet S,int lo,int hi>
inline constexpr bool hull_covers = S::is_static && S::hull_beg <= lo && S::hull_end != dyn && S::hull_end >= hi;

/// the width of an impl, vector or boolean
template<class I> struct WidthOf;
template<class T,int N,class Arch>  struct WidthOf<SimdVecImpl<T,N,Arch>>   { static constexpr int value = N; };
template<int N,int IS,class Arch>   struct WidthOf<SimdBoolImpl<N,IS,Arch>> { static constexpr int value = N; };
template<class I> inline constexpr int width_of = WidthOf<std::remove_cvref_t<I>>::value;

/// the item size of a boolean impl, 0 for a vector
template<class I> struct ItemSizeOfImpl { static constexpr int value = 0; };
template<int N,int IS,class Arch> struct ItemSizeOfImpl<SimdBoolImpl<N,IS,Arch>> { static constexpr int value = IS; };
template<class I> inline constexpr int item_size_of = ItemSizeOfImpl<std::remove_cvref_t<I>>::value;

/// a lane of all ones, in T's own bits
template<class T> constexpr T all_ones_lane() {
    using U = typename PI_<8 * sizeof( T )>::T;
    return std::bit_cast<T>( U( ~U( 0 ) ) );
}

/// all ones on the lanes of a STATIC set, zero elsewhere -- what a bitwise and keeps, and what
/// `vmaskmov` reads. A bitwise and rather than a multiply by 0/1: the lanes outside the set may
/// hold anything, a NaN included, and `NaN * 0` is `NaN`.
template<LaneSet S,class T,int N>
struct LanePattern {
    static constexpr std::array<T,N> make() {
        std::array<T,N> r{};
        for ( int i = 0; i < N; ++i ) r[ i ] = S().has( i ) ? all_ones_lane<T>() : T( 0 );
        return r;
    }
    alignas( 64 ) static constexpr std::array<T,N> v = make();
};

/// PARTIAL LOAD AND STORE: the lanes of the set, and NOT ONE BYTE OUTSIDE THEM -- this is the
/// operation for the tail of a buffer, where the lane past the end is another page. A loaded
/// lane outside the set is unspecified; a stored one is left alone. Declared here so that
/// `SimdVec` can name them; defined in `SimdOps.h`, where the variants are.
template<LaneSet Set,class T,int N,class Arch>
SimdVecImpl<T,N,Arch> load_partial( const T *ptr, const Set &set, S<SimdVecImpl<T,N,Arch>> );
template<LaneSet Set,class T,int N,class Arch>
void store_partial( T *ptr, const SimdVecImpl<T,N,Arch> &v, const Set &set );

/// the halves of an impl, as types
template<class I> using Half0 = std::remove_cvref_t<decltype( std::declval<I>().data.split.v0 )>;
template<class I> using Half1 = std::remove_cvref_t<decltype( std::declval<I>().data.split.v1 )>;

/// CAN `f` BE APPLIED HALF BY HALF? Every operand and the result must split at the same width,
/// and `f` on the halves must yield exactly the half of the result -- a comparison at half
/// width may come back in another mask flavour than at full width, in which case it cannot be
/// assembled and the operation runs whole.
///
/// A function rather than a variable template: naming `A::split_size_0` on an impl that has no
/// split is a hard error, not a false, so each question has to wait for the previous answer.
template<class F,class R,class... A>
constexpr bool splits_alike() {
    if constexpr ( ( HasSplit<A> && ... ) && HasSplit<R> ) {
        if constexpr ( ( ( A::split_size_0 == R::split_size_0 ) && ... ) )
            return std::is_same_v<std::invoke_result_t<F,const Half0<A>&...>,Half0<R>>
                && std::is_same_v<std::invoke_result_t<F,const Half1<A>&...>,Half1<R>>;
        else
            return false;
    } else {
        return false;
    }
}

/// THE PRUNING. `f` is a lane-wise operation on impls; `a...` its operands. The result's lanes
/// outside `set` are unspecified.
template<LaneSet S,class F,class... A>
auto prune( const S &set, const F &f, const A &...a ) {
    using R = std::invoke_result_t<F,const A&...>;
    if constexpr ( splits_alike<F,R,A...>() && ! hull_covers<S,0,width_of<R>> ) {
        constexpr int n0 = R::split_size_0, N = n0 + R::split_size_1;
        R res;
        if constexpr ( hull_disjoint<S,n0,N> ) {
            res.data.split.v0 = prune( set, f, a.data.split.v0... );
        } else if constexpr ( hull_disjoint<S,0,n0> ) {
            res.data.split.v1 = prune( set.template shifted<n0>(), f, a.data.split.v1... );
        } else {
            // both halves may be wanted. A dynamic set gets a BRANCH here, but only in front of
            // real work: a half that is itself several registers, or a lane loop -- both of
            // which are what a split half looks like. In front of a single instruction the
            // branch would cost what it saves, so the half is computed.
            if constexpr ( ! S::is_static && ( HasSplit<Half0<R>> || HasSplit<Half1<R>> ) ) {
                if ( set.empty( n0, N ) ) { res.data.split.v0 = prune( set, f, a.data.split.v0... ); return res; }
                if ( set.empty( 0, n0 ) ) { res.data.split.v1 = prune( set.template shifted<n0>(), f, a.data.split.v1... ); return res; }
            }
            // ... and each half gets the set seen from its own lane 0, so that the pruning goes
            // on below: `[0,9)` on four registers is two of them, then one, not four.
            res.data.split.v0 = prune( set, f, a.data.split.v0... );
            res.data.split.v1 = prune( set.template shifted<n0>(), f, a.data.split.v1... );
        }
        return res;
    } else {
        return f( a... );
    }
}

} // namespace internal
} // namespace asimd

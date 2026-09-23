#pragma once

// =============================================================================================
// THE GENERIC VARIANTS -- rank 0, one per operation, always available. This is what guarantees
// that every operation has an answer at every (type, width, architecture), whatever the backends
// register or fail to register above it.
//
// A generic form is correct first. Where it can be built from operations that already have
// register forms (`fma` from `mul` and `add`) it is; otherwise it walks `values` lane by lane.
// =============================================================================================

#include "Key.h"
#include "../LaneSet.h"

namespace asimd {

// =====================================================================================
// GENERIC VARIANTS -- rank 0. Always available: this is what guarantees a fallback.
// =====================================================================================

/// THROUGH `mul` AND `add`, not through a lane loop. There is no fused multiply-add on any x86
/// integer type, so this generic form is what every `fma` on an integer lands on -- and written
/// as a loop over `values` it was a scalar loop unless the compiler happened to offer vector
/// arithmetic. `mul` and `add` have register forms; use them, and the fallback inherits them.
///
/// It costs nothing on floating point: where a real `vfmadd` exists the REGISTER variant wins,
/// and where it does not, a multiply followed by an add is exactly what the hardware would do.
template<class T,int N,class Arch>
struct sel::Variant<ops::fma,Key<T,N,Arch>,sel::GENERIC> {
    static constexpr bool available = true;
    using V = internal::SimdVecImpl<T,N,Arch>;
    static V run( const V &a, const V &b, const V &c ) {
        return internal::add( internal::mul( a, b ), c );
    }
};

template<class T,int N,class Arch>
struct sel::Variant<ops::permute,Key<T,N,Arch>,sel::GENERIC> {
    static constexpr bool available = true;
    using V = internal::SimdVecImpl<T,N,Arch>;
    using I = internal::SimdVecImpl<SI32,N,Arch>;
    static V run( const V &v, const I &idx ) {
        T tmp[ N ];
        for ( int i = 0; i < N; ++i ) tmp[ i ] = v.data.values[ i ];
        V res;
        // A MODULO, AND AN UNSIGNED ONE.
        //
        // `% N` and not `& ( N - 1 )`, because the mask is a modulo only at a power of two and
        // arbitrary widths are what asimd is for: at N = 5 the mask was `& 4`, so reversing five
        // lanes returned `50 10 10 10 10`.
        //
        // UNSIGNED, because that is what the hardware does. `vpermps` keeps the low three bits of
        // each index, so index -1 selects lane 7 -- and ARM's `TBL` control is masked the same
        // way, so the two backends agree without either being told about the other. The signed
        // `j < 0 ? j + N : j` written here sent it to lane 7 as well at N = 8 -- by coincidence
        // -- but to a different lane at any other width, so the generic and register forms
        // disagreed. Reading the index as unsigned makes them agree at every power of two, and is
        // one instruction cheaper: at a constant power-of-two N the compiler turns `% N` back
        // into the `and`.
        for ( int i = 0; i < N; ++i )
            res.data.values[ i ] = tmp[ PI32( idx.data.values[ i ] ) % PI32( N ) ];
        return res;
    }
};

template<int LANE,class T,int N,class Arch>
struct sel::Variant<ops::bcast_lane<LANE>,Key<T,N,Arch>,sel::GENERIC> {
    static constexpr bool available = true;
    using V = internal::SimdVecImpl<T,N,Arch>;
    static V run( const V &v ) {
        V res;
        for ( int i = 0; i < N; ++i ) res.data.values[ i ] = v.data.values[ LANE ];
        return res;
    }
};

/// through a copy, because `values` may alias itself in a register impl's union.
template<int K,int n,class T,int N,class Arch>
struct sel::Variant<ops::rotate_lanes<K,n>,Key<T,N,Arch>,sel::GENERIC> {
    static constexpr bool available = true;
    using V = internal::SimdVecImpl<T,N,Arch>;
    static V run( const V &v ) {
        T tmp[ N ];
        for ( int i = 0; i < N; ++i ) tmp[ i ] = v.data.values[ i ];
        V res;
        for ( int i = 0; i < N; ++i ) res.data.values[ i ] = tmp[ i < n ? ( i + K ) % n : i ];
        return res;
    }
};

template<int K,class T,int N,class Arch>
struct sel::Variant<ops::ext_lanes<K>,Key<T,N,Arch>,sel::GENERIC> {
    static constexpr bool available = true;
    using V = internal::SimdVecImpl<T,N,Arch>;
    static V run( const V &a, const V &b ) {
        V res;
        for ( int i = 0; i < N; ++i )
            res.data.values[ i ] = i + K < N ? a.data.values[ i + K ] : b.data.values[ i + K - N ];
        return res;
    }
};

/// Across a split: recurse, skipping a half the set does not reach -- statically, or behind a
/// branch, since the alternative at the register is a lane loop, not an instruction. On one
/// register: the plain load when the set is known to cover it, lane by lane otherwise. The lane
/// loop is what makes the contract hold everywhere: it reads exactly the lanes of the set.
template<class T,int N,class Arch>
struct sel::Variant<ops::load_partial,Key<T,N,Arch>,sel::GENERIC> {
    static constexpr bool available = true;
    using V = internal::SimdVecImpl<T,N,Arch>;
    template<LaneSet S>
    static V run( const T *ptr, const S &set ) {
        V res;
        if constexpr ( internal::hull_covers<S,0,N> ) {
            res = internal::load_unaligned( ptr, asimd::S<V>() );
        } else if constexpr ( internal::HasSplit<V> ) {
            constexpr int n0 = V::split_size_0;
            const auto lo = [ & ] { res.data.split.v0 = sel::call<ops::load_partial,Key<T,n0,Arch>>( ptr, set ); };
            const auto hi = [ & ] { res.data.split.v1 = sel::call<ops::load_partial,Key<T,N - n0,Arch>>( ptr + n0, set.template shifted<n0>() ); };
            if constexpr ( internal::hull_disjoint<S,n0,N> ) lo();
            else if constexpr ( internal::hull_disjoint<S,0,n0> ) hi();
            else if constexpr ( S::is_static ) { lo(); hi(); }
            else {
                if ( ! set.empty( 0, n0 ) ) lo();
                if ( ! set.empty( n0, N ) ) hi();
            }
        } else {
            for ( int i = 0; i < N; ++i )
                if ( set.has( i ) ) res.data.values[ i ] = ptr[ i ];
        }
        return res;
    }
};

template<class T,int N,class Arch>
struct sel::Variant<ops::store_partial,Key<T,N,Arch>,sel::GENERIC> {
    static constexpr bool available = true;
    using V = internal::SimdVecImpl<T,N,Arch>;
    template<LaneSet S>
    static void run( T *ptr, const V &v, const S &set ) {
        if constexpr ( internal::hull_covers<S,0,N> ) {
            internal::store_unaligned( ptr, v );
        } else if constexpr ( internal::HasSplit<V> ) {
            constexpr int n0 = V::split_size_0;
            const auto lo = [ & ] { sel::call<ops::store_partial,Key<T,n0,Arch>>( ptr, v.data.split.v0, set ); };
            const auto hi = [ & ] { sel::call<ops::store_partial,Key<T,N - n0,Arch>>( ptr + n0, v.data.split.v1, set.template shifted<n0>() ); };
            if constexpr ( internal::hull_disjoint<S,n0,N> ) lo();
            else if constexpr ( internal::hull_disjoint<S,0,n0> ) hi();
            else if constexpr ( S::is_static ) { lo(); hi(); }
            else {
                if ( ! set.empty( 0, n0 ) ) lo();
                if ( ! set.empty( n0, N ) ) hi();
            }
        } else {
            for ( int i = 0; i < N; ++i )
                if ( set.has( i ) ) ptr[ i ] = T( internal::at( v, i ) );
        }
    }
};

template<class T,int N,class Arch,int IS>
struct sel::Variant<ops::select,Key<T,N,Arch,IS>,sel::GENERIC> {
    static constexpr bool available = true;
    using V = internal::SimdVecImpl<T,N,Arch>;
    using M = internal::SimdBoolImpl<N,IS,Arch>;
    static V run( const M &m, const V &a, const V &b ) {
        V res;
        for ( int i = 0; i < N; ++i )
            res.data.values[ i ] = m.data.values[ i ] ? a.data.values[ i ] : b.data.values[ i ];
        return res;
    }
};

/// `to_bits` returns a PI64, not an `unsigned`. A mask can be 64 lanes wide (`SI8` on
/// AVX-512BW), and `1u << i` is undefined for i >= 32 -- so the generic form was undefined
/// behaviour on exactly the widths that need it most.
template<int N,class Arch,int IS>
struct sel::Variant<ops::to_bits,Key<void,N,Arch,IS>,sel::GENERIC> {
    static constexpr bool available = true;
    using M = internal::SimdBoolImpl<N,IS,Arch>;
    static PI64 run( const M &m ) {
        PI64 res = 0;
        for ( int i = 0; i < N; ++i )
            res |= PI64( bool( m.data.values[ i ] ) ) << i;
        return res;
    }
};

template<int N,class Arch>
struct sel::Variant<ops::mask_from_bits,Key<void,N,Arch>,sel::GENERIC> {
    static constexpr bool available = true;
    static internal::SimdBoolImpl<N,32,Arch> run( PI64 b ) {
        internal::SimdBoolImpl<N,32,Arch> res;
        for ( int i = 0; i < N; ++i )
            res.data.values[ i ] = ( b >> i ) & 1 ? ~PI32( 0 ) : PI32( 0 );
        return res;
    }
};

/// The four comparisons. asimd only had `lt` and `gt`, and only as lazy expressions. `eq` is
/// needed to DESIGNATE A LANE (`iota == i` is the portable way of saying "lane i", where
/// intrinsics take an immediate mask) and `ge` to bring indices back into range.
#define ASIMD_OPS_GENERIC_CMP( TAG, OP )                                                       \
    template<class T,int N,class Arch>                                                          \
    struct sel::Variant<ops::TAG,Key<T,N,Arch>,sel::GENERIC> {                                  \
        static constexpr bool available = true;                                                 \
        using V = internal::SimdVecImpl<T,N,Arch>;                                              \
        static internal::SimdBoolImpl<N,32,Arch> run( const V &a, const V &b ) {                \
            internal::SimdBoolImpl<N,32,Arch> res;                                              \
            for ( int i = 0; i < N; ++i )                                                       \
                res.data.values[ i ] = ( a.data.values[ i ] OP b.data.values[ i ] ) ? ~PI32( 0 ) \
                                                                                    : PI32( 0 ); \
            return res;                                                                         \
        }                                                                                       \
    }

ASIMD_OPS_GENERIC_CMP( cmp_gt, >  );
ASIMD_OPS_GENERIC_CMP( cmp_lt, <  );
ASIMD_OPS_GENERIC_CMP( cmp_eq, == );
ASIMD_OPS_GENERIC_CMP( cmp_ge, >= );

#undef ASIMD_OPS_GENERIC_CMP

} // namespace asimd

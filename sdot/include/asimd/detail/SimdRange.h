#pragma once

#include "SimdVec.h"
#include "support/N.h"
#include "support/Int.h"

#include <algorithm>
#include <utility>

namespace asimd {

/**
  next_size = size/2 => use of smaller SIMD instruction at the end before size 1 (might be faster in some cases, but needing more instructions)
*/
template<int size,int next_size=1>
struct SimdRange {
    /// func( TI index, N<simd_size> )
    /// Version with beg defined with known alignment + offset
    template<class TB,class TE,int a,int o,class Func>
    static void for_each( Int<TB,a,o> beg, TE end, Func &&func ) {
        // first aligned index
        Int<TB,size,0> cur;

        // if possible to have `beg` non aligned to `size`, go to an aligned one
        if constexpr ( a % size || o % size ) {
            if ( TB mod = a % size ? beg.get() % size : o % size ) {
                TB nxt = std::min( beg.get() + size - mod, end );
                SimdRange<next_size>::for_each( beg, nxt, std::forward<Func>( func ) );
                cur.set( nxt );
            } else
                cur.set( beg.get() );
        } else
            cur.set( beg.get() );

        // aligned loop
        for( Int<TB,size,0> nxt; ; cur = nxt ) {
            nxt = cur + N<size>();
            if ( nxt > end )
                return SimdRange<next_size>::for_each( cur, end, std::forward<Func>( func ) );

            func( cur, N<size>() );
        }
    }

    /// Version with no proberty on beg
    template<class TB,class TE,class Func>
    static void for_each( TB beg, TE end, Func &&func ) {
        return for_each( Int<TB,1,0>( beg ), end, std::forward<Func>( func ) );
    }

    /// Version which starts at 0
    template<class TE,class Func>
    static void for_each( TE end, Func &&func ) {
        return for_each( Int<TE,size,0>( 0 ), end, std::forward<Func>( func ) );
    }
};

//
template<int next_size>
struct SimdRange<1,next_size> {
    template<class TB,int a,int o,class TE,class Func>
    static void for_each( Int<TB,a,o> beg, TE end, Func &&func ) {
        for( Int<TB,1,0> cur = beg.get(); cur < end; cur.al_incr() )
            func( cur, N<1>() );
    }

    template<class TB,class TE,class Func>
    static void for_each( TB beg, TE end, Func &&func ) {
        return for_each( Int<TB,1,0>( beg ), end, std::forward<Func>( func ) );
    }

    /// Version which starts at 0
    template<class TE,class Func>
    static void for_each( TE end, Func &&func ) {
        return for_each( Int<TE,1,0>( 0 ), end, std::forward<Func>( func ) );
    }
};

} // namespace asimd

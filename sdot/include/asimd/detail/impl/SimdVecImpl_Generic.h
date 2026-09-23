#pragma once

#include "SimdBoolImpl_Generic.h"
#include "ASIMD_DEBUG_ON_OP.h"
#include "../support/common_types.h"
#include "../support/VecValues.h"
#include "../support/S.h"

#include <type_traits>
#include <algorithm>
#include <bit>

namespace asimd {
template<class T,int size,class Arch> struct SimdVec;

namespace internal {

// SimdVecImpl ---------------------------------------------------------
/// Splittable version
template<class T,int size,class Arch>
struct SimdVecImpl {
    static constexpr int split_size_0 = prev_pow_2( size );
    static constexpr int split_size_1 = size - split_size_0;
    struct Split {
        SimdVecImpl<T,split_size_0,Arch> v0;
        SimdVecImpl<T,split_size_1,Arch> v1;
    };
    union {
        T     values[ size ];
        Split split;
    } data;
};

/// Atomic version
template<class T,class Arch>
struct SimdVecImpl<T,1,Arch> {
    union {
        T values[ 1 ];
    } data;
};

/// DOES THIS IMPL HAVE A SPLIT VIEW? A splittable impl is a pair of halves and the generic forms
/// recurse into them; a REGISTER impl deliberately has no `split` at all -- putting one in the
/// union sends every vector through memory (see the comment on SIMD_VEC_IMPL_REG just below).
///
/// So the generic forms cannot simply recurse: at a register-backed width they must go through
/// `values` instead. Before this trait existed they recursed unconditionally, which made every
/// operation WITHOUT a register form a hard compile error rather than a slow path -- `.sum()`,
/// `mul` and `div` on every integer type, the strided `iota`, `gather` and `scatter` below AVX2.
template<class I>
concept HasSplit = requires ( I i ) { i.data.split.v0; };

/// ... and does it have a whole-vector `values` (the `vector_size` typedef of a register impl)
/// that arithmetic can be applied to in one go, rather than a plain array?
template<class I>
concept HasVectorValues = requires ( I a, I b ) { { a.data.values + b.data.values }; };

/// Helper to make Impl with a register
// THE LAYOUT OF A REGISTER IMPL DECIDES THE ABI, hence the performance.
//
// The obvious form -- `union { T values[ SIZE ]; Split split; TREG reg; }` -- sends EVERY vector
// through MEMORY as soon as it crosses a call. The SysV x86-64 rule: an aggregate larger than two
// eightbytes is passed in registers only if the first eightbyte is SSE and ALL the following ones
// are SSEUP. But `float[ 8 ]` classifies as SSE,SSE,SSE,SSE -- an array is not a vector -- and a
// union takes the WORST class among its members. A `struct { __m128 v0, v1; }` is no better:
// SSE,SSEUP,SSE,SSEUP, and the third eightbyte alone sends the whole thing back to memory.
//
// MEASURED, on a plain addition passed by value:
//     bare `__m256`                            4 instructions, 0 memory accesses
//     `union { vector; TREG; }`                4 instructions, 0
//     `union { T[ SIZE ]; TREG; }`             8 instructions, 4
//     `union { ...; Split; TREG; }`            8 instructions, 4
//
// Hence this form: `values` typed as a VECTOR (gcc/clang's `vector_size` extension, which keeps
// `values[ i ]` indexing) and `Split` TAKEN OUT of the union. Operations with no register form
// then go lane by lane through `values` rather than through the split.
//
// `tests/xmake.lua` checks this on every build -- a `static_assert` can do nothing about an ABI,
// and the regression changes no result, only the speed, by a factor of two.
#define SIMD_VEC_IMPL_REG( COND, T, SIZE, TREG ) \
    template<class Arch> requires ( Arch::template Has<features::COND>::value ) \
    struct SimdVecImpl<T,SIZE,Arch> { \
        static constexpr int split_size_0 = prev_pow_2( SIZE ); \
        static constexpr int split_size_1 = SIZE - split_size_0; \
        ASIMD_VALUES_TYPE( Values, T, SIZE ); \
        union { \
            Values values; \
            TREG   reg; \
        } data; \
    }

// at ------------------------------------------------------------------------
//
// BY VALUE, NOT BY REFERENCE, and that is forced. When `values` is a `vector_size` type there is
// no object to bind a reference to: clang rejects `T &r = v.data.values[ i ]` outright ("non-const
// reference cannot bind to vector element") where gcc allows it as an extension. So reading a
// lane returns a copy -- it is one scalar -- and writing one goes through `set_at`.
template<class T,int size,class Arch> HaD
T at( const SimdVecImpl<T,size,Arch> &vec, int i ) {
    return vec.data.values[ i ];
}

template<class T,int size,class Arch> HaD
void set_at( SimdVecImpl<T,size,Arch> &vec, int i, T value ) {
    vec.data.values[ i ] = value;
}

/// The address of lane 0, for `begin()` / `end()`. Same reason: `&vec.data.values[ 0 ]` is not
/// something you may write on a vector type, and the union is contiguous anyway.
template<class T,int size,class Arch> HaD
const T *lane_ptr( const SimdVecImpl<T,size,Arch> &vec ) {
    return reinterpret_cast<const T *>( &vec.data );
}

// init ----------------------------------------------------------------------
template<class T,int size,class Arch,class G> HaD
void init_sc( SimdVecImpl<T,size,Arch> &vec, G a, G b, G c, G d, G e, G f, G g, G h ) {
    vec.data.values[ 0 ] = a;
    vec.data.values[ 1 ] = b;
    vec.data.values[ 2 ] = c;
    vec.data.values[ 3 ] = d;
    vec.data.values[ 4 ] = e;
    vec.data.values[ 5 ] = f;
    vec.data.values[ 6 ] = g;
    vec.data.values[ 7 ] = h;
}

template<class T,int size,class Arch,class G> HaD
void init_sc( SimdVecImpl<T,size,Arch> &vec, G a, G b, G c, G d, G e ) {
    vec.data.values[ 0 ] = a;
    vec.data.values[ 1 ] = b;
    vec.data.values[ 2 ] = c;
    vec.data.values[ 3 ] = d;
    vec.data.values[ 4 ] = e;
}

template<class T,int size,class Arch,class G> HaD
void init_sc( SimdVecImpl<T,size,Arch> &vec, G a, G b, G c, G d ) {
    vec.data.values[ 0 ] = a;
    vec.data.values[ 1 ] = b;
    vec.data.values[ 2 ] = c;
    vec.data.values[ 3 ] = d;
}

template<class T,int size,class Arch,class G> HaD
void init_sc( SimdVecImpl<T,size,Arch> &vec, G a, G b ) {
    vec.data.values[ 0 ] = a;
    vec.data.values[ 1 ] = b;
}

/// THE SCALAR BROADCAST, and it has to go through the split like everything else. Written as a
/// loop over `values` it wrote 32 floats to memory one at a time for a `SimdVec<float,32>` --
/// 25 instructions where two `vbroadcastss` do the job -- because a splittable impl stores a real
/// array and there was no register form to reach at that width. Recursing lands on
/// SIMD_VEC_IMPL_REG_INIT_1 at each half instead.
template<class T,int size,class Arch,class G> HaD
void init_sc( SimdVecImpl<T,size,Arch> &vec, G a ) {
    if constexpr ( HasSplit<SimdVecImpl<T,size,Arch>> ) {
        init_sc( vec.data.split.v0, a );
        init_sc( vec.data.split.v1, a );
    } else {
        for( int i = 0; i < size; ++i )
            vec.data.values[ i ] = a;
    }
}

template<class T,int size,int part,class Arch> HaD
void init_sc( SimdVecImpl<T,size,Arch> &vec, SimdVecImpl<T,part,Arch> a, SimdVecImpl<T,size-part,Arch> b ) {
    vec.data.split.v0 = a;
    vec.data.split.v1 = b;
}

#define SIMD_VEC_IMPL_REG_INIT_1( COND, T, SIZE, FUNC ) \
    template<class Arch> requires ( Arch::template Has<features::COND>::value ) HaD \
    void init_sc( SimdVecImpl<T,SIZE,Arch> &vec, T a ) { \
        ASIMD_DEBUG_ON_OP("init_sc_1",#COND,#FUNC) vec.data.reg = FUNC; \
    }

#define SIMD_VEC_IMPL_REG_INIT_2( COND, T, SIZE, FUNC ) \
    template<class Arch> requires ( Arch::template Has<features::COND>::value ) HaD \
    void init_sc( SimdVecImpl<T,SIZE,Arch> &vec, T a, T b ) { \
        ASIMD_DEBUG_ON_OP("init_sc_2",#COND,#FUNC) vec.data.reg = FUNC; \
    }

#define SIMD_VEC_IMPL_REG_INIT_4( COND, T, SIZE, FUNC ) \
    template<class Arch> requires ( Arch::template Has<features::COND>::value ) HaD \
    void init_sc( SimdVecImpl<T,SIZE,Arch> &vec, T a, T b, T c, T d ) { \
        ASIMD_DEBUG_ON_OP("init_sc_4",#COND,#FUNC) vec.data.reg = FUNC; \
    }

#define SIMD_VEC_IMPL_REG_INIT_8( COND, T, SIZE, FUNC ) \
    template<class Arch> requires ( Arch::template Has<features::COND>::value ) HaD \
    void init_sc( SimdVecImpl<T,SIZE,Arch> &vec, T a, T b, T c, T d, T e, T f, T g, T h ) { \
        ASIMD_DEBUG_ON_OP("init_sc_8",#COND,#FUNC) vec.data.reg = FUNC; \
    }

#define SIMD_VEC_IMPL_REG_INIT_16( COND, T, SIZE, FUNC ) \
    template<class Arch> requires ( Arch::template Has<features::COND>::value ) HaD \
    void init_sc( SimdVecImpl<T,SIZE,Arch> &vec, T a, T b, T c, T d, T e, T f, T g, T h, T i, T j, T k, T l, T m, T n, T o, T p ) { \
        ASIMD_DEBUG_ON_OP("init_sc_16",#COND,#FUNC) vec.data.reg = FUNC; \
    }

// prefetch ----------------------------------------------------------------------
template<int len,class N_len,class S_Arch>
void prefetch( const void *, N_len, S_Arch ) {
}

// load_unaligned ----------------------------------------------------------------------------
template<class G,class T,int size,class Arch> HaD
SimdVecImpl<T,size,Arch> load_unaligned( const G *data, S<SimdVecImpl<T,size,Arch>> ) {
    SimdVecImpl<T,size,Arch> res;
    if constexpr ( HasSplit<SimdVecImpl<T,size,Arch>> ) {
        res.data.split.v0 = load_unaligned( data                                         , S<SimdVecImpl<T,SimdVecImpl<T,size,Arch>::split_size_0,Arch>>() );
        res.data.split.v1 = load_unaligned( data + SimdVecImpl<T,size,Arch>::split_size_0, S<SimdVecImpl<T,SimdVecImpl<T,size,Arch>::split_size_1,Arch>>() );
    } else {
        for ( int i = 0; i < size; ++i )
            res.data.values[ i ] = data[ i ];
    }
    return res;
}

template<class G,class T,class Arch> HaD
SimdVecImpl<T,1,Arch> load_unaligned( const G *data, S<SimdVecImpl<T,1,Arch>> ) {
    SimdVecImpl<T,1,Arch> res;
    res.data.values[ 0 ] = *data;
    return res;
}

#define SIMD_VEC_IMPL_REG_LOAD_UNALIGNED( COND, T, SIZE, FUNC ) \
    template<class Arch> requires ( Arch::template Has<features::COND>::value ) HaD \
    SimdVecImpl<T,SIZE,Arch> load_unaligned( const T *data, S<SimdVecImpl<T,SIZE,Arch>> ) { \
        ASIMD_DEBUG_ON_OP("load_unaligned",#COND,#FUNC) SimdVecImpl<T,SIZE,Arch> res; res.data.reg = FUNC; return res; \
    }

#define SIMD_VEC_IMPL_REG_LOAD_FOT( COND, T, G, SIZE, FUNC ) \
    template<class Arch> requires ( Arch::template Has<features::COND>::value ) HaD \
    SimdVecImpl<T,SIZE,Arch> load_unaligned( const G *data, S<SimdVecImpl<T,SIZE,Arch>> ) { \
        ASIMD_DEBUG_ON_OP("store_unaligned_fot",#COND,#FUNC) SimdVecImpl<T,SIZE,Arch> res; res.data.reg = FUNC; return res; \
    }

// load_aligned ---------------------------------------------------------------------------------
template<class G,class T,int size,class Arch> HaD
SimdVecImpl<T,size,Arch> load_aligned( const G *data, S<SimdVecImpl<T,size,Arch>> ) {
    SimdVecImpl<T,size,Arch> res;
    if constexpr ( HasSplit<SimdVecImpl<T,size,Arch>> ) {
        res.data.split.v0 = load_aligned( data                                         , S<SimdVecImpl<T,SimdVecImpl<T,size,Arch>::split_size_0,Arch>>() );
        res.data.split.v1 = load_aligned( data + SimdVecImpl<T,size,Arch>::split_size_0, S<SimdVecImpl<T,SimdVecImpl<T,size,Arch>::split_size_1,Arch>>() );
    } else {
        for ( int i = 0; i < size; ++i )
            res.data.values[ i ] = data[ i ];
    }
    return res;
}

template<class G,class T,class Arch> HaD
SimdVecImpl<T,1,Arch> load_aligned( const G *data, S<SimdVecImpl<T,1,Arch>> ) {
    SimdVecImpl<T,1,Arch> res;
    res.data.values[ 0 ] = *data;
    return res;
}

template<class G,class T,int size,class Arch> HaD
SimdVecImpl<T,size,Arch> load_aligned_stream( const G *data, S<SimdVecImpl<T,size,Arch>> ) {
    SimdVecImpl<T,size,Arch> res;
    if constexpr ( HasSplit<SimdVecImpl<T,size,Arch>> ) {
        res.data.split.v0 = load_aligned_stream( data                                         , S<SimdVecImpl<T,SimdVecImpl<T,size,Arch>::split_size_0,Arch>>() );
        res.data.split.v1 = load_aligned_stream( data + SimdVecImpl<T,size,Arch>::split_size_0, S<SimdVecImpl<T,SimdVecImpl<T,size,Arch>::split_size_1,Arch>>() );
    } else {
        for ( int i = 0; i < size; ++i )
            res.data.values[ i ] = data[ i ];
    }
    return res;
}

template<class G,class T,class Arch> HaD
SimdVecImpl<T,1,Arch> load_aligned_stream( const G *data, S<SimdVecImpl<T,1,Arch>> ) {
    SimdVecImpl<T,1,Arch> res;
    res.data.values[ 0 ] = *data;
    return res;
}

template<class P,class T,int size,class Arch> HaD
SimdVecImpl<T,size,Arch> load( const P &data, S<SimdVecImpl<T,size,Arch>> ) {
    SimdVecImpl<T,size,Arch> res;
    if constexpr ( HasSplit<SimdVecImpl<T,size,Arch>> ) {
        res.data.split.v0 = load( data                                              , S<SimdVecImpl<T,SimdVecImpl<T,size,Arch>::split_size_0,Arch>>() );
        res.data.split.v1 = load( data + N<SimdVecImpl<T,size,Arch>::split_size_0>(), S<SimdVecImpl<T,SimdVecImpl<T,size,Arch>::split_size_1,Arch>>() );
    } else {
        const auto *p = data.get();
        for ( int i = 0; i < size; ++i )
            res.data.values[ i ] = p[ i ];
    }
    return res;
}

template<class P,class T,class Arch> HaD
SimdVecImpl<T,1,Arch> load( const P &data, S<SimdVecImpl<T,1,Arch>> ) {
    SimdVecImpl<T,1,Arch> res;
    res.data.values[ 0 ] = *data.get();
    return res;
}

template<class P,class T,int size,class Arch> HaD
SimdVecImpl<T,size,Arch> load_stream( const P &data, S<SimdVecImpl<T,size,Arch>> ) {
    SimdVecImpl<T,size,Arch> res;
    if constexpr ( HasSplit<SimdVecImpl<T,size,Arch>> ) {
        res.data.split.v0 = load_stream( data                                              , S<SimdVecImpl<T,SimdVecImpl<T,size,Arch>::split_size_0,Arch>>() );
        res.data.split.v1 = load_stream( data + N<SimdVecImpl<T,size,Arch>::split_size_0>(), S<SimdVecImpl<T,SimdVecImpl<T,size,Arch>::split_size_1,Arch>>() );
    } else {
        const auto *p = data.get();
        for ( int i = 0; i < size; ++i )
            res.data.values[ i ] = p[ i ];
    }
    return res;
}

template<class P,class T,class Arch> HaD
SimdVecImpl<T,1,Arch> load_stream( const P &data, S<SimdVecImpl<T,1,Arch>> ) {
    SimdVecImpl<T,1,Arch> res;
    res.data.values[ 0 ] = *data.get();
    return res;
}

#define SIMD_VEC_IMPL_REG_LOAD_ALIGNED( COND, T, SIZE, MIN_ALIG, FUNC ) \
    template<class Arch> requires ( Arch::template Has<features::COND>::value ) HaD \
    SimdVecImpl<T,SIZE,Arch> load_aligned( const T *data, S<SimdVecImpl<T,SIZE,Arch>> ) { \
        ASIMD_DEBUG_ON_OP("load_aligned",#COND,#FUNC) SimdVecImpl<T,SIZE,Arch> res; res.data.reg = FUNC; return res; \
    } \
    template<class P,class Arch> HaD \
    SimdVecImpl<T,SIZE,Arch> load( const P &data, S<SimdVecImpl<T,SIZE,Arch>> s ) requires ( std::is_same<T,typename std::decay<decltype(*data)>::type>::value && Arch::template Has<features::COND>::value ) { \
        return P::alignment % MIN_ALIG || P::offset % MIN_ALIG ? load_unaligned( data.get(), s ) : load_aligned( data.get(), s ); \
    }

#define SIMD_VEC_IMPL_REG_LOAD_ALIGNED_STREAM( COND, T, SIZE, MIN_ALIG, FUNC ) \
    template<class Arch> requires ( Arch::template Has<features::COND>::value ) HaD \
    SimdVecImpl<T,SIZE,Arch> load_aligned_stream( const T *data, S<SimdVecImpl<T,SIZE,Arch>> ) { \
        ASIMD_DEBUG_ON_OP("load_aligned_stream",#COND,#FUNC) SimdVecImpl<T,SIZE,Arch> res; res.data.reg = FUNC; return res; \
    } \
    template<class P,class Arch> HaD \
    SimdVecImpl<T,SIZE,Arch> load_stream( const P &data, S<SimdVecImpl<T,SIZE,Arch>> s ) requires ( std::is_same<T,typename std::decay<decltype(*data)>::type>::value && Arch::template Has<features::COND>::value ) { \
        return P::alignment % MIN_ALIG || P::offset % MIN_ALIG ? load_unaligned( data.get(), s ) : load_aligned_stream( data.get(), s ); \
    }

#define SIMD_VEC_IMPL_REG_LOAD_ALIGNED_FOT( COND, T, G, SIZE, MIN_ALIG, FUNC ) \
    template<class Arch> requires ( Arch::template Has<features::COND>::value ) HaD \
    SimdVecImpl<T,SIZE,Arch> load_aligned( const G *data, S<SimdVecImpl<T,SIZE,Arch>> ) { \
        ASIMD_DEBUG_ON_OP("load_aligned_fot",#COND,#FUNC) SimdVecImpl<T,SIZE,Arch> res; res.data.reg = FUNC; return res; \
    } \
    template<class P,int a> requires ( std::is_same<G,typename std::decay<decltype(*data)>::type>::value && Arch::template Has<features::COND>::value ) HaD \
    SimdVecImpl<T,SIZE,Arch> load( const P &data, S<SimdVecImpl<T,SIZE,Arch>> s ) { \
        return P::alignment % MIN_ALIG || P::offset % MIN_ALIG ? load_unaligned( data.get(), s ) : load_aligned( data.get(), s ); \
    }

// store and init unaligned -----------------------------------------------------------------------
template<class G,class T,int size,class Arch> HaD
void store_unaligned( G *data, const SimdVecImpl<T,size,Arch> &impl ) {
    if constexpr ( HasSplit<SimdVecImpl<T,size,Arch>> ) {
        store_unaligned( data                    , impl.data.split.v0 );
        store_unaligned( data + impl.split_size_0, impl.data.split.v1 );
    } else {
        for ( int i = 0; i < size; ++i )
            data[ i ] = impl.data.values[ i ];
    }
}

template<class G,class T,class Arch> HaD
void store_unaligned( G *data, const SimdVecImpl<T,1,Arch> &impl ) {
    *data = impl.data.values[ 0 ];
}

template<class G,class T,int size,class Arch> HaD
void init_unaligned( G *data, const SimdVecImpl<T,size,Arch> &impl ) {
    if constexpr ( HasSplit<SimdVecImpl<T,size,Arch>> ) {
        init_unaligned( data                    , impl.data.split.v0 );
        init_unaligned( data + impl.split_size_0, impl.data.split.v1 );
    } else {
        for ( int i = 0; i < size; ++i )
            new ( data + i ) G( impl.data.values[ i ] );
    }
}

template<class G,class T,class Arch> HaD
void init_unaligned( G *data, const SimdVecImpl<T,1,Arch> &impl ) {
    new ( data ) G( impl.data.values[ 0 ] );
}

#define SIMD_VEC_IMPL_REG_STORE_UNALIGNED( COND, T, SIZE, FUNC ) \
    template<class Arch> requires ( Arch::template Has<features::COND>::value ) HaD \
    void store_unaligned( T *data, const SimdVecImpl<T,SIZE,Arch> &impl ) { \
        ASIMD_DEBUG_ON_OP("store_unaligned",#COND,#FUNC) FUNC; \
    } \
    template<class Arch> requires ( Arch::template Has<features::COND>::value ) HaD \
    void init_unaligned( T *data, const SimdVecImpl<T,SIZE,Arch> &impl ) { \
        ASIMD_DEBUG_ON_OP("init_unaligned",#COND,#FUNC) FUNC; \
    }

// store and init aligned -----------------------------------------------------------------------
template<class G,class T,int size,class Arch> HaD
void store_aligned( G *data, const SimdVecImpl<T,size,Arch> &impl ) {
    if constexpr ( HasSplit<SimdVecImpl<T,size,Arch>> ) {
        store_aligned( data                    , impl.data.split.v0 );
        store_aligned( data + impl.split_size_0, impl.data.split.v1 );
    } else {
        for ( int i = 0; i < size; ++i )
            data[ i ] = impl.data.values[ i ];
    }
}

template<class G,class T,class Arch> HaD
void store_aligned( G *data, const SimdVecImpl<T,1,Arch> &impl ) {
    *data = impl.data.values[ 0 ];
}

template<class P,class T,int size,class Arch> HaD
void store( const P &data, const SimdVecImpl<T,size,Arch> &impl ) {
    if constexpr ( HasSplit<SimdVecImpl<T,size,Arch>> ) {
        store( data                         , impl.data.split.v0 );
        store( data + N<impl.split_size_0>(), impl.data.split.v1 );
    } else {
        auto *p = data.get();
        for ( int i = 0; i < size; ++i )
            p[ i ] = impl.data.values[ i ];
    }
}

template<class P,class T,class Arch> HaD
void store( const P &data, const SimdVecImpl<T,1,Arch> &impl ) {
    *data = impl.data.values[ 0 ];
}

template<class G,class T,int size,class Arch> HaD
void store_aligned_stream( G *data, const SimdVecImpl<T,size,Arch> &impl ) {
    if constexpr ( HasSplit<SimdVecImpl<T,size,Arch>> ) {
        store_aligned_stream( data                    , impl.data.split.v0 );
        store_aligned_stream( data + impl.split_size_0, impl.data.split.v1 );
    } else {
        for ( int i = 0; i < size; ++i )
            data[ i ] = impl.data.values[ i ];
    }
}

template<class G,class T,class Arch> HaD
void store_aligned_stream( G *data, const SimdVecImpl<T,1,Arch> &impl ) {
    *data = impl.data.values[ 0 ];
}

template<class P,class T,int size,class Arch> HaD
void store_stream( const P &data, const SimdVecImpl<T,size,Arch> &impl ) {
    if constexpr ( HasSplit<SimdVecImpl<T,size,Arch>> ) {
        store_stream( data                                              , impl.data.split.v0 );
        store_stream( data + N<SimdVecImpl<T,size,Arch>::split_size_0>(), impl.data.split.v1 );
    } else {
        auto *p = data.get();
        for ( int i = 0; i < size; ++i )
            p[ i ] = impl.data.values[ i ];
    }
}

template<class P,class T,class Arch> HaD
void store_stream( const P &data, const SimdVecImpl<T,1,Arch> &impl ) {
    *data = impl.data.values[ 0 ];
}

template<class G,class T,int size,class Arch> HaD
void init_aligned( G *data, const SimdVecImpl<T,size,Arch> &impl ) {
    if constexpr ( HasSplit<SimdVecImpl<T,size,Arch>> ) {
        init_aligned( data                    , impl.data.split.v0 );
        init_aligned( data + impl.split_size_0, impl.data.split.v1 );
    } else {
        for ( int i = 0; i < size; ++i )
            new ( data + i ) G( impl.data.values[ i ] );
    }
}

template<class G,class T,class Arch> HaD
void init_aligned( G *data, const SimdVecImpl<T,1,Arch> &impl ) {
    new ( data ) G( impl.data.values[ 0 ] );
}

template<class P,class T,int size,class Arch> HaD
void init( const P &data, const SimdVecImpl<T,size,Arch> &impl ) {
    if constexpr ( HasSplit<SimdVecImpl<T,size,Arch>> ) {
        init( data                         , impl.data.split.v0 );
        init( data + N<impl.split_size_0>(), impl.data.split.v1 );
    } else {
        using G = typename std::decay<decltype( *data )>::type;
        auto *p = data.get();
        for ( int i = 0; i < size; ++i )
            new ( p + i ) G( impl.data.values[ i ] );
    }
}

template<class P,class T,class Arch> HaD
void init( const P &data, const SimdVecImpl<T,1,Arch> &impl ) {
    using G = typename std::decay<decltype(*data)>::type;
    new ( data.get() ) G( impl.data.values[ 0 ] );
}

template<class G,class T,int size,class Arch> HaD
void init_aligned_stream( G *data, const SimdVecImpl<T,size,Arch> &impl ) {
    if constexpr ( HasSplit<SimdVecImpl<T,size,Arch>> ) {
        init_aligned_stream( data                    , impl.data.split.v0 );
        init_aligned_stream( data + impl.split_size_0, impl.data.split.v1 );
    } else {
        for ( int i = 0; i < size; ++i )
            new ( data + i ) G( impl.data.values[ i ] );
    }
}

template<class G,class T,class Arch> HaD
void init_aligned_stream( G *data, const SimdVecImpl<T,1,Arch> &impl ) {
    new ( data ) G( impl.data.values[ 0 ] );
}

template<class P,class T,int size,class Arch> HaD
void init_stream( const P &data, const SimdVecImpl<T,size,Arch> &impl ) {
    if constexpr ( HasSplit<SimdVecImpl<T,size,Arch>> ) {
        init_stream( data                         , impl.data.split.v0 );
        init_stream( data + N<impl.split_size_0>(), impl.data.split.v1 );
    } else {
        using G = typename std::decay<decltype( *data )>::type;
        auto *p = data.get();
        for ( int i = 0; i < size; ++i )
            new ( p + i ) G( impl.data.values[ i ] );
    }
}

template<class P,class T,class Arch> HaD
void init_stream( const P &data, const SimdVecImpl<T,1,Arch> &impl ) {
    using G = typename std::decay<decltype(*data)>::type;
    new ( data.get() ) G( impl.data.values[ 0 ] );
}

#define SIMD_VEC_IMPL_REG_STORE_ALIGNED( COND, T, SIZE, MIN_ALIG, FUNC ) \
    template<class Arch> requires ( Arch::template Has<features::COND>::value ) HaD \
    void store_aligned( T *data, const SimdVecImpl<T,SIZE,Arch> &impl ) { \
        ASIMD_DEBUG_ON_OP("store_aligned",#COND,#FUNC) FUNC; \
    } \
    template<class P,class Arch> HaD \
    void store( const P &data, const SimdVecImpl<T,SIZE,Arch> &impl ) requires ( std::is_same<T,typename std::decay<decltype(*data)>::type>::value && Arch::template Has<features::COND>::value ) { \
        P::alignment % MIN_ALIG || P::offset % MIN_ALIG ? store_unaligned( data.get(), impl ) : store_aligned( data.get(), impl ); \
    } \
    template<class Arch> requires ( Arch::template Has<features::COND>::value ) HaD \
    void init_aligned( T *data, const SimdVecImpl<T,SIZE,Arch> &impl ) { \
        ASIMD_DEBUG_ON_OP("init_aligned",#COND,#FUNC) FUNC; \
    } \
    template<class P,class Arch> HaD \
    void init( const P &data, const SimdVecImpl<T,SIZE,Arch> &impl ) requires ( std::is_same<T,typename std::decay<decltype(*data)>::type>::value && Arch::template Has<features::COND>::value ) { \
        P::alignment % MIN_ALIG || P::offset % MIN_ALIG ? init_unaligned( data.get(), impl ) : init_aligned( data.get(), impl ); \
    }

#define SIMD_VEC_IMPL_REG_STORE_ALIGNED_STREAM( COND, T, SIZE, MIN_ALIG, FUNC ) \
    template<class Arch> requires( Arch::template Has<features::COND>::value ) HaD \
    void store_aligned_stream( T *data, const SimdVecImpl<T,SIZE,Arch> &impl ) { \
        ASIMD_DEBUG_ON_OP("store_aligned_stream",#COND,#FUNC) FUNC; \
    } \
    template<class P,class Arch> HaD \
    void store_stream( const P &data, const SimdVecImpl<T,SIZE,Arch> &impl ) requires ( std::is_same<T,typename std::decay<decltype(*data)>::type>::value && Arch::template Has<features::COND>::value ) { \
        P::alignment % MIN_ALIG || P::offset % MIN_ALIG ? store_unaligned( data.get(), impl ) : store_aligned_stream( data.get(), impl ); \
    } \
    template<class Arch> requires( Arch::template Has<features::COND>::value ) HaD \
    void init_aligned_stream( T *data, const SimdVecImpl<T,SIZE,Arch> &impl ) { \
        ASIMD_DEBUG_ON_OP("init_aligned_stream",#COND,#FUNC) FUNC; \
    } \
    template<class P,class Arch> HaD \
    void init_stream( const P &data, const SimdVecImpl<T,SIZE,Arch> &impl ) requires( std::is_same<T,typename std::decay<decltype(*data)>::type>::value && Arch::template Has<features::COND>::value ) { \
        P::alignment % MIN_ALIG || P::offset % MIN_ALIG ? init_unaligned( data.get(), impl ) : init_aligned_stream( data.get(), impl ); \
    }

// arithmetic operations -------------------------------------------------------------
//
// THREE BRANCHES, chosen at compile time, and the middle one is the reason this is not just a
// recursion any more:
//
//   `split`         a splittable impl is a pair of halves -- recurse into them.
//   whole `values`  a REGISTER impl has no split (README section 3) but its `values` IS a vector
//                   type, so the operation applies to it in one go. gcc and clang lower that to
//                   the right instruction on their own, which is how `SI32 x 8` gets a `vpmulld`
//                   without anybody registering a `mul` backend for it.
//   lane by lane    everything else, including the one-lane tail of an odd width.
//
// Before this, the first branch was unconditional. At a register-backed width, an operation with
// no register form was therefore a HARD COMPILE ERROR rather than a slow path: `.sum()`, `mul`
// and `div` on every integer type, the strided `iota`, `gather` and `scatter` below AVX2. That
// was 150 of the 408 cells of `tests/compile_matrix.sh`.
#define SIMD_VEC_IMPL_ARITHMETIC_OP( NAME, OP ) \
    template<class T,int size,class Arch> HaD \
    SimdVecImpl<T,size,Arch> NAME( const SimdVecImpl<T,size,Arch> &a, const SimdVecImpl<T,size,Arch> &b ) { \
        SimdVecImpl<T,size,Arch> res; \
        if constexpr ( HasSplit<SimdVecImpl<T,size,Arch>> ) { \
            res.data.split.v0 = NAME( a.data.split.v0, b.data.split.v0 ); \
            res.data.split.v1 = NAME( a.data.split.v1, b.data.split.v1 ); \
        /* THE TEST ASKS WHETHER THE RESULT CAN BE ASSIGNED BACK, not merely whether the */ \
        /* operator parses -- and the difference is ARRAY-TO-POINTER DECAY. On MSVC, and under */ \
        /* ASIMD_NO_COMPILER_VECTORS, `values` is a plain `T[ N ]`; `a.data.values - b.data.values` */ \
        /* then decays to POINTER SUBTRACTION, which is perfectly valid and yields a ptrdiff_t. */ \
        /* So the old test said yes for `sub`, this branch was taken, and assigning an integer */ \
        /* to an array is a hard error: */ \
        /*     error C3863: array type 'Values' is not assignable            (MSVC)          */ \
        /*     error: array type 'Values' (aka 'long[4]') is not assignable  (clang, same)   */ \
        /* It only bites where a register impl exists WITHOUT a register form of the operation, */ \
        /* which on x86 is 256-bit integer `add`/`sub` -- AVX gives the impl, AVX2 gives the */ \
        /* instructions. That is MSVC's `/arch:AVX` exactly, and no `run_all_isa.sh` row matched */ \
        /* it: the MSVC-path rows were SSE2 (where 128-bit int add/sub are registered) and */ \
        /* native (where the 256-bit ones are). There is an `-mavx` row now. */ \
        /* `+`, `*`, `/` and `&` are ill-formed on pointers, so `sub` was the only one affected. */ \
        } else if constexpr ( requires { res.data.values = a.data.values OP b.data.values; } ) { \
            res.data.values = a.data.values OP b.data.values; \
        } else { \
            for ( int i = 0; i < size; ++i ) \
                res.data.values[ i ] = a.data.values[ i ] OP b.data.values[ i ]; \
        } \
        return res; \
    } \
    \
    template<class T,class Arch> HaD \
    SimdVecImpl<T,1,Arch> NAME( const SimdVecImpl<T,1,Arch> &a, const SimdVecImpl<T,1,Arch> &b ) { \
        SimdVecImpl<T,1,Arch> res; \
        res.data.values[ 0 ] = a.data.values[ 0 ] OP b.data.values[ 0 ]; \
        return res; \
    }

    SIMD_VEC_IMPL_ARITHMETIC_OP( sll, << )
    SIMD_VEC_IMPL_ARITHMETIC_OP( add, +  )
    SIMD_VEC_IMPL_ARITHMETIC_OP( sub, -  )
    SIMD_VEC_IMPL_ARITHMETIC_OP( mul, *  )
    SIMD_VEC_IMPL_ARITHMETIC_OP( div, /  )

#undef SIMD_VEC_IMPL_ARITHMETIC_OP

/// `and` is a BITWISE operation, so on a floating point type it is neither `a & b` (ill-formed)
/// nor an arithmetic and: it is an and on the bit patterns. `SimdVec<float,5> & ...` used to fail
/// to compile for exactly that reason -- the one-lane tail of the split reached `float & float`.
template<class T> HaD
T lane_and( T a, T b ) {
    if constexpr ( std::is_integral<T>::value ) {
        return a & b;
    } else {
        using U = typename PI_<8 * sizeof( T )>::T;
        return std::bit_cast<T>( U( std::bit_cast<U>( a ) & std::bit_cast<U>( b ) ) );
    }
}

template<class T,int size,class Arch> HaD
SimdVecImpl<T,size,Arch> anb( const SimdVecImpl<T,size,Arch> &a, const SimdVecImpl<T,size,Arch> &b ) {
    SimdVecImpl<T,size,Arch> res;
    if constexpr ( HasSplit<SimdVecImpl<T,size,Arch>> ) {
        res.data.split.v0 = anb( a.data.split.v0, b.data.split.v0 );
        res.data.split.v1 = anb( a.data.split.v1, b.data.split.v1 );
    } else if constexpr ( requires { res.data.values = a.data.values & b.data.values; } ) {
        // assignability, not parseability -- see the note in SIMD_VEC_IMPL_ARITHMETIC_OP. `&` on
        // two pointers is ill-formed so this one was never wrong, but the two tests should not
        // differ in what they ask.
        res.data.values = a.data.values & b.data.values;
    } else {
        for ( int i = 0; i < size; ++i )
            res.data.values[ i ] = lane_and( T( a.data.values[ i ] ), T( b.data.values[ i ] ) );
    }
    return res;
}

template<class T,class Arch> HaD
SimdVecImpl<T,1,Arch> anb( const SimdVecImpl<T,1,Arch> &a, const SimdVecImpl<T,1,Arch> &b ) {
    SimdVecImpl<T,1,Arch> res;
    res.data.values[ 0 ] = lane_and( a.data.values[ 0 ], b.data.values[ 0 ] );
    return res;
}

#define SIMD_VEC_IMPL_REG_ARITHMETIC_OP( COND, T, SIZE, NAME, FUNC ) \
    template<class Arch> requires ( Arch::template Has<features::COND>::value ) HaD \
    SimdVecImpl<T,SIZE,Arch> NAME( const SimdVecImpl<T,SIZE,Arch> &a, const SimdVecImpl<T,SIZE,Arch> &b ) { \
        ASIMD_DEBUG_ON_OP(#NAME,#COND,#FUNC) SimdVecImpl<T,SIZE,Arch> res; res.data.reg = FUNC( a.data.reg, b.data.reg ); return res; \
    }

// cmp operations ------------------------------------------------------------------
#define SIMD_VEC_IMPL_CMP_OP( NAME, OP ) \
    /* _as_a_simd_bool */ \
    /* THE THIRD CONDITION IS NOT REDUNDANT, and it was missing. */ \
    /* */ \
    /* This form returns the BIT flavour of mask, and recursing into the split hands each half */ \
    /* to whatever `NAME##_as_a_simd_bool` resolves to at half the width -- which, wherever a */ \
    /* backend registers a register form through SIMD_VEC_IMPL_CMP_OP_SIMDVEC, is the LANE */ \
    /* flavour. `SimdBoolImpl<8,32>` does not assign to a `SimdBoolImpl<8,1>`, so the whole */ \
    /* cell was a hard compile error rather than a slow path -- the same shape of hole as the */ \
    /* `data.split` recursion of README section 3, one mechanism over. */ \
    /* */ \
    /* It needed 16 lanes AND a register form at 8 to show up, which is why it survived: the */ \
    /* bit-flavoured mask only splits from 16 items up, and `to_bits( a > b )` and */ \
    /* `select( a > b, ... )` go through `ops::cmp_gt` and never reach this function at all. */ \
    /* `any( a > b )` and `all( a > b )` do. On x86 it reproduces at `SimdVec<float,16>` under */ \
    /* -mavx, which no test happened to ask for; on ARM, where the register stops at 128 bits, */ \
    /* it is `SimdVec<SI16,16>` -- an ordinary width. */ \
    /* */ \
    /* The `requires` asks the only question that matters -- do the halves give something this */ \
    /* mask can hold -- and falls back to the lane loop when they do not. */ \
    template<class T,int size,class Arch> HaD \
    SimdBoolImpl<size,1,Arch> NAME##_as_a_simd_bool( const SimdVecImpl<T,size,Arch> &a, const SimdVecImpl<T,size,Arch> &b ) { \
        SimdBoolImpl<size,1,Arch> res; \
        if constexpr ( HasSplit<SimdVecImpl<T,size,Arch>> && HasSplit<SimdBoolImpl<size,1,Arch>> \
                    && requires { res.data.split.v0 = NAME##_as_a_simd_bool( a.data.split.v0, b.data.split.v0 ); } ) { \
            res.data.split.v0 = NAME##_as_a_simd_bool( a.data.split.v0, b.data.split.v0 ); \
            res.data.split.v1 = NAME##_as_a_simd_bool( a.data.split.v1, b.data.split.v1 ); \
        } else { \
            res.data.values.set_value( false ); \
            for ( int i = 0; i < size; ++i ) \
                if ( a.data.values[ i ] OP b.data.values[ i ] ) res.data.values.set_bit( i ); \
        } \
        return res; \
    } \
    template<class T,class Arch> HaD \
    SimdBoolImpl<8,1,Arch> NAME##_as_a_simd_bool( const SimdVecImpl<T,8,Arch> &a, const SimdVecImpl<T,8,Arch> &b ) { \
        SimdBoolImpl<8,1,Arch> res; \
        res.data.values.set_values( \
            a.data.values[ 0 ] OP b.data.values[ 0 ], a.data.values[ 1 ] OP b.data.values[ 1 ], a.data.values[ 2 ] OP b.data.values[ 2 ], a.data.values[ 3 ] OP b.data.values[ 3 ], \
            a.data.values[ 4 ] OP b.data.values[ 4 ], a.data.values[ 5 ] OP b.data.values[ 5 ], a.data.values[ 6 ] OP b.data.values[ 6 ], a.data.values[ 7 ] OP b.data.values[ 7 ]  \
        ); \
        return res; \
    } \
    template<class T,class Arch> HaD \
    SimdBoolImpl<4,1,Arch> NAME##_as_a_simd_bool( const SimdVecImpl<T,4,Arch> &a, const SimdVecImpl<T,4,Arch> &b ) { \
        SimdBoolImpl<4,1,Arch> res; \
        res.data.values.set_values( \
            a.data.values[ 0 ] OP b.data.values[ 0 ], a.data.values[ 1 ] OP b.data.values[ 1 ], a.data.values[ 2 ] OP b.data.values[ 2 ], a.data.values[ 3 ] OP b.data.values[ 3 ] \
        ); \
        return res; \
    } \
    template<class T,class Arch> HaD \
    SimdBoolImpl<2,1,Arch> NAME##_as_a_simd_bool( const SimdVecImpl<T,2,Arch> &a, const SimdVecImpl<T,2,Arch> &b ) { \
        SimdBoolImpl<2,1,Arch> res; \
        res.data.values.set_values( \
            a.data.values[ 0 ] OP b.data.values[ 0 ], a.data.values[ 1 ] OP b.data.values[ 1 ] \
        ); \
        return res; \
    } \
    template<class T,class Arch> HaD \
    SimdBoolImpl<1,1,Arch> NAME##_as_a_simd_bool( const SimdVecImpl<T,1,Arch> &a, const SimdVecImpl<T,1,Arch> &b ) { \
        SimdBoolImpl<1,1,Arch> res; \
        res.data.values.set_values( \
            a.data.values[ 0 ] OP b.data.values[ 0 ] \
        ); \
        return res; \
    } \
    /* _as_a_simd_vec */ \
    template<class T,int size,class Arch,class I> HaD \
    SimdVecImpl<I,size,Arch> NAME##_as_a_simd_vec( const SimdVecImpl<T,size,Arch> &a, const SimdVecImpl<T,size,Arch> &b, S<SimdVecImpl<I,size,Arch>> ) { \
        SimdVecImpl<I,size,Arch> res; \
        if constexpr ( HasSplit<SimdVecImpl<T,size,Arch>> && HasSplit<SimdVecImpl<I,size,Arch>> ) { \
            res.data.split.v0 = NAME##_as_a_simd_vec( a.data.split.v0, b.data.split.v0, S<SimdVecImpl<I,a.split_size_0,Arch>>() ); \
            res.data.split.v1 = NAME##_as_a_simd_vec( a.data.split.v1, b.data.split.v1, S<SimdVecImpl<I,a.split_size_1,Arch>>() ); \
        } else { \
            for ( int i = 0; i < size; ++i ) \
                res.data.values[ i ] = a.data.values[ i ] OP b.data.values[ i ] ? ~I( 0 ) : I( 0 ); \
        } \
        return res; \
    } \
    template<class T,class Arch,class I> HaD \
    SimdVecImpl<I,1,Arch> NAME##_as_a_simd_vec( const SimdVecImpl<T,1,Arch> &a, const SimdVecImpl<T,1,Arch> &b, S<SimdVecImpl<I,1,Arch>> ) { \
        SimdVecImpl<I,1,Arch> res; \
        res.data.values[ 0 ] = a.data.values[ 0 ] OP b.data.values[ 0 ] ? ~I( 0 ) : I( 0 ); \
        return res; \
    } \
    template<class T,int s,class Arch> \
    struct Op_##NAME { \
        bool operator[]( PI index ) const { return a.data.values[ index ] OP b.data.values[ index ]; } \
        PI size() const { return s; } \
        \
        SimdVecImpl<T,s,Arch> a, b; \
    }; \
    \
    template<class T,int size,class Arch> HaD \
    Op_##NAME<T,size,Arch> NAME( const SimdVecImpl<T,size,Arch> &a, const SimdVecImpl<T,size,Arch> &b ) { \
        return { a, b }; \
    }

SIMD_VEC_IMPL_CMP_OP( lt, < )
SIMD_VEC_IMPL_CMP_OP( gt, > )

#undef SIMD_VEC_IMPL_CMP_OP

#define SIMD_VEC_IMPL_CMP_OP_SIMDVEC( COND, T, NB_ITEMS, ITEM_SIZE, NAME, FUNC ) \
    template<class Arch> requires( Arch::template Has<features::COND>::value ) HaD \
    auto NAME##_as_a_simd_bool( const SimdVecImpl<T,NB_ITEMS,Arch> &a, const SimdVecImpl<T,NB_ITEMS,Arch> &b ) { \
        ASIMD_DEBUG_ON_OP(#NAME,#COND,#FUNC) SimdBoolImpl<NB_ITEMS,ITEM_SIZE,Arch> res; res.data.reg = FUNC; return res; \
    }

/// The same, with an EXCLUSION -- "this feature, and NOT that one".
///
/// NEEDED FOR THE SAME REASON `ASIMD_X86_REQ_EXCL` is, one mechanism over. `Selection.h`'s KNOWN
/// LIMITS note says two backends registering the same (Op, Key, RANK) are as ambiguous as two
/// overloads would be -- and these ARE two overloads, with nothing to order them. `FP32 x 4` and
/// `FP64 x 2` were registered twice on any AVX target: once here by SSE2 (`cmpps`, two operands)
/// and once by AVX (`vcmpps`, three operands and a predicate). Both constraints are satisfied,
/// neither subsumes the other, so the call was AMBIGUOUS -- a hard error, on the most ordinary
/// widths there are.
///
/// It survived because nothing reached it: `to_bits( a > b )` and `select( a > b, ... )` both go
/// through `ops::cmp_gt` and its rank-ordered variants, and only `any( a > b )` / `all( a > b )`
/// use this function at all.
#define SIMD_VEC_IMPL_CMP_OP_SIMDVEC_EXCL( COND, CNOT, T, NB_ITEMS, ITEM_SIZE, NAME, FUNC ) \
    template<class Arch> requires( Arch::template Has<features::COND>::value \
                                && ! Arch::template Has<features::CNOT>::value ) HaD \
    auto NAME##_as_a_simd_bool( const SimdVecImpl<T,NB_ITEMS,Arch> &a, const SimdVecImpl<T,NB_ITEMS,Arch> &b ) { \
        ASIMD_DEBUG_ON_OP(#NAME,#COND,#FUNC) SimdBoolImpl<NB_ITEMS,ITEM_SIZE,Arch> res; res.data.reg = FUNC; return res; \
    }

#define SIMD_VEC_IMPL_CMP_OP_SIMDVEC_VEC( COND, T, I, SIZE, NAME, FUNC ) \
    template<class Arch> requires( Arch::template Has<features::COND>::value ) HaD \
    auto NAME##_as_a_simd_vec( const SimdVecImpl<T,SIZE,Arch> &a, const SimdVecImpl<T,SIZE,Arch> &b, S<SimdVecImpl<I,SIZE,Arch>> ) { \
        ASIMD_DEBUG_ON_OP(#NAME,#COND,#FUNC) SimdVecImpl<I,SIZE,Arch> res; res.data.reg = FUNC; return res; \
    }

// iota( beg ) --------------------------------------------------------------------------
//
// THROUGH `values`, NO LONGER THROUGH `split`. A register impl has no split view any more -- it
// used to send the whole vector back to memory (see the comment on `SIMD_VEC_IMPL_REG`) -- and
// since `values` is now a VECTOR type, writing it element by element folds to a constant at
// compile time. Register forms live in `SimdVecImpl_AVX.h` and `_AVX2.h`: without them gcc
// materializes this loop as a chain of `vpinsrd` instead of a constant load.
template<class T,int size,class Arch> HaD
SimdVecImpl<T,size,Arch> iota( T beg, S<SimdVecImpl<T,size,Arch>> ) {
    SimdVecImpl<T,size,Arch> res;
    for ( int i = 0; i < size; ++i )
        res.data.values[ i ] = beg + T( i );
    return res;
}

template<class T,class Arch> HaD
SimdVecImpl<T,1,Arch> iota( T beg, S<SimdVecImpl<T,1,Arch>> ) {
    SimdVecImpl<T,1,Arch> res;
    res.data.values[ 0 ] = beg;
    return res;
}

// iota( beg, mul ) ---------------------------------------------------------------------
template<class T,int size,class Arch> HaD
SimdVecImpl<T,size,Arch> iota( T beg, T mul, S<SimdVecImpl<T,size,Arch>> ) {
    SimdVecImpl<T,size,Arch> res;
    if constexpr ( HasSplit<SimdVecImpl<T,size,Arch>> ) {
        // `T( ... )` ON THE SECOND ROW, AND IT IS NOT DECORATION. `beg + n * mul` is subject to
        // the INTEGER PROMOTIONS: on a 16- or 8-bit lane type it has type `int`, so the recursive
        // call became `iota( int, short, S<...> )` and `T` could be deduced as both `int` and
        // `short` at once -- no matching function, a hard error rather than a slow path.
        //
        // It only ever showed on the narrow types, and only at a width that SPLITS, which is why
        // it survived: `tests/compile_matrix.sh` stopped at 32-bit lanes, and on ARM the native
        // width of `SI16` is eight -- a register impl, so this branch is not taken. It is
        // `SimdVec<SI16,16>` and `SimdVec<SI8,32>` that reach it, on either architecture.
        res.data.split.v0 = iota( beg                                                  , mul, S<SimdVecImpl<T,SimdVecImpl<T,size,Arch>::split_size_0,Arch>>() );
        res.data.split.v1 = iota( T( beg + SimdVecImpl<T,size,Arch>::split_size_0 * mul ), mul, S<SimdVecImpl<T,SimdVecImpl<T,size,Arch>::split_size_1,Arch>>() );
    } else {
        // as for `iota( beg )`: written through `values`, which is a vector type here, so the
        // whole thing folds to a constant load plus one multiply-add at compile time.
        for ( int i = 0; i < size; ++i )
            res.data.values[ i ] = beg + T( i ) * mul;
    }
    return res;
}

template<class T,class Arch> HaD
SimdVecImpl<T,1,Arch> iota( T beg, T /*mul*/, S<SimdVecImpl<T,1,Arch>> ) {
    SimdVecImpl<T,1,Arch> res;
    res.data.values[ 0 ] = beg;
    return res;
}

// sum -----------------------------------------------------------------------------
template<class T,int size,class Arch> HaD
T horizontal_sum( const SimdVecImpl<T,size,Arch> &impl ) {
    if constexpr ( HasSplit<SimdVecImpl<T,size,Arch>> ) {
        // ADD THE HALVES FIRST, then reduce once -- not reduce twice and add the scalars. When
        // the two halves have the same width that is one vector add plus one ladder, against two
        // ladders: measured 17 instructions down to 9 on `float x 8` under SSE2.
        constexpr int n0 = SimdVecImpl<T,size,Arch>::split_size_0;
        constexpr int n1 = SimdVecImpl<T,size,Arch>::split_size_1;
        if constexpr ( n0 == n1 )
            return horizontal_sum( add( impl.data.split.v0, impl.data.split.v1 ) );
        else
            return horizontal_sum( impl.data.split.v0 ) + horizontal_sum( impl.data.split.v1 );
    } else {
        // pairwise rather than a running accumulator: same instruction count on an integer type,
        // and a shorter dependency chain plus a reproducible order on a floating point one.
        T acc[ size ];
        for ( int i = 0; i < size; ++i ) acc[ i ] = impl.data.values[ i ];
        for ( int n = size; n > 1; n = ( n + 1 ) / 2 )
            for ( int i = 0; i < n / 2; ++i ) acc[ i ] += acc[ i + ( n + 1 ) / 2 ];
        return acc[ 0 ];
    }
}

template<class T,class Arch> HaD
T horizontal_sum( const SimdVecImpl<T,1,Arch> &impl ) {
    return impl.data.values[ 0 ];
}

/// Builds an impl from a bare register. Brace-initialising the union would hit its FIRST member,
/// `values`, not `reg` -- which silently works for the float types (the vector typedef and the
/// intrinsic type are compatible) and does not compile for the integer ones.
template<class Impl,class R> HaD
Impl impl_from_reg( R r ) { Impl res; res.data.reg = r; return res; }

/// A register form for the horizontal sum. There was none, at any width or any type: `.sum()` on
/// eight floats was 44 instructions, because a pairwise tree written over `values[ i ]` uses no
/// vector arithmetic and neither compiler can put it back together.
#define SIMD_VEC_IMPL_REG_HSUM( COND, T, SIZE, FUNC ) \
    template<class Arch> requires ( Arch::template Has<features::COND>::value ) HaD \
    T horizontal_sum( const SimdVecImpl<T,SIZE,Arch> &impl ) { \
        ASIMD_DEBUG_ON_OP("horizontal_sum",#COND,#FUNC) return FUNC; \
    }

// scatter/gather -----------------------------------------------------------------------
template<class G,class V,class T,int size,class Arch> HaD
void scatter( G *ptr, const V &ind, const SimdVecImpl<T,size,Arch> &vec ) {
    if constexpr ( HasSplit<SimdVecImpl<T,size,Arch>> && HasSplit<V> ) {
        scatter( ptr, ind.data.split.v0, vec.data.split.v0 );
        scatter( ptr, ind.data.split.v1, vec.data.split.v1 );
    } else {
        for ( int i = 0; i < size; ++i )
            ptr[ ind.data.values[ i ] ] = vec.data.values[ i ];
    }
}

template<class G,class V,class T,class Arch> HaD
void scatter( G *ptr, const V &ind, const SimdVecImpl<T,1,Arch> &vec ) {
    ptr[ ind.data.values[ 0 ] ] = vec.data.values[ 0 ];
}

#define SIMD_VEC_IMPL_REG_SCATTER( COND, T, I, SIZE, FUNC ) \
    template<class Arch> requires( Arch::template Has<features::COND>::value ) HaD \
    auto scatter( T *data, const SimdVecImpl<I,SIZE,Arch> &ind, const SimdVecImpl<T,SIZE,Arch> &vec ) { \
        ASIMD_DEBUG_ON_OP("scatter",#COND,#FUNC); FUNC; \
    }


template<class G,class V,class T,int size,class Arch> HaD
SimdVecImpl<T,size,Arch> gather( const G *data, const V &ind, S<SimdVecImpl<T,size,Arch>> ) {
    SimdVecImpl<T,size,Arch> res;
    if constexpr ( HasSplit<SimdVecImpl<T,size,Arch>> && HasSplit<V> ) {
        res.data.split.v0 = gather( data, ind.data.split.v0, S<SimdVecImpl<T,SimdVecImpl<T,size,Arch>::split_size_0,Arch>>() );
        res.data.split.v1 = gather( data, ind.data.split.v1, S<SimdVecImpl<T,SimdVecImpl<T,size,Arch>::split_size_1,Arch>>() );
    } else {
        for ( int i = 0; i < size; ++i )
            res.data.values[ i ] = data[ ind.data.values[ i ] ];
    }
    return res;
}

template<class G,class V,class T,class Arch> HaD
SimdVecImpl<T,1,Arch> gather( const G *data, const V &ind, S<SimdVecImpl<T,1,Arch>> ) {
    SimdVecImpl<T,1,Arch> res;
    res.data.values[ 0 ] = data[ ind.data.values[ 0 ] ];
    return res;
}

#define SIMD_VEC_IMPL_REG_GATHER( COND, T, I, SIZE, FUNC ) \
    template<class Arch> requires ( Arch::template Has<features::COND>::value ) HaD \
    SimdVecImpl<T,SIZE,Arch> gather( const T *data, const SimdVecImpl<I,SIZE,Arch> &ind, S<SimdVecImpl<T,SIZE,Arch>> ) { \
        ASIMD_DEBUG_ON_OP("gather",#COND,#FUNC) SimdVecImpl<T,SIZE,Arch> res; res.data.reg = FUNC; return res; \
    }

// min/max ---------------------------------------------------------------------
#define SIMD_VEC_IMPL_ARITHMETIC_FUNC( NAME, HELPER ) \
    template<class T,int size,class Arch> HaD \
    SimdVecImpl<T,size,Arch> NAME( const SimdVecImpl<T,size,Arch> &a, const SimdVecImpl<T,size,Arch> &b ) { \
        HELPER; \
        SimdVecImpl<T,size,Arch> res; \
        if constexpr ( HasSplit<SimdVecImpl<T,size,Arch>> ) { \
            res.data.split.v0 = NAME( a.data.split.v0, b.data.split.v0 ); \
            res.data.split.v1 = NAME( a.data.split.v1, b.data.split.v1 ); \
        } else { \
            for ( int i = 0; i < size; ++i ) \
                res.data.values[ i ] = NAME( T( a.data.values[ i ] ), T( b.data.values[ i ] ) ); \
        } \
        return res; \
    } \
    \
    template<class T,class Arch> HaD \
    SimdVecImpl<T,1,Arch> NAME( const SimdVecImpl<T,1,Arch> &a, const SimdVecImpl<T,1,Arch> &b ) { \
        HELPER; \
        SimdVecImpl<T,1,Arch> res; \
        res.data.values[ 0 ] = NAME( a.data.values[ 0 ], b.data.values[ 0 ] ); \
        return res; \
    }

    SIMD_VEC_IMPL_ARITHMETIC_FUNC( min, using std::min )
    SIMD_VEC_IMPL_ARITHMETIC_FUNC( max, using std::max )
#undef SIMD_VEC_IMPL_ARITHMETIC_FUNC


} // namespace internal
} // namespace asimd


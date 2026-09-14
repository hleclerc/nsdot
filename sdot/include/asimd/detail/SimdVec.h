#pragma once

#include "impl/SimdVecImpl_Generic.h"
#include "impl/SimdVecImpl_X86.h" // IWYU pragma: export
#include "impl/SimdVecImpl_Arm.h" // IWYU pragma: export

#include "SimdBool.h"
#include "LaneSet.h"
#include "SimdSize.h"
#include "Ptr.h"   // the `load( P )` / `store( P )` overloads below are written against it

namespace asimd {

template<class T_,int size_,class Arch> struct SimdVec;

// The lane rotations, declared here so that the methods below can name them; defined in
// `SimdOps.h`, next to the other operations that go through `Selection.h`.
template<int K,class T,int W,class Arch>       SimdVec<T,W,Arch> ext_lanes   ( const SimdVec<T,W,Arch> &a, const SimdVec<T,W,Arch> &b, N<K> );
template<int k,int n,class T,int W,class Arch> SimdVec<T,W,Arch> rotate_lanes( const SimdVec<T,W,Arch> &v, N<k>, N<n> );
template<int k,class T,int W,class Arch>       SimdVec<T,W,Arch> rotate_lanes( const SimdVec<T,W,Arch> &v, N<k> );
template<int n,class T,int W,class Arch>       SimdVec<T,W,Arch> rotate_lanes( const SimdVec<T,W,Arch> &v, int k, N<n> );
template<class T,int W,class Arch>             SimdVec<T,W,Arch> rotate_lanes( const SimdVec<T,W,Arch> &v, int k, int n );
template<class T,int W,class Arch>             SimdVec<T,W,Arch> rotate_lanes( const SimdVec<T,W,Arch> &v, int k );
template<int k,class T,int W,class Arch>       SimdVec<T,W,Arch> rotate_lanes( const SimdVec<T,W,Arch> &v, N<k>, int n );

// The arithmetic with a lane set (`LaneSet.h`), likewise.
template<class T,int W,class Arch,LaneSet S>   SimdVec<T,W,Arch> add( const SimdVec<T,W,Arch> &a, const SimdVec<T,W,Arch> &b, const S &s );
template<class T,int W,class Arch,LaneSet S>   SimdVec<T,W,Arch> sub( const SimdVec<T,W,Arch> &a, const SimdVec<T,W,Arch> &b, const S &s );
template<class T,int W,class Arch,LaneSet S>   SimdVec<T,W,Arch> mul( const SimdVec<T,W,Arch> &a, const SimdVec<T,W,Arch> &b, const S &s );
template<class T,int W,class Arch,LaneSet S>   SimdVec<T,W,Arch> div( const SimdVec<T,W,Arch> &a, const SimdVec<T,W,Arch> &b, const S &s );
template<class T,int W,class Arch,LaneSet S>   SimdVec<T,W,Arch> min( const SimdVec<T,W,Arch> &a, const SimdVec<T,W,Arch> &b, const S &s );
template<class T,int W,class Arch,LaneSet S>   SimdVec<T,W,Arch> max( const SimdVec<T,W,Arch> &a, const SimdVec<T,W,Arch> &b, const S &s );
template<class T,int W,class Arch,LaneSet S>   SimdVec<T,W,Arch> fma( const SimdVec<T,W,Arch> &a, const SimdVec<T,W,Arch> &b, const SimdVec<T,W,Arch> &c, const S &s );
template<class T,int W,class Arch,LaneSet S>   T                 sum( const SimdVec<T,W,Arch> &v, const S &s );

/**
  Simd vector.
*/
template<class T_,int size_=SimdSize<T_,NativeCpu>::value,class Arch=NativeCpu>
struct SimdVec {
    using                                        Impl                  = internal::SimdVecImpl<T_,size_,Arch>;
    using                                        T                     = T_;

    HaD                                          SimdVec               ( T a, T b, T c, T d, T e, T f, T g, T h ) { init_sc( impl, a, b, c, d, e, f, g, h ); }
    HaD                                          SimdVec               ( T a, T b, T c, T d, T e ) { init_sc( impl, a, b, c, d, e ); }
    HaD                                          SimdVec               ( T a, T b, T c, T d ) { init_sc( impl, a, b, c, d ); }
    HaD                                          SimdVec               ( T a, T b ) { init_sc( impl, a, b ); }
    HaD                                          SimdVec               ( T a ) { init_sc( impl, a ); }
    HaD                                          SimdVec               ( Impl impl ) : impl( impl ) {}
    HaD                                          SimdVec               () {}

    static HaD constexpr int                     size                  () { return size_; }

    // static ctors
    static HaD SimdVec                           iota                  ( T beg, T mul ) { return internal::iota( beg, mul, S<Impl>() ); }
    static HaD SimdVec                           iota                  ( T beg = 0 ) { return internal::iota( beg, S<Impl>() ); }

    //
    static void                                  prefetch              ( const T *beg ) { internal::prefetch( beg, N<sizeof(Impl)>(), S<Arch>() ); }

    // static versions of load/store
    template<class G> static HaD SimdVec         load_unaligned        ( const G *ptr ) { return internal::load_unaligned( ptr, S<Impl>() ); }
    template<class G> static HaD SimdVec         load_aligned          ( const G *ptr ) { return internal::load_aligned( ptr, S<Impl>() ); }
    template<class P> static HaD SimdVec         load                  ( const P &ptr ) { return internal::load( _chk_ptr( ptr ), S<Impl>() ); } ///< P::alignment, P::offset, data.get()

    template<class G> static HaD SimdVec         load_unaligned_stream ( const G *ptr ) { return internal::load_unaligned( ptr, S<Impl>() ); }
    template<class G> static HaD SimdVec         load_aligned_stream   ( const G *ptr ) { return internal::load_aligned_stream( ptr, S<Impl>() ); }
    template<class P> static HaD SimdVec         load_stream           ( const P &ptr ) { return internal::load_stream( _chk_ptr( ptr ), S<Impl>() ); } ///< P::alignment, P::offset, data.get()

    // the lanes of a set, and NOT ONE BYTE OUTSIDE THEM -- the load and store for the tail of a
    // buffer. `n` alone means the first `n` lanes. Loaded lanes outside the set are unspecified.
    template<LaneSet S> static HaD SimdVec       load_partial          ( const T *ptr, const S &s ) { return internal::load_partial( ptr, s, asimd::S<Impl>() ); }
    static HaD SimdVec                           load_partial          ( const T *ptr, int n ) { return load_partial( ptr, LaneRange<0>( n ) ); }
    template<LaneSet S> static HaD void          store_partial         ( T *ptr, const SimdVec &vec, const S &s ) { internal::store_partial( ptr, vec.impl, s ); }
    static HaD void                              store_partial         ( T *ptr, const SimdVec &vec, int n ) { store_partial( ptr, vec, LaneRange<0>( n ) ); }
    template<LaneSet S> HaD void                 store_partial         ( T *ptr, const S &s ) const { internal::store_partial( ptr, impl, s ); }
    HaD void                                     store_partial         ( T *ptr, int n ) const { store_partial( ptr, LaneRange<0>( n ) ); }

    static HaD void                              init_unaligned        ( T *ptr, const SimdVec &vec ) { internal::init_unaligned( ptr, vec.impl ); }
    static HaD void                              init_aligned          ( T *ptr, const SimdVec &vec ) { internal::init_aligned( ptr, vec.impl ); }
    template<class P> static HaD void            init                  ( const P &ptr, const SimdVec &vec ) { internal::init( _chk_ptr( ptr ), vec.impl ); }

    static HaD void                              init_unaligned_stream ( T *ptr, const SimdVec &vec ) { internal::init_unaligned( ptr, vec.impl ); }
    static HaD void                              init_aligned_stream   ( T *ptr, const SimdVec &vec ) { internal::init_aligned_stream( ptr, vec.impl ); }
    template<class P> static HaD void            init_stream           ( const P &ptr, const SimdVec &vec ) { internal::init_stream( _chk_ptr( ptr ), vec.impl ); }

    static HaD void                              store_unaligned       ( T *ptr, const SimdVec &vec ) { internal::store_unaligned( ptr, vec.impl ); }
    static HaD void                              store_aligned         ( T *ptr, const SimdVec &vec ) { internal::store_aligned( ptr, vec.impl ); }
    template<class P> static HaD void            store                 ( const P &ptr, const SimdVec &vec ) { internal::store( _chk_ptr( ptr ), vec.impl ); }

    static HaD void                              store_unaligned_stream( T *ptr, const SimdVec &vec ) { internal::store_unaligned( ptr, vec.impl ); }
    static HaD void                              store_aligned_stream  ( T *ptr, const SimdVec &vec ) { internal::store_aligned_stream( ptr, vec.impl ); }
    template<class P> static HaD void            store_stream          ( const P &ptr, const SimdVec &vec ) { internal::store_stream( _chk_ptr( ptr ), vec.impl ); }

    // dynamic versions of load/store
    HaD void                                     store_unaligned       ( T *ptr ) const { store_unaligned( ptr, *this ); }
    HaD void                                     store_aligned         ( T *ptr ) const { store_aligned( ptr, *this ); }
    template<class P> HaD void                   store                 ( const P &ptr ) const { _chk_ptr( ptr ); store( ptr, *this ); }

    HaD void                                     store_unaligned_stream( T *ptr ) const { store_unaligned_stream( ptr, *this ); }
    HaD void                                     store_aligned_stream  ( T *ptr ) const { store_aligned_stream( ptr, *this ); }
    template<class P> HaD void                   store_stream          ( const P &ptr ) const { _chk_ptr( ptr ); store_stream( ptr, *this ); }

    HaD void                                     init_unaligned        ( T *ptr ) const { init_unaligned( ptr, *this ); }
    HaD void                                     init_aligned          ( T *ptr ) const { init_aligned( ptr, *this ); }
    template<class P> HaD void                   init                  ( const P &ptr ) const { init( ptr, *this ); }

    HaD void                                     init_unaligned_stream ( T *ptr ) const { init_unaligned( ptr, *this ); }
    HaD void                                     init_aligned_stream   ( T *ptr ) const { init_aligned_stream( ptr, *this ); }
    template<class P> HaD void                   init_stream           ( const P &ptr ) const { init_stream( ptr, *this ); }

    // scatter/gather
    template<class G,class V> static HaD void    scatter               ( G *ptr, const V &ind, const SimdVec &vec ) { internal::scatter( ptr, ind.impl, vec.impl ); }
    template<class G,class V> static HaD SimdVec gather                ( const G *data, const V &ind ) { return internal::gather( data, ind.impl, S<Impl>() ); }

    // selection
    /// Reading a lane yields a VALUE. A `vector_size` lane is not an object one can point at --
    /// see the note on `at` in SimdVecImpl_Generic.h -- so writing one goes through a proxy that
    /// keeps `v[ i ] = x` working without ever forming a `T &`.
    struct LaneProxy {
        HaD operator T   () const { return internal::at( *impl, i ); }
        HaD LaneProxy &operator=( T value ) { internal::set_at( *impl, i, value ); return *this; }
        Impl *impl; int i;
    };

    HaD T                                        operator[]            ( int i ) const { return internal::at( impl, i ); }
    HaD LaneProxy                                operator[]            ( int i ) { return { &impl, i }; }
    HaD auto                                     sub_vec               ( N<size_> ) const { return *this; }
    HaD auto&                                    sub_vec               ( N<size_> ) { return *this; }
    template<int s> HaD auto                     sub_vec               ( N<s> ) const { return SimdVec<T,size_/2,Arch>( impl.data.split.v0 ).sub_vec( N<s>() ); }
    template<int s> HaD auto&                    sub_vec               ( N<s> ) { return reinterpret_cast<SimdVec<T,size_/2,Arch> &>( impl.data.split.v0 ).sub_vec( N<s>() ); }
    HaD const T*                                 begin                 () const { return internal::lane_ptr( impl ); }
    HaD const T*                                 end                   () const { return begin() + size(); }

    // arithmetic operators
    HaD SimdVec                                  operator<<            ( const SimdVec &that ) const { return internal::sll( impl, that.impl ); }
    HaD SimdVec                                  operator&             ( const SimdVec &that ) const { return internal::anb( impl, that.impl ); }
    HaD SimdVec                                  operator+             ( const SimdVec &that ) const { return internal::add( impl, that.impl ); }
    HaD SimdVec                                  operator-             ( const SimdVec &that ) const { return internal::sub( impl, that.impl ); }
    HaD SimdVec                                  operator*             ( const SimdVec &that ) const { return internal::mul( impl, that.impl ); }
    HaD SimdVec                                  operator/             ( const SimdVec &that ) const { return internal::div( impl, that.impl ); }

    // comparison: return an Op_... that can be converted to a SimdBool or a SimdVec
    HaD auto                                     operator>             ( const SimdVec &that ) const { return internal::gt ( impl, that.impl );  }
    HaD auto                                     operator<             ( const SimdVec &that ) const { return internal::lt ( impl, that.impl );  }

    // self arithmetic operators
    HaD SimdVec&                                 operator+=            ( const auto &that ) { *this = *this + that; return *this; }
    HaD SimdVec&                                 operator-=            ( const auto &that ) { *this = *this - that; return *this; }
    HaD SimdVec&                                 operator*=            ( const auto &that ) { *this = *this * that; return *this; }
    HaD SimdVec&                                 operator/=            ( const auto &that ) { *this = *this / that; return *this; }

    HaD T                                        sum                   () const { return internal::horizontal_sum( impl ); }

    // the arithmetic restricted to a lane set -- see `LaneSet.h`. Lanes outside are unspecified.
    template<LaneSet S> HaD SimdVec              add                   ( const SimdVec &b, const S &s ) const { return asimd::add( *this, b, s ); }
    template<LaneSet S> HaD SimdVec              sub                   ( const SimdVec &b, const S &s ) const { return asimd::sub( *this, b, s ); }
    template<LaneSet S> HaD SimdVec              mul                   ( const SimdVec &b, const S &s ) const { return asimd::mul( *this, b, s ); }
    template<LaneSet S> HaD SimdVec              div                   ( const SimdVec &b, const S &s ) const { return asimd::div( *this, b, s ); }
    template<LaneSet S> HaD SimdVec              min                   ( const SimdVec &b, const S &s ) const { return asimd::min( *this, b, s ); }
    template<LaneSet S> HaD SimdVec              max                   ( const SimdVec &b, const S &s ) const { return asimd::max( *this, b, s ); }
    template<LaneSet S> HaD SimdVec              fma                   ( const SimdVec &b, const SimdVec &c, const S &s ) const { return asimd::fma( *this, b, c, s ); }
    template<LaneSet S> HaD T                    sum                   ( const S &s ) const { return asimd::sum( *this, s ); }

    // lane rotations -- see `SimdOps.h`. `k` and `n` are each an `int` or an `N<>`.
    template<class K,class NN> HaD SimdVec       rotate_lanes          ( K k, NN n ) const { return asimd::rotate_lanes( *this, k, n ); }
    template<class K> HaD SimdVec                rotate_lanes          ( K k ) const { return asimd::rotate_lanes( *this, k ); }
    template<int K> HaD SimdVec                  ext_lanes             ( const SimdVec &b, N<K> ) const { return asimd::ext_lanes( *this, b, N<K>() ); }

    template<class P> static const P&            _chk_ptr              ( const P &ptr ) { static_assert( P::alignment && ( P::offset != P::alignment ), "this method is expecting a asimd::Ptr<> like object (with `alignment`, `offset` static attributes and a `get` method)" ); return ptr; }

    Impl                                         impl;                 ///<
};

#define SIMD_VEC_IMPL_CMP_OP( NAME, OP ) \
    template<class T,int size,class Arch> auto as_a_simd_bool( const internal::Op_##NAME<T,size,Arch> &op ) { return simd_bool_from_simd_bool_impl( internal::NAME##_as_a_simd_bool( op.a, op.b ) ); } \
    template<class T,int size,class Arch> auto as_a_simd_vec( const internal::Op_##NAME<T,size,Arch> &op ) { using P = typename PI_<8*sizeof(T)>::T; return SimdVec<P,size,Arch>( internal::NAME##_as_a_simd_vec( op.a, op.b, S<internal::SimdVecImpl<P,size,Arch>>() ) ); } \
    template<class T,int size,class Arch> bool any( const internal::Op_##NAME<T,size,Arch> &op ) { return any( as_a_simd_bool( op ) ); } \
    template<class T,int size,class Arch> bool all( const internal::Op_##NAME<T,size,Arch> &op ) { return all( as_a_simd_bool( op ) ); } \

SIMD_VEC_IMPL_CMP_OP( lt, < )
SIMD_VEC_IMPL_CMP_OP( gt, > )

#undef SIMD_VEC_IMPL_CMP_OP

template<class T,int size,class Arch> HaD
SimdVec<T,size,Arch> min( const SimdVec<T,size,Arch> &a, const SimdVec<T,size,Arch> &b ) { return internal::min( a.impl, b.impl ); }

template<class T,int size,class Arch> HaD
SimdVec<T,size,Arch> max( const SimdVec<T,size,Arch> &a, const SimdVec<T,size,Arch> &b ) { return internal::max( a.impl, b.impl ); }

} // namespace asimd

#ifndef ASIMD_Integer_H
#define ASIMD_Integer_H

#include <algorithm>
#include "N.h" // declares `Int` (with its default argument) to break the include cycle

namespace asimd {

/**
  Pointer with integer value guaranteed to be writable as
    `alignment * n + offset` with `n` an integer
*/
template<class T_,int _aligment,int _offset>
struct Int {
    static constexpr int               alignment = _aligment;
    static constexpr int               offset    = _offset;
    using                              T         = T_;

    /**/                               Int       ( T value = offset );

    T                                  get       () const;
    operator                           T         () const;

    Int&                               al_incr   () { value += alignment; return *this; }
    Int&                               al_decr   () { value -= alignment; return *this; }
    void                               set       ( T v ) { value = v; }

    template<class D> bool             operator> ( const D &that ) const { return value >  that; }
    template<class D> bool             operator>=( const D &that ) const { return value >= that; }
    template<class D> bool             operator< ( const D &that ) const { return value <  that; }
    template<class D> bool             operator<=( const D &that ) const { return value <= that; }
    template<class D> bool             operator==( const D &that ) const { return value == that; }

    T                                  value;
};

} // namespace asimd

template<class T,int a,int o,class U,int b,int p> auto operator+ ( const asimd::Int<T,a,o> &, const asimd::Int<U,b,p> & );
template<class T,int a,int o,class U,int b,int p> auto operator- ( const asimd::Int<T,a,o> &, const asimd::Int<U,b,p> & );
template<class T,int a,int o,class U,int b,int p> auto operator* ( const asimd::Int<T,a,o> &, const asimd::Int<U,b,p> & );

template<class T,int a,int o,int d>               auto operator+ ( const asimd::Int<T,a,o> &, asimd::N<d> );
template<class T,int a,int o,int d>               auto operator- ( const asimd::Int<T,a,o> &, asimd::N<d> );
template<class T,int a,int o,int d>               auto operator* ( const asimd::Int<T,a,o> &, asimd::N<d> );
template<class T,int a,int o>                     auto operator* ( const asimd::Int<T,a,o> &, asimd::N<0> );

template<int d,class T,int a,int o>               auto operator+ ( asimd::N<d>, const asimd::Int<T,a,o> & );
template<int d,class T,int a,int o>               auto operator- ( asimd::N<d>, const asimd::Int<T,a,o> & );
template<int d,class T,int a,int o>               auto operator* ( asimd::N<d>, const asimd::Int<T,a,o> & );
template<class T,int a,int o>                     auto operator* ( asimd::N<0>, const asimd::Int<T,a,o> & );

template<class T,int a,int o,class D>             auto operator+ ( const asimd::Int<T,a,o> &, const D & );
template<class T,int a,int o,class D>             auto operator- ( const asimd::Int<T,a,o> &, const D & );
template<class T,int a,int o,class D>             auto operator* ( const asimd::Int<T,a,o> &, const D & );

template<class D,class T,int a,int o>             auto operator+ ( const D &, const asimd::Int<T,a,o> & );
template<class D,class T,int a,int o>             auto operator- ( const D &, const asimd::Int<T,a,o> & );
template<class D,class T,int a,int o>             auto operator* ( const D &, const asimd::Int<T,a,o> & );

#include "Int.tcc"

#endif // ASIMD_Integer_H

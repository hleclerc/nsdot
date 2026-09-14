#include "Int.h"
#include "gcd.h"

namespace asimd {

template<class T,int a,int o>
Int<T,a,o>::Int( T value ) : value( value ) {
}

template<class T,int a,int o>
Int<T,a,o>::operator T() const {
    return value;
}

template<class T,int a,int o>
T Int<T,a,o>::get() const {
    return value;
}

} // namespace asimd

//-------------------------------------------------------
template<class T,int a,int o,class U,int b,int p>
auto operator+( const asimd::Int<T,a,o> &f, const asimd::Int<U,b,p> &that ) {
    constexpr int na = asimd::gcd( a, b );
    constexpr int no = ( ( o + p ) % na + na ) % na;
    return asimd::Int<T,na,no>( f.value + that.get() );
}

template<class T,int a,int o,class U,int b,int p>
auto operator-( const asimd::Int<T,a,o> &f, const asimd::Int<U,b,p> &that ) {
    constexpr int na = asimd::gcd( a, b );
    constexpr int no = ( ( o - p ) % na + na ) % na;
    return asimd::Int<T,na,no>( f.value - that.get() );
}

template<class T,int a,int o,class U,int b,int p>
auto operator*( const asimd::Int<T,a,o> &f, const asimd::Int<U,b,p> &that ) {
    constexpr int na = asimd::gcd( asimd::gcd( a * b, o * b ), a * p );
    constexpr int no = o * p;
    return asimd::Int<T,na,no>( f.value * that.get() );
}

//-------------------------------------------------------
template<class T,int a,int o,int d>
auto operator+( const asimd::Int<T,a,o> &f, asimd::N<d> ) {
    constexpr int no = ( ( o + d ) % a + a ) % a;
    return asimd::Int<T,a,no>( f.value + d );
}

template<class T,int a,int o,int d>
auto operator-( const asimd::Int<T,a,o> &f, asimd::N<d> ) {
    constexpr int no = ( ( o - d ) % a + a ) % a;
    return asimd::Int<T,a,no>( f.value - d );
}

template<class T,int a,int o,int d>
auto operator*( const asimd::Int<T,a,o> &f, asimd::N<d> ) {
    return asimd::Int<T,a*d,o*d>( f.value * d );
}

template<class T,int a,int o>
auto operator*( const asimd::Int<T,a,o> &, asimd::N<0> ) {
    return asimd::N<0>();
}

//-------------------------------------------------------
template<int d,class T,int a,int o>
auto operator+( asimd::N<d>, const asimd::Int<T,a,o> &f ) {
    constexpr int no = ( ( o + d ) % a + a ) % a;
    return asimd::Int<T,a,no>( f.value + d );
}

template<int d,class T,int a,int o>
auto operator-( asimd::N<d>,const asimd::Int<T,a,o> &f ) {
    constexpr int no = ( ( d - o ) % a + a ) % a;
    return asimd::Int<T,a,no>( d - f.value );
}

template<int d,class T,int a,int o>
auto operator*( asimd::N<d>, const asimd::Int<T,a,o> &f ) {
    return asimd::Int<T,a*d,o*d>( f.value * d );
}

template<class T,int a,int o>
auto operator*( asimd::N<0>, const asimd::Int<T,a,o> & ) {
    return asimd::N<0>();
}

//-------------------------------------------------------
template<class T,int a,int o,class D>
auto operator+( const asimd::Int<T,a,o> &f, const D &that ) {
    using R = decltype( f.value + that );
    return asimd::Int<R,1,0>( f.value + that );
}

template<class T,int a,int o,class D>
auto operator-( const asimd::Int<T,a,o> &f, const D &that ) {
    using R = decltype( f.value - that );
    return asimd::Int<R,1,0>( f.value - that );
}

template<class T,int a,int o,class D>
auto operator*( const asimd::Int<T,a,o> &f, const D &that ) {
    using R = decltype( f.value * that );
    constexpr int na = asimd::gcd( a, o );
    return asimd::Int<R,na,0>( f.value * that );
}

//-------------------------------------------------------
template<class D,class T,int a,int o>
auto operator+( const D &that, const asimd::Int<T,a,o> &f ) {
    using R = decltype( f.value + that );
    return asimd::Int<R,1,0>( f.value + that );
}

template<class D,class T,int a,int o>
auto operator-( const D &that, const asimd::Int<T,a,o> &f ) {
    using R = decltype( that - f.value );
    return asimd::Int<R,1,0>( that - f.value );
}

template<class D,class T,int a,int o>
auto operator*( const D &that, const asimd::Int<T,a,o> &f ) {
    using R = decltype( f.value * that );
    constexpr int na = asimd::gcd( a, o );
    return asimd::Int<R,na,0>( f.value * that );
}

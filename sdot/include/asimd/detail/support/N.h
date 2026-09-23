#ifndef ASIMD_PAREX_N_H
#define ASIMD_PAREX_N_H

#include <cstdint>
#include <cstddef>

namespace asimd {

/// FORWARD DECLARATION, and it is load-bearing. `N.h` and `Int.h` include each other: entering
/// through `Int.h` -- which is what `Ptr.h` does -- left `Int` undeclared by the time the
/// operators at the bottom of this file named it, and `Ptr.h` had not compiled since.
/// The default argument lives HERE, so `Int.h` must not repeat it.
template<class T_,int _aligment,int _offset=0> struct Int;

/**
  like std::integral_constant
*/
template<int n>
struct N {
    enum {                  value          = n };

    constexpr operator      std::size_t    () const { return n; }
    constexpr operator      bool           () const { return n; }
    constexpr operator      int            () const { return n; }

    template<class OS> void write_to_stream( OS &os ) const { os << n; }

    constexpr int           get            () const { return n; }

    template<class T> N     operator=      ( const T & ) const { return *this; }
    template<int m> N       operator=      ( N<m> ) const { static_assert( n == m, "" ); return *this; }

    constexpr N<-n>         operator-      () const { return {}; }
};

template<class T> struct IsN { enum { value = false }; };
template<int n> struct IsN<N<n>> { enum { value = true }; };
template<int n> struct IsN<N<n> &> { enum { value = true }; };
template<int n> struct IsN<const N<n> &> { enum { value = true }; };

template<class T> constexpr bool isN( const T & ) { return false; }
template<int n> constexpr bool isN( N<n> ) { return true; }

} // namespace asimd

#include "Int.h"

template<int n,int m> constexpr auto operator+ ( asimd::N<n>, asimd::N<m> ) { return asimd::N<n+m   >{}; }
template<int n,int m> constexpr auto operator- ( asimd::N<n>, asimd::N<m> ) { return asimd::N<n-m   >{}; }
template<int n,int m> constexpr auto operator* ( asimd::N<n>, asimd::N<m> ) { return asimd::N<n*m   >{}; }
template<int n,int m> constexpr auto operator/ ( asimd::N<n>, asimd::N<m> ) { return asimd::N<n/m   >{}; }
template<int n,int m> constexpr auto operator< ( asimd::N<n>, asimd::N<m> ) { return asimd::N<(n< m)>{}; }
template<int n,int m> constexpr auto operator<=( asimd::N<n>, asimd::N<m> ) { return asimd::N<(n<=m)>{}; }
template<int n,int m> constexpr auto operator> ( asimd::N<n>, asimd::N<m> ) { return asimd::N<(n> m)>{}; }
template<int n,int m> constexpr auto operator>=( asimd::N<n>, asimd::N<m> ) { return asimd::N<(n>=m)>{}; }
template<int n,int m> constexpr auto operator==( asimd::N<n>, asimd::N<m> ) { return asimd::N<(n==m)>{}; }
template<int n,int m> constexpr auto operator&&( asimd::N<n>, asimd::N<m> ) { return asimd::N<(n&&m)>{}; }

template<int n,class T> constexpr auto operator+ ( asimd::N<n>, const T &val ) { return n +  val; }
template<int n,class T> constexpr auto operator- ( asimd::N<n>, const T &val ) { return n -  val; }
template<int n,class T> constexpr auto operator* ( asimd::N<n>, const T &val ) { return asimd::Int<T,n>( n *  val ); }
template<int n,class T> constexpr auto operator/ ( asimd::N<n>, const T &val ) { return n /  val; }
template<int n,class T> constexpr bool operator< ( asimd::N<n>, const T &val ) { return n <  val; }
template<int n,class T> constexpr bool operator<=( asimd::N<n>, const T &val ) { return n <= val; }
template<int n,class T> constexpr bool operator> ( asimd::N<n>, const T &val ) { return n >  val; }
template<int n,class T> constexpr bool operator>=( asimd::N<n>, const T &val ) { return n >= val; }
template<int n,class T> constexpr bool operator==( asimd::N<n>, const T &val ) { return n == val; }
template<int n,class T> constexpr bool operator&&( asimd::N<n>, const T &val ) { return n && val; }

template<class T,int m> constexpr auto operator+ ( const T &val, asimd::N<m> ) { return val +  m; }
template<class T,int m> constexpr auto operator- ( const T &val, asimd::N<m> ) { return val -  m; }
template<class T,int m> constexpr auto operator* ( const T &val, asimd::N<m> ) { return asimd::Int<T,m>( val *  m ); }
template<class T,int m> constexpr auto operator/ ( const T &val, asimd::N<m> ) { return val /  m; }
template<class T,int m> constexpr bool operator< ( const T &val, asimd::N<m> ) { return val <  m; }
template<class T,int m> constexpr bool operator<=( const T &val, asimd::N<m> ) { return val <= m; }
template<class T,int m> constexpr bool operator> ( const T &val, asimd::N<m> ) { return val >  m; }
template<class T,int m> constexpr bool operator>=( const T &val, asimd::N<m> ) { return val >= m; }
template<class T,int m> constexpr bool operator==( const T &val, asimd::N<m> ) { return val == m; }
template<class T,int m> constexpr bool operator&&( const T &val, asimd::N<m> ) { return val && m; }

template<class T,int m> auto &operator+=( T &val, asimd::N<m> ) { val += m; return val; }
template<class T,int m> auto &operator-=( T &val, asimd::N<m> ) { val -= m; return val; }
template<class T,int m> auto &operator*=( T &val, asimd::N<m> ) { val *= m; return val; }
template<class T,int m> auto &operator/=( T &val, asimd::N<m> ) { val /= m; return val; }

#endif // ASIMD_PAREX_N_H

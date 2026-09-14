#pragma once

#include "Int.h"

namespace asimd {

template<class T,class U>
constexpr T ceil( const T &a, const U &m ) {
    return ( a + m - 1 ) / m * m;
}

template<class T,int m>
constexpr auto ceil( const T &a, N<m> ) {
    auto res = ( a + m - 1 ) / m * m;
    return Int<decltype(res),m>( res );
}

template<int a,int m>
constexpr auto ceil( N<a>, N<m> ) {
    constexpr int res = ( a + m - 1 ) / m * m;
    return N<res>();
}

} // namespace asimd

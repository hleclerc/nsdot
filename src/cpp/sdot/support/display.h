#pragma once

#include "common_macros.h"
#include <iostream>

namespace sdot {

T_T void display( std::ostream &os, const T &value );

// priority tag for ordered SFINAE dispatch (replaces C++20 requires{} chains)
template<int N> struct _disp_priority : _disp_priority<N-1> {};
template<>      struct _disp_priority<0> {};

// sentinel callable: accepted by any for_each_item / for_each to probe availability
struct _DisplaySentinel { T_T void operator()( const T & ) const {} };

// dispatch overloads — highest priority wins (tried in order 4 → 0)
T_T auto _display( std::ostream &os, const T &v, _disp_priority<4> )
    -> decltype( v.display( os ) )
    { v.display( os ); }

T_T auto _display( std::ostream &os, const T &v, _disp_priority<3> )
    -> decltype( v.for_each_item( _DisplaySentinel{} ), void() ) {
    std::size_t cpt = 0;
    v.for_each_item( [&]( const auto &item ) {
        display( os << ( cpt++ ? ", " : "" ), item );
    } );
}

T_T auto _display( std::ostream &os, const T &v, _disp_priority<2> )
    -> decltype( v.shape().size(), void() ) {
    constexpr int rank = DECAYED_TYPE_OF( v.shape().size() )::value;
    const auto shape   = v.shape();
    if constexpr ( rank == 0 ) {
        os << v.value();
    } else if constexpr ( rank == 1 ) {
        for ( std::size_t i = 0; i < shape[ 0 ]; ++i )
            display( os << ( i ? ", " : "" ), v[ i ] );
    } else {
        for ( std::size_t i = 0; i < shape[ 0 ]; ++i )
            display( os << "\n  ", v( i ) );
    }
}

T_T auto _display( std::ostream &os, const T &v, _disp_priority<1> )
    -> decltype( os << v, void() )
    { os << v; }

T_T auto _display( std::ostream &os, const T &v, _disp_priority<0> )
    -> decltype( std::begin( v ), void() ) {
    auto iter = std::begin( v );
    if ( iter != std::end( v ) ) {
        os << *iter;
        while ( ++iter != std::end( v ) )
            os << ", " << *iter;
    } else
        os << "[]";
}

T_T void _display( std::ostream &os, const T &v, ... ) {
    os << "TODO: display of " << typeid( v ).name();
}

T_T void display( std::ostream &os, const T &value ) {
    _display( os, value, _disp_priority<4>{} );
}

} // namespace sdot

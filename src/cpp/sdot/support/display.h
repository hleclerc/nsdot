#pragma once

#include "util/Priority.h"
#include "common_macros.h"
#include <iostream>

namespace sdot {

void display( std::ostream &os, const auto &value );

// dispatch overloads — highest priority wins (tried in order 4 → 0)
auto display( std::ostream &os, const auto &v, Priority<5> ) -> decltype( v.display( os ) ) {
    v.display( os );
}

auto display( std::ostream &os, const auto &v, Priority<4> ) -> decltype( v.for_each_item( []( const auto & ) {} ), void() ) {
    std::size_t cpt = 0;
    v.for_each_item( [&]( const auto &item ) {
        display( os << ( cpt++ ? ", " : "" ), item );
    } );
}

auto display( std::ostream &os, const auto &v, Priority<3> ) -> decltype( v.shape().size(), void() ) {
    constexpr int rank = DECAYED_TYPE_OF( v.shape().size() )::value;
    const auto shape = v.shape();
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

auto display( std::ostream &os, const auto &v, Priority<2> ) -> decltype( os << v, void() ) {
    os << v;
}

auto display( std::ostream &os, const auto &v, Priority<1> ) -> decltype( std::begin( v ), void() ) {
    auto iter = std::begin( v );
    if ( iter != std::end( v ) ) {
        os << *iter;
        while ( ++iter != std::end( v ) )
            os << ", " << *iter;
    } else
        os << "[]";
}

void display( std::ostream &os, const auto &value, Priority<0> ) {
    os << "TODO: display of " << typeid( value ).name();
}

void display( std::ostream &os, const auto &value ) {
    display( os, value, Priority<5>{} );
}

} // namespace sdot

#pragma once

#include "apply_values.h" // FORWARD

namespace sdot {

auto for_each_item( auto &&list, auto &&func ) requires ( requires { list.for_each_item( FORWARD( func ) ); } || requires { apply_values( FORWARD( list ), []( auto &&... ) {} ); } )  {
    if constexpr ( requires { list.for_each_item( FORWARD( func ) ); } )
        list.for_each_item( FORWARD( func ) );
    else {
        apply_values( FORWARD( list ), [&]( auto &&...values ) {
            ( func( values ), ... );
        } );
    }
}

} // namespace sdot

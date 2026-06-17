#pragma once

#include "string/read_arg_name.h"
#include "common_macros.h" // HD, GD, DECAYED_TYPE_OF
#include "display.h"
#include <cstddef>         // std::size_t (common_types.h includes us -> can't include it back)
#include <cstdio>          // printf (device + host)
#include <mutex>

namespace sdot {

// ----------------- host path -----------------

T_VT void __print_with_mutex( std::ostream &os, std::string_view arg_names, const T &...arg_values ) {
    static std::mutex m;
    m.lock();

    // write
    int cpt = 0;
    auto get_item = [&]( const auto &arg_value ) {
        if ( cpt++ )
            os << "\t";
        display( os << "\033[90m" << read_arg_name( arg_names ) << ":\033[0m ", arg_value );
    };
    ( get_item( arg_values ), ... );
    os << std::endl;

    //
    m.unlock();
}

// ----------------- device path -----------------
// No std::ostream on the device, so values are formatted with printf. Mirrors display.h's dispatch
// (member display, scalars, compile-time constants, item/value ranges) on a best-effort basis.
#ifdef __CUDACC__
GD void __display_device( const auto &value ) {
    using V = DECAYED_TYPE_OF( value );
    if constexpr ( requires { value.display_device(); } ) {
        value.display_device();
    } else if constexpr ( std::is_same_v<V,bool> ) {
        printf( value ? "true" : "false" );
    } else if constexpr ( std::is_floating_point_v<V> ) {
        printf( "%g", double( value ) );
    } else if constexpr ( std::is_integral_v<V> ) {
        printf( "%lld", (long long)( value ) );
    } else if constexpr ( requires { V::value; } ) { // compile-time constant, e.g. Ct<T,v>
        __display_device( ( decltype( V::value ) )V::value );
    } else if constexpr ( requires { DECAYED_TYPE_OF( value.shape().size() )::value; } ) { // TensorView: rank-aware
        constexpr int rank = DECAYED_TYPE_OF( value.shape().size() )::value;
        if constexpr ( rank == 0 ) {
            __display_device( value.value() );                                    // rank-0: never operator[] (would squeeze an empty shape)
        } else if constexpr ( rank == 1 ) {
            printf( "[" );
            for ( std::size_t i = 0; i < std::size_t( value.shape()[ 0 ] ); ++i ) { if ( i ) printf( ", " ); __display_device( value[ i ] ); }
            printf( "]" );
        } else {
            for ( std::size_t i = 0; i < std::size_t( value.shape()[ 0 ] ); ++i ) { printf( "\n  " ); __display_device( value( i ) ); }
        }
    } else if constexpr ( requires { value.apply_values( []( auto... ) {} ); } ) { // Tuple
        int cpt = 0;
        printf( "(" );
        value.apply_values( [&]( const auto &...vs ) { ( ( printf( cpt++ ? ", " : "" ), __display_device( vs ) ), ... ); } );
        printf( ")" );
    } else if constexpr ( requires { value.for_each_item( []( const auto & ) {} ); } ) {
        int cpt = 0;
        printf( "[" );
        value.for_each_item( [&]( const auto &item ) { if ( cpt++ ) printf( ", " ); __display_device( item ); } );
        printf( "]" );
    } else if constexpr ( requires { value.size(); value[ std::size_t( 0 ) ]; } ) { // indexable (e.g. Vector)
        int cpt = 0;
        printf( "[" );
        for ( std::size_t i = 0; i < std::size_t( value.size() ); ++i ) { if ( cpt++ ) printf( ", " ); __display_device( value[ i ] ); }
        printf( "]" );
    } else {
        printf( "<?>" );
    }
}
#endif

// ----------------- unified entry -----------------
// HD so the same info(...) works in host and device code (the macro expands identically in both);
// the body picks std::cout (host) or printf (device, prefixed with block/thread id) at compile time.
T_VT HD void __info( const char *arg_names, const T &...arg_values ) {
#ifdef __CUDA_ARCH__
    printf( "\033[90m[gpu blk=%d thr=%d]\033[0m %s = ", blockIdx.x, threadIdx.x, arg_names );
    int cpt = 0;
    ( ( printf( cpt++ ? " | " : "" ), __display_device( arg_values ) ), ... );
    printf( "\n" );
#else
    __print_with_mutex( std::cout, arg_names, arg_values... );
#endif
}

#define info( ... ) sdot::__info( #__VA_ARGS__, __VA_ARGS__ )

} // namespace sdot

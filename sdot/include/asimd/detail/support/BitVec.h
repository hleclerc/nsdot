#pragma once

#include "common_types.h"
#include "N.h"

namespace asimd {

template<int nb_items>
class BitVec {
public:
    template      <int off>
    void          for_each_int( const auto &func, N<off> ) const; ///< func( value, mask ), starting from off
    template      <int off>
    void          for_each_int( const auto &func, N<off> ); ///< func( value, mask ), starting from off

    void          for_each_int( const auto &func ) const { for_each_int( func, N<0>() ); } ///< func( value, mask )
    void          for_each_int( const auto &func ) { for_each_int( func, N<0>() ); } ///< func( value, mask )

    void          set_values  ( bool a, bool b, bool c, bool d, bool e, bool f, bool g, bool h ) { data[ 0 ] = a + ( b << 1 ) + ( c << 2 ) + ( d << 3 ) + ( e << 4 ) + ( f << 5 ) + ( g << 6 ) + ( h << 7 ); }
    void          set_values  ( bool a, bool b, bool c, bool d ) { data[ 0 ] = a + ( b << 1 ) + ( c << 2 ) + ( d << 3 ); }
    void          set_values  ( bool a, bool b ) { data[ 0 ] = a + ( b << 1 ); }
    void          set_values  ( bool a ) { data[ 0 ] = a; }
 
    void          set_value   ( bool a ) { if ( a ) for_each_int( []( auto &v, auto m ) { v = m; } ); else for_each_int( []( auto &v, auto ) { v = 0; } ); }
 
    void          set_bit     ( int index, bool a = true ) { PI8 m = PI8( 1 ) << ( index % 8 ); if ( a ) data[ index / 8 ] |= m; else data[ index / 8 ] &= PI8( ~m ); }
 
    bool          operator[]  ( int index ) const { return data[ index / 8 ] & ( 1 << ( index % 8 ) ); }
    constexpr int size        () const { return nb_items; }

    bool          all         () const { bool res = 1; for_each_int( [&]( auto v, auto m ) { res &= ( v == m ); } ); return res; }
    /// the mask argument is unnamed on purpose: `any` does not need it, and naming it was the one
    /// `-Wextra` warning left in the build.
    bool          any         () const { bool res = 0; for_each_int( [&]( auto v, auto ) { res |= ( v != 0 ); } ); return res; }
    
private:
    PI8           data[ ( nb_items + 7 ) / 8 ];
};

template<int nb_items> template<int off>
void BitVec<nb_items>::for_each_int( const auto &func, N<off> ) const {
    static_assert( off % 8 == 0 ); // TODO
    
    if      constexpr( off + 64 <= nb_items ) { func( *reinterpret_cast<const PI64 *>( data + off / 8 ),  ~PI64( 0 ) ); for_each_int( func, N<off + 64>() ); }
    else if constexpr( off + 32 <= nb_items ) { func( *reinterpret_cast<const PI32 *>( data + off / 8 ),  ~PI32( 0 ) ); for_each_int( func, N<off + 32>() ); }
    else if constexpr( off + 16 <= nb_items ) { func( *reinterpret_cast<const PI16 *>( data + off / 8 ),  ~PI16( 0 ) ); for_each_int( func, N<off + 16>() ); }
    else if constexpr( off +  8 <= nb_items ) { func( *reinterpret_cast<const PI8  *>( data + off / 8 ), PI8( 0xFF ) ); for_each_int( func, N<off + 8 >() ); }
    else if constexpr( off +  1 <= nb_items ) { func( *reinterpret_cast<const PI8  *>( data + off / 8 ), PI8( 0xFF ) >> ( 8 - nb_items + off ) ); }
}

template<int nb_items> template<int off>
void BitVec<nb_items>::for_each_int( const auto &func, N<off> ) {
    static_assert( off % 8 == 0 ); // TODO
    
    if      constexpr( off + 64 <= nb_items ) { func( *reinterpret_cast<PI64 *>( data + off / 8 ),  ~PI64( 0 ) ); for_each_int( func, N<off + 64>() ); }
    else if constexpr( off + 32 <= nb_items ) { func( *reinterpret_cast<PI32 *>( data + off / 8 ),  ~PI32( 0 ) ); for_each_int( func, N<off + 32>() ); }
    else if constexpr( off + 16 <= nb_items ) { func( *reinterpret_cast<PI16 *>( data + off / 8 ),  ~PI16( 0 ) ); for_each_int( func, N<off + 16>() ); }
    else if constexpr( off +  8 <= nb_items ) { func( *reinterpret_cast<PI8  *>( data + off / 8 ), PI8( 0xFF ) ); for_each_int( func, N<off + 8 >() ); }
    else if constexpr( off +  1 <= nb_items ) { func( *reinterpret_cast<PI8  *>( data + off / 8 ), PI8( 0xFF ) >> ( 8 - nb_items + off ) ); }
}

} // namespace asimd

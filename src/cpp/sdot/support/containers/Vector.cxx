#pragma once

#include <algorithm>
#include <utility>

#include "../algorithms/for_each_item.h"
#include "Vector.h"

namespace sdot {

#define UTPH template<class T,int ct_size> HD
#define UTP template<class T,int ct_size>
#define DTP Vector<T,ct_size>

UTP template<class U,class E> HD DTP::Vector( const U &values ) {
    PI i = 0;
    sdot::for_each_item( values, [&]( const auto &item ) {
        if ( i < ct_size )
            new ( data() + i++ ) T( item );
    } );
    for(; i < ct_size; ++i )
        new ( data() + i ) T;
}

UTP T_VA HD DTP::Vector( FillWith, A &&...ctor_args ) {
    for( auto &v : *this )
        new ( &v ) T( ctor_args... );
}

UTPH DTP::Vector( Reserved ) {
}

UTPH DTP::Vector() {
    for( auto &v : *this )
        new ( &v ) T;
}

UTP T_VA HD DTP::Vector( Values, A &&...values ) : Vector( Reserved() ) {
    PI i = 0;
    auto append = [&]( auto &&value ) {
        new ( data() + i++ ) T( FORWARD( value ) );
    };
    ( append( FORWARD( values ) ), ... );
}

UTPH DTP::Vector( const Vector &that ) : Vector( Reserved() ) {
    for ( PI i = 0; i < that.size(); ++i )
        new ( data() + i ) T( that[ i ] );
}

UTPH DTP::Vector( Vector &&that ) noexcept {
    for ( PI i = 0; i < PI( ct_size ); ++i )
        new ( data() + i ) T( std::move( that[ i ] ) );
}

UTPH Vector<T,ct_size>& DTP::operator=( const Vector &that ) {
    if ( this != &that )
        for ( PI i = 0; i < size(); ++i )
            operator[]( i ) = that[ i ];
    return *this;
}

UTPH DTP& DTP::operator=( Vector &&that ) noexcept {
    if ( this != &that )
        for ( PI i = 0; i < PI( ct_size ); ++i )
            operator[]( i ) = std::move( that[ i ] );
    return *this;
}

UTPH DTP::~Vector() {
    for ( PI i = 0; i < size(); ++i )
        data()[ i ].~T();
}

UTPH const T& DTP::operator[]( PI index ) const {
    return data()[ index ];
}

UTPH T& DTP::operator[]( PI index ) {
    return data()[ index ];
}

// UTP bool DTP::operator<( const Vector &that ) const {
//     return std::ranges::lexicographical_compare( operator std::span<T>(), that.operator std::span<T>() );
// }

UTP T_U HD DTP DTP::with_func( U &&func ) {
    Vector res;
    for ( PI i = 0; i < ct_size; ++i )
        new ( res.data() + i ) T( func( i ) );
    return res;
}

UTPH DTP DTP::zeros() {
    return with_func( [] HD ( PI ) { return T( 0 ); } );
}

UTPH DTP DTP::ones() {
    return with_func( [] HD ( PI ) { return T( 1 ); } );
}

UTPH DTP DTP::with_value_at( PI index, T value ) {
    return with_func( [=] HD ( PI i ) { return i == index ? value : T( 0 ); } );
}

UTPH auto DTP::with_pushed_value( T value ) const {
    Vector<T,ct_size+1> res( Reserved{} );
    for( PI i = 0; i < size(); ++i )
        new ( res.data() + i ) T( operator[]( i ) );
    new ( res.data() + size() ) T( value );
    return res;
}

UTPH auto DTP::without_index( PI ind_to_remove ) const {
    Vector<T,ct_size-1> res( Reserved{} );
    for( PI i = 0; i < ind_to_remove; ++i )
        new ( res.data() + i ) T( operator[]( i ) );
    for( PI i = ind_to_remove + 1; i < size(); ++i )
        new ( res.data() + i - 1 ) T( operator[]( i ) );
    return res;
}

UTPH const T* DTP::data() const {
    return reinterpret_cast<const T *>( this->_storage );
}

UTPH T* DTP::data() {
    return reinterpret_cast<T *>( this->_storage );
}

UTPH auto DTP::begin() const {
    return data();
}

UTPH auto DTP::begin() {
    return data();
}

UTPH auto DTP::end() const {
    return data() + size();
}

UTPH auto DTP::end() {
    return data() + size();
}

UTPH PI DTP::arg_max() const {
    PI res = 0;
    for( PI i = 1; i < size(); ++i )
        if ( operator[]( res ) < operator[]( i ) )
            res = i;
    return res;
}

UTPH T DTP::max() const {
    T res = operator[]( 0 );
    for( PI i = 1; i < size(); ++i )
        if ( res < operator[]( i ) )
            res = operator[]( i );
    return res;
}

#undef UTPH
#undef UTP
#undef DTP

} // namespace sdot

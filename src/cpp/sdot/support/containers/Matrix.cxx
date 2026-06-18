#pragma once

#include "Matrix.h"
#include <cmath>

namespace sdot {

#define UTP  template<class T,int ct_size>
#define DTP  Matrix<T,ct_size>

UTP HD Matrix<T,ct_size-1> DTP::without_row_and_col( PI wr, PI wc ) const {
    Matrix<T,ct_size-1> res;
    for( PI r = 0; r < res.nb_rows(); ++r )
        for( PI c = 0; c < res.nb_cols(); ++c )
            res( r, c ) = operator()( r + ( r >= wr ), c + ( c >= wc ) );
    return res;
}

UTP T_U HD DTP DTP::with_func( U &&func ) {
    Matrix res;
    for( PI r = 0; r < ct_size; ++r )
        for( PI c = 0; c < ct_size; ++c )
            res( r, c ) = func( r, c );
    return res;
}

UTP HD DTP DTP::with_replaced_col( PI c, const Vec &col ) const {
    Matrix res = *this;
    for( PI r = 0; r < nb_rows(); ++r )
        res( r, c ) = col[ r ];
    return res;
}

UTP HD typename DTP::Vec DTP::diagonal() const {
    Vec res( Size(), nb_rows() );
    for( PI i = 0; i < nb_rows(); ++i )
        res[ i ] = operator()( i, i );
    return res;
}

UTP HD T DTP::determinant() const {
    if constexpr ( ct_size == 1 ) {
        return operator()( 0, 0 );
    } else {
        T sgn = 1, res = 0;
        for( PI r = 0; r < nb_rows(); ++r, sgn = -sgn )
            res += sgn * operator()( r, 0 ) * without_row_and_col( r, 0 ).determinant();
        return res;
    }
}

UTP HD DTP DTP::cholesky() const {
    const PI nd = nb_rows();
    Matrix L( nd );
    for ( PI i = 0; i < nd; ++i )
        for ( PI j = 0; j < nd; ++j )
            L( i, j ) = T( 0 );

    for ( PI j = 0; j < nd; ++j ) {
        // diagonal
        T s = operator()( j, j );
        for ( PI k = 0; k < j; ++k ) s -= L( j, k ) * L( j, k );
        L( j, j ) = std::sqrt( s );

        // column below diagonal
        const T inv_ljj = T( 1 ) / L( j, j );
        for ( PI i = j + 1; i < nd; ++i ) {
            T t = operator()( i, j );
            for ( PI k = 0; k < j; ++k ) t -= L( i, k ) * L( j, k );
            L( i, j ) = t * inv_ljj;
        }
    }
    return L;
}

UTP HD typename DTP::Vec DTP::solve( const Vec &vec ) const {
    T d = determinant();
    T sgn = 1;
    Vec res;
    for( PI c = 0; c < nb_rows(); ++c, sgn = -sgn )
        res[ c ] = with_replaced_col( c, vec ).determinant() / d;
    return res;
}

UTP HD typename DTP::Vec DTP::solve_ge( Vec b ) const {
    const PI n = nb_rows();
    Matrix A = *this;

    // forward elimination with partial pivoting
    for ( PI p = 0; p < n; ++p ) {
        PI pivot = p;
        for ( PI r = p + 1; r < n; ++r )
            if ( std::abs( A( r, p ) ) > std::abs( A( pivot, p ) ) )
                pivot = r;
        for ( PI c = p; c < n; ++c ) std::swap( A( p, c ), A( pivot, c ) );
        std::swap( b[ p ], b[ pivot ] );

        if ( A( p, p ) == T( 0 ) ) continue;  // zero pivot: degenerate row, leave as 0

        const T inv = T( 1 ) / A( p, p );
        for ( PI r = p + 1; r < n; ++r ) {
            const T factor = A( r, p ) * inv;
            for ( PI c = p + 1; c < n; ++c )
                A( r, c ) -= factor * A( p, c );
            b[ r ] -= factor * b[ p ];
        }
    }

    // back substitution (x initialised to 0 so zero-pivot rows stay 0)
    Vec x;
    for ( PI i = 0; i < n; ++i )
        x[ i ] = T( 0 );
    for ( PI p = n; p-- > 0; ) {
        if ( A( p, p ) == T( 0 ) )
            continue;
        T s = b[ p ];
        for ( PI q = p + 1; q < n; ++q )
            s -= A( p, q ) * x[ q ];
        x[ p ] = s / A( p, p );
    }
    return x;
}

UTP HD DTP DTP::inverse() const {
    const PI n = nb_rows();
    Matrix A = *this;
    Matrix inv = with_func( []( PI r, PI c ) -> T { return r == c ? T(1) : T(0); } );

    for ( PI p = 0; p < n; ++p ) {
        // partial pivot
        PI pivot = p;
        for ( PI r = p + 1; r < n; ++r )
            if ( std::abs( A( r, p ) ) > std::abs( A( pivot, p ) ) )
                pivot = r;
        for ( PI c = 0; c < n; ++c )
            std::swap( A( p, c ), A( pivot, c ) );
        for ( PI c = 0; c < n; ++c )
            std::swap( inv( p, c ), inv( pivot, c ) );

        if ( A( p, p ) == T( 0 ) )
            continue;

        const T inv_diag = T( 1 ) / A( p, p );
        for ( PI c = 0; c < n; ++c ) {
            A( p, c ) *= inv_diag;
            inv( p, c ) *= inv_diag;
        }

        for ( PI r = 0; r < n; ++r ) {
            if ( r == p ) continue;
            const T f = A( r, p );
            for ( PI c = 0; c < n; ++c ) {
                A( r, c ) -= f * A( p, c );
                inv( r, c ) -= f * inv( p, c );
            }
        }
    }
    return inv;
}


#undef UTPH
#undef UTP
#undef DTP

} // namespace sdot

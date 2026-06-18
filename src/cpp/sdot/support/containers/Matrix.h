#pragma once

#include "Vector.h"

namespace sdot {

// _size
template<class T,int ct_size>
class Matrix {
public:
    struct                EigenSystem             { Vector<T,ct_size> values; /* ascending order */ Matrix<T,ct_size> vectors; /* row i = eigenvector i */ };
    using                 value_type              = T;
    using                 Content                 = Vector<T,ct_size*ct_size>;
    using                 Vec                     = Vector<T,ct_size>;

    HD                    Matrix                  ( FillWith, T value ) : _content( FillWith(), value ) {}
    HD                    Matrix                  () {}

    T_U HD static Matrix  with_func               ( U &&func );

    HD const T&           operator()              ( PI r, PI c ) const { return _content[ r * ct_size + c ]; }
    HD T&                 operator()              ( PI r, PI c ) { return _content[ r * ct_size + c ]; }

    HD auto               without_row_and_col     ( PI r, PI c ) const -> Matrix<T,ct_size-1>;
    HD auto               with_replaced_col       ( PI c, const Vec &col ) const -> Matrix;
    EigenSystem           eigen_system            () const;
    HD T                  determinant             () const;
    Vec                   diagonal                () const;
    Matrix                cholesky                () const;  ///< returns L s.t. *this = L * L^T (H must be SPD)
    HD Vec                solve_ge                ( Vec b ) const;   ///< Gaussian elimination with partial pivoting; zero pivot → x[p]=0 (handles degenerate cells)
    Matrix                inverse                 () const;  ///< Gauss-Jordan on [A | I]; zero pivot row → identity row in result
    Vec                   solve                   ( const Vec &vec ) const;

    HD constexpr auto     nb_rows                 () const { return Ct<int,ct_size>(); }
    HD constexpr auto     nb_cols                 () const { return Ct<int,ct_size>(); }
    T_U HD constexpr auto shape                   ( U ) const { return Ct<int,ct_size>(); }

    HD const T*           data                    () const { return _content.data(); }
    HD T*                 data                    () { return _content.data(); }

    HD auto               begin                   () const { return _content.begin(); }
    HD auto               begin                   () { return _content.begin(); }
    HD auto               end                     () const { return _content.end(); }
    HD auto               end                     () { return _content.end(); }

    Content               _content;
};

} // namespace sdot

#include "Matrix.cxx"

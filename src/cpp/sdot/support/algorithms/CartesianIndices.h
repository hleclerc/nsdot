#pragma once

#include "../containers/Tuple.h" // tuple, product, with_appended_value, without_index, apply_values, 1_c
#include "../common_macros.h"
#include "min.h"

namespace sdot {

namespace detail {
    // unravel d'un index plat en multi-indice (ordre colonne) pour `shape`
    auto unravel_index( auto flat, auto &&res_so_far, auto &&shape ) {
        auto coeff = shape.apply_values( []( auto &&...values ) { return ( 1_c * ... * values ); } );
        auto res   = res_so_far.with_appended_value( flat / coeff );
        if constexpr ( DECAYED_TYPE_OF( shape )::ct_size )
            return unravel_index( flat % coeff, res, shape.without_index( 0_c ) );
        else
            return res;
    }
}

/// Ensemble des multi-indices d'une forme (cf. `CartesianIndices` en Julia).
/// Utilisée comme item_list de `run_parallel` : le kernel reçoit `item_list[flat]`, c.-à-d. le
/// multi-indice correspondant. Trivialement copiable (ne porte que la forme) -> capturable kernel.
template<class Shape>
struct CartesianIndices {
    auto size          () const { return product( shape ); }
    auto operator[]    ( auto flat ) const { return detail::unravel_index( flat, tuple(), shape.without_index( 0_c ) ); }
    auto make_available( auto &&/*queue*/, auto &&/*io_category*/, auto &&cont ) const { return cont( *this ); }

    /// intersection des parcours : min terme à terme des formes (mêmes rangs).
    auto intersection  ( const auto &other ) const {
        auto s = shape.apply_values( [&]( auto &&...as ) {
            return other.shape.apply_values( [&]( auto &&...bs ) {
                return tuple( min( as, bs )... );
            } );
        } );
        return CartesianIndices<DECAYED_TYPE_OF( s )>{ s };
    }

    Shape shape;
};

} // namespace sdot

#pragma once

#include "../kernels/make_avaiable.h"
#include "../kernels/transfer_cost.h"
#include "../common_types.h"

namespace sdot {

/// View on a `ShapeVar`: a tensor of counts (rank 0 for a scalar count, rank > 0 for ragged
/// ones) plus its capacity bound `max`.
///
/// Assigning writes the count(s) -- a scalar broadcasts over all elements of `view` -- and is
/// where the capacity check lives: a count must not exceed `max` (the reservation that sizes
/// the tensors depending on this ShapeVar). `max == -1` means "unbounded" (a pure output with
/// no reservation).
template<class View>
struct ShapeVarView {
    View view;
    SI   max;

    ShapeVarView &operator=( auto v ) {
        // TODO(error-buffer): when a written count exceeds `max`, record this ShapeVar in the
        // error buffer so the kernel can be relaunched with a larger capacity.
        view = v;
        return *this;
    }

    operator SI() const { return view.value(); }

    /// remise à zéro d'un compteur de sortie. Passe par la queue (donc par un kernel) : sur un
    /// device, le buffer n'est pas accessible depuis l'hôte -- ce n'est pas une boucle hôte.
    void fill_with( auto &&queue, auto v ) { view.fill_with( FORWARD( queue ), v ); }

    // comme argument de `run_parallel` : on rend la vue disponible, et on reconstruit la même
    // `ShapeVarView` autour (le `max` est une constante, il suit la vue dans le kernel).
    auto transfer_cost ( const auto &queue, auto io_category ) const {
        return sdot::transfer_cost( queue, io_category, view );
    }

    auto make_available( auto &&queue, auto io_category, auto &&cont ) const {
        return sdot::make_available( queue, io_category, view, [&]( auto &&kernel_view ) {
            return cont( ShapeVarView<DECAYED_TYPE_OF( kernel_view )>{ FORWARD( kernel_view ), max } );
        } );
    }
};

template<class View>
auto make_shape_var_view( View view, SI max ) { return ShapeVarView<View>{ view, max }; }

} // namespace sdot

#pragma once

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
};

template<class View>
auto make_shape_var_view( View view, SI max ) { return ShapeVarView<View>{ view, max }; }

} // namespace sdot

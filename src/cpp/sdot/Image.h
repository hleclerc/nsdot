#pragma once

// the members + generated methods (`operator()`, `make_available`, ...) as macros this struct
// drops in, plus the axes it names -- written to the build include tree by `CallArg_Aggregate`.
#include <sdot/generated/aggregates/Image.h>
#include "support/common_macros.h"

namespace sdot {

SDOT_TEMPLATE_DECL_FOR_Image
struct Image {
    SDOT_ATTRIBUTES_OF_Image

    SCInt  ct_dim       = DECAYED_TYPE_OF( nb_dims )::value;
    using  TF           = DECAYED_TYPE_OF( values )::TF;

    // build a FULLY-POPULATED image -- each of `origin` / `frame` / `knots` that is unbound
    // (a `NoneTensor`) replaced by its documented default -- and hand it to `cont`. Lets the
    // methods below be written ONCE against a complete image instead of gating on `is_valid()`.
    auto   with_defaults( auto &&cont ) const;

    // total measure of the piecewise-constant function: sum over cells of value * cell volume.
    TF     measure      () const;

    // unidimension piece
    struct Udp          { SI index; TF pos; TF mass; TF y; };
    auto   udp_start    () const;
    auto   udp_cont     ( auto &&udp, auto mass_to_take, auto &&cb_parts ) const;
};

}

#include "Image.cxx"

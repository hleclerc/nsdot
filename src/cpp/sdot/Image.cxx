#pragma once

#include "support/algorithms/CartesianIndices.h" // iterate the (dynamic-rank) cell grid
#include "support/containers/IotaTensor.h"       // IotaTensor<TF> (knots = 0,1,2) -- default `knots`
#include "support/containers/Vector.h"           // Vector<TF,d>::zeros()          -- default `origin`
#include "support/containers/Matrix.h"
#include "CstUdPiece.h"
#include "Image.h"
#include <cmath>

#define UTP SDOT_TEMPLATE_DECL_FOR_Image
#define DTP Image<SDOT_TEMPLATE_ARGS_FOR_Image>

namespace sdot {

UTP auto DTP::with_defaults( auto &&cont ) const {
    // Substitute one absent member per step, then recurse -- exactly the shape of
    // `Cell::init_as_hypercube`'s default handling, but rebuilding the WHOLE aggregate (a new
    // template instantiation, deduced by C++20 aggregate CTAD, like the generated `make_available`)
    // rather than passing individual defaults down. We must rebuild, not mutate: a `NoneTensor` has
    // a fixed type and no `operator=`, so replacing its VALUE means replacing its TYPE.
    //
    // The brace-init is POSITIONAL, so its order must match `SDOT_ATTRIBUTES_OF_Image` (the field
    // declaration order across the MRO, axes skipped): target_mass, nb_dims, shape, values, origin,
    // frame, knots, current_mass. `::sdot::Image` (qualified) names the TEMPLATE so CTAD re-deduces;
    // bare `Image` would mean the current instantiation and defeat the substitution.
    if constexpr ( ! CT_VALUE( origin.is_valid() ) )
        return ::sdot::Image{ target_mass, nb_dims, shape, values, Vector<TF,ct_dim>::zeros(), frame, knots, current_mass }.with_defaults( FORWARD( cont ) );
    else if constexpr ( ! CT_VALUE( frame.is_valid() ) )
        return ::sdot::Image{ target_mass, nb_dims, shape, values, origin, Matrix<TF,ct_dim>::identity(), knots, current_mass }.with_defaults( FORWARD( cont ) );
    else if constexpr ( ! CT_VALUE( knots.is_valid() ) )
        return ::sdot::Image{ target_mass, nb_dims, shape, values, origin, frame, IotaTensor<TF>{}, current_mass }.with_defaults( FORWARD( cont ) );
    else
        return cont( *this );
}

UTP typename DTP::TF DTP::measure() const {
    return with_defaults( []( auto &&img ) {
        using ImgT = DECAYED_TYPE_OF( img );
        using TF = typename ImgT::TF;
        constexpr int d = ImgT::ct_dim;

        // |det(frame)| -- rebuild a dense d*d matrix so this works whether `frame` is a `Matrix`
        // (the identity default) or a user-given `TensorView`.
        const auto F = Matrix<TF,d>::with_func( [&]( auto r, auto c ) { return TF( img.frame( r, c ) ); } );
        const TF fdet = std::abs( F.determinant() );

        // sum over the cells (extent per axis = number of cells along it = `values.shape()`).
        // Each cell contributes value * |det(frame)| * Prod_axis ( knots(axis, i+1) - knots(axis, i) ).
        // `values` is an ordinary dense rank-d view (the `img_pos` AxisList unrolls into d named
        // dimensions), so it and `knots` are both indexed positionally.
        auto shape = img.values.shape();
        CartesianIndices<DECAYED_TYPE_OF( shape )> cells{ shape };
        TF sum = 0;
        for ( PI flat = 0; flat < cells.size(); ++flat ) {
            const TF cell = cells[ flat ].apply_values( [&]( auto ...i ) {
                // running axis counter over the index pack (axes 0..d-1, in order)
                PI axis = 0;
                TF spacing = 1;
                ( ( spacing *= TF( img.knots( axis, i + 1 ) ) - TF( img.knots( axis, i ) ), ++axis ), ... );
                return TF( img.values( i... ) ) * spacing;
            } );
            sum += cell * fdet;
        }
        return sum;
    } );
}

UTP auto DTP::udp_start() const {
    return Udp{
        .index = 0,
        .pos = knots( 0 ),
        .mass = ( knots( 1 ) - knots( 0 ) ) * values[ 0 ],
        .y = values[ 0 ]
    };
}

UTP auto DTP::udp_cont( auto &&udp, auto mass_to_take, auto &&cb_parts ) const {
    while( mass_to_take >= udp.mass ) {
        const SI new_index = udp.index + 1;
        const TF x1 = knots( new_index );
        if ( const TF y = udp.y )
            cb_parts( CstUdPiece{ .x0 = udp.pos, .x1 = x1, .y = y } );

        udp.index = new_index;
        udp.pos = x1;
        udp.y = values[ new_index ];

        udp.mass = ( knots( new_index + 1 ) - x1 ) * udp.y;
    }

    const TF x1 = udp.pos + mass_to_take / udp.y;
    cb_parts( CstUdPiece{ .x0 = udp.pos, .x1 = x1, .y = udp.y } );
    udp.pos = x1;
}

}

#undef UTP
#undef DTP

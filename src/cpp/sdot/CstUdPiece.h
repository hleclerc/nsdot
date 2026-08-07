#pragma once

namespace sdot {

template<class TF>
struct CstUdPiece {
    // mass carried by the piece: Integral_{x0}^{x1} y dx
    TF mass() const { return y * ( x1 - x0 ); }

    // first moment: Integral_{x0}^{x1} x * y dx (used to build target barycenters)
    TF first_moment() const { return y * ( x1 * x1 - x0 * x0 ) / 2; }

    // Integral_{x0}^{x1} (x - p)^2 dx = [ (x1-p)^3 - (x0-p)^3 ] / 3, WITHOUT the density factor
    // (the direct term of d cost / d y_c, which survives even where y = 0).
    auto second_moment_about( auto &&p ) const {
        const TF a = x1 - p, b = x0 - p;
        return ( a * a * a - b * b * b ) / 3;
    }

    // Integral_{x0}^{x1} (x - dirac_pos)^2 * y dx, expanded via a^3-b^3 = (a-b)(a^2+ab+b^2) with
    // a = x1-dirac_pos, b = x0-dirac_pos and specialized for the piecewise-CONSTANT density this
    // piece always carries (a single `y`, not the general piecewise-linear y0/y1 the earlier form
    // priced for) -- about half the multiplies of the generic formula.
    auto w2_dist( auto &&dirac_pos ) const {
        const TF dp = dirac_pos;
        return ( x0 - x1 ) * y * ( 3 * dp * ( x0 + x1 ) - 3 * dp * dp - x0 * x0 - x1 * x1 - x0 * x1 ) / 3;
    }

    TF x0, x1;
    TF y;
};

}

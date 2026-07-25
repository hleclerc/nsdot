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

    auto w2_dist( auto &&dirac_pos ) const {
        // TODO: optimize
        const TF y0 = y;
        const TF y1 = y;
        return ( x0 - x1 ) * (
            + 4 * dirac_pos * ( x0 * ( 2 * y0 + y1 ) + x1 * ( y0 + 2 * y1 ) )
            - 6 * dirac_pos * dirac_pos * ( y0 + y1 )
            - x0 * x0 * ( 3 * y0 + y1 )
            - x1 * x1 * ( y0 + 3 * y1 )
            - 2 * x0 * x1 * ( y0 + y1 )
        ) / 12;
    }

    TF x0, x1;
    TF y;
};

}

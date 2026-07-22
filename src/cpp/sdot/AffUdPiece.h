#pragma once

namespace sdot {

template<class TF>
struct CstUdPiece {
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

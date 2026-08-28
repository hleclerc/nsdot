#pragma once

#include <loom/support/common_types.h>

namespace sdot {

/// The accelerator that accelerates nothing: it hands over every seed, in index order.
///
/// Not a placeholder -- it is what makes "no acceleration" an ORDINARY case of the accelerated
/// code rather than a second implementation of `make_cell` to keep in step with the first. There
/// is exactly one cell builder, and what it enumerates is a parameter of it (see
/// `SpatialAccelerator.py` for the concept it implements).
///
/// It is also the only implementer with no Python side at all: it has nothing to carry, so there
/// is no aggregate to lower, no buffer to allocate, and `PowerDiagram::every_seed()` simply makes
/// one on the spot from a count it already holds.
struct EverySeed {
    SI nb_seeds;

    /// Every seed but `i0`. `may_cut` is not even asked: a bound that is never used to skip
    /// anything is a bound not worth computing, and the vertex sweep it costs is precisely what
    /// this class is here to NOT pay when there is no tree to prune with.
    void for_each_candidate( const auto &/*from*/, SI i0, auto &&/*scratch*/,
                             auto &&/*may_cut*/, auto &&cut_with ) const {
        for ( SI i1 = 0; i1 < nb_seeds; ++i1 )
            if ( i1 != i0 && ! cut_with( i1 ) )
                return;
    }
};

}

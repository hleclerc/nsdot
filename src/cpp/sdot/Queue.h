#pragma once

#include <SYCL/sycl.hpp>

namespace sdot {

/// Thin wrapper around sycl::queue
struct Queue {
    void parallel_for( auto batch_axes, auto &&func );


    sycl::queue q;
};

} // namespace sdot

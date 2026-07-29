#pragma once

#include <SYCL/sycl.hpp>

namespace sdot {

/// Atomic `target += value`, for scattering a value that MANY work-items contribute to the same slot
/// of (e.g. a ProjectedSumOfDiracs backward: every angle adds d cost / d position onto the SAME
/// shared 2D-point gradient). `relaxed` order + `device` scope is all we need -- correctness of the
/// sum, not any ordering. `generic_space` so it works whatever memory the buffer lives in.
template<class T>
void atomic_add( T &target, T value ) {
    sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device,
                     sycl::access::address_space::generic_space> ref( target );
    ref.fetch_add( value );
}

}

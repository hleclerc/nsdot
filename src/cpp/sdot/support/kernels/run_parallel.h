#pragma once

#include "../containers/Tuple.h" // IWYU pragma: export
#include "CpuQueue.h" // IWYU pragma: export

namespace sdot {

/// call func for each list item, parallel way.
///   func may define directly (in method) or indirectly (via surdefinitions) the limits in terms of nb threads, ...
///
/// On sélectionne la sycl::queue en fonction des arguments
///
/// On transforme tous les objets en LocalMemory pour le kernel
///
/// run_parallel( range(), []( auto idx, auto &&a, auto &&b, auto &&v ) { a = b; ... },
///   OutList(), a
///   InpList(), b, 34
/// 13.53
/// 31.43
/// 1.0‘“4
/// )
void run_parallel( auto &&queue_list, auto &&item_list, auto &&func, auto &&...args );

} // namespace sdot

#include "run_parallel.cxx" // IWYU pragma: export

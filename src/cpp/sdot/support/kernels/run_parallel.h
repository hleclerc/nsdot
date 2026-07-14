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
/// )
/// `second` = item_list, ou un `Dependencies` (via `after(...)`) suivi de l'item_list.
///
/// `queue_list` peut aussi être une queue seule (`run_parallel( queue, ... )`) : c'est le cas
/// courant d'un kernel généré, qui n'a qu'un contexte d'exécution -- la liste ne sert que quand
/// il y a un choix à faire (on prend alors le moins coûteux, transferts compris).
auto run_parallel( auto &&queue_list, auto &&second, auto &&...rest );

} // namespace sdot

#include "run_parallel.cxx" // IWYU pragma: export

#pragma once

#include "../containers/Tuple.h" // IWYU pragma: export
#include "../common_macros.h" // IWYU pragma: export  (FORWARD, DECAYED_TYPE_OF)
#include "CpuQueue.h" // IWYU pragma: export

namespace sdot {

/// Wrap a functor with an explicit `max_nb_threads` cap that `run_parallel` reads (`_run_kernel`) to
/// bound the launched work-items to `min( nb_items, cap )` -- so a body's PER-THREAD scratch is sized
/// on threads, not items. A generated body's LAMBDA cannot carry the hook, and a LOCAL struct cannot
/// have the templated `operator()`/`max_nb_threads` a kernel needs (C++ forbids member templates in a
/// local class); hence this wrapper lives at namespace scope. Stateless but for the int cap -> cheap
/// to capture by value into the SYCL kernel. `operator()` just forwards to the wrapped functor (plain,
/// non-`HD`, exactly like the generated lambda it wraps), so the optional `thread_index`/`nb_threads`
/// args (see `_do_submit`) pass straight through.
template<class Func>
struct MaxThreads {
    int  cap;
    Func func;
    int  max_nb_threads( auto &&... ) const { return cap; }
    void operator()( auto &&...args ) const { func( FORWARD( args )... ); }
};
template<class Func>
MaxThreads<std::decay_t<Func>> with_max_threads( int cap, Func &&func ) {
    return { cap, FORWARD( func ) };
}

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

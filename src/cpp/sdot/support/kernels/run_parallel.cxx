#pragma once

#include "../algorithms/for_each_item.h"
#include "../algorithms/min.h"
#include "sdot/support/common_macros.h"
#include "../algorithms/min.h"
#include "transfer_cost.h"
#include "make_avaiable.h"
#include "run_parallel.h"
#include "QueueEvent.h"
#include "kernel_cost.h"
#include "IoCategory.h"
#include "../Ct.h"

namespace sdot {

namespace detail::RunParallel {
    template<int nb_args>
    auto _map_reduce_run_arg( const auto &map, const auto &reduce, auto io_category, Ct<int,nb_args>, auto &&head, auto &&...tail ) {
        if constexpr ( nb_args == 0 )
            return reduce( FORWARD( head ), FORWARD( tail )... );
        else if constexpr ( std::is_same_v<DECAYED_TYPE_OF( head ),UndefList> || std::is_same_v<DECAYED_TYPE_OF( head ),OutList> || std::is_same_v<DECAYED_TYPE_OF( head ),MutList> || std::is_same_v<DECAYED_TYPE_OF( head ),InpList> )
            return _map_reduce_run_arg( map, reduce, head, Ct<int,nb_args-1>(), FORWARD( tail )... );
        else {
            return map( io_category, FORWARD( head ), [&]( auto &&mapped ) {
                return _map_reduce_run_arg( map, reduce, io_category, Ct<int,nb_args-1>(), FORWARD( tail )..., FORWARD( mapped ) );
            } );
        }
    }

    QueueEvent _run_kernel( auto &&queue, auto &&deps, auto &&func, auto &&item_list, auto &&...args ) {
        const int nb_items = item_list.size();
        int max_nb_threads = nb_items;
        if constexpr ( requires { func.max_nb_threads( args... ); } )
            max_nb_threads = func.max_nb_threads( args... );

        // launch at most `max_nb_threads` work items, each handling a strided slice of the items
        const int nb_threads = min( nb_items, max_nb_threads );
        if ( nb_threads <= 0 )
            return {};

        // submit (et pas parallel_for direct) pour pouvoir déclarer les dépendances `deps`.
        // capture PAR VALEUR du kernel : il est asynchrone (on ne wait pas ici), donc autonome —
        // sinon nb_items/nb_threads/args capturés par référence seraient détruits.
        return queue.queue.submit( [&]( sycl::handler &h ) {
            for ( const auto &e : deps.events )
                h.depends_on( e );
            h.parallel_for( nb_threads, [=]( sycl::id<1> thread_id ) {
                for ( int index = thread_id; index < nb_items; index += nb_threads )
                    func( index, args... );
            } );
        } );
    }

    // corps de run_parallel, avec dépendances explicites `deps` (peut être Dependencies<0>)
    auto _run_parallel( auto &&queue_list, auto &&deps, auto &&item_list, auto &&func, auto &&...args ) {
        // costs
        auto costs = apply_values( FORWARD( queue_list ), [&]( auto &&...queues ) {
            auto cost_for = [&]( auto &&queue ) {
                return _map_reduce_run_arg( [&]( auto io_category, const auto &arg, auto &&cont ) {
                    return cont( transfer_cost( queue, io_category, arg ) );
                }, [&]( auto &&...map_out ) {
                    return ( map_out + ... + kernel_cost( func, queue, item_list, args... ) );
                }, InpList(), Ct<int,sizeof...(args)+2>(), item_list, UndefList(), args... );
            };
            return tuple( cost_for( queues )... );
        } );

        // first cost == min cost
        double min_cost = costs.apply_values( []( auto...values ) { return min( values... ); } );
        int index_in_queue = 0;
        bool done = false;
        QueueEvent result; ///< event du contexte choisi (RAII : wait à la destruction si non consommé)
        for_each_item( queue_list, [&]( auto &&queue ) {
            double cost = costs[ index_in_queue++ ];
            if ( done || cost > min_cost )
                return;
            done = true;

            result = _map_reduce_run_arg( [&]( auto io_category, auto &&arg, auto &&cont ) {
                return make_available( queue, io_category, FORWARD( arg ), FORWARD( cont ) );
            }, [&]( auto &&...args ) {
                return _run_kernel( queue, deps, FORWARD( func ), FORWARD( args )... );
            }, InpList(), Ct<int,sizeof...(args)+2>(), item_list, UndefList(), args... );
        } );
        return result;
    }
}

// `second` = soit un Dependencies (déps explicites via after(...)), soit l'item_list (pas de déps).
auto run_parallel( auto &&queue_list, auto &&second, auto &&...rest ) {
    if constexpr ( is_dependencies<DECAYED_TYPE_OF( second )> )
        return detail::RunParallel::_run_parallel( FORWARD( queue_list ), FORWARD( second ), FORWARD( rest )... );
    else
        return detail::RunParallel::_run_parallel( FORWARD( queue_list ), Dependencies<0>{}, FORWARD( second ), FORWARD( rest )... );
}

// namespace RunDetails {
//     // force max_cpu_threads to 1
//     template<class Func>
//     struct RunSequentialWrapper : RunTraits::RunFunctorWrapper<Func> {
//         T_VA int max_cpu_threads( A &&.../* args */ ) { return 1_c; }
//         T_VA int max_gpu_threads( A &&.../* args */ ) { return 1_c; }
//     };

//     // decl
//     template<class ES,class F,class L,class... A>        void run_parallel_from( const ES &execution_space, Ct<int,0>, F &&func, L &&list, A &&...args );
//     template<int n,class ES,class F,class H,class... A>  void run_parallel_from( const ES &execution_space, Ct<int,n> cn, F &&func, H &&head, A &&...tail );
//     template<int n,class ES,class F,class H,class... A>  void run_parallel_from( const ES &execution_space, Ct<int,n> cn, F &&func, Inp, H &&head, A &&...tail );
//     template<int n,class ES,class F,class H,class... A>  void run_parallel_from( const ES &execution_space, Ct<int,n> cn, F &&func, Out, H &&head, A &&...tail );
//     template<int n,class ES,class F,class H,class... A>  void run_parallel_from( const ES &execution_space, Ct<int,n> cn, F &&func, Mut, H &&head, A &&...tail );

//     // end
//     template<class ES,class F,class L,class... A> void run_parallel_from( const ES &execution_space, Ct<int,0>, F &&func, L &&list, A &&...args ) {
//         execution_space.run_parallel( FORWARD( list ), func, FORWARD( args )... );
//     }

//     // Inp
//     template<int n,class ES,class F,class H,class... A> void run_parallel_from( const ES &execution_space, Ct<int,n> cn, F &&func, Inp, H &&head, A &&...tail ) {
//         make_accessible( execution_space, FORWARD( head ), 1_b, 0_b, [&]( auto &&head ) {
//             run_parallel_from( execution_space, cn - 2_c, FORWARD( func ), FORWARD( tail )..., FORWARD( head ) );
//         } );
//     }

//     // Out
//     template<int n,class ES,class F,class H,class... A> void run_parallel_from( const ES &execution_space, Ct<int,n> cn, F &&func, Out, H &&head, A &&...tail ) {
//         make_accessible( execution_space, FORWARD( head ), 0_b, 1_b, [&]( auto &&head ) {
//             run_parallel_from( execution_space, cn - 2_c, FORWARD( func ), FORWARD( tail )..., FORWARD( head ) );
//         } );
//     }

//     // Mut
//     template<int n,class ES,class F,class H,class... A> void run_parallel_from( const ES &execution_space, Ct<int,n> cn, F &&func, Mut, H &&head, A &&...tail ) {
//         make_accessible( execution_space, FORWARD( head ), 1_b, 1_b, [&]( auto &&head ) {
//             run_parallel_from( execution_space, cn - 2_c, FORWARD( func ), FORWARD( tail )..., FORWARD( head ) );
//         } );
//     }

//     // raw
//     template<int n,class ES,class F,class H,class... A> void run_parallel_from( const ES &execution_space, Ct<int,n> cn, F &&func, H &&head, A &&...tail ) {
//         make_accessible( execution_space, FORWARD( head ), 0_b, 0_b, [&]( auto &&head ) {
//             run_parallel_from( execution_space, cn - 1_c, FORWARD( func ), FORWARD( tail )..., FORWARD( head ) );
//         } );
//     }
// } // namespace RunDetails

// template<class L,class F,class... A> void run_parallel( L &&list, F &&func, A &&...args ) {
//     // statically chosen from the args memory spaces (single type -> only this branch compiles)
//     auto execution_space = execution_space_for( args... );

//     //
//     RunDetails::run_parallel_from( execution_space, Ct<int,2+sizeof...( args )>(), func, Inp(), FORWARD( list ), FORWARD( args )... );

//     // make every arg accessible from that space (pass-through or transfer), then run
//     // RunDetails::_get_args_on( execution_space, Ct<int,1+sizeof...( args )>(), FORWARD( list ), FORWARD( args )...,
//     //     RunDetails::LaunchOn<DECAYED_TYPE_OF( execution_space ),DECAYED_TYPE_OF( func )>{ execution_space, func }
//     // );
// }

// template<class L,class F,class... A> void run_sequential( L &&list, F &&func, A &&...args ) {
//     run_parallel( FORWARD( list ), RunDetails::RunSequentialWrapper<DECAYED_TYPE_OF(func)>{ FORWARD( func ) }, FORWARD( args )... );
// }

} // namespace sdot

#pragma once

#include "loom/support/common_types.h"
#include "loom/support/Ct.h"
#include "loom/support/kernels/make_avaiable.h"
#include "loom/support/kernels/transfer_cost.h"
#include "sdot/generated/axes/num_nbr.h"
#include "sdot/generated/axes/cell_0.h"

#define SDOT_TEMPLATE_DECL_FOR_Neighbors template<class T_ids, class T_vals, class T_nb_nbrs>


#define SDOT_TEMPLATE_ARGS_FOR_Neighbors T_ids, T_vals, T_nb_nbrs

namespace sdot {
template<class T_ids, class T_vals, class T_nb_nbrs>
struct Neighbors_io {
    static constexpr bool is_io_policy = true;
    T_ids ids;
    T_vals vals;
    T_nb_nbrs nb_nbrs;
};
}

#define SDOT_ATTRIBUTES_OF_Neighbors \
    T_ids ids; \
    T_vals vals; \
    T_nb_nbrs nb_nbrs; \
 \
    HD auto operator()( const auto &...index ) const { \
        return ::sdot::Neighbors{ ids( index... ), vals( index... ), nb_nbrs( index... ) }; \
    } \
 \
    static constexpr auto io_of_ids( auto io ) { if constexpr ( requires { io.ids; } ) return io.ids; else return io; } \
    static constexpr auto io_of_vals( auto io ) { if constexpr ( requires { io.vals; } ) return io.vals; else return io; } \
    static constexpr auto io_of_nb_nbrs( auto io ) { if constexpr ( requires { io.nb_nbrs; } ) return io.nb_nbrs; else return io; } \
 \
    auto transfer_cost( const auto &queue, auto io ) const { \
        return sdot::transfer_cost( queue, io_of_ids( io ), ids ) \
             + sdot::transfer_cost( queue, io_of_vals( io ), vals ) \
             + sdot::transfer_cost( queue, io_of_nb_nbrs( io ), nb_nbrs ) \
             + Ct<double,0.0>(); \
    } \
 \
    auto kernel_form( auto &&queue, auto io ) const { \
        return ::sdot::Neighbors{ sdot::kernel_form( queue, io_of_ids( io ), ids ), sdot::kernel_form( queue, io_of_vals( io ), vals ), sdot::kernel_form( queue, io_of_nb_nbrs( io ), nb_nbrs ) }; \
    }

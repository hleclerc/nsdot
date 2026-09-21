#pragma once

#include "loom/support/common_types.h"
#include "loom/support/Ct.h"
#include "loom/support/kernels/make_avaiable.h"
#include "loom/support/kernels/transfer_cost.h"
#include "sdot/generated/axes/num_point.h"
#include "sdot/generated/axes/dim.h"

#define SDOT_TEMPLATE_DECL_FOR__BspCloud template<class T_positions, class T_weights, class T_order, class T_nb_points, class T_nb_dims>


#define SDOT_TEMPLATE_ARGS_FOR__BspCloud T_positions, T_weights, T_order, T_nb_points, T_nb_dims

namespace sdot {
template<class T_positions, class T_weights, class T_order, class T_nb_points, class T_nb_dims>
struct _BspCloud_io {
    static constexpr bool is_io_policy = true;
    T_positions positions;
    T_weights weights;
    T_order order;
    T_nb_points nb_points;
    T_nb_dims nb_dims;
};
}

#define SDOT_ATTRIBUTES_OF__BspCloud \
    T_positions positions; \
    T_weights weights; \
    T_order order; \
    T_nb_points nb_points; \
    T_nb_dims nb_dims; \
 \
    HD auto operator()( const auto &...index ) const { \
        return ::sdot::_BspCloud{ positions( index... ), weights( index... ), order( index... ), nb_points( index... ), nb_dims( index... ) }; \
    } \
 \
    static constexpr auto io_of_positions( auto io ) { if constexpr ( requires { io.positions; } ) return io.positions; else return io; } \
    static constexpr auto io_of_weights( auto io ) { if constexpr ( requires { io.weights; } ) return io.weights; else return io; } \
    static constexpr auto io_of_order( auto io ) { if constexpr ( requires { io.order; } ) return io.order; else return io; } \
    static constexpr auto io_of_nb_points( auto io ) { if constexpr ( requires { io.nb_points; } ) return io.nb_points; else return io; } \
    static constexpr auto io_of_nb_dims( auto io ) { if constexpr ( requires { io.nb_dims; } ) return io.nb_dims; else return io; } \
 \
    auto transfer_cost( const auto &queue, auto io ) const { \
        return sdot::transfer_cost( queue, io_of_positions( io ), positions ) \
             + sdot::transfer_cost( queue, io_of_weights( io ), weights ) \
             + sdot::transfer_cost( queue, io_of_order( io ), order ) \
             + sdot::transfer_cost( queue, io_of_nb_points( io ), nb_points ) \
             + sdot::transfer_cost( queue, io_of_nb_dims( io ), nb_dims ) \
             + Ct<double,0.0>(); \
    } \
 \
    auto kernel_form( auto &&queue, auto io ) const { \
        return ::sdot::_BspCloud{ sdot::kernel_form( queue, io_of_positions( io ), positions ), sdot::kernel_form( queue, io_of_weights( io ), weights ), sdot::kernel_form( queue, io_of_order( io ), order ), sdot::kernel_form( queue, io_of_nb_points( io ), nb_points ), sdot::kernel_form( queue, io_of_nb_dims( io ), nb_dims ) }; \
    }

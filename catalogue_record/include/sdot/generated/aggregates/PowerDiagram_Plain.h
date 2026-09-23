#pragma once

#include "loom/support/common_types.h"
#include "loom/support/Ct.h"
#include "loom/support/kernels/make_avaiable.h"
#include "loom/support/kernels/transfer_cost.h"
#include "sdot/generated/axes/num_point.h"
#include "sdot/generated/axes/num_boundary.h"
#include "sdot/generated/axes/dim.h"

#define SDOT_TEMPLATE_DECL_FOR_PowerDiagram_Plain template<class T_box_min, class T_box_max, class T_bnd_directions, class T_bnd_offsets, class T_nb_points, class T_nb_boundaries, class T_nb_dims, class T_positions, class T_weights>


#define SDOT_TEMPLATE_ARGS_FOR_PowerDiagram_Plain T_box_min, T_box_max, T_bnd_directions, T_bnd_offsets, T_nb_points, T_nb_boundaries, T_nb_dims, T_positions, T_weights

namespace sdot {
template<class T_box_min, class T_box_max, class T_bnd_directions, class T_bnd_offsets, class T_nb_points, class T_nb_boundaries, class T_nb_dims, class T_positions, class T_weights>
struct PowerDiagram_Plain_io {
    static constexpr bool is_io_policy = true;
    T_box_min box_min;
    T_box_max box_max;
    T_bnd_directions bnd_directions;
    T_bnd_offsets bnd_offsets;
    T_nb_points nb_points;
    T_nb_boundaries nb_boundaries;
    T_nb_dims nb_dims;
    T_positions positions;
    T_weights weights;
};
}

#define SDOT_ATTRIBUTES_OF_PowerDiagram_Plain \
    T_box_min box_min; \
    T_box_max box_max; \
    T_bnd_directions bnd_directions; \
    T_bnd_offsets bnd_offsets; \
    T_nb_points nb_points; \
    T_nb_boundaries nb_boundaries; \
    T_nb_dims nb_dims; \
    T_positions positions; \
    T_weights weights; \
 \
    HD auto operator()( const auto &...index ) const { \
        return ::sdot::PowerDiagram_Plain{ box_min( index... ), box_max( index... ), bnd_directions( index... ), bnd_offsets( index... ), nb_points( index... ), nb_boundaries( index... ), nb_dims( index... ), positions( index... ), weights( index... ) }; \
    } \
 \
    static constexpr auto io_of_box_min( auto io ) { if constexpr ( requires { io.box_min; } ) return io.box_min; else return io; } \
    static constexpr auto io_of_box_max( auto io ) { if constexpr ( requires { io.box_max; } ) return io.box_max; else return io; } \
    static constexpr auto io_of_bnd_directions( auto io ) { if constexpr ( requires { io.bnd_directions; } ) return io.bnd_directions; else return io; } \
    static constexpr auto io_of_bnd_offsets( auto io ) { if constexpr ( requires { io.bnd_offsets; } ) return io.bnd_offsets; else return io; } \
    static constexpr auto io_of_nb_points( auto io ) { if constexpr ( requires { io.nb_points; } ) return io.nb_points; else return io; } \
    static constexpr auto io_of_nb_boundaries( auto io ) { if constexpr ( requires { io.nb_boundaries; } ) return io.nb_boundaries; else return io; } \
    static constexpr auto io_of_nb_dims( auto io ) { if constexpr ( requires { io.nb_dims; } ) return io.nb_dims; else return io; } \
    static constexpr auto io_of_positions( auto io ) { if constexpr ( requires { io.positions; } ) return io.positions; else return io; } \
    static constexpr auto io_of_weights( auto io ) { if constexpr ( requires { io.weights; } ) return io.weights; else return io; } \
 \
    auto transfer_cost( const auto &queue, auto io ) const { \
        return sdot::transfer_cost( queue, io_of_box_min( io ), box_min ) \
             + sdot::transfer_cost( queue, io_of_box_max( io ), box_max ) \
             + sdot::transfer_cost( queue, io_of_bnd_directions( io ), bnd_directions ) \
             + sdot::transfer_cost( queue, io_of_bnd_offsets( io ), bnd_offsets ) \
             + sdot::transfer_cost( queue, io_of_nb_points( io ), nb_points ) \
             + sdot::transfer_cost( queue, io_of_nb_boundaries( io ), nb_boundaries ) \
             + sdot::transfer_cost( queue, io_of_nb_dims( io ), nb_dims ) \
             + sdot::transfer_cost( queue, io_of_positions( io ), positions ) \
             + sdot::transfer_cost( queue, io_of_weights( io ), weights ) \
             + Ct<double,0.0>(); \
    } \
 \
    auto kernel_form( auto &&queue, auto io ) const { \
        return ::sdot::PowerDiagram_Plain{ sdot::kernel_form( queue, io_of_box_min( io ), box_min ), sdot::kernel_form( queue, io_of_box_max( io ), box_max ), sdot::kernel_form( queue, io_of_bnd_directions( io ), bnd_directions ), sdot::kernel_form( queue, io_of_bnd_offsets( io ), bnd_offsets ), sdot::kernel_form( queue, io_of_nb_points( io ), nb_points ), sdot::kernel_form( queue, io_of_nb_boundaries( io ), nb_boundaries ), sdot::kernel_form( queue, io_of_nb_dims( io ), nb_dims ), sdot::kernel_form( queue, io_of_positions( io ), positions ), sdot::kernel_form( queue, io_of_weights( io ), weights ) }; \
    }

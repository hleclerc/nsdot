#pragma once

#include "loom/support/common_types.h"
#include "loom/support/Ct.h"
#include "loom/support/kernels/make_avaiable.h"
#include "loom/support/kernels/transfer_cost.h"
#include "sdot/generated/axes/num_knot.h"
#include "sdot/generated/axes/dim.h"
#include "sdot/generated/axes/dir.h"
#include "sdot/generated/axes/num_cell_cum.h"
#include "sdot/generated/axes/img_pos_0.h"
#include "sdot/generated/axes/img_pos_1.h"
#include "sdot/generated/axes/img_pos_2.h"

#define SDOT_TEMPLATE_DECL_FOR_Image template<class T_target_mass, class T_nb_dims, class T_shape, class T_values, class T_origin, class T_frame, class T_knots, class T_current_mass, class T_nb_cells_cum, class T_cell_cum_mass>


#define SDOT_TEMPLATE_ARGS_FOR_Image T_target_mass, T_nb_dims, T_shape, T_values, T_origin, T_frame, T_knots, T_current_mass, T_nb_cells_cum, T_cell_cum_mass

namespace sdot {
template<class T_target_mass, class T_nb_dims, class T_shape, class T_values, class T_origin, class T_frame, class T_knots, class T_current_mass, class T_nb_cells_cum, class T_cell_cum_mass>
struct Image_io {
    static constexpr bool is_io_policy = true;
    T_target_mass target_mass;
    T_nb_dims nb_dims;
    T_shape shape;
    T_values values;
    T_origin origin;
    T_frame frame;
    T_knots knots;
    T_current_mass current_mass;
    T_nb_cells_cum nb_cells_cum;
    T_cell_cum_mass cell_cum_mass;
};
}

#define SDOT_ATTRIBUTES_OF_Image \
    T_target_mass target_mass; \
    T_nb_dims nb_dims; \
    T_shape shape; \
    T_values values; \
    T_origin origin; \
    T_frame frame; \
    T_knots knots; \
    T_current_mass current_mass; \
    T_nb_cells_cum nb_cells_cum; \
    T_cell_cum_mass cell_cum_mass; \
 \
    HD auto operator()( const auto &...index ) const { \
        return ::sdot::Image{ target_mass( index... ), nb_dims( index... ), shape( index... ), values( index... ), origin( index... ), frame( index... ), knots( index... ), current_mass( index... ), nb_cells_cum( index... ), cell_cum_mass( index... ) }; \
    } \
 \
    static constexpr auto io_of_target_mass( auto io ) { if constexpr ( requires { io.target_mass; } ) return io.target_mass; else return io; } \
    static constexpr auto io_of_nb_dims( auto io ) { if constexpr ( requires { io.nb_dims; } ) return io.nb_dims; else return io; } \
    static constexpr auto io_of_shape( auto io ) { if constexpr ( requires { io.shape; } ) return io.shape; else return io; } \
    static constexpr auto io_of_values( auto io ) { if constexpr ( requires { io.values; } ) return io.values; else return io; } \
    static constexpr auto io_of_origin( auto io ) { if constexpr ( requires { io.origin; } ) return io.origin; else return io; } \
    static constexpr auto io_of_frame( auto io ) { if constexpr ( requires { io.frame; } ) return io.frame; else return io; } \
    static constexpr auto io_of_knots( auto io ) { if constexpr ( requires { io.knots; } ) return io.knots; else return io; } \
    static constexpr auto io_of_current_mass( auto io ) { if constexpr ( requires { io.current_mass; } ) return io.current_mass; else return io; } \
    static constexpr auto io_of_nb_cells_cum( auto io ) { if constexpr ( requires { io.nb_cells_cum; } ) return io.nb_cells_cum; else return io; } \
    static constexpr auto io_of_cell_cum_mass( auto io ) { if constexpr ( requires { io.cell_cum_mass; } ) return io.cell_cum_mass; else return io; } \
 \
    auto transfer_cost( const auto &queue, auto io ) const { \
        return sdot::transfer_cost( queue, io_of_target_mass( io ), target_mass ) \
             + sdot::transfer_cost( queue, io_of_nb_dims( io ), nb_dims ) \
             + sdot::transfer_cost( queue, io_of_shape( io ), shape ) \
             + sdot::transfer_cost( queue, io_of_values( io ), values ) \
             + sdot::transfer_cost( queue, io_of_origin( io ), origin ) \
             + sdot::transfer_cost( queue, io_of_frame( io ), frame ) \
             + sdot::transfer_cost( queue, io_of_knots( io ), knots ) \
             + sdot::transfer_cost( queue, io_of_current_mass( io ), current_mass ) \
             + sdot::transfer_cost( queue, io_of_nb_cells_cum( io ), nb_cells_cum ) \
             + sdot::transfer_cost( queue, io_of_cell_cum_mass( io ), cell_cum_mass ) \
             + Ct<double,0.0>(); \
    } \
 \
    auto kernel_form( auto &&queue, auto io ) const { \
        return ::sdot::Image{ sdot::kernel_form( queue, io_of_target_mass( io ), target_mass ), sdot::kernel_form( queue, io_of_nb_dims( io ), nb_dims ), sdot::kernel_form( queue, io_of_shape( io ), shape ), sdot::kernel_form( queue, io_of_values( io ), values ), sdot::kernel_form( queue, io_of_origin( io ), origin ), sdot::kernel_form( queue, io_of_frame( io ), frame ), sdot::kernel_form( queue, io_of_knots( io ), knots ), sdot::kernel_form( queue, io_of_current_mass( io ), current_mass ), sdot::kernel_form( queue, io_of_nb_cells_cum( io ), nb_cells_cum ), sdot::kernel_form( queue, io_of_cell_cum_mass( io ), cell_cum_mass ) }; \
    }

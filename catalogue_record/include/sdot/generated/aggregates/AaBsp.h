#pragma once

#include "loom/support/common_types.h"
#include "loom/support/Ct.h"
#include "loom/support/kernels/make_avaiable.h"
#include "loom/support/kernels/transfer_cost.h"
#include "sdot/generated/axes/num_bsp_seed.h"
#include "sdot/generated/axes/num_bsp_node.h"
#include "sdot/generated/axes/num_lohi.h"
#include "sdot/generated/axes/dim.h"

#define SDOT_TEMPLATE_DECL_FOR_AaBsp template<class T_seed_indices, class T_node_left, class T_node_right, class T_node_begin, class T_node_end, class T_node_box, class T_node_wa, class T_node_wb, class T_nb_bsp_seeds, class T_nb_bsp_nodes, class T_nb_lohi, class T_nb_dims>


#define SDOT_TEMPLATE_ARGS_FOR_AaBsp T_seed_indices, T_node_left, T_node_right, T_node_begin, T_node_end, T_node_box, T_node_wa, T_node_wb, T_nb_bsp_seeds, T_nb_bsp_nodes, T_nb_lohi, T_nb_dims

namespace sdot {
template<class T_seed_indices, class T_node_left, class T_node_right, class T_node_begin, class T_node_end, class T_node_box, class T_node_wa, class T_node_wb, class T_nb_bsp_seeds, class T_nb_bsp_nodes, class T_nb_lohi, class T_nb_dims>
struct AaBsp_io {
    static constexpr bool is_io_policy = true;
    T_seed_indices seed_indices;
    T_node_left node_left;
    T_node_right node_right;
    T_node_begin node_begin;
    T_node_end node_end;
    T_node_box node_box;
    T_node_wa node_wa;
    T_node_wb node_wb;
    T_nb_bsp_seeds nb_bsp_seeds;
    T_nb_bsp_nodes nb_bsp_nodes;
    T_nb_lohi nb_lohi;
    T_nb_dims nb_dims;
};
}

#define SDOT_ATTRIBUTES_OF_AaBsp \
    T_seed_indices seed_indices; \
    T_node_left node_left; \
    T_node_right node_right; \
    T_node_begin node_begin; \
    T_node_end node_end; \
    T_node_box node_box; \
    T_node_wa node_wa; \
    T_node_wb node_wb; \
    T_nb_bsp_seeds nb_bsp_seeds; \
    T_nb_bsp_nodes nb_bsp_nodes; \
    T_nb_lohi nb_lohi; \
    T_nb_dims nb_dims; \
 \
    HD auto operator()( const auto &...index ) const { \
        return ::sdot::AaBsp{ seed_indices( index... ), node_left( index... ), node_right( index... ), node_begin( index... ), node_end( index... ), node_box( index... ), node_wa( index... ), node_wb( index... ), nb_bsp_seeds( index... ), nb_bsp_nodes( index... ), nb_lohi( index... ), nb_dims( index... ) }; \
    } \
 \
    static constexpr auto io_of_seed_indices( auto io ) { if constexpr ( requires { io.seed_indices; } ) return io.seed_indices; else return io; } \
    static constexpr auto io_of_node_left( auto io ) { if constexpr ( requires { io.node_left; } ) return io.node_left; else return io; } \
    static constexpr auto io_of_node_right( auto io ) { if constexpr ( requires { io.node_right; } ) return io.node_right; else return io; } \
    static constexpr auto io_of_node_begin( auto io ) { if constexpr ( requires { io.node_begin; } ) return io.node_begin; else return io; } \
    static constexpr auto io_of_node_end( auto io ) { if constexpr ( requires { io.node_end; } ) return io.node_end; else return io; } \
    static constexpr auto io_of_node_box( auto io ) { if constexpr ( requires { io.node_box; } ) return io.node_box; else return io; } \
    static constexpr auto io_of_node_wa( auto io ) { if constexpr ( requires { io.node_wa; } ) return io.node_wa; else return io; } \
    static constexpr auto io_of_node_wb( auto io ) { if constexpr ( requires { io.node_wb; } ) return io.node_wb; else return io; } \
    static constexpr auto io_of_nb_bsp_seeds( auto io ) { if constexpr ( requires { io.nb_bsp_seeds; } ) return io.nb_bsp_seeds; else return io; } \
    static constexpr auto io_of_nb_bsp_nodes( auto io ) { if constexpr ( requires { io.nb_bsp_nodes; } ) return io.nb_bsp_nodes; else return io; } \
    static constexpr auto io_of_nb_lohi( auto io ) { if constexpr ( requires { io.nb_lohi; } ) return io.nb_lohi; else return io; } \
    static constexpr auto io_of_nb_dims( auto io ) { if constexpr ( requires { io.nb_dims; } ) return io.nb_dims; else return io; } \
 \
    auto transfer_cost( const auto &queue, auto io ) const { \
        return sdot::transfer_cost( queue, io_of_seed_indices( io ), seed_indices ) \
             + sdot::transfer_cost( queue, io_of_node_left( io ), node_left ) \
             + sdot::transfer_cost( queue, io_of_node_right( io ), node_right ) \
             + sdot::transfer_cost( queue, io_of_node_begin( io ), node_begin ) \
             + sdot::transfer_cost( queue, io_of_node_end( io ), node_end ) \
             + sdot::transfer_cost( queue, io_of_node_box( io ), node_box ) \
             + sdot::transfer_cost( queue, io_of_node_wa( io ), node_wa ) \
             + sdot::transfer_cost( queue, io_of_node_wb( io ), node_wb ) \
             + sdot::transfer_cost( queue, io_of_nb_bsp_seeds( io ), nb_bsp_seeds ) \
             + sdot::transfer_cost( queue, io_of_nb_bsp_nodes( io ), nb_bsp_nodes ) \
             + sdot::transfer_cost( queue, io_of_nb_lohi( io ), nb_lohi ) \
             + sdot::transfer_cost( queue, io_of_nb_dims( io ), nb_dims ) \
             + Ct<double,0.0>(); \
    } \
 \
    auto kernel_form( auto &&queue, auto io ) const { \
        return ::sdot::AaBsp{ sdot::kernel_form( queue, io_of_seed_indices( io ), seed_indices ), sdot::kernel_form( queue, io_of_node_left( io ), node_left ), sdot::kernel_form( queue, io_of_node_right( io ), node_right ), sdot::kernel_form( queue, io_of_node_begin( io ), node_begin ), sdot::kernel_form( queue, io_of_node_end( io ), node_end ), sdot::kernel_form( queue, io_of_node_box( io ), node_box ), sdot::kernel_form( queue, io_of_node_wa( io ), node_wa ), sdot::kernel_form( queue, io_of_node_wb( io ), node_wb ), sdot::kernel_form( queue, io_of_nb_bsp_seeds( io ), nb_bsp_seeds ), sdot::kernel_form( queue, io_of_nb_bsp_nodes( io ), nb_bsp_nodes ), sdot::kernel_form( queue, io_of_nb_lohi( io ), nb_lohi ), sdot::kernel_form( queue, io_of_nb_dims( io ), nb_dims ) }; \
    }

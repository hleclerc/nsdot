#pragma once

#include "loom/support/common_types.h"
#include "loom/support/Ct.h"
#include "loom/support/kernels/make_avaiable.h"
#include "loom/support/kernels/transfer_cost.h"
#include "sdot/generated/axes/num_vertex.h"
#include "sdot/generated/axes/num_cut.h"
#include "sdot/generated/axes/dim_axis.h"

#define SDOT_TEMPLATE_DECL_FOR_Cell_N template<class T_vertex_positions, class T_vertex_cuts, class T_vertex_nbrs, class T_cut_ids, class T_nb_vertices, class T_nb_cuts, class T_nb_dims>


#define SDOT_TEMPLATE_ARGS_FOR_Cell_N T_vertex_positions, T_vertex_cuts, T_vertex_nbrs, T_cut_ids, T_nb_vertices, T_nb_cuts, T_nb_dims

namespace sdot {
template<class T_vertex_positions, class T_vertex_cuts, class T_vertex_nbrs, class T_cut_ids, class T_nb_vertices, class T_nb_cuts, class T_nb_dims>
struct Cell_N_io {
    static constexpr bool is_io_policy = true;
    T_vertex_positions vertex_positions;
    T_vertex_cuts vertex_cuts;
    T_vertex_nbrs vertex_nbrs;
    T_cut_ids cut_ids;
    T_nb_vertices nb_vertices;
    T_nb_cuts nb_cuts;
    T_nb_dims nb_dims;
};
}

#define SDOT_ATTRIBUTES_OF_Cell_N \
    T_vertex_positions vertex_positions; \
    T_vertex_cuts vertex_cuts; \
    T_vertex_nbrs vertex_nbrs; \
    T_cut_ids cut_ids; \
    T_nb_vertices nb_vertices; \
    T_nb_cuts nb_cuts; \
    T_nb_dims nb_dims; \
 \
    HD auto operator()( const auto &...index ) const { \
        return ::sdot::Cell_N{ vertex_positions( index... ), vertex_cuts( index... ), vertex_nbrs( index... ), cut_ids( index... ), nb_vertices( index... ), nb_cuts( index... ), nb_dims( index... ) }; \
    } \
 \
    static constexpr auto io_of_vertex_positions( auto io ) { if constexpr ( requires { io.vertex_positions; } ) return io.vertex_positions; else return io; } \
    static constexpr auto io_of_vertex_cuts( auto io ) { if constexpr ( requires { io.vertex_cuts; } ) return io.vertex_cuts; else return io; } \
    static constexpr auto io_of_vertex_nbrs( auto io ) { if constexpr ( requires { io.vertex_nbrs; } ) return io.vertex_nbrs; else return io; } \
    static constexpr auto io_of_cut_ids( auto io ) { if constexpr ( requires { io.cut_ids; } ) return io.cut_ids; else return io; } \
    static constexpr auto io_of_nb_vertices( auto io ) { if constexpr ( requires { io.nb_vertices; } ) return io.nb_vertices; else return io; } \
    static constexpr auto io_of_nb_cuts( auto io ) { if constexpr ( requires { io.nb_cuts; } ) return io.nb_cuts; else return io; } \
    static constexpr auto io_of_nb_dims( auto io ) { if constexpr ( requires { io.nb_dims; } ) return io.nb_dims; else return io; } \
 \
    auto transfer_cost( const auto &queue, auto io ) const { \
        return sdot::transfer_cost( queue, io_of_vertex_positions( io ), vertex_positions ) \
             + sdot::transfer_cost( queue, io_of_vertex_cuts( io ), vertex_cuts ) \
             + sdot::transfer_cost( queue, io_of_vertex_nbrs( io ), vertex_nbrs ) \
             + sdot::transfer_cost( queue, io_of_cut_ids( io ), cut_ids ) \
             + sdot::transfer_cost( queue, io_of_nb_vertices( io ), nb_vertices ) \
             + sdot::transfer_cost( queue, io_of_nb_cuts( io ), nb_cuts ) \
             + sdot::transfer_cost( queue, io_of_nb_dims( io ), nb_dims ) \
             + Ct<double,0.0>(); \
    } \
 \
    auto kernel_form( auto &&queue, auto io ) const { \
        return ::sdot::Cell_N{ sdot::kernel_form( queue, io_of_vertex_positions( io ), vertex_positions ), sdot::kernel_form( queue, io_of_vertex_cuts( io ), vertex_cuts ), sdot::kernel_form( queue, io_of_vertex_nbrs( io ), vertex_nbrs ), sdot::kernel_form( queue, io_of_cut_ids( io ), cut_ids ), sdot::kernel_form( queue, io_of_nb_vertices( io ), nb_vertices ), sdot::kernel_form( queue, io_of_nb_cuts( io ), nb_cuts ), sdot::kernel_form( queue, io_of_nb_dims( io ), nb_dims ) }; \
    }

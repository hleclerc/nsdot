#pragma once

#include "loom/support/common_types.h"
#include "loom/support/Ct.h"
#include "loom/support/kernels/make_avaiable.h"
#include "loom/support/kernels/transfer_cost.h"
#include "sdot/generated/axes/num_lohi.h"
#include "sdot/generated/axes/dim.h"
#include "sdot/generated/axes/bspnode_0.h"

#define SDOT_TEMPLATE_DECL_FOR__BspLevel template<class T_begin, class T_end, class T_mid, class T_box, class T_wa, class T_wb, class T_nb_lohi, class T_nb_dims>


#define SDOT_TEMPLATE_ARGS_FOR__BspLevel T_begin, T_end, T_mid, T_box, T_wa, T_wb, T_nb_lohi, T_nb_dims

namespace sdot {
template<class T_begin, class T_end, class T_mid, class T_box, class T_wa, class T_wb, class T_nb_lohi, class T_nb_dims>
struct _BspLevel_io {
    static constexpr bool is_io_policy = true;
    T_begin begin;
    T_end end;
    T_mid mid;
    T_box box;
    T_wa wa;
    T_wb wb;
    T_nb_lohi nb_lohi;
    T_nb_dims nb_dims;
};
}

#define SDOT_ATTRIBUTES_OF__BspLevel \
    T_begin begin; \
    T_end end; \
    T_mid mid; \
    T_box box; \
    T_wa wa; \
    T_wb wb; \
    T_nb_lohi nb_lohi; \
    T_nb_dims nb_dims; \
 \
    HD auto operator()( const auto &...index ) const { \
        return ::sdot::_BspLevel{ begin( index... ), end( index... ), mid( index... ), box( index... ), wa( index... ), wb( index... ), nb_lohi( index... ), nb_dims( index... ) }; \
    } \
 \
    static constexpr auto io_of_begin( auto io ) { if constexpr ( requires { io.begin; } ) return io.begin; else return io; } \
    static constexpr auto io_of_end( auto io ) { if constexpr ( requires { io.end; } ) return io.end; else return io; } \
    static constexpr auto io_of_mid( auto io ) { if constexpr ( requires { io.mid; } ) return io.mid; else return io; } \
    static constexpr auto io_of_box( auto io ) { if constexpr ( requires { io.box; } ) return io.box; else return io; } \
    static constexpr auto io_of_wa( auto io ) { if constexpr ( requires { io.wa; } ) return io.wa; else return io; } \
    static constexpr auto io_of_wb( auto io ) { if constexpr ( requires { io.wb; } ) return io.wb; else return io; } \
    static constexpr auto io_of_nb_lohi( auto io ) { if constexpr ( requires { io.nb_lohi; } ) return io.nb_lohi; else return io; } \
    static constexpr auto io_of_nb_dims( auto io ) { if constexpr ( requires { io.nb_dims; } ) return io.nb_dims; else return io; } \
 \
    auto transfer_cost( const auto &queue, auto io ) const { \
        return sdot::transfer_cost( queue, io_of_begin( io ), begin ) \
             + sdot::transfer_cost( queue, io_of_end( io ), end ) \
             + sdot::transfer_cost( queue, io_of_mid( io ), mid ) \
             + sdot::transfer_cost( queue, io_of_box( io ), box ) \
             + sdot::transfer_cost( queue, io_of_wa( io ), wa ) \
             + sdot::transfer_cost( queue, io_of_wb( io ), wb ) \
             + sdot::transfer_cost( queue, io_of_nb_lohi( io ), nb_lohi ) \
             + sdot::transfer_cost( queue, io_of_nb_dims( io ), nb_dims ) \
             + Ct<double,0.0>(); \
    } \
 \
    auto kernel_form( auto &&queue, auto io ) const { \
        return ::sdot::_BspLevel{ sdot::kernel_form( queue, io_of_begin( io ), begin ), sdot::kernel_form( queue, io_of_end( io ), end ), sdot::kernel_form( queue, io_of_mid( io ), mid ), sdot::kernel_form( queue, io_of_box( io ), box ), sdot::kernel_form( queue, io_of_wa( io ), wa ), sdot::kernel_form( queue, io_of_wb( io ), wb ), sdot::kernel_form( queue, io_of_nb_lohi( io ), nb_lohi ), sdot::kernel_form( queue, io_of_nb_dims( io ), nb_dims ) }; \
    }

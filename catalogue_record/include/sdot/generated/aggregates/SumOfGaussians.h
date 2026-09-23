#pragma once

#include "loom/support/common_types.h"
#include "loom/support/Ct.h"
#include "loom/support/kernels/make_avaiable.h"
#include "loom/support/kernels/transfer_cost.h"
#include "sdot/generated/axes/num_gaussian.h"
#include "sdot/generated/axes/dim.h"

#define SDOT_TEMPLATE_DECL_FOR_SumOfGaussians template<class T_target_mass, class T_nb_gaussians, class T_nb_dims, class T_positions, class T_sigmas, class T_weights, class T_current_mass>


#define SDOT_TEMPLATE_ARGS_FOR_SumOfGaussians T_target_mass, T_nb_gaussians, T_nb_dims, T_positions, T_sigmas, T_weights, T_current_mass

namespace sdot {
template<class T_target_mass, class T_nb_gaussians, class T_nb_dims, class T_positions, class T_sigmas, class T_weights, class T_current_mass>
struct SumOfGaussians_io {
    static constexpr bool is_io_policy = true;
    T_target_mass target_mass;
    T_nb_gaussians nb_gaussians;
    T_nb_dims nb_dims;
    T_positions positions;
    T_sigmas sigmas;
    T_weights weights;
    T_current_mass current_mass;
};
}

#define SDOT_ATTRIBUTES_OF_SumOfGaussians \
    T_target_mass target_mass; \
    T_nb_gaussians nb_gaussians; \
    T_nb_dims nb_dims; \
    T_positions positions; \
    T_sigmas sigmas; \
    T_weights weights; \
    T_current_mass current_mass; \
 \
    HD auto operator()( const auto &...index ) const { \
        return ::sdot::SumOfGaussians{ target_mass( index... ), nb_gaussians( index... ), nb_dims( index... ), positions( index... ), sigmas( index... ), weights( index... ), current_mass( index... ) }; \
    } \
 \
    static constexpr auto io_of_target_mass( auto io ) { if constexpr ( requires { io.target_mass; } ) return io.target_mass; else return io; } \
    static constexpr auto io_of_nb_gaussians( auto io ) { if constexpr ( requires { io.nb_gaussians; } ) return io.nb_gaussians; else return io; } \
    static constexpr auto io_of_nb_dims( auto io ) { if constexpr ( requires { io.nb_dims; } ) return io.nb_dims; else return io; } \
    static constexpr auto io_of_positions( auto io ) { if constexpr ( requires { io.positions; } ) return io.positions; else return io; } \
    static constexpr auto io_of_sigmas( auto io ) { if constexpr ( requires { io.sigmas; } ) return io.sigmas; else return io; } \
    static constexpr auto io_of_weights( auto io ) { if constexpr ( requires { io.weights; } ) return io.weights; else return io; } \
    static constexpr auto io_of_current_mass( auto io ) { if constexpr ( requires { io.current_mass; } ) return io.current_mass; else return io; } \
 \
    auto transfer_cost( const auto &queue, auto io ) const { \
        return sdot::transfer_cost( queue, io_of_target_mass( io ), target_mass ) \
             + sdot::transfer_cost( queue, io_of_nb_gaussians( io ), nb_gaussians ) \
             + sdot::transfer_cost( queue, io_of_nb_dims( io ), nb_dims ) \
             + sdot::transfer_cost( queue, io_of_positions( io ), positions ) \
             + sdot::transfer_cost( queue, io_of_sigmas( io ), sigmas ) \
             + sdot::transfer_cost( queue, io_of_weights( io ), weights ) \
             + sdot::transfer_cost( queue, io_of_current_mass( io ), current_mass ) \
             + Ct<double,0.0>(); \
    } \
 \
    auto kernel_form( auto &&queue, auto io ) const { \
        return ::sdot::SumOfGaussians{ sdot::kernel_form( queue, io_of_target_mass( io ), target_mass ), sdot::kernel_form( queue, io_of_nb_gaussians( io ), nb_gaussians ), sdot::kernel_form( queue, io_of_nb_dims( io ), nb_dims ), sdot::kernel_form( queue, io_of_positions( io ), positions ), sdot::kernel_form( queue, io_of_sigmas( io ), sigmas ), sdot::kernel_form( queue, io_of_weights( io ), weights ), sdot::kernel_form( queue, io_of_current_mass( io ), current_mass ) }; \
    }

#pragma once

#include "loom/support/common_types.h"
#include "loom/support/Ct.h"
#include "loom/support/kernels/make_avaiable.h"
#include "loom/support/kernels/transfer_cost.h"
#include "sdot/generated/axes/num_thread.h"
#include "sdot/generated/axes/num_word.h"

#define SDOT_TEMPLATE_DECL_FOR_CellScratch template<class T_words, class T_nb_threads, class T_nb_words, class T_kernel_fp_size>


#define SDOT_TEMPLATE_ARGS_FOR_CellScratch T_words, T_nb_threads, T_nb_words, T_kernel_fp_size

namespace sdot {
template<class T_words, class T_nb_threads, class T_nb_words, class T_kernel_fp_size>
struct CellScratch_io {
    static constexpr bool is_io_policy = true;
    T_words words;
    T_nb_threads nb_threads;
    T_nb_words nb_words;
    T_kernel_fp_size kernel_fp_size;
};
}

#define SDOT_ATTRIBUTES_OF_CellScratch \
    T_words words; \
    T_nb_threads nb_threads; \
    T_nb_words nb_words; \
    T_kernel_fp_size kernel_fp_size; \
 \
    HD auto operator()( const auto &...index ) const { \
        return ::sdot::CellScratch{ words( index... ), nb_threads( index... ), nb_words( index... ), kernel_fp_size( index... ) }; \
    } \
 \
    static constexpr auto io_of_words( auto io ) { if constexpr ( requires { io.words; } ) return io.words; else return io; } \
    static constexpr auto io_of_nb_threads( auto io ) { if constexpr ( requires { io.nb_threads; } ) return io.nb_threads; else return io; } \
    static constexpr auto io_of_nb_words( auto io ) { if constexpr ( requires { io.nb_words; } ) return io.nb_words; else return io; } \
    static constexpr auto io_of_kernel_fp_size( auto io ) { if constexpr ( requires { io.kernel_fp_size; } ) return io.kernel_fp_size; else return io; } \
 \
    auto transfer_cost( const auto &queue, auto io ) const { \
        return sdot::transfer_cost( queue, io_of_words( io ), words ) \
             + sdot::transfer_cost( queue, io_of_nb_threads( io ), nb_threads ) \
             + sdot::transfer_cost( queue, io_of_nb_words( io ), nb_words ) \
             + sdot::transfer_cost( queue, io_of_kernel_fp_size( io ), kernel_fp_size ) \
             + Ct<double,0.0>(); \
    } \
 \
    auto kernel_form( auto &&queue, auto io ) const { \
        return ::sdot::CellScratch{ sdot::kernel_form( queue, io_of_words( io ), words ), sdot::kernel_form( queue, io_of_nb_threads( io ), nb_threads ), sdot::kernel_form( queue, io_of_nb_words( io ), nb_words ), sdot::kernel_form( queue, io_of_kernel_fp_size( io ), kernel_fp_size ) }; \
    }

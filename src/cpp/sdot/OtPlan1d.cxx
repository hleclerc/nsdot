#pragma once

#include "OtPlan1d.h"
#include <numeric>

#define UTP SDOT_TEMPLATE_DECL_FOR_OtPlan1d
#define DTP OtPlan1d<SDOT_TEMPLATE_ARGS_FOR_OtPlan1d>

namespace sdot {

UTP void DTP::update_outputs( auto &&sorted_indices ) {
    std::iota( sorted_indices.begin(), sorted_indices.end(), 0l );
    std::sort( sorted_indices.begin(), sorted_indices.end(), [&]( auto a, auto b ) {
        return src_dist.positions[ a ][ 0_c ] < src_dist.positions[ b ][ 0_c ];
    } );

    nb_diracs.set( src_dist.nb_diracs );
    TF local_cost = 0;
    dst_dist.with_defaults( [&]( auto &&dst_dist ) {
        auto udp = dst_dist.udp_start();
        const SI nb_diracs = sorted_indices.size();
        for( SI num_sorted_dirac = 0; num_sorted_dirac < nb_diracs; ++num_sorted_dirac ) {
            const SI num_dirac = sorted_indices[ num_sorted_dirac ];
            const TF dirac_pos = src_dist.positions( ::num_dirac = num_dirac, dim = 0 );
            const TF mass = TF( 1 ) / nb_diracs;
            dst_dist.udp_cont( udp, mass, [&]( auto &&item ) {
                INFO( num_sorted_dirac, item.x0 );
                local_cost += item.w2_dist( dirac_pos );
            } );
        }

    } );

    cost = local_cost;
}

}

#undef UTP
#undef DTP

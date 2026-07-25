#pragma once

#include "OtPlan1d.h"
#include <algorithm>
#include <numeric>

#define UTP SDOT_TEMPLATE_DECL_FOR_OtPlan1d
#define DTP OtPlan1d<SDOT_TEMPLATE_ARGS_FOR_OtPlan1d>

namespace sdot {

UTP void DTP::update_outputs( auto &&sorted_indices ) {
    // Sort the dirac indices by position, in place on the INTEGER `sorted_indices` buffer. `dtype`
    // must be integer (a float buffer would sort/index on a floating key and collapse all indices
    // onto 0), and the iterator must be reliable (see StridedIterator: element strides, fixed
    // `operator-`) for `std::iota`/`std::sort` to persist their writes.
    std::iota( sorted_indices.begin(), sorted_indices.end(), 0ll );
    std::sort( sorted_indices.begin(), sorted_indices.end(), [&]( auto a, auto b ) {
        return src_dist.positions( ::num_dirac = a, dim = 0 ) < src_dist.positions( ::num_dirac = b, dim = 0 );
    } );

    const SI nb = src_dist.weights.size();
    nb_diracs.set( src_dist.nb_diracs );
    TF local_cost = 0;
    dst_dist.with_defaults( [&]( auto &&dst_dist ) {
        auto udp = dst_dist.udp_start();
        for( SI k = 0; k < nb; ++k ) {
            const SI num_dirac = sorted_indices( k );
            const TF dirac_pos = src_dist.positions( ::num_dirac = num_dirac, dim = 0 );
            const TF mass = src_dist.weights( ::num_dirac = num_dirac );
            TF moment = 0;
            dst_dist.udp_cont( udp, mass, [&]( auto &&item ) {
                local_cost += item.w2_dist( dirac_pos );
                moment += item.first_moment();
            } );
            // barycenter = center of mass of the target slice assigned to this dirac
            barycenters( ::num_dirac = num_dirac, dim = 0 ) = moment / mass;
        }
    } );

    cost = local_cost;
}

UTP void DTP::update_outputs_bwd( auto &&grad_plan, auto &&sorted_indices ) const {
    //   cost = Sum_k Integral_{t_k}^{t_{k+1}} y(x) (x - p_{s(k)})^2 dx,
    // with diracs sorted by position (order s = `sorted_indices`, reused as a forward residual --
    // no re-sort here), and t_k = M^{-1}(W_k) the target quantile at the cumulative source mass
    // W_k = Sum_{j<k} w_{s(j)}.
    //
    // Each gradient below is guarded at compile time on `is_valid()`: an unperturbed input reaches
    // us as a `NoneTensor` (no `operator=`), so its block must vanish -- see [[differentiation]].
    // Everything runs inside a SYCL kernel, so NO std::vector / dynamic allocation: the per-dirac
    // suffix sum is carried by two scalars instead of an array.
    const TF g = grad_plan.cost; // scalar cotangent seeding `cost`
    const SI nb = src_dist.weights.size();

    // --- d cost / d positions -------------------------------------------------------------------
    // The boundaries t_k depend only on the (fixed) masses, so only p_i moves the integrand:
    //   d cost / d p_i = Integral_{S_i} 2 (p_i - x) y dx = 2 w_i ( p_i - b_i ),  b_i = barycenters(i).
    // Order-independent, no walk, one closed-form write per dirac.
    if constexpr ( CT_VALUE( grad_plan.src_dist.positions.is_valid() ) ) {
        for( SI i = 0; i < nb; ++i ) {
            const TF p = src_dist.positions( ::num_dirac = i, dim = 0 );
            const TF b = barycenters( ::num_dirac = i, dim = 0 );
            const TF w = src_dist.weights( ::num_dirac = i );
            grad_plan.src_dist.positions( ::num_dirac = i, dim = 0 ) = g * 2 * w * ( p - b );
        }
    }

    // --- d cost / d weights and d cost / d values ----------------------------------------------
    // Both flow through the moving boundaries t_m. The potential jump at t_m is (the y(t_m) factor
    // cancels against dt_m/dW_m = 1/y(t_m)):
    //   phi_m = (t_m - p_{s(m-1)})^2 - (t_m - p_{s(m)})^2   for 1 <= m <= n-1  (interior bounds),
    //   phi_n = (t_n - p_{s(n-1)})^2                        for the last endpoint t_n = M^{-1}(Sum w),
    // which moves too since w (and y) are free at the kernel level -- normalization lives upstream.
    // With Phi_k = Sum_{m=k+1}^{n} phi_m = phi_total - Sum_{m<=k} phi_m:
    //   d cost / d w_{s(k)} = Phi_k,   and, per emitted piece (one cell c, sorted dirac k),
    //   d cost / d y_c     += second_moment_about(p_{s(k)}) - Phi_k * (x1 - x0).
    if constexpr ( CT_VALUE( grad_plan.src_dist.weights.is_valid() ) ||
                   CT_VALUE( grad_plan.dst_dist.values.is_valid() ) ) {
        dst_dist.with_defaults( [&]( auto &&img ) {
            // NB: read each weight into a scalar `TF w` before passing it to `udp_cont`. That method
            // MUTATES its `mass_to_take` argument (`mass_to_take -= udp.mass`); handing it the tensor
            // view `src_dist.weights( ... )` directly would decrement the shared weights buffer and
            // corrupt every later read (the forward avoids this via its own `const TF mass`).

            // pass A: phi_total = Sum_{m=1}^n phi_m (a single scalar; phi_n from the last endpoint t_n).
            TF phi_total = 0;
            {
                auto udp = img.udp_start();
                TF p_prev = 0;
                for( SI k = 0; k < nb; ++k ) {
                    const SI di = sorted_indices( k );
                    const TF pk = src_dist.positions( ::num_dirac = di, dim = 0 );
                    const TF w = src_dist.weights( ::num_dirac = di );
                    if ( k >= 1 ) {
                        const TF t = udp.pos; // boundary t_k
                        phi_total += ( t - p_prev ) * ( t - p_prev ) - ( t - pk ) * ( t - pk );
                    }
                    img.udp_cont( udp, w, []( auto && ) {} );
                    p_prev = pk;
                }
                phi_total += ( udp.pos - p_prev ) * ( udp.pos - p_prev ); // phi_n, udp.pos == t_n
            }

            if constexpr ( CT_VALUE( grad_plan.dst_dist.values.is_valid() ) ) {
                const SI nb_cells = img.values.size();
                for( SI c = 0; c < nb_cells; ++c )
                    grad_plan.dst_dist.values( c ) = 0; // accumulated below, one += per piece
            }

            // pass B: running prefix `pref = Sum_{m<=k} phi_m` gives Phi_k = phi_total - pref, feeding
            // grad_w (per dirac) and grad_y (per piece) in the same walk.
            auto udp = img.udp_start();
            TF p_prev = 0, pref = 0;
            for( SI k = 0; k < nb; ++k ) {
                const SI di = sorted_indices( k );
                const TF pk = src_dist.positions( ::num_dirac = di, dim = 0 );
                const TF w = src_dist.weights( ::num_dirac = di );
                if ( k >= 1 ) {
                    const TF t = udp.pos;
                    pref += ( t - p_prev ) * ( t - p_prev ) - ( t - pk ) * ( t - pk );
                }
                const TF Phi_k = phi_total - pref;

                if constexpr ( CT_VALUE( grad_plan.src_dist.weights.is_valid() ) )
                    grad_plan.src_dist.weights( ::num_dirac = di ) = g * Phi_k;

                if constexpr ( CT_VALUE( grad_plan.dst_dist.values.is_valid() ) )
                    // `udp.index` is the cell of the piece being emitted (read before the walker
                    // advances); every piece lives in a single cell.
                    img.udp_cont( udp, w, [&]( auto &&item ) {
                        grad_plan.dst_dist.values( udp.index ) += g * ( item.second_moment_about( pk ) - Phi_k * ( item.x1 - item.x0 ) );
                    } );
                else
                    img.udp_cont( udp, w, []( auto && ) {} );

                p_prev = pk;
            }
        } );
    }
}

}

#undef UTP
#undef DTP

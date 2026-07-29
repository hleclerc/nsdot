#pragma once

#include "OtPlan1d.h"
#include <cstdint>
#include <bit>

#define UTP SDOT_TEMPLATE_DECL_FOR_OtPlan1d
#define DTP OtPlan1d<SDOT_TEMPLATE_ARGS_FOR_OtPlan1d>

namespace sdot {

UTP void DTP::sort_diracs( auto &&sorted_indices, auto &&radix_tmp, auto &&sorted_pos ) const {
    // Sort the diracs by position INTO `sorted_indices` (a per-thread scratch row). We do NOT argsort
    // (`std::sort` on indices with a comparator that reads `positions( a )`/`positions( b )`): that is
    // O(n log n) RANDOM gathers into `positions` and, at large n, the kernel's dominant cost (measured
    // ~18x the sweep's own gather). Instead we PACK, in each int64 slot, an order-preserving key of the
    // position in the high bits + the index in the low bits, and sort those CONTIGUOUSLY. The key is
    // float32-precision; positions it cannot separate (and exact ties) are ordered by index ->
    // deterministic, and harmless for the OT cost (near-equal positions map to near-equal targets).
    //
    // Shared by the forward AND the backward: the per-thread scratch is transient (not a residual), so
    // the backward RE-DERIVES the order here into its own fresh scratch rather than reading it back.
    const SI nb = src_dist.weights.size();
    for ( SI i = 0; i < nb; ++i ) {
        const float f = float( src_dist.position( i ) );
        uint32_t u = std::bit_cast<uint32_t>( f );
        u ^= ( u & 0x80000000u ) ? 0xFFFFFFFFu : 0x80000000u;       // IEEE float -> order-preserving uint32
        sorted_indices( i ) = ( SI( u >> 1 ) << 32 ) | SI( i );     // key (31 bits) high, index (32 bits) low; stays positive
    }

    // LSD radix sort (stable, O(n) -- no comparisons) over the 4 bytes of the packed KEY (bits 32..62;
    // the low 32 index bits ride along, ties keep input order). Beats the contiguous comparison sort by
    // avoiding the n log n compares; the 256-bucket scatter stays write-combining-friendly. `count` is a
    // FIXED-size stack array (no dynamic allocation). 4 passes = even -> the result lands back in
    // `sorted_indices`; `radix_tmp` is the ping-pong scratch.
    auto radix_pass = [&]( auto &&src, auto &&dst, int shift ) {
        SI count[ 256 ] = {};
        for ( SI i = 0; i < nb; ++i ) { const SI v = src( i ); count[ ( v >> shift ) & 0xFF ]++; }
        SI off = 0;
        for ( int b = 0; b < 256; ++b ) { const SI c = count[ b ]; count[ b ] = off; off += c; }
        for ( SI i = 0; i < nb; ++i ) { const SI v = src( i ); dst( count[ ( v >> shift ) & 0xFF ]++ ) = v; }
    };
    radix_pass( sorted_indices, radix_tmp, 32 );
    radix_pass( radix_tmp, sorted_indices, 40 );
    radix_pass( sorted_indices, radix_tmp, 48 );
    radix_pass( radix_tmp, sorted_indices, 56 );

    // decode the index (drop the packed key) AND cache the sorted 1D position CONTIGUOUSLY, so the
    // sweep (and the two backward passes) STREAM `sorted_pos( k )` instead of re-gathering the point
    // and recomputing the projection `src_dist.position( sorted_indices( k ) )` each time -- the one
    // random gather here replaces several. `sorted_pos` is a per-thread scratch like the others.
    for ( SI k = 0; k < nb; ++k ) {
        const SI di = sorted_indices( k ) & 0xFFFFFFFFll;
        sorted_indices( k ) = di;
        sorted_pos( k ) = src_dist.position( di );
    }
}

UTP void DTP::update_outputs( auto &&sorted_indices, auto &&radix_tmp, auto &&sorted_pos ) {
    // Forward = COST, plus the barycenters ONLY when `barycenters` is a bound output (the caller set
    // `with_barycenters`). When it is not, it is a NoneTensor (no `operator=`) so the guarded write
    // and the `moment` it needs both vanish at compile time -- the forward then does cost only, and
    // the backward recomputes any b_i it needs (see `update_outputs_bwd`). `barycenters` is a per-
    // (angle x dirac) buffer ([nb_angles, n] = 80GB at scale), hence off by default.
    const SI nb = src_dist.weights.size();
    sort_diracs( sorted_indices, radix_tmp, sorted_pos );

    nb_diracs.set( src_dist.nb_diracs );
    TF local_cost = 0;
    dst_dist.with_defaults( [&]( auto &&dst_dist ) {
        auto udp = dst_dist.udp_start();
        for( SI k = 0; k < nb; ++k ) {
            const SI num_dirac = sorted_indices( k );
            const TF dirac_pos = sorted_pos( k );          // streamed (contiguous), not re-projected
            const TF mass = src_dist.weights( ::num_dirac = num_dirac );
            TF moment = 0;
            dst_dist.udp_cont( udp, mass, [&]( auto &&item ) {
                local_cost += item.w2_dist( dirac_pos );
                if constexpr ( CT_VALUE( barycenters.is_valid() ) )
                    moment += item.first_moment();
            } );
            if constexpr ( CT_VALUE( barycenters.is_valid() ) )
                barycenters( ::num_dirac = num_dirac, dim = 0 ) = moment / mass;   // center of mass of the slice
        }
    } );

    cost = local_cost;
}

UTP void DTP::update_outputs_bwd( auto &&grad_plan, auto &&sorted_indices, auto &&radix_tmp, auto &&sorted_pos ) const {
    //   cost = Sum_k Integral_{t_k}^{t_{k+1}} y(x) (x - p_{s(k)})^2 dx,
    // with diracs sorted by position (order s = `sorted_indices`), and t_k = M^{-1}(W_k) the target
    // quantile at the cumulative source mass W_k = Sum_{j<k} w_{s(j)}. The scratch is PER-THREAD and
    // transient (not a forward residual), so the order is RE-DERIVED here via `sort_diracs` -- but
    // only in the weights/values block that actually walks it; the positions gradient is
    // order-independent (closed form per dirac) and needs no sort.
    //
    // Each gradient below is guarded at compile time on `is_valid()`: an unperturbed input reaches
    // us as a `NoneTensor` (no `operator=`), so its block must vanish -- see [[differentiation]].
    // Everything runs inside a SYCL kernel, so NO std::vector / dynamic allocation: the per-dirac
    // suffix sum is carried by two scalars instead of an array.
    const TF g = grad_plan.cost; // scalar cotangent seeding `cost`
    const SI nb = src_dist.weights.size();

    // --- d cost / d positions -------------------------------------------------------------------
    //   d cost / d p_i = Integral_{S_i} 2 (p_i - x) y dx = 2 w_i ( p_i - b_i ),  b_i the barycenter of
    // the target slice assigned to dirac i. Guarded at compile time on the source so the whole block
    // is skipped when the position gradient is not wanted; `add_position_grad` then scatters (per-angle
    // write for SumOfDiracs, atomic to the shared 2D points for a projected source).
    if constexpr ( CT_VALUE( src_dist.position_grad_wanted( grad_plan.src_dist ) ) ) {
        if constexpr ( CT_VALUE( barycenters.is_valid() ) ) {
            // b_i was stored by the forward (`with_barycenters`): read it, no walk, order-independent.
            for( SI i = 0; i < nb; ++i ) {
                const TF p = src_dist.position( i );
                const TF b = barycenters( ::num_dirac = i, dim = 0 );
                const TF w = src_dist.weights( ::num_dirac = i );
                src_dist.add_position_grad( grad_plan.src_dist, i, g * 2 * w * ( p - b ) );
            }
        } else {
            // b_i NOT stored: RECOMPUTE it by the same re-sort + sweep the forward did (trades the
            // [nb_angles, n] residual for compute). `sorted_pos` streamed, `sorted_indices` gives the
            // dirac to scatter onto.
            sort_diracs( sorted_indices, radix_tmp, sorted_pos );
            dst_dist.with_defaults( [&]( auto &&img ) {
                auto udp = img.udp_start();
                for( SI k = 0; k < nb; ++k ) {
                    const SI num_dirac = sorted_indices( k );
                    const TF dirac_pos = sorted_pos( k );
                    const TF mass = src_dist.weights( ::num_dirac = num_dirac );
                    TF moment = 0;
                    img.udp_cont( udp, mass, [&]( auto &&item ) { moment += item.first_moment(); } );
                    const TF b = moment / mass;
                    src_dist.add_position_grad( grad_plan.src_dist, num_dirac, g * 2 * mass * ( dirac_pos - b ) );
                }
            } );
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
        sort_diracs( sorted_indices, radix_tmp, sorted_pos );   // re-derive order + cache sorted positions
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
                    const TF pk = src_dist.position( di );
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
                const TF pk = sorted_pos( k );          // streamed (contiguous), not re-projected
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

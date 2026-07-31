#pragma once

#include "OtPlan1d.h"
#include <SYCL/sycl.hpp>
#include <cstdint>
#include <bit>

#define UTP SDOT_TEMPLATE_DECL_FOR_OtPlan1d
#define DTP OtPlan1d<SDOT_TEMPLATE_ARGS_FOR_OtPlan1d>

namespace sdot {

UTP void DTP::sort_diracs( auto &&sorted_indices, auto &&radix_tmp, auto &&sorted_pos,
                            int local_index, int local_size, auto &&group, auto &&local_scratch ) const {
    // Sort the diracs by position INTO `sorted_indices` (a per-GROUP scratch row, shared by every
    // work-item of the group). We do NOT argsort (`std::sort` on indices with a comparator that reads
    // `positions( a )`/`positions( b )`): that is O(n log n) RANDOM gathers into `positions` and, at
    // large n, the kernel's dominant cost (measured ~18x the sweep's own gather). Instead we PACK, in
    // each int64 slot, an order-preserving key of the position in the high bits + the index in the low
    // bits, and sort those CONTIGUOUSLY. The key is float32-precision; positions it cannot separate
    // (and exact ties) are ordered by index -> deterministic, and harmless for the OT cost (near-equal
    // positions map to near-equal targets).
    //
    // Shared by the forward AND the backward: the per-group scratch is transient (not a residual), so
    // the backward RE-DERIVES the order here into its own fresh scratch rather than reading it back.
    //
    // COOPERATIVE: `local_size` work-items split `[0,nb)` into CONTIGUOUS chunks (`lo`..`hi`), one per
    // work-item, and cooperate via `local_scratch` (a shared `int32` local-memory buffer) + barriers.
    // At `local_size == 1` this degenerates EXACTLY to the single-thread sequential algorithm it
    // generalizes (one chunk spanning the whole range) -- that degeneration is the correctness anchor.
    const SI nb = src_dist.weights.size();
    const SI lo = ( nb * SI( local_index ) ) / local_size;
    const SI hi = ( nb * SI( local_index + 1 ) ) / local_size;

    // packing: each dirac's key depends only on its own position -> no cooperation needed here.
    for ( SI i = lo; i < hi; ++i ) {
        const float f = float( src_dist.position( i ) );
        uint32_t u = std::bit_cast<uint32_t>( f );
        u ^= ( u & 0x80000000u ) ? 0xFFFFFFFFu : 0x80000000u;       // IEEE float -> order-preserving uint32
        sorted_indices( i ) = ( SI( u >> 1 ) << 32 ) | SI( i );     // key (31 bits) high, index (32 bits) low; stays positive
    }
    sycl::group_barrier( group );

    // LSD radix sort (stable, O(n) -- no comparisons) over the 4 bytes of the packed KEY (bits 32..62;
    // the low 32 index bits ride along, ties keep input order). 4 passes = even -> the result lands
    // back in `sorted_indices`; `radix_tmp` is the ping-pong scratch.
    //
    // Chunked-histogram-scatter, the standard allocation-free parallel counting-sort scatter -- NO
    // ATOMICS anywhere: `local_scratch` holds `local_size` rows of 256 int32 counts, ROW `local_index`
    // is this work-item's PRIVATE chunk histogram (only it ever reads/writes it), plus one extra row
    // (index `local_size`) for the cross-chunk bucket offsets (`bucket_start`, computed once by the
    // leader). `local_rank` is a per-work-item STACK array (no dynamic allocation, same pattern as the
    // original `count[256]`). The final 4-pass result is bit-identical regardless of `local_size` (the
    // packed key already embeds the index as an explicit tie-break) -- only intermediate per-pass
    // orderings of radix-tied elements can differ across chunk counts, which doesn't affect the result.
    auto radix_pass = [&]( auto &&src, auto &&dst, int shift ) {
        const SI hist_row    = SI( local_index ) * 256;
        const SI bucket_start_row = SI( local_size ) * 256;

        for ( SI b = local_index; b < SI( local_size ) * 256; b += local_size )
            local_scratch[ b ] = 0;
        sycl::group_barrier( group );

        for ( SI i = lo; i < hi; ++i ) {
            const SI v = src( i );
            local_scratch[ hist_row + SI( ( v >> shift ) & 0xFF ) ]++;
        }
        sycl::group_barrier( group );

        // leader-serial two-level exclusive scan, O(local_size*256): within-bucket, across chunks
        // (rewrites each row's count to its CHUNK-LOCAL offset in place), then across buckets (the
        // running total lands in `bucket_start_row`). Cheap first-correctness-pass; the first thing to
        // optimize once correctness lands, per the design plan.
        if ( local_index == 0 ) {
            SI off = 0;
            for ( int b = 0; b < 256; ++b ) {
                SI run = 0;
                for ( int u = 0; u < local_size; ++u ) {
                    const SI c = local_scratch[ SI( u ) * 256 + b ];
                    local_scratch[ SI( u ) * 256 + b ] = std::int32_t( run );
                    run += c;
                }
                local_scratch[ bucket_start_row + b ] = std::int32_t( off );
                off += run;
            }
        }
        sycl::group_barrier( group );

        SI local_rank[ 256 ];
        for ( int b = 0; b < 256; ++b )
            local_rank[ b ] = local_scratch[ hist_row + b ];
        for ( SI i = lo; i < hi; ++i ) {
            const SI v = src( i );
            const int b = int( ( v >> shift ) & 0xFF );
            dst( local_scratch[ bucket_start_row + b ] + local_rank[ b ]++ ) = v;
        }
        sycl::group_barrier( group );
    };
    radix_pass( sorted_indices, radix_tmp, 32 );
    radix_pass( radix_tmp, sorted_indices, 40 );
    radix_pass( sorted_indices, radix_tmp, 48 );
    radix_pass( radix_tmp, sorted_indices, 56 );

    // decode the index (drop the packed key) AND cache the sorted 1D position CONTIGUOUSLY, so the
    // sweep (and the two backward passes) STREAM `sorted_pos( k )` instead of re-gathering the point
    // and recomputing the projection `src_dist.position( sorted_indices( k ) )` each time -- the one
    // random gather here replaces several. `sorted_pos` is a per-group scratch like the others.
    // Embarrassingly parallel over `k` (no cooperation needed) -- each work-item decodes its own chunk.
    for ( SI k = lo; k < hi; ++k ) {
        const SI di = sorted_indices( k ) & 0xFFFFFFFFll;
        sorted_indices( k ) = di;
        sorted_pos( k ) = src_dist.position( di );
    }
    // the caller (`update_outputs`/`update_outputs_bwd`) then reads the FULL `[0,nb)` range from a
    // single (leader) work-item -- every work-item must have finished writing its chunk first.
    sycl::group_barrier( group );
}

UTP void DTP::update_outputs( auto &&sorted_indices, auto &&radix_tmp, auto &&sorted_pos,
                               int local_index, int local_size, auto &&group, auto &&local_scratch ) {
    // Forward = COST, plus the barycenters ONLY when `barycenters` is a bound output (the caller set
    // `with_barycenters`). When it is not, it is a NoneTensor (no `operator=`) so the guarded write
    // and the `moment` it needs both vanish at compile time -- the forward then does cost only, and
    // the backward recomputes any b_i it needs (see `update_outputs_bwd`). `barycenters` is a per-
    // (angle x dirac) buffer ([nb_angles, n] = 80GB at scale), hence off by default.
    const SI nb = src_dist.weights.size();
    sort_diracs( sorted_indices, radix_tmp, sorted_pos, local_index, local_size, group, local_scratch );

    // the sweep below is a STATEFUL sequential cursor over the target measure (`udp`'s state at step
    // k+1 depends on step k) -- not cooperatively parallelizable without redesigning `udp_cont` itself,
    // so it runs on the group's LEADER only (`local_index == 0`), one full walk over `[0,nb)`.
    if ( local_index == 0 ) {
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
    // the group's scratch row (`sorted_indices`/`radix_tmp`/`sorted_pos`) is REUSED for the next angle
    // this group strides onto (see FfiCodeParallel's `group_index` docstring): every work-item must
    // wait for the leader to finish reading it above before any of them starts overwriting it.
    sycl::group_barrier( group );
}

UTP void DTP::update_outputs_bwd( auto &&grad_plan, auto &&sorted_indices, auto &&radix_tmp, auto &&sorted_pos,
                                   int local_index, int local_size, auto &&group, auto &&local_scratch ) const {
    //   cost = Sum_k Integral_{t_k}^{t_{k+1}} y(x) (x - p_{s(k)})^2 dx,
    // with diracs sorted by position (order s = `sorted_indices`), and t_k = M^{-1}(W_k) the target
    // quantile at the cumulative source mass W_k = Sum_{j<k} w_{s(j)}. The scratch is PER-GROUP and
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
    const SI lo = ( nb * SI( local_index ) ) / local_size;
    const SI hi = ( nb * SI( local_index + 1 ) ) / local_size;

    // --- d cost / d positions -------------------------------------------------------------------
    //   d cost / d p_i = Integral_{S_i} 2 (p_i - x) y dx = 2 w_i ( p_i - b_i ),  b_i the barycenter of
    // the target slice assigned to dirac i. Guarded at compile time on the source so the whole block
    // is skipped when the position gradient is not wanted; `add_position_grad` then scatters (per-angle
    // write for SumOfDiracs, atomic to the shared 2D points for a projected source).
    if constexpr ( CT_VALUE( src_dist.position_grad_wanted( grad_plan.src_dist ) ) ) {
        if constexpr ( CT_VALUE( barycenters.is_valid() ) ) {
            // b_i was stored by the forward (`with_barycenters`): read it, no walk, order-independent --
            // EMBARRASSINGLY parallel over i, every work-item does its own chunk, no scratch touched.
            for( SI i = lo; i < hi; ++i ) {
                const TF p = src_dist.position( i );
                const TF b = barycenters( ::num_dirac = i, dim = 0 );
                const TF w = src_dist.weights( ::num_dirac = i );
                src_dist.add_position_grad( grad_plan.src_dist, i, g * 2 * w * ( p - b ) );
            }
        } else {
            // b_i NOT stored: RECOMPUTE it by the same re-sort + sweep the forward did (trades the
            // [nb_angles, n] residual for compute). `sorted_pos` streamed, `sorted_indices` gives the
            // dirac to scatter onto. The sweep itself is a sequential cursor -> leader only.
            sort_diracs( sorted_indices, radix_tmp, sorted_pos, local_index, local_size, group, local_scratch );
            if ( local_index == 0 ) {
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
            // the scratch row is about to be overwritten by the weights/values block's OWN `sort_diracs`
            // call below (if it runs) -- every work-item must wait for the leader to finish reading it.
            sycl::group_barrier( group );
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
        // re-derive order + cache sorted positions (cooperative); the two-pass walk below is a
        // sequential cursor over `img` (same reason as the positions-gradient recompute above) ->
        // leader only.
        sort_diracs( sorted_indices, radix_tmp, sorted_pos, local_index, local_size, group, local_scratch );
        if ( local_index == 0 ) {
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
    // scratch row reused by the next angle this group strides onto -- every work-item must wait for
    // the leader (if it ran above) before any of them starts overwriting it.
    sycl::group_barrier( group );
}

}

#undef UTP
#undef DTP

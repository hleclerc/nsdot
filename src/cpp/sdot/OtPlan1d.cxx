#pragma once

#include "OtPlan1d.h"
#include "support/atomic_add.h"
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

    // LSD radix sort (stable, O(n) -- no comparisons) over the top 32 bits of the packed KEY (bits
    // 32..63; the low 32 index bits ride along, ties keep input order), `NB_BITS` at a time.
    // `NB_PASSES` is EVEN -> the result lands back in `sorted_indices`; `radix_tmp` is the ping-pong
    // scratch. `NB_BITS = 8` (256 buckets, 4 passes), NOT smaller: measured on an RTX 2080 Ti (600
    // angles, n up to 2e5) that `group_size` (this cooperative sort's OWN width) is a NET NEGATIVE on
    // CUDA -- the per-group forward cost is FLAT across `group_size` 1..128 (dominated entirely by the
    // leader-only sequential `udp_cont` sweep below, not by the sort), and the backward gets
    // MONOTONICALLY WORSE as `group_size` grows (2.2s -> 3.9s at n=2e5): a bigger work-group reserves
    // more of an SM's threads/registers for the group's ENTIRE lifetime while only ONE work-item is
    // ever doing useful work during that sweep, which REDUCES how many independent angle-groups can be
    // concurrently resident on the same SM -- exactly backwards from what this workload needs (many
    // small independent leader-serial walks, not fewer big cooperating ones). `CudaGpu.group_size`
    // therefore now defaults to `1` (no cooperation): smaller buckets bought group_size headroom that
    // turned out to be worthless, so there is no reason to pay their 2x-passes cost -- see
    // [[group-cooperative-sort]]. The cooperative code PATH is kept (generic, degenerates exactly to
    // the sequential algorithm at `group_size == 1`, and remains available for a future kernel whose
    // per-group work is NOT leader-serial).
    constexpr int NB_BITS    = 8;
    constexpr int NB_BUCKETS = 1 << NB_BITS;
    constexpr int NB_PASSES  = 32 / NB_BITS;
    static_assert( NB_PASSES % 2 == 0 );

    // Chunked-histogram-scatter, the standard allocation-free parallel counting-sort scatter -- NO
    // ATOMICS anywhere: `local_scratch` holds `local_size` rows of `NB_BUCKETS` int32 counts, ROW
    // `local_index` is this work-item's PRIVATE chunk histogram (only it ever reads/writes it), plus
    // one extra row (index `local_size`) for the cross-chunk bucket offsets (`bucket_start`, computed
    // once by the leader). `local_rank` is a per-work-item STACK array (no dynamic allocation). The
    // final result is bit-identical regardless of `local_size` (the packed key already embeds the
    // index as an explicit tie-break) -- only intermediate per-pass orderings of radix-tied elements
    // can differ across chunk counts, which doesn't affect the result.
    auto radix_pass = [&]( auto &&src, auto &&dst, int shift ) {
        const SI hist_row    = SI( local_index ) * NB_BUCKETS;
        const SI bucket_start_row = SI( local_size ) * NB_BUCKETS;

        for ( SI b = local_index; b < SI( local_size ) * NB_BUCKETS; b += local_size )
            local_scratch[ b ] = 0;
        sycl::group_barrier( group );

        for ( SI i = lo; i < hi; ++i ) {
            const SI v = src( i );
            local_scratch[ hist_row + SI( ( v >> shift ) & ( NB_BUCKETS - 1 ) ) ]++;
        }
        sycl::group_barrier( group );

        // leader-serial two-level exclusive scan, O(local_size*NB_BUCKETS): within-bucket, across
        // chunks (rewrites each row's count to its CHUNK-LOCAL offset in place), then across buckets
        // (the running total lands in `bucket_start_row`). Cheap first-correctness-pass; the first
        // thing to optimize further once GPU numbers exist for THIS bucket count, per the design plan.
        if ( local_index == 0 ) {
            SI off = 0;
            for ( int b = 0; b < NB_BUCKETS; ++b ) {
                SI run = 0;
                for ( int u = 0; u < local_size; ++u ) {
                    const SI c = local_scratch[ SI( u ) * NB_BUCKETS + b ];
                    local_scratch[ SI( u ) * NB_BUCKETS + b ] = std::int32_t( run );
                    run += c;
                }
                local_scratch[ bucket_start_row + b ] = std::int32_t( off );
                off += run;
            }
        }
        sycl::group_barrier( group );

        SI local_rank[ NB_BUCKETS ];
        for ( int b = 0; b < NB_BUCKETS; ++b )
            local_rank[ b ] = local_scratch[ hist_row + b ];
        for ( SI i = lo; i < hi; ++i ) {
            const SI v = src( i );
            const int b = int( ( v >> shift ) & ( NB_BUCKETS - 1 ) );
            dst( local_scratch[ bucket_start_row + b ] + local_rank[ b ]++ ) = v;
        }
        sycl::group_barrier( group );
    };
    for ( int p = 0; p < NB_PASSES; ++p ) {
        const int shift = 32 + p * NB_BITS;
        if ( p % 2 == 0 ) radix_pass( sorted_indices, radix_tmp, shift );
        else              radix_pass( radix_tmp, sorted_indices, shift );
    }

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

UTP typename DTP::TF DTP::chunked_weight_prefix( auto &&sorted_indices, auto &&group_scan, SI lo, SI hi,
                                                   int local_index, int local_size, auto &&group ) const {
    // This work-item's cumulative SOURCE weight at its OWN chunk start `lo` (sorted order) -- the
    // `Image::udp_at` argument letting it jump straight to its chunk's starting `Udp` state. Much
    // cheaper than `Image::cell_cum_mass`: only `local_size` partials are combined (the CHUNK
    // totals), not a full per-index array -- only the starting boundary of each chunk is ever needed.
    TF local_sum = 0;
    for ( SI k = lo; k < hi; ++k )
        local_sum += TF( src_dist.weights( ::num_dirac = sorted_indices( k ) ) );
    group_scan( local_index ) = local_sum;
    sycl::group_barrier( group );

    if ( local_index == 0 ) {
        TF run = 0;
        for ( int t = 0; t < local_size; ++t ) {
            const TF v = group_scan( t );
            group_scan( t ) = run;
            run += v;
        }
    }
    sycl::group_barrier( group );

    return group_scan( local_index ); // W_lo_t
}

UTP void DTP::update_outputs( auto &&sorted_indices, auto &&radix_tmp, auto &&sorted_pos,
                               auto &&group_scan,
                               int local_index, int local_size, auto &&group, auto &&local_scratch ) {
    // Forward = COST, plus the barycenters ONLY when `barycenters` is a bound output (the caller set
    // `with_barycenters`). When it is not, it is a NoneTensor (no `operator=`) so the guarded write
    // and the `moment` it needs both vanish at compile time -- the forward then does cost only, and
    // the backward recomputes any b_i it needs (see `update_outputs_bwd`). `barycenters` is a per-
    // (angle x dirac) buffer ([nb_angles, n] = 80GB at scale), hence off by default.
    const SI nb = src_dist.weights.size();
    sort_diracs( sorted_indices, radix_tmp, sorted_pos, local_index, local_size, group, local_scratch );

    const SI lo = ( nb * SI( local_index ) ) / local_size;
    const SI hi = ( nb * SI( local_index + 1 ) ) / local_size;
    const TF w_lo = chunked_weight_prefix( sorted_indices, group_scan, lo, hi, local_index, local_size, group );

    // PARALLEL sweep: each work-item JUMPS straight to its own chunk's starting `Udp` state (via
    // `Image::udp_at`, no need to walk there step by step) and walks ONLY `[lo,hi)` -- no leader, no
    // idle work-items (contrast the old leader-only, `local_index==0`-guarded, full-`[0,nb)` walk this
    // replaces). `barycenters` writes stay race-free exactly as before (disjoint per-dirac indices);
    // `cost` is now a per-work-item PARTIAL sum, combined below.
    TF local_cost = 0;
    dst_dist.with_defaults( [&]( auto &&dst_dist ) {
        auto udp = dst_dist.udp_at( dst_dist.cell_cum_mass, w_lo );
        for( SI k = lo; k < hi; ++k ) {
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

    // combine the `local_size` partial costs into the group's `cost` -- same barrier+group_scan
    // reduction idiom as the cooperative scans above.
    group_scan( local_index ) = local_cost;
    sycl::group_barrier( group );
    if ( local_index == 0 ) {
        nb_diracs.set( src_dist.nb_diracs );
        TF total = 0;
        for ( int t = 0; t < local_size; ++t )
            total += TF( group_scan( t ) );
        cost = total;
    }
    // the group's scratch rows are REUSED for the next angle this group strides onto (see
    // FfiCodeParallel's `group_index` docstring): every work-item must wait for the leader to finish
    // reading `group_scan` above before any of them starts overwriting it.
    sycl::group_barrier( group );
}

UTP void DTP::update_outputs_bwd( auto &&grad_plan, auto &&sorted_indices, auto &&radix_tmp, auto &&sorted_pos,
                                   auto &&group_scan,
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
            // b_i NOT stored: RECOMPUTE it by the same re-sort the forward did (trades the
            // [nb_angles, n] residual for compute). PARALLEL sweep: each work-item jumps straight to
            // its own chunk's start (`udp_at`, no leader) and walks only `[lo,hi)` -- per-dirac scatter,
            // disjoint indices, no reduction needed.
            sort_diracs( sorted_indices, radix_tmp, sorted_pos, local_index, local_size, group, local_scratch );
            const TF w_lo = chunked_weight_prefix( sorted_indices, group_scan, lo, hi, local_index, local_size, group );
            dst_dist.with_defaults( [&]( auto &&img ) {
                auto udp = img.udp_at( img.cell_cum_mass, w_lo );
                for( SI k = lo; k < hi; ++k ) {
                    const SI num_dirac = sorted_indices( k );
                    const TF dirac_pos = sorted_pos( k );
                    const TF mass = src_dist.weights( ::num_dirac = num_dirac );
                    TF moment = 0;
                    img.udp_cont( udp, mass, [&]( auto &&item ) { moment += item.first_moment(); } );
                    const TF b = moment / mass;
                    src_dist.add_position_grad( grad_plan.src_dist, num_dirac, g * 2 * mass * ( dirac_pos - b ) );
                }
            } );
            // the scratch rows are about to be overwritten by the weights/values block's OWN
            // `sort_diracs`/`chunked_weight_prefix` calls below (if it runs) -- every work-item must
            // wait for every other to finish reading them first.
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
    //
    // Phi_k needs the GRAND TOTAL phi_total, itself a sum over every dirac -- a genuine global
    // dependency, not a sequential-implementation artifact, so this still needs two passes over the
    // walk. But each pass is now CHUNK-LOCAL and runs on every work-item in parallel: phase 1 gets each
    // work-item's own chunk phi SUBTOTAL (one local walk, jump-started via `udp_at`) and combines the
    // `local_size` subtotals into both `phi_total` (broadcast) and each work-item's exclusive prefix
    // `pref_lo` (its phase-2 starting offset) from the SAME scan -- same idiom as `sort_diracs`'s bucket
    // scan producing both `bucket_start` and per-row chunk-local offsets together. Phase 2 re-derives
    // its own start state (cheap, O(log nb_cells) -- simpler than threading `Udp` state across the
    // barrier) and re-walks its own chunk once more, this time actually writing `grad_w`/`grad_y`.
    if constexpr ( CT_VALUE( grad_plan.src_dist.weights.is_valid() ) ||
                   CT_VALUE( grad_plan.dst_dist.values.is_valid() ) ) {
        sort_diracs( sorted_indices, radix_tmp, sorted_pos, local_index, local_size, group, local_scratch );
        const TF w_lo = chunked_weight_prefix( sorted_indices, group_scan, lo, hi, local_index, local_size, group );

        dst_dist.with_defaults( [&]( auto &&img ) {
            // NB: read each weight into a scalar `TF w` before passing it to `udp_cont`. That method
            // MUTATES its `mass_to_take` argument (`mass_to_take -= udp.mass`); handing it the tensor
            // view `src_dist.weights( ... )` directly would decrement the shared weights buffer and
            // corrupt every later read (the forward avoids this via its own `const TF mass`).

            if constexpr ( CT_VALUE( grad_plan.dst_dist.values.is_valid() ) ) {
                // cooperative zeroing, chunked over CELLS -- accumulated below via atomic `+=`,
                // one per piece.
                const SI nb_cells = img.values.size();
                const SI lo_c = ( nb_cells * SI( local_index ) ) / local_size;
                const SI hi_c = ( nb_cells * SI( local_index + 1 ) ) / local_size;
                for ( SI c = lo_c; c < hi_c; ++c )
                    grad_plan.dst_dist.values( c ) = 0;
                sycl::group_barrier( group ); // every zero-write visible before any += below
            }

            // phase 1: this work-item's chunk phi SUBTOTAL.
            TF phi_sub = 0;
            {
                auto udp = img.udp_at( img.cell_cum_mass, w_lo );
                TF p_prev = ( lo > 0 ) ? TF( sorted_pos( lo - 1 ) ) : TF( 0 );
                for( SI k = lo; k < hi; ++k ) {
                    const SI di = sorted_indices( k );
                    const TF pk = sorted_pos( k );          // streamed (contiguous), not re-projected
                    const TF w = src_dist.weights( ::num_dirac = di );
                    if ( k >= 1 ) { // GLOBAL index check: every dirac except the very first has a boundary
                        const TF t = udp.pos;
                        phi_sub += ( t - p_prev ) * ( t - p_prev ) - ( t - pk ) * ( t - pk );
                    }
                    img.udp_cont( udp, w, []( auto && ) {} );
                    p_prev = pk;
                }
                if ( hi == nb ) // this work-item owns the last dirac -> also owns phi_n
                    phi_sub += ( udp.pos - p_prev ) * ( udp.pos - p_prev );
            }

            group_scan( local_index ) = phi_sub;
            sycl::group_barrier( group );
            if ( local_index == 0 ) {
                TF run = 0;
                for ( int t = 0; t < local_size; ++t ) {
                    const TF v = group_scan( t );
                    group_scan( t ) = run; // exclusive prefix, in place
                    run += v;
                }
                group_scan( local_size ) = run; // broadcast slot: phi_total
            }
            sycl::group_barrier( group );
            const TF pref_lo = group_scan( local_index );
            const TF phi_total = group_scan( local_size );

            // phase 2: re-derive this work-item's own start state and re-walk its chunk, this time
            // actually writing grad_w (disjoint per dirac, safe) / grad_y (see the atomic note below).
            auto udp = img.udp_at( img.cell_cum_mass, w_lo );
            TF p_prev = ( lo > 0 ) ? TF( sorted_pos( lo - 1 ) ) : TF( 0 );
            TF pref = pref_lo;
            for( SI k = lo; k < hi; ++k ) {
                const SI di = sorted_indices( k );
                const TF pk = sorted_pos( k );
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
                    // advances); every piece lives in a single cell. ATOMIC: unlike the sequential
                    // original, a cell straddling a CHUNK BOUNDARY now gets one write from the end of
                    // chunk t-1 and one from the start of chunk t, on two concurrent work-items --
                    // everywhere else (grad_w, barycenters, add_position_grad) stays disjoint per dirac.
                    img.udp_cont( udp, w, [&]( auto &&item ) {
                        atomic_add( grad_plan.dst_dist.values( udp.index ).ref(),
                                    TF( g * ( item.second_moment_about( pk ) - Phi_k * ( item.x1 - item.x0 ) ) ) );
                    } );
                else
                    img.udp_cont( udp, w, []( auto && ) {} );

                p_prev = pk;
            }
        } );
    }
    // scratch rows reused by the next angle this group strides onto -- every work-item must wait for
    // every other to finish reading them before any of them starts overwriting it.
    sycl::group_barrier( group );
}

}

#undef UTP
#undef DTP

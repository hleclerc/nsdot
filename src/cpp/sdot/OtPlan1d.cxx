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
                            int local_index, int local_size, auto &&group, auto &&local_scratch, auto &&sub_group ) const {
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

    // Sub-group (warp) cooperative histogram: ONE shared row of `NB_BUCKETS` counts per SUB-GROUP,
    // not per work-item -- shrinks the histogram's shared-memory footprint from
    // `local_size * NB_BUCKETS` to `num_sg * NB_BUCKETS`, letting `group_size` grow far past the ~32
    // the old one-row-per-thread scheme allowed under the fixed 49152B/block budget (`ncu` measured
    // only 1 resident warp/SM at that clamp -- see `CudaGpu.group_size`'s docstring). `sg_id`/`num_sg`
    // re-chunk `[0,nb)` into `num_sg` CONTIGUOUS, INDEX-ORDERED ranges (one per sub-group), exactly
    // like `lo`/`hi` above did per work-item -- this determinism (chunk = a fixed function of index,
    // not of execution timing) is what keeps the LSD radix sort's pass-to-pass stability correct even
    // though the histogram COUNTING and the SCATTER below are built from parallel, warp-cooperative
    // pieces instead of one sequential walk. Degenerates EXACTLY to the original per-work-item scheme
    // when `sgs == 1` (always true on CPU, where `group_size` -- hence `local_size` -- is itself
    // always 1, see `Cpu.group_size`): `num_sg == local_size`, one lane per "sub-group", every wave
    // below is a single element with `intra_rank == 0`, i.e. the same sequential increment idiom.
    const int sg_lid = int( sub_group.get_local_linear_id() );
    const int sgs    = int( sub_group.get_local_linear_range() );
    const int sg_id  = int( sub_group.get_group_linear_id() );
    const int num_sg = ( local_size + sgs - 1 ) / sgs;
    const SI  lo_s    = ( nb * SI( sg_id ) ) / num_sg;
    const SI  hi_s    = ( nb * SI( sg_id + 1 ) ) / num_sg;

    // `local_scratch` layout: `num_sg` histogram/cursor rows, one fixed cross-chunk `bucket_start`
    // row, then `num_sg` MATCH-MASK rows (see the scatter phase below) -- `(2*num_sg+1)*NB_BUCKETS`
    // elements total, all zeroed together every pass (kept in sync BY HAND with `OtPlan1d.py`'s
    // `local_mem_elems`/`_scratch`'s `group_size` shared-memory budget).
    auto radix_pass = [&]( auto &&src, auto &&dst, int shift ) {
        const SI hist_row        = SI( sg_id ) * NB_BUCKETS;
        const SI bucket_start_row = SI( num_sg ) * NB_BUCKETS;
        const SI match_row       = SI( num_sg + 1 ) * NB_BUCKETS + SI( sg_id ) * NB_BUCKETS;

        for ( SI b = local_index; b < SI( 2 * num_sg + 1 ) * NB_BUCKETS; b += local_size )
            local_scratch[ b ] = 0;
        sycl::group_barrier( group );

        // histogram BUILD: every lane of this sub-group strides over ITS sub-group's chunk, each
        // element bumping its digit's shared counter via a LOCAL atomic -- safe/order-independent
        // (only the FINAL total per bucket matters here, unlike the scatter below, which DOES need
        // index-order-stable ranks).
        for ( SI i = lo_s + sg_lid; i < hi_s; i += sgs ) {
            const SI v = src( i );
            atomic_add_local( local_scratch[ hist_row + SI( ( v >> shift ) & ( NB_BUCKETS - 1 ) ) ], std::int32_t( 1 ) );
        }
        sycl::group_barrier( group );

        // two-level exclusive scan, split so its O(NB_BUCKETS) part runs IN PARALLEL across the group
        // instead of leader-serial. Each work-item now owns a DISJOINT slice of buckets `b` and, for
        // that slice, scans across the `num_sg` sub-group rows itself -- no data dependency between
        // buckets, so this parallelizes trivially. Only the (much smaller, O(NB_BUCKETS)) across-bucket
        // prefix stays leader-serial. Identical in STRUCTURE to a plain per-chunk scan, just over
        // `num_sg` rows instead of `local_size` of them.
        for ( int b = local_index; b < NB_BUCKETS; b += local_size ) {
            SI run = 0;
            for ( int u = 0; u < num_sg; ++u ) {
                const SI c = local_scratch[ SI( u ) * NB_BUCKETS + b ];
                local_scratch[ SI( u ) * NB_BUCKETS + b ] = std::int32_t( run );
                run += c;
            }
            local_scratch[ bucket_start_row + b ] = std::int32_t( run ); // bucket TOTAL, for now
        }
        sycl::group_barrier( group );

        if ( local_index == 0 ) {
            SI off = 0;
            for ( int b = 0; b < NB_BUCKETS; ++b ) {
                const SI c = local_scratch[ bucket_start_row + b ];
                local_scratch[ bucket_start_row + b ] = std::int32_t( off );
                off += c;
            }
        }
        sycl::group_barrier( group );

        // SCATTER, wave by wave (one `sgs`-sized wave per iteration, in increasing index order --
        // deterministic, the same property that makes the outer chunking stable). Finding each lane's
        // STABLE rank among same-digit lanes of THIS wave in O(1) (not O(sgs)) needs a CUDA warp-match
        // intrinsic in the textbook version (`__match_any_sync`) -- unavailable portably in SYCL, so
        // this replicates it via the ATOMIC-OR technique CUB itself falls back to when match-any isn't
        // used (`cub::BlockRadixRankMatch`'s `WARP_MATCH_ATOMIC_OR` path, see
        // cub/block/block_radix_rank.cuh): `match_row[b]` is a per-bucket bitmask, one bit per lane --
        // every lane ORs its own bit into ITS bucket's cell (safe/commutative, no ordering needed),
        // then, after a barrier makes every OR visible, each lane reads back its OWN bucket's mask to
        // learn EXACTLY which lanes share its digit, in one shot. `popcount(mask & lanes-at-or-before-
        // me)` is this lane's 1-indexed rank within that set; the HIGHEST-numbered matching lane is
        // elected "leader" (it alone sees the group's full popcount) and is the one that atomically
        // bumps the shared per-digit cursor -- ONE atomic per DISTINCT digit in the wave, not one per
        // lane. The reservation base itself needs no shuffle: `hist_row[b]` is already shared local
        // memory, so EVERY matching lane just reads it directly (redundant across ties, but cheap and
        // avoids relying on `sycl::select_from_group`'s availability). `dest = bucket_start_row[b]`
        // (this bucket's GROUP-global base, fixed since the scan above) `+ base` (the shared
        // reservation, read before the leader's bump) `+ popcount - 1` (this lane's 0-indexed rank
        // within the reservation) -- the same quantity a sequential `local_scratch[hist_row+b]++`
        // idiom computes one element at a time, resolved for a whole wave in O(1) collectives instead.
        for ( SI base_i = lo_s; base_i < hi_s; base_i += sgs ) {
            // `b < 0` doubles as the "inactive lane" flag (a real digit is always 0..NB_BUCKETS-1) --
            // one fewer live register than a separate `bool active` carried the whole wave through.
            const SI  i = base_i + sg_lid;
            SI  v = 0;
            int b = -1;
            if ( i < hi_s ) {
                v = src( i );
                b = int( ( v >> shift ) & ( NB_BUCKETS - 1 ) );
                atomic_or_local( local_scratch[ match_row + b ], std::int32_t( std::uint32_t( 1 ) << sg_lid ) );
            }
            sycl::group_barrier( sub_group ); // every lane's OR visible before any lane reads its mask

            // `leader` only matters to decide THIS lane's own yes/no -- collapse it to `is_leader`
            // immediately instead of carrying the full lane index across the barrier below.
            int  popc = 0;
            SI   base = 0;
            bool is_leader = false;
            if ( b >= 0 ) {
                const std::uint32_t bin_mask = std::uint32_t( local_scratch[ match_row + b ] );
                const std::uint32_t lanemask_le =
                    ( sg_lid == 31 ) ? 0xFFFFFFFFu : ( ( std::uint32_t( 1 ) << ( sg_lid + 1 ) ) - 1u );
                popc = std::popcount( bin_mask & lanemask_le );
                is_leader = ( sg_lid == 31 - std::countl_zero( bin_mask ) );
                base = local_scratch[ hist_row + b ]; // reservation base -- READ before anyone bumps it
            }
            // every active lane's READ of `base` above must complete before the leader's WRITE below --
            // NOT guaranteed by SIMT reconvergence alone on independent-thread-scheduling hardware
            // (Volta+, includes this Turing target) without an explicit barrier between the two.
            sycl::group_barrier( sub_group );

            if ( is_leader ) {
                atomic_add_local( local_scratch[ hist_row + b ], std::int32_t( popc ) ); // popc == the group's TOTAL size for the leader
                local_scratch[ match_row + b ] = 0; // reset for the next wave/pass
            }

            // `dst` only needs `base`/`popc`, both already captured in registers above -- no ordering
            // dependency on the leader's bump, so this can run without waiting for it.
            if ( b >= 0 )
                dst( local_scratch[ bucket_start_row + b ] + base + popc - 1 ) = v;
            sycl::group_barrier( sub_group ); // cursor bump + mask reset visible before the next wave reads them
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
                               int local_index, int local_size, auto &&group, auto &&local_scratch, auto &&sub_group ) {
    // Forward = COST, plus the barycenters ONLY when `barycenters` is a bound output (the caller set
    // `with_barycenters`). When it is not, it is a NoneTensor (no `operator=`) so the guarded write
    // and the `moment` it needs both vanish at compile time -- the forward then does cost only, and
    // the backward recomputes any b_i it needs (see `update_outputs_bwd`). `barycenters` is a per-
    // (angle x dirac) buffer ([nb_angles, n] = 80GB at scale), hence off by default.
    const SI nb = src_dist.weights.size();
    sort_diracs( sorted_indices, radix_tmp, sorted_pos, local_index, local_size, group, local_scratch, sub_group );

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
                                   int local_index, int local_size, auto &&group, auto &&local_scratch, auto &&sub_group ) const {
    //   cost = Sum_k Integral_{t_k}^{t_{k+1}} y(x) (x - p_{s(k)})^2 dx,
    // with diracs sorted by position (order s = `sorted_indices`), and t_k = M^{-1}(W_k) the target
    // quantile at the cumulative source mass W_k = Sum_{j<k} w_{s(j)}. The scratch is PER-GROUP and
    // transient (not a forward residual), so the order is RE-DERIVED here via `sort_diracs` -- ONCE,
    // hoisted below and shared by whichever of the position-grad (barycenters not stored) and
    // weights/values-grad blocks actually run, since both need the exact same sorted order and
    // sorting twice would be a pure waste.
    //
    // Each gradient below is guarded at compile time on `is_valid()`: an unperturbed input reaches
    // us as a `NoneTensor` (no `operator=`), so its block must vanish -- see [[differentiation]].
    // Everything runs inside a SYCL kernel, so NO std::vector / dynamic allocation: the per-dirac
    // suffix sum is carried by two scalars instead of an array.
    const TF g = grad_plan.cost; // scalar cotangent seeding `cost`
    const SI nb = src_dist.weights.size();
    const SI lo = ( nb * SI( local_index ) ) / local_size;
    const SI hi = ( nb * SI( local_index + 1 ) ) / local_size;

    // The sort depends only on POSITION, never on which gradient is being computed -- so when BOTH
    // the position-grad block (barycenters not stored) AND the weights/values-grad block need it,
    // sort ONCE here and let both read the same `sorted_indices`/`sorted_pos`, instead of paying a
    // full redundant radix sort per block.
    constexpr bool need_position_resort  = CT_VALUE( src_dist.position_grad_wanted( grad_plan.src_dist ) )
                                         && ! CT_VALUE( barycenters.is_valid() );
    constexpr bool need_weight_value_grad = CT_VALUE( grad_plan.src_dist.weights.is_valid() )
                                          || CT_VALUE( grad_plan.dst_dist.values.is_valid() );
    if constexpr ( need_position_resort || need_weight_value_grad )
        sort_diracs( sorted_indices, radix_tmp, sorted_pos, local_index, local_size, group, local_scratch, sub_group );

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
            // b_i NOT stored: RECOMPUTE it from the sort hoisted above (trades the [nb_angles, n]
            // residual for compute). PARALLEL sweep: each work-item jumps straight to its own chunk's
            // start (`udp_at`, no leader) and walks only `[lo,hi)` -- per-dirac scatter, disjoint
            // indices, no reduction needed.
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
    if constexpr ( need_weight_value_grad ) {
        // sorted above (hoisted, shared with the position-grad block when both run).
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

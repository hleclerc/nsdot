#pragma once

#include <sdot/generated/aggregates/OtPlan1d.h>
#include <loom/support/common_macros.h>

namespace sdot {

SDOT_TEMPLATE_DECL_FOR_OtPlan1d
struct OtPlan1d {
    SDOT_ATTRIBUTES_OF_OtPlan1d

    SCInt ct_dim        = DECAYED_TYPE_OF( nb_dims )::value;
    using TF            = DECAYED_TYPE_OF( cost )::TF;

    // `local_index`/`local_size`/`group`/`local_scratch`: the work-group cooperating on ONE angle
    // (see run_parallel.h/FfiCodeParallel's `group_size` docstring) -- `group_size == 1` degenerates
    // exactly to the single-work-item algorithm these generalize. `sub_group`: the warp cooperating
    // within that work-group, used by the radix histogram (one shared row per sub-group instead of
    // per work-item, see `sort_diracs`'s docstring).
    void  sort_diracs( auto &&sorted_indices, auto &&radix_tmp, auto &&sorted_pos,
                        int local_index, int local_size, auto &&group, auto &&local_scratch, auto &&sub_group ) const;

    // Cooperative chunked-scan helper for the SWEEP (see `update_outputs`/`update_outputs_bwd`).
    // `group_scan` is a per-group TF-valued scratch row (>= `local_size + 1` elements): `local_scratch`
    // is a fixed `int32` local accessor sized for `sort_diracs`'s radix buckets and cannot carry TF
    // sums. Uses the SAME shape as `sort_diracs`'s bucket scan (each work-item reduces its own
    // contiguous sub-range, one barrier, a leader-serial exclusive scan over `local_size` partials,
    // one barrier) and reuses `group_scan`'s slots freely across calls -- each slot `t` is only ever
    // written/read by work-item `t` outside the leader's own already-fenced scan, so no extra
    // synchronization is needed beyond each call's own two barriers.
    //
    // This work-item's cumulative SOURCE weight at its OWN chunk start `lo` (sorted order) -- the
    // `Image::udp_at` argument that lets it jump straight to its chunk's starting `Udp` state.
    TF    chunked_weight_prefix( auto &&sorted_indices, auto &&group_scan, SI lo, SI hi,
                                  int local_index, int local_size, auto &&group ) const;

    // Sort-independent halves of the forward/backward -- the sweep only, given a (however obtained)
    // sorted order. Shared by the internally-sorting entry points below and by the `_presorted` ones,
    // which take that order as already provided (e.g. from `jnp.argsort`, see
    // `OtPlan1d.py::update_outputs_presorted`; [[jax-sort-lax-scan]]).
    void  sweep_outputs( auto &&sorted_indices, auto &&sorted_pos, auto &&group_scan,
                          int local_index, int local_size, auto &&group );
    void  sweep_outputs_bwd( auto &&grad_plan, auto &&sorted_indices, auto &&sorted_pos, auto &&group_scan,
                              int local_index, int local_size, auto &&group ) const;

    void  update_outputs( auto &&sorted_indices, auto &&radix_tmp, auto &&sorted_pos,
                           auto &&group_scan,
                           int local_index, int local_size, auto &&group, auto &&local_scratch, auto &&sub_group );
    void  update_outputs_bwd( auto &&grad_plan, auto &&sorted_indices, auto &&radix_tmp, auto &&sorted_pos,
                               auto &&group_scan,
                               int local_index, int local_size, auto &&group, auto &&local_scratch, auto &&sub_group ) const;

    void  update_outputs_presorted( auto &&sorted_indices, auto &&sorted_pos, auto &&group_scan,
                                     int local_index, int local_size, auto &&group );
    void  update_outputs_bwd_presorted( auto &&grad_plan, auto &&sorted_indices, auto &&sorted_pos, auto &&group_scan,
                                         int local_index, int local_size, auto &&group ) const;
};

}

#include "OtPlan1d.cxx"

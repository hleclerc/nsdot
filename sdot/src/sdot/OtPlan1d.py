from typing import TYPE_CHECKING, cast, overload

from sdot.distributions.Distribution import Distribution
from sdot.distributions.SumOfDiracs import SumOfDiracs

from loom.tensor import CtShapeVar
from loom.tensor import ShapeVar
from loom.tensor import AxisList
from loom.tensor import Tensor
from loom.tensor import Axis

from loom.compilation.FfiCode import FfiCodeParallel
from loom.util import Aggregate
from loom.drivers.driver import driver


class OtPlan1d( Aggregate ):
    """
    """

    # axes
    nb_diracs        : ShapeVar
    nb_dims          : CtShapeVar

    num_dirac        : Axis[ "nb_diracs" ]
    dim              : Axis[ "nb_dims" ]

    # inputs -- both typed by the BASE so any concrete distribution injected here is REUSED (the
    # `__base_init__` idempotency guard reuses an instance that `isinstance`s the annotation; a
    # concrete `SumOfDiracs` annotation would instead try to CONSTRUCT one from a ProjectedSumOfDiracs).
    src_dist         : Distribution
    dst_dist         : Distribution

    # outputs. `barycenters` is produced+stored ONLY when asked for (`with_barycenters`): it is a
    # per-(angle x dirac) buffer ([nb_angles, n] = 80GB at scale). When NOT stored it stays a
    # NoneTensor, and the backward RECOMPUTES the b_i it needs (re-sort + re-sweep) rather than
    # reading it -- so `loss`/`grad` never materialize it. See [[projected-source-fusion]].
    barycenters      : Tensor[ "num_dirac", "dim" ]
    cost             : Tensor

    def __init__( self, src_dist, dst_dist, with_barycenters = False, auto_update = True ):
        # `with_barycenters`: produce and STORE the OT barycenters as an output (readable via
        # `plan.barycenters`). Off by default -- it is an [nb_angles, n] buffer, and the backward can
        # recompute the b_i it needs. Turn it on when you actually want the barycenters, or to trade
        # memory for a faster backward (a stored b_i skips the backward's re-sort + re-sweep).
        self._with_barycenters = bool( with_barycenters )

        # normalization
        src_dist = src_dist.normalized_version()
        dst_dist = dst_dist.normalized_version()
        # dispatch on CAPABILITY, not a concrete type: any `_is_dirac_source` distribution answers the
        # C++ `position(i)` / `add_position_grad` contract OtPlan1d needs (SumOfDiracs reads a buffer,
        # ProjectedSumOfDiracs computes the projection on the fly). Keep the diracs on the src side.
        if getattr( dst_dist, "_is_dirac_source", False ):
            dst_dist, src_dist = src_dist, dst_dist
        if not getattr( src_dist, "_is_dirac_source", False ):
            raise RuntimeError( "For now, at least one of the 2 distributions must be a sum of diracs" )

        # attr init. The batch axes are carried by the (already batched) inputs -- reuse THEM so the
        # plan is co-iterated with its distributions: `apply_batch_axes` batches our own output
        # (`cost`) while the idempotency guard leaves the injected `src_dist`/`dst_dist` untouched
        # (they are already batched over the same axes).
        self.__base_init__(
            nb_diracs = src_dist.nb_diracs.value,
            nb_dims = src_dist.nb_dims.value,
            src_dist = src_dist,
            dst_dist = dst_dist,
            batch_axes = src_dist.batch_axes,
        )

        # computations. `auto_update = False`: skip -- for callers that will instead drive
        # `update_outputs_presorted` themselves (e.g. a `jnp.argsort`-fed, sort-free evaluation path,
        # see [[jax-sort-lax-scan]]); every existing caller keeps the default (unaffected).
        if auto_update:
            self.update_outputs()

    def _scratch( self ):
        """The PER-GROUP sort + sweep scratch tensors + the group axis + the group-SIZE marker tensor,
        shared by every OtPlan1d kernel.

        NOT per-angle: `[ num_group, num_dirac ]`, so memory scales with CONCURRENT ANGLES (a RAM
        budget), not the total angle count (2*80GB -> ~2*nt*n*8 at 1e7 diracs x 1000 angles). Each
        work-GROUP indexes its own row `scratch( group_index )`, race-free (`group_index` unique and
        STABLE over the strided item loop, see FfiCodeParallel) -- `local_size` work-items inside that
        group then cooperate on the row via local memory + barriers (see `OtPlan1d.cxx::sort_diracs`),
        instead of one lone work-item doing it all. `nt`/`gs` are HOST decisions: `nt` (min of cores,
        a RAM budget, the batch size) is UNCHANGED by cooperation -- coop doesn't shrink the O(n)-per-
        concurrent-angle scratch, it puts more workers on each one. `gs` defaults to 1 (degenerates to
        the pre-cooperative algorithm; only CUDA overrides it, see `Device.group_size`).

        Both `num_group` and `num_local` are PLAIN ShapeVars PRESCRIBED to `nt`/`gs`, NOT `CtShapeVar`s:
        the values size the allocation (`static_count` resolves it forward AND in the backward's fresh
        scratch, the ShapeVars being shared) but are READ AT RUN TIME by the kernel -- so the compiled
        kernel is independent of `nt`/`gs`, i.e. of the machine/RAM (a `CtShapeVar` would bake them in
        the type and recompile per box). `num_local_marker` carries NO real data -- it exists purely so
        `group_size`/`thread_cap` can read `.shape( 0 )` off an already-mapped call argument, exactly
        like `thread_cap` already does for `nt` via `sorted_indices.shape( 0 )` (never a literal, so one
        compiled kernel serves every machine). See [[otplan1d-kernel-profile]].

        `group_scan` backs the PARALLEL sweep (see `chunked_weight_prefix`/`update_outputs`): a small
        TF-valued per-group scratch row (`gs+1` entries) for the cooperative chunked-reduction idiom
        shared by every scan in there (weight-prefix, phi-subtotal scan, cost reduction) --
        `local_scratch` (the sort's local-memory accessor) is a fixed `int32` local buffer sized for
        the radix buckets and cannot carry a `TF` sum, hence this separate GLOBAL per-group tensor.
        `cell_cum_mass` itself is NOT scratch anymore: it lives on `dst_dist` (`Image.cell_cum_mass`,
        a cached `ComputedAttribute`, see `Image.ensure_cell_cum_mass`), computed once instead of
        rebuilt by every forward/backward call here.
        """
        n  = int( self.nb_diracs.value )
        nt = driver.device.nb_threads( batch_axes = self.batch_axes, nb_local_bytes_per_thread = 3 * 8 * n )
        # `NB_BUCKETS = 256` (8-bit radix digit, see `OtPlan1d.cxx::sort_diracs`) -- MUST match that
        # constant by hand. The histogram is now ONE ROW PER SUB-GROUP (warp), not per work-item (see
        # `sort_diracs`'s docstring) -- `nb_shared_bytes_per_subgroup` tells `group_size` to budget
        # `ceil( gs / subgroup_size )` rows instead of `gs` of them. `2 *`: each sub-group needs TWO
        # rows now, not one -- a histogram/cursor row AND a match-mask row (the atomic-OR warp-match
        # substitute the scatter phase uses to find same-digit lanes in O(1), see `sort_diracs`'s
        # docstring). `+1` row (`nb_shared_bytes_fixed`): `update_outputs`'s `local_mem_elems`
        # allocates one extra shared row (the cross-chunk bucket offsets) on top -- tell `group_size`
        # about that fixed overhead so its shared-memory budget check matches what actually gets
        # allocated.
        gs = driver.device.group_size( nb_shared_bytes_per_subgroup = 2 * 256 * 4, nb_shared_bytes_fixed = 256 * 4 )
        num_group = Axis( ShapeVar( nt ), name = "num_group" )
        num_local = Axis( ShapeVar( gs ), name = "num_local" )
        num_scan  = Axis( ShapeVar( gs + 1 ), name = "num_local_scan" )   # `+1` = the phi_total broadcast slot
        return {
            "sorted_indices":   Tensor[ num_group, self.num_dirac, dict( dtype = int ) ](),
            "radix_tmp":        Tensor[ num_group, self.num_dirac, dict( dtype = int ) ](),
            "sorted_pos":       Tensor[ num_group, self.num_dirac ](),
            "num_local_marker": Tensor[ num_local, dict( dtype = int ) ](),
            "group_scan":       Tensor[ num_group, num_scan ](),
        }

    def update_outputs( self ):
        # give the target distribution first refusal: e.g. `Image.try_update_otplan1d` solves
        # the whole thing in closed-form pure JAX (no driver.call/C++ kernel at all) when the
        # combination qualifies (JAX backend, 1D, <=1 batch axis, no barycenters, a dirac
        # source that can supply plain positions/weights) -- see [[pure-jax-otplan1d]]. `False`
        # means "unsupported combination", not "error" -- fall through to the general path.
        if self.dst_dist.try_update_otplan1d( self ):
            return

        # second refusal, this time from the BATCH shape itself: when there is one, loop over
        # it with `jax.lax.map` (one `driver.call` per angle) instead of one call handling every
        # angle's memory at once -- see `_update_outputs_via_angle_loop`'s docstring for why.
        if self.batch_axes and self._update_outputs_via_angle_loop():
            return

        # `dst_dist.cell_cum_mass` is read (not built) by the C++ below -- make sure it is materialized
        # BEFORE this call, once, instead of being rebuilt by every forward/backward invocation of it.
        self.dst_dist.ensure_cell_cum_mass()

        # `barycenters` is produced only when asked for: including it in `output_attributes` binds it
        # as an OUTPUT (the forward writes it, guarded on `is_valid()` C++-side); leaving it out keeps
        # it a NoneTensor, and the backward recomputes b_i. This is what the flag decides.
        barycenters_out = [ "plan.barycenters" ] if self._with_barycenters else []
        group_size_expr = "num_local_marker.shape( 0 )"
        driver.call(
            FfiCodeParallel( name = "update_outputs_OtPlan1d",
                # `scratch( group_index )` yields this work-GROUP's shared rank-1 row (`scratch( k )`
                # inside, cooperatively). `plan( batch_index )` still picks the angle. The backward
                # RE-SORTS into its OWN fresh scratch, so it needs all of them -- they are no longer
                # residuals (per-group, their forward content is transient). `local_index`/`local_size`/
                # `group`/`local_scratch` are the reserved cooperative params (see FfiCodeParallel).
                # `group_scan` backs the parallel sweep (see `_scratch`'s docstring); `cell_cum_mass`
                # is read straight off `plan(batch_index).dst_dist`, not passed in here.
                fwd_code = ( "plan( batch_index ).update_outputs( sorted_indices( group_index ), radix_tmp( group_index ), sorted_pos( group_index ), "
                            "group_scan( group_index ), "
                            "local_index, local_size, group, local_scratch, sub_group );" ),
                bwd_code = ( "plan( batch_index ).update_outputs_bwd( grad_for_plan( batch_index ), sorted_indices( group_index ), radix_tmp( group_index ), sorted_pos( group_index ), "
                            "group_scan( group_index ), "
                            "local_index, local_size, group, local_scratch, sub_group );" ),
                # `grad_for_plan.src_dist`'s points gradient (a `ProjectedSumOfDiracs`) is SHARED and
                # ATOMICALLY ACCUMULATED across every angle (see `ProjectedSumOfDiracs::add_position_grad`)
                # -- unlike a per-angle output, Jax/XLA does not guarantee a fresh FFI result buffer
                # starts zeroed, so without this pre-pass the atomic adds land on whatever device memory
                # happened to be there (confirmed: reusing the same call site's output buffer picks up
                # unrelated leftover content). Run ONCE, BEFORE the scaffolded `run_parallel` above -- a
                # per-angle body cannot express a once-only pre-pass (see `FfiCode._setup_code`). Routed
                # through `Tensor::fill_with( queue, ... )` (-> `run_parallel`) inside
                # `zero_position_grad` itself, so this goes through the same, already-proven
                # device-kernel launch path as every other kernel -- no hand-rolled `sycl::queue::submit`
                # here. `zero_position_grad` is a template method (`auto&&` params, like
                # `add_position_grad`), so its OWN `if constexpr` on `position_grad_wanted` genuinely
                # discards the `NoneTensor` branch, exactly like `update_outputs_bwd` gets for
                # `add_position_grad` -- no external gate needed. A no-op for a plain `SumOfDiracs` src
                # (`SumOfDiracs::zero_position_grad`), so this call is unconditional here.
                bwd_setup_code = "plan.src_dist.zero_position_grad( queue, grad_for_plan.src_dist );",
                thread_cap = "sorted_indices.shape( 0 )",
                group_size = group_size_expr,
                # rows: `2 * ceil( local_size / subgroup_size )` shared per-WARP rows -- a histogram/
                # cursor row AND a match-mask row per sub-group cooperating on the radix histogram/
                # scatter (see `sort_diracs`'s docstring) -- plus one shared row for the cross-chunk
                # bucket offsets (see `OtPlan1d.cxx::sort_diracs::radix_pass`). `256` = `NB_BUCKETS`
                # there (kept in sync by hand); `subgroup_size` kept in sync with `Device.subgroup_size`
                # for this device (a baked-in literal -- the compiled kernel already targets one
                # concrete device/backend). Unrelated to `group_scan` (ordinary global scratch, not
                # local memory -- this stays sized for the radix sort only).
                local_mem_elems = (
                    f"( 2 * ( ( ( { group_size_expr } ) + { driver.device.subgroup_size - 1 } ) "
                    f"/ { driver.device.subgroup_size } ) + 1 ) * 256" ),
            ),
            output_attributes = barycenters_out + [ "plan.cost", "plan.nb_diracs", "sorted_indices", "radix_tmp", "sorted_pos", "num_local_marker", "group_scan" ],
            # the scratch buffers are per-group transient: the backward re-allocates them fresh instead
            # of reading the forward's (stale) values as residuals -- see `_call_backward`. `sorted_pos`
            # caches the sorted 1D positions so the sweeps STREAM them (no re-projection).
            # `num_local_marker`/`group_scan` are transient too.
            scratch_attributes = [ "sorted_indices", "radix_tmp", "sorted_pos", "num_local_marker", "group_scan" ],
            # every output count is prescribed from `nb_diracs`, so no capacity can overflow: skip the
            # per-call run-time overflow check (a device->host sync under jit). See driver.call.
            has_dynamic_capacity = False,
            plan = self,
            **self._scratch(),
        )

    def _update_outputs_via_angle_loop( self ):
        """Loop over the (single) batch axis with `jax.lax.map` + `jax.checkpoint`, building a
        FRESH, unbatched `OtPlan1d` (one `driver.call` -- sort AND sweep, unaffected by this)
        per angle, instead of one call whose internal scratch/output memory scales with how many
        CONCURRENT angles it processes at once (`_scratch`'s `nt`, or an `[ nb_angles, n, dim ]`
        `barycenters` output). Mirrors `Image.try_update_otplan1d`'s own `lax.map`, extended to
        the driver.call/C++ path: 'we will always have a large n', so bounding peak memory to
        ONE angle -- regardless of the total angle count -- matters more than batching several
        angles' GPU work concurrently (confirmed: `with_barycenters=True` OOMs today's single
        batched call at n=1e8 x 20 angles; looping fixes it, see [[pure-jax-otplan1d]]). Declines
        (returns False, caller keeps the single batched call) when: not JAX, more than one batch
        axis, or either distribution cannot slice itself (`batch_slice`).
        """
        if len( self.batch_axes ) != 1 or driver.framework.module_name != "jax":
            return False
        # probed with a concrete Python int (outside any trace, so this is free): confirms both
        # sides know how to slice themselves before committing to the loop.
        if self.src_dist.batch_slice( 0 ) is None or self.dst_dist.batch_slice( 0 ) is None:
            return False

        import jax
        import jax.numpy as jnp

        with_barycenters = self._with_barycenters

        @jax.checkpoint
        def body( index ):
            src_i = self.src_dist.batch_slice( index )
            dst_i = self.dst_dist.batch_slice( index )
            plan_i = OtPlan1d( src_i, dst_i, with_barycenters = with_barycenters )
            # `.raw`, not `.tensor`: `plan.nb_diracs` is itself a kernel OUTPUT (re-confirmed
            # device-side, see `update_outputs`'s `output_attributes`), so under this trace it is
            # no longer the static host int `.tensor` needs to slice `barycenters` down to its
            # logical shape (`.tensor`'s own docstring: "a kernel-written count is a device value
            # under a trace, where Python cannot slice by it"). `.raw` reads the buffer as
            # allocated instead -- exactly the logical shape here anyway, since every count in
            # this call is prescribed from `nb_diracs` with no capacity slack.
            if with_barycenters:
                return plan_i.cost.raw, plan_i.barycenters.raw
            return plan_i.cost.raw

        n_batch = self.batch_axes[ 0 ].max
        result = jax.lax.map( body, jnp.arange( n_batch ) )
        if with_barycenters:
            cost, barycenters = result
            self.barycenters = barycenters
        else:
            cost = result
        self.cost = cost
        return True

    def update_outputs_presorted( self, sorted_indices, sorted_pos ):
        """Sort-free variant of `update_outputs`: `sorted_indices`/`sorted_pos` are ALREADY the sorted
        order for this instance's diracs (e.g. from `jnp.argsort`, computed upstream -- one angle at a
        time under an outer `lax.scan`) -- the C++ side skips `sort_diracs` entirely and only runs the
        sweep (`OtPlan1d.cxx::update_outputs_presorted`/`update_outputs_bwd_presorted`). Evaluates
        whether letting XLA's own (whole-device) sort replace the per-angle single-work-group radix
        sort -- the scaling bottleneck at large n, see `sort_diracs`'s docstring -- is worthwhile; see
        [[jax-sort-lax-scan]]. `self` must be a SINGLE (unbatched) instance: one angle per call, so the
        caller supplies the outer angle loop (a `lax.scan`, not this method).

        `sorted_indices`: int array, `sorted_pos`: float array, both shape `[nb_diracs]` (this angle's
        diracs only, already in sorted order -- typically `jnp.argsort(positions)` and
        `positions[jnp.argsort(positions)]` upstream).
        """
        self.dst_dist.ensure_cell_cum_mass()
        group_size_expr = "num_local_marker.shape( 0 )"
        gs = driver.device.group_size( nb_shared_bytes_per_subgroup = 0, nb_shared_bytes_fixed = 0 )
        # ONE group only: this call handles a SINGLE angle (the caller's `lax.scan` supplies the outer
        # angle loop), so there is no "concurrent angles" axis to size `num_group` on (contrast
        # `_scratch`, sized on `nt` concurrent angles for the internally-sorting, all-angles-at-once path).
        num_group = Axis( ShapeVar( 1 ), name = "num_group" )
        num_local = Axis( ShapeVar( gs ), name = "num_local" )
        num_scan  = Axis( ShapeVar( gs + 1 ), name = "num_local_scan" )
        driver.call(
            FfiCodeParallel( name = "update_outputs_presorted_OtPlan1d",
                fwd_code = ( "plan( batch_index ).update_outputs_presorted( sorted_indices( group_index ), sorted_pos( group_index ), "
                            "group_scan( group_index ), "
                            "local_index, local_size, group );" ),
                bwd_code = ( "plan( batch_index ).update_outputs_bwd_presorted( grad_for_plan( batch_index ), sorted_indices( group_index ), sorted_pos( group_index ), "
                            "group_scan( group_index ), "
                            "local_index, local_size, group );" ),
                bwd_setup_code = "plan.src_dist.zero_position_grad( queue, grad_for_plan.src_dist );",
                thread_cap = "sorted_indices.shape( 0 )",
                group_size = group_size_expr,
                # no `sort_diracs` here (order already given) -> no radix-bucket local scratch needed.
                local_mem_elems = "0",
            ),
            output_attributes = [ "plan.cost", "plan.nb_diracs", "num_local_marker", "group_scan" ],
            scratch_attributes = [ "num_local_marker", "group_scan" ],
            has_dynamic_capacity = False,
            plan = self,
            # `Tensor.wrap` only takes axis NAME strings (it mints fresh, detached axes) -- use the SAME
            # names as `num_group`/`self.num_dirac` above so `CallArgsAnalysis` unifies them by name
            # with this call's other tensors instead of minting disconnected ShapeVars.
            sorted_indices = Tensor.wrap( sorted_indices if sorted_indices.ndim == 2 else sorted_indices[ None ],
                                           [ num_group.name, self.num_dirac.name ], dtype = int ),
            sorted_pos = Tensor.wrap( sorted_pos if sorted_pos.ndim == 2 else sorted_pos[ None ],
                                       [ num_group.name, self.num_dirac.name ] ),
            num_local_marker = Tensor[ num_local, dict( dtype = int ) ](),
            group_scan = Tensor[ num_group, num_scan ](),
        )

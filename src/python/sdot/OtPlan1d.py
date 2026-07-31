from typing import TYPE_CHECKING, cast, overload

from sdot.distributions.Distribution import Distribution
from sdot.distributions.SumOfDiracs import SumOfDiracs

from .tensor.CtShapeVar import CtShapeVar
from .tensor.ShapeVar import ShapeVar
from .tensor.AxisList import AxisList
from .tensor.Tensor import Tensor
from .tensor.Axis import Axis

from .compilation.FfiCode import FfiCodeParallel
from .util.Aggregate import Aggregate
from .drivers.driver import driver


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

    def __init__( self, src_dist, dst_dist, with_barycenters = False ):
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

        # computations
        self.update_outputs()

    def _scratch( self ):
        """The three PER-GROUP sort scratch tensors + the group axis + the group-SIZE marker tensor,
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
        """
        n  = int( self.nb_diracs.value )
        nt = driver.device.nb_threads( batch_axes = self.batch_axes, nb_local_bytes_per_thread = 3 * 8 * n )
        # `+1` row: `update_outputs`'s `local_mem_elems` allocates one extra shared row (the
        # cross-chunk bucket offsets) on top of the one-per-work-item rows -- tell `group_size` about
        # that fixed overhead so its shared-memory budget check matches what actually gets allocated.
        gs = driver.device.group_size( nb_shared_bytes_per_group_item = 256 * 4, nb_shared_bytes_fixed = 256 * 4 )
        num_group = Axis( ShapeVar( nt ), name = "num_group" )
        num_local = Axis( ShapeVar( gs ), name = "num_local" )
        return {
            "sorted_indices":   Tensor[ num_group, self.num_dirac, dict( dtype = int ) ](),
            "radix_tmp":        Tensor[ num_group, self.num_dirac, dict( dtype = int ) ](),
            "sorted_pos":       Tensor[ num_group, self.num_dirac ](),
            "num_local_marker": Tensor[ num_local, dict( dtype = int ) ](),
        }

    def update_outputs( self ):
        # `barycenters` is produced only when asked for: including it in `output_attributes` binds it
        # as an OUTPUT (the forward writes it, guarded on `is_valid()` C++-side); leaving it out keeps
        # it a NoneTensor, and the backward recomputes b_i. This is what the flag decides.
        barycenters_out = [ "plan.barycenters" ] if self._with_barycenters else []
        group_size_expr = "num_local_marker.shape( 0 )"
        driver.call(
            FfiCodeParallel( name = "update_outputs_OtPlan1d",
                # `scratch( group_index )` yields this work-GROUP's shared rank-1 row (`scratch( k )`
                # inside, cooperatively). `plan( batch_index )` still picks the angle. The backward
                # RE-SORTS into its OWN fresh scratch, so it needs all three -- they are no longer
                # residuals (per-group, their forward content is transient). `local_index`/`local_size`/
                # `group`/`local_scratch` are the reserved cooperative params (see FfiCodeParallel).
                fwd_code = ( "plan( batch_index ).update_outputs( sorted_indices( group_index ), radix_tmp( group_index ), sorted_pos( group_index ), "
                            "local_index, local_size, group, local_scratch );" ),
                bwd_code = ( "plan( batch_index ).update_outputs_bwd( grad_for_plan( batch_index ), sorted_indices( group_index ), radix_tmp( group_index ), sorted_pos( group_index ), "
                            "local_index, local_size, group, local_scratch );" ),
                thread_cap = "sorted_indices.shape( 0 )",
                group_size = group_size_expr,
                # +1 rows: `local_size` private per-work-item histogram rows, plus one shared row for
                # the cross-chunk bucket offsets (see `OtPlan1d.cxx::sort_diracs::radix_pass`).
                local_mem_elems = f"( { group_size_expr } + 1 ) * 256",
            ),
            output_attributes = barycenters_out + [ "plan.cost", "plan.nb_diracs", "sorted_indices", "radix_tmp", "sorted_pos", "num_local_marker" ],
            # the three scratch buffers are per-group transient: the backward re-allocates them fresh
            # instead of reading the forward's (stale) values as residuals -- see `_call_backward`.
            # `sorted_pos` caches the sorted 1D positions so the sweeps STREAM them (no re-projection).
            # `num_local_marker` is transient too -- it carries no data, only a shape.
            scratch_attributes = [ "sorted_indices", "radix_tmp", "sorted_pos", "num_local_marker" ],
            # every output count is prescribed from `nb_diracs`, so no capacity can overflow: skip the
            # per-call run-time overflow check (a device->host sync under jit). See driver.call.
            has_dynamic_capacity = False,
            plan = self,
            **self._scratch(),
        )

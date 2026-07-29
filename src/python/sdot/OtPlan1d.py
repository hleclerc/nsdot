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
        """The three PER-THREAD sort scratch tensors + the thread axis, shared by every OtPlan1d
        kernel (forward, and the on-demand `compute_barycenters`).

        NOT per-angle: `[ num_thread, num_dirac ]`, so memory scales with CORES not angles
        (2*80GB -> ~2*nt*n*8 at 1e7 diracs x 1000 angles). Each work-item indexes its own row
        `scratch( thread_index )`, race-free (thread_index unique over the strided item loop, see
        FfiCodeParallel). `nt` is a HOST decision (min of cores, a RAM budget, the batch size).
        `num_thread` is a PLAIN ShapeVar PRESCRIBED to `nt`, NOT a `CtShapeVar`: the value sizes the
        allocation (`static_count` resolves it forward AND in the backward's fresh scratch, the
        ShapeVar being shared) but is READ AT RUN TIME by the kernel -- so the compiled kernel is
        independent of `nt`, i.e. of the machine/RAM (a `CtShapeVar` would bake `nt` in the type and
        recompile per box). The work-item cap likewise READS the scratch extent at run time
        (`thread_cap = "sorted_indices.shape( 0 )"`), never a literal. See [[otplan1d-kernel-profile]].
        """
        n  = int( self.nb_diracs.value )
        nt = driver.device.nb_threads( batch_axes = self.batch_axes, nb_local_bytes_per_thread = 2 * 8 * n )
        num_thread = Axis( ShapeVar( nt ), name = "num_thread" )
        return {
            "sorted_indices": Tensor[ num_thread, self.num_dirac, dict( dtype = int ) ](),
            "radix_tmp":      Tensor[ num_thread, self.num_dirac, dict( dtype = int ) ](),
            "sorted_pos":     Tensor[ num_thread, self.num_dirac ](),
        }

    def update_outputs( self ):
        # `barycenters` is produced only when asked for: including it in `output_attributes` binds it
        # as an OUTPUT (the forward writes it, guarded on `is_valid()` C++-side); leaving it out keeps
        # it a NoneTensor, and the backward recomputes b_i. This is what the flag decides.
        barycenters_out = [ "plan.barycenters" ] if self._with_barycenters else []
        driver.call(
            FfiCodeParallel( name = "update_outputs_OtPlan1d",
                # `scratch( thread_index )` yields this work-item's private rank-1 row (`scratch( k )`
                # inside). `plan( batch_index )` still picks the angle. The backward RE-SORTS into its
                # OWN fresh scratch, so it needs all three -- they are no longer residuals (per-thread,
                # their forward content is transient).
                fwd_code = "plan( batch_index ).update_outputs( sorted_indices( thread_index ), radix_tmp( thread_index ), sorted_pos( thread_index ) );",
                bwd_code = "plan( batch_index ).update_outputs_bwd( grad_for_plan( batch_index ), sorted_indices( thread_index ), radix_tmp( thread_index ), sorted_pos( thread_index ) );",
                thread_cap = "sorted_indices.shape( 0 )",
            ),
            output_attributes = barycenters_out + [ "plan.cost", "plan.nb_diracs", "sorted_indices", "radix_tmp", "sorted_pos" ],
            # the three scratch buffers are per-thread transient: the backward re-allocates them fresh
            # instead of reading the forward's (stale) values as residuals -- see `_call_backward`.
            # `sorted_pos` caches the sorted 1D positions so the sweeps STREAM them (no re-projection).
            scratch_attributes = [ "sorted_indices", "radix_tmp", "sorted_pos" ],
            # every output count is prescribed from `nb_diracs`, so no capacity can overflow: skip the
            # per-call run-time overflow check (a device->host sync under jit). See driver.call.
            has_dynamic_capacity = False,
            plan = self,
            **self._scratch(),
        )

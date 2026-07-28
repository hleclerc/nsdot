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

    # inputs
    src_dist         : SumOfDiracs
    dst_dist         : Distribution

    # outputs
    barycenters      : Tensor[ "num_dirac", "dim" ]
    cost             : Tensor

    def __init__( self, src_dist, dst_dist ):
        # normalization
        src_dist = src_dist.normalized_version()
        dst_dist = dst_dist.normalized_version()
        if isinstance( dst_dist, SumOfDiracs ):
            dst_dist, src_dist = src_dist, dst_dist
        if not isinstance( src_dist, SumOfDiracs ):
            raise RuntimeError( "For now, at least one of the 2 distributions must be a sum of diracs" )

        # attr init. The batch axes are carried by the (already batched) inputs -- reuse THEM so the
        # plan is co-iterated with its distributions: `apply_batch_axes` batches our own outputs
        # (`cost`, `barycenters`) while the idempotency guard leaves the injected `src_dist`/`dst_dist`
        # untouched (they are already batched over the same axes).
        self.__base_init__(
            nb_diracs = src_dist.nb_diracs.value,
            nb_dims = src_dist.nb_dims.value,
            src_dist = src_dist,
            dst_dist = dst_dist,
            batch_axes = src_dist.batch_axes,
        )

        # computations
        self.update_outputs()

    def update_outputs( self ):
        # PER-THREAD scratch (not per-angle): the sort buffers are `[ num_thread, num_dirac ]`,
        # shared across the batch, so their memory scales with CORES not angles (2*80GB -> ~2*nt*n*8
        # at 1e7 diracs x 1000 angles). Each work-item indexes its own row `scratch( thread_index )`,
        # race-free because `thread_index` is unique over the strided item loop (see FfiCodeParallel).
        # `nt` is a HOST decision (min of cores, a RAM budget for the 2 int64 buffers, and the batch
        # size). `num_thread` is a PLAIN ShapeVar PRESCRIBED to `nt`, NOT a `CtShapeVar`: the value
        # sizes the allocation (`static_count` resolves it forward AND in the backward's fresh scratch,
        # the ShapeVar being shared) but is READ AT RUN TIME by the kernel -- so the compiled kernel is
        # independent of `nt`, i.e. of the machine/RAM (a `CtShapeVar` would bake `nt` into the type
        # and force a recompile per box). The work-item cap likewise READS the scratch extent at run
        # time (`thread_cap = "sorted_indices.shape( 0 )"`), never a literal. See [[otplan1d-kernel-profile]].
        n  = int( self.nb_diracs.value )
        nt = driver.device.nb_threads( batch_axes = self.batch_axes, nb_local_bytes_per_thread = 2 * 8 * n )
        num_thread = Axis( ShapeVar( nt ), name = "num_thread" )

        driver.call(
            FfiCodeParallel( name = "update_outputs_OtPlan1d",
                # `scratch( thread_index )` yields this work-item's private rank-1 row (`scratch( k )`
                # inside). `plan( batch_index )` still picks the angle. The backward RE-SORTS into its
                # OWN fresh scratch (below), so it needs `radix_tmp` too -- `sorted_indices` is no
                # longer a residual (per-thread, its forward content is transient).
                fwd_code = "plan( batch_index ).update_outputs( sorted_indices( thread_index ), radix_tmp( thread_index ) );",
                bwd_code = "plan( batch_index ).update_outputs_bwd( grad_for_plan( batch_index ), sorted_indices( thread_index ), radix_tmp( thread_index ) );",
                thread_cap = "sorted_indices.shape( 0 )",
            ),
            output_attributes = [ "plan.barycenters", "plan.cost", "plan.nb_diracs", "sorted_indices", "radix_tmp" ],
            # both scratch buffers are per-thread transient: the backward re-allocates them fresh
            # instead of reading the forward's (stale) values as residuals -- see `_call_backward`.
            scratch_attributes = [ "sorted_indices", "radix_tmp" ],
            # every output count is prescribed from `nb_diracs`, so no capacity can overflow: skip the
            # per-call run-time overflow check (a device->host sync under jit). See driver.call.
            has_dynamic_capacity = False,
            sorted_indices = Tensor[ num_thread, self.num_dirac, dict( dtype = int ) ](),
            radix_tmp = Tensor[ num_thread, self.num_dirac, dict( dtype = int ) ](),
            plan = self
        )

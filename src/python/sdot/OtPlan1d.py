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
        driver.call(
            FfiCodeParallel( name = "update_outputs_OtPlan1d",
                # `sorted_indices( batch_index )` yields the per-batch rank-1 scratch the C++ expects
                # (`begin()/end()`, `sorted_indices( k )`); with no batch the optional index falls
                # through and the whole tensor is returned, so this covers both cases.
                fwd_code = "plan( batch_index ).update_outputs( sorted_indices( batch_index ) );",
                bwd_code = "plan( batch_index ).update_outputs_bwd( grad_for_plan( batch_index ), sorted_indices( batch_index ) );",
            ),
            # output_capacities = { "plan.nb_diracs": self.src.nb_diracs.value },
            output_attributes = [ "plan.barycenters", "plan.cost", "plan.nb_diracs", "sorted_indices" ],
            sorted_indices = Tensor[ *self.batch_axes, self.num_dirac, dict( dtype = int ) ](),
            plan = self
        )

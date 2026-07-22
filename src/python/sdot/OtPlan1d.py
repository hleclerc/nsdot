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

        # attr init
        self.__base_init__(
            nb_diracs = src_dist.nb_diracs.value,
            nb_dims = src_dist.nb_dims.value,
            src_dist = src_dist,
            dst_dist = dst_dist,
        )

        # computations
        self.update_outputs()

    def update_outputs( self ):
        driver.call(
            FfiCodeParallel( name = "update_outputs_OtPlan1d",
                fwd_code = "plan( batch_index ).update_outputs( sorted_indices );",
            ),
            # output_capacities = { "plan.nb_diracs": self.src.nb_diracs.value },
            output_attributes = [ "plan.barycenters", "plan.cost", "plan.nb_diracs", "sorted_indices" ],
            sorted_indices = Tensor[ self.num_dirac, *self.batch_axes ](),
            plan = self
        )

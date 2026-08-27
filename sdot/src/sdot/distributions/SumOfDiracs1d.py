from typing import TYPE_CHECKING, cast, overload

from loom.tensor import CtShapeVar
from loom.tensor import ShapeVar
from loom.tensor import RealTensor
from loom.tensor import Axis

# from loom.compilation.FfiCode import FfiCodeParallel
# from loom.util import Aggregate
# from loom.drivers.driver import driver

from .Distribution import Distribution
from .SumOfDiracs import SumOfDiracs


class SumOfDiracs1d( Distribution ):
    """
    """

    nb_diracs        : ShapeVar
    nb_dims          : CtShapeVar

    num_dirac        : Axis[ "nb_diracs" ]

    positions        : RealTensor[ "num_dirac" ]
    weights          : RealTensor[ "num_dirac" ]


    def __init__( self, positions, weights = None, **kwargs ):
        self.__base_init__( positions = positions, weights = weights, nb_dims = 1, **kwargs )

    def normalized_version( self ):
        dim = Axis[ self.nb_dims ]()
        return SumOfDiracs(
            positions = self.positions.append_axis( dim ),
            weights = self.weights,
            batch_axes = self.batch_axes,
        ).normalized_version()

    @property
    def measure( self ):
        if self.weights.is_defined:
            return self.weights.sum()
        return self.nb_diracs.value

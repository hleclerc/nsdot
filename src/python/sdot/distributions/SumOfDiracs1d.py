from typing import TYPE_CHECKING, cast, overload

from ..tensor.CtShapeVar import CtShapeVar
from ..tensor.ShapeVar import ShapeVar
from ..tensor.Tensor import Tensor
from ..tensor.Axis import Axis

# from ..compilation.FfiCode import FfiCodeParallel
# from ..util.Aggregate import Aggregate
# from ..drivers.driver import driver

from .Distribution import Distribution
from .SumOfDiracs import SumOfDiracs


class SumOfDiracs1d( Distribution ):
    """
    """

    nb_diracs        : ShapeVar
    nb_dims          : CtShapeVar

    num_dirac        : Axis[ "nb_diracs" ]

    positions        : Tensor[ "num_dirac" ]
    weights          : Tensor[ "num_dirac" ]


    def __init__( self, positions, weights = None ):
        self.__base_init__( positions = positions, weights = weights, nb_dims = 1 )

    def normalized_version( self ):
        dim = Axis[ self.nb_dims ]()
        return SumOfDiracs(
            positions = self.positions.append_axis( dim ),
            weights = self.weights,
        ).normalized_version()

    @property
    def measure( self ):
        if self.weights.is_defined:
            return self.weights.sum()
        return self.nb_diracs.value

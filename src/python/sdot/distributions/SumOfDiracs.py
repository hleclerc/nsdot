from typing import TYPE_CHECKING, cast, overload

from ..tensor.CtShapeVar import CtShapeVar
from ..tensor.ShapeVar import ShapeVar
from ..tensor.AxisList import AxisList
from ..tensor.Tensor import Tensor
from ..tensor.Axis import Axis

from ..compilation.FfiCode import FfiCodeParallel

from .Distribution import Distribution


class SumOfDiracs( Distribution ):
    """
    """

    nb_diracs        : ShapeVar
    nb_dims          : CtShapeVar

    num_dirac        : Axis[ "nb_diracs" ]
    dim              : Axis[ "nb_dims" ]

    positions        : Tensor[ "num_dirac", "dim" ]
    weights          : Tensor[ "num_dirac" ]


    def __init__( self, positions, weights = None ):
        self.__base_init__( positions = positions, weights = weights, nb_dims = 1 )

    def normalized_version( self ):
        return self

    @property
    def measure( self ):
        if self.weights.is_defined:
            return self.weights.sum()
        return self.nb_diracs.value

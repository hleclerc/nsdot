# from typing import TYPE_CHECKING, cast, overload

# from ..tensor.CtShapeVar import CtShapeVar
# from ..tensor.ShapeVar import ShapeVar
# from ..tensor.AxisList import AxisList
# from ..tensor.Tensor import Tensor
# from ..tensor.Axis import Axis

# from ..compilation.FfiCode import FfiCodeParallel
# from ..drivers.driver import driver

from ..util.Aggregate import Aggregate


class Distribution( Aggregate ):
    """
    """

    @property
    def measure( self ):
        raise NotImplementedError

    def normalized_version( self ):
        return self

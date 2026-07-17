from typing import TYPE_CHECKING, cast, overload

from .tensor.CtShapeVar import CtShapeVar
from .tensor.ShapeVar import ShapeVar
from .tensor.AxisList import AxisList
from .tensor.Tensor import Tensor
from .tensor.Axis import Axis

from .compilation.FfiCode import FfiCodeParallel
from .util.aggregate import Aggregate
from .drivers.driver import driver


class Image( Aggregate ):
    nb_dims          : CtShapeVar
    shape            : ShapeVar[ "dim" ]

    num_knot         : Axis[ "shape + 1" ]
    img_pos          : AxisList[ "dim", "shape" ]
    dim              : Axis[ "nb_dims" ]

    origin           : Tensor[ "dim" ]
    values           : Tensor[ "img_pos..." ]
    knots            : Tensor[ "dim", "num_knot" ]

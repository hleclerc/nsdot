from typing import TYPE_CHECKING, cast, overload

from .tensor.CtShapeVar import CtShapeVar
from .tensor.ShapeVar import ShapeVar
from .tensor.AxisList import AxisList
from .tensor.Tensor import Tensor
from .tensor.Axis import Axis

from .compilation.FfiCode import FfiCodeParallel
from .util.Aggregate import Aggregate
from .drivers.driver import driver


class Image( Aggregate ):
    """
        Piecewise constant function.

        Each square/cube/hypercube is defined by `origin` and `frame( dir ) * knots( ... )`

        By default, knots is equal to 0, 1, ... for each dim.
    """

    nb_dims          : CtShapeVar
    shape            : ShapeVar[ "dim" ]

    num_knot         : Axis[ "shape + 1" ]
    img_pos          : AxisList[ "dim", "shape" ]
    dim              : Axis[ "nb_dims" ]
    dir              : Axis[ "nb_dims" ]

    values           : Tensor[ "img_pos..." ]

    origin           : Tensor[ "dim" ]
    frame            : Tensor[ "dir", "dim" ]
    knots            : Tensor[ "dim", "num_knot" ]

    @property
    def measure( self ):
        res = Tensor[ tuple( self.batch_axes ) ]()
        driver.call(
            FfiCodeParallel( name = "measure",
                fwd_code = "res( batch_index ) = image( batch_index ).measure();",
                # bwd_code = "cell( batch_index ).measure_bwd( res( batch_index ), item_map( batch_index ), nb_map_items, "
                           # "grad_for_res( batch_index ), grad_for_cell( batch_index ).vertex_positions );",
            ),
            output_attributes = [ "res" ],
            image = self,
            res = res
        )
        return res

from typing import TYPE_CHECKING, cast, overload

from ..tensor.CtShapeVar import CtShapeVar
from ..tensor.ShapeVar import ShapeVar
from ..tensor.AxisList import AxisList
from ..tensor.Tensor import Tensor
from ..tensor.Axis import Axis

from ..util.ComputedAttribute import ComputedAttribute
from ..compilation.FfiCode import FfiCodeParallel
from ..drivers.driver import driver

from .Distribution import Distribution


class Image( Distribution ):
    """
        Piecewise constant function on a grid.

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

    current_mass     : ComputedAttribute[ Tensor, ( "values", "frame", "knots" ) ]

    def __init__( self, values, **kwargs ) -> None:
        self.__base_init__( values = values, target_mass = 1.0, **kwargs )

    def normalized_version( self ):
        # update mass
        mass = self.mass

        # normalize
        if self.target_mass.is_defined:
            return Image(
                nb_dims = self.nb_dims.value,
                shape = self.shape.value,

                values = self.target_mass / mass * self.values,

                origin = self.origin,
                frame = self.frame,
                knots = self.knots,

                current_mass = self.target_mass
            )

        return self

    def _update_current_mass( self ):
        # res = Tensor[ tuple( self.batch_axes ) ]()
        driver.call(
            FfiCodeParallel( name = "measure",
                fwd_code = "image.current_mass( batch_index ) = image( batch_index ).measure();",
                # bwd_code = "cell( batch_index ).measure_bwd( res( batch_index ), item_map( batch_index ), nb_map_items, "
                           # "grad_for_res( batch_index ), grad_for_cell( batch_index ).vertex_positions );",
            ),
            output_attributes = [ "image.current_mass" ],
            image = self,
        )

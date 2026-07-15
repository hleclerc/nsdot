from typing import TYPE_CHECKING, cast, overload

from .tensor.CtShapeVar import CtShapeVar
from .tensor.ShapeVar import ShapeVar
from .tensor.Tensor import Tensor
from .tensor.Axis import Axis

from .compilation.FfiCode import FfiCodeParallel
from .util.aggregate import aggregate
from .drivers.driver import driver


@aggregate
class Cell:
    nb_vertices      : ShapeVar #
    nb_edges         : ShapeVar #
    nb_cuts          : ShapeVar #

    nb_dims          : CtShapeVar #

    num_vertex       : Axis[ "nb_vertices" ]
    num_edge         : Axis[ "nb_edges" ]
    num_cut          : Axis[ "nb_cuts" ]
    ein              : Axis[ "nb_dims + 1" ]
    dim              : Axis[ "nb_dims" ]

    is_fully_bounded : Tensor
    vertex_positions : Tensor[ "num_vertex", "dim" ]
    vertex_indices   : Tensor[ "num_vertex", "dim", dict( dtype = int ) ]
    edge_indices     : Tensor[ "num_vertex", "ein", dict( dtype = int ) ]
    cut_vectors      : Tensor[ "num_cut", "dim" ]
    cut_offsets      : Tensor[ "num_cut" ]
    cut_ids          : Tensor[ "num_cut", dict( dtype = int ) ]


    if TYPE_CHECKING:
        def __base_init__( self, *args, **kwargs ): ...

    def __init__( self, nb_dims, init_unbounded = True ):
        self.__base_init__( nb_dims = nb_dims )

        if init_unbounded:
            self.init_unbounded()

    def init_unbounded( self ):
        driver.call(
            FfiCodeParallel( name = "init_unbounded", fwd_code = "cell( batch_index ).init_unbounded();" ),
            output_attributes = [
                "cell.cut_vectors", "cell.cut_offsets", "cell.cut_ids",
                "cell.nb_vertices", "cell.nb_cuts", "cell.nb_edges",
                "cell.vertex_positions", "cell.is_fully_bounded",
            ],
            capacities = { "cell.nb_vertices": 8, "cell.nb_cuts": 8, "cell.nb_edges": 8 },
            cell = self,
        )

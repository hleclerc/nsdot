from typing import TYPE_CHECKING, cast, overload

from .tensor.CtShapeVar import CtShapeVar
from .tensor.ShapeVar import ShapeVar
from .tensor.Tensor import Tensor
from .tensor.Axis import Axis

from .compilation.FfiCode import FfiCodeParallel
from .util.aggregate import aggregate
from .drivers.driver import driver

INFINITE = -2
BOUNDARY = -1


@aggregate
class Cell:
    nb_vertices      : ShapeVar #
    nb_edges         : ShapeVar #
    nb_cuts          : ShapeVar #

    nb_dims          : CtShapeVar #

    num_vertex       : Axis[ "nb_vertices" ]
    num_edge         : Axis[ "nb_edges" ]
    num_axis         : Axis[ "nb_dims" ]
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


    def __init__( self, nb_dims, init_as_unbounded = True ):
        self.__base_init__( nb_dims = nb_dims )

        if init_as_unbounded:
            self.init_as_unbounded()

    @staticmethod
    def make_hypercube( nb_dims, origin = None, axes = None, cut_id = BOUNDARY ):
        res = Cell( nb_dims, init_as_unbounded = False )
        res.init_as_hypercube( origin, axes, cut_id )
        return res

    @staticmethod
    def make_unbounded( nb_dims ):
        return Cell( nb_dims )

    def init_as_unbounded( self ):
        driver.call(
            FfiCodeParallel( name = "init_as_unbounded", fwd_code = "cell( batch_index ).init_as_unbounded();" ),
            capacities = { "cell.nb_vertices": 8, "cell.nb_cuts": 8, "cell.nb_edges": 8 },
            output_attribute_exceptions = self._output_attribute_exceptions(),
            output_attributes = [ "cell" ],
            cell = self,
        )

    def init_as_hypercube( self, origin = None, axes = None, cut_id = BOUNDARY ):
        origin = Tensor[ self.dim ]( origin )
        axes = Tensor[ self.num_axis, self.dim ]( axes )

        driver.call(
            FfiCodeParallel( name = "init_as_hypercube", fwd_code = "cell( batch_index ).init_as_hypercube( origin, axes, cut_id );" ),
            capacities = { "cell.nb_vertices": 8, "cell.nb_cuts": 8, "cell.nb_edges": 8 },
            output_attribute_exceptions = self._output_attribute_exceptions(),
            output_attributes = [ "cell" ],
            cut_id = cut_id,
            origin = origin,
            axes = axes,
            cell = self,
        )

    def _output_attribute_exceptions( self ):
        if self.nb_dims <= 2:
            return [ "cell.vertex_indices", "cell.edge_indices" ]
        return []

    # def init_as_hypercube( self ):
    #     output_attribute_exceptions = []
    #     if self.nb_dims <= 2:
    #         output_attribute_exceptions = [ "cell.vertex_indices", "cell.edge_indices" ]

    #     driver.call(
    #         FfiCodeParallel( name = "init_as_unbounded", fwd_code = "cell( batch_index ).init_as_unbounded();" ),
    #         capacities = { "cell.nb_vertices": 8, "cell.nb_cuts": 8, "cell.nb_edges": 8 },
    #         output_attribute_exceptions = output_attribute_exceptions,
    #         output_attributes = [ "cell" ],
    #         cell = self,
    #     )

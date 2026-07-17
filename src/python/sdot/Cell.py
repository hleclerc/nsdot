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
    nb_vertices      : ShapeVar
    nb_edges         : ShapeVar
    nb_cuts          : ShapeVar

    nb_dims          : CtShapeVar

    num_vertex       : Axis[ "nb_vertices" ]
    num_edge         : Axis[ "nb_edges" ]
    num_axis         : Axis[ "nb_dims" ]
    num_cut          : Axis[ "nb_cuts" ]
    ein              : Axis[ "nb_dims + 1" ]
    dim              : Axis[ "nb_dims" ]

    is_fully_bounded : Tensor

    vertex_positions : Tensor[ "num_vertex", "dim" ]
    vertex_indices   : Tensor[ "num_vertex", "dim", dict( dtype = int ) ]

    edge_indices     : Tensor[ "num_edge", "ein", dict( dtype = int ) ]

    cut_directions   : Tensor[ "num_cut", "dim" ]
    cut_offsets      : Tensor[ "num_cut" ]
    cut_ids          : Tensor[ "num_cut", dict( dtype = int ) ]

    if TYPE_CHECKING:
        def __base_init__( self, *args, **kwargs ): ...
        def apply_batch_axes( self, batch_axes ): ...
        batch_axes: list


    def __init__( self, nb_dims, init_as_unbounded = True, batch_axes = None ):
        # `batch_axes = [ new_batch_axis( n ), ... ]` batches this cell: every declared tensor gains
        # those leading axes. `__base_init__` does the work (and records them on `self.batch_axes`);
        # each axis already carries its size (a prescribed `ShapeVar`), so there is nothing else to
        # thread here.
        self.__base_init__( nb_dims = nb_dims, batch_axes = batch_axes )

        if init_as_unbounded:
            self.init_as_unbounded()

    @classmethod
    def make_hypercube( cls, nb_dims, origin = None, axes = None, cut_id = BOUNDARY, batch_axes = None ):
        res = cls( nb_dims, init_as_unbounded = False, batch_axes = batch_axes )
        res.init_as_hypercube( origin, axes, cut_id )
        return res

    @classmethod
    def make_unbounded( cls, nb_dims, batch_axes = None ):
        return cls( nb_dims, batch_axes = batch_axes )


    def init_as_unbounded( self, batch_axes = None ):
        if batch_axes is not None:
            self.apply_batch_axes( batch_axes )

        driver.call(
            FfiCodeParallel( name = "init_as_unbounded", fwd_code = "cell( batch_index ).init_as_unbounded();" ),
            output_capacities = self._init_capacities(),
            output_exceptions = self._output_attribute_exceptions(),
            output_attributes = [ "cell" ],
            cell = self,
        )

    def init_as_hypercube( self, origin = None, axes = None, cut_id = BOUNDARY, batch_axes = None ):
        # batching an already-built cell: prepend the axes to its tensors before we allocate them
        # (`origin` / `axes` stay unbatched -- shared across items). Harmless if the cell was already
        # batched at construction and none is passed here.
        if batch_axes is not None:
            self.apply_batch_axes( batch_axes )

        origin = Tensor[ self.dim ]( origin )
        axes = Tensor[ self.num_axis, self.dim ]( axes )

        driver.call(
            FfiCodeParallel( name = "init_as_hypercube",
                fwd_code = "cell( batch_index ).init_as_hypercube( origin, axes, cut_id );",
                bwd_code = "cell( batch_index ).init_as_hypercube_bwd( origin, axes, grad_for_cell( batch_index ), grad_for_origin( batch_index ), grad_for_axes( batch_index ) );"
            ),
            output_exceptions = self._output_attribute_exceptions(),
            output_capacities = self._init_capacities(),
            output_attributes = [ "cell" ],
            cut_id = cut_id,
            origin = origin,
            axes = axes,
            cell = self,
        )

    @property
    def measure( self ):
        nb_map_items = ShapeVar()
        nb_threads = ShapeVar()

        # named axes for `item_map`: giving them a name lets the body index them by keyword
        # (`item_map( num_map_item = 0, ... )`). Using them in the tensor spec is enough to register
        # them -- `CallArg_Tensor.cpp_axis_names()` emits their `DEFINE_AXIS`; they need NOT be
        # passed as call kwargs (an `Axis` lowers to no data anyway). Built DIRECTLY, not via
        # `Axis[ sv ]( name = ... )`: `Parametrized.__call__` would fold `name` into template_kwargs,
        # never reaching `AbstractAxis.__init__`'s `name`, and the axis would stay anonymous (`a0`).
        num_map_item = Axis( nb_map_items, name = "num_map_item" )
        num_thread = Axis( nb_threads, name = "num_thread" )

        nt = driver.device.nb_threads(
            nb_local_bytes_per_thread = self._measure_bytes_per_thread(),
            batch_axes = self.batch_axes,
        )

        # `res` and `item_map` are outputs the method builds itself (not aggregate members), so they
        # must gain the batch axes explicitly -- `Tensor[ *batch_axes, ... ]`, empty for a plain
        # `Cell`. The body then indexes `res` by `batch_index` just like `cell`: a no-op when
        # unbatched (empty multi-index), one write per item when batched.
        res = Tensor[ tuple( self.batch_axes ) ]()

        driver.call(
            FfiCodeParallel( name = "measure",
                fwd_code = "cell( batch_index ).measure( res( batch_index ), item_map( batch_index ), nb_map_items );",
                bwd_code = "cell( batch_index ).measure_bwd( res( batch_index ), item_map( batch_index ), nb_map_items, "
                           "grad_for_res( batch_index ), grad_for_cell( batch_index ).vertex_positions );",
            ),
            output_capacities = { "nb_map_items": self._nb_map_items(), "nb_threads": nt, },
            output_attributes = [ "res", "item_map", "nb_map_items" ],
            # `measure` reads only `vertex_positions` (and `is_fully_bounded`); the nD gradient
            # will do the same (fan triangulation, not the facet-normal formula) -- so none of
            # these ever cross the FFI or become a differentiable primal for this call.
            input_exceptions = [
                "cell.vertex_indices", "cell.edge_indices",
                "cell.cut_directions", "cell.cut_offsets", "cell.cut_ids",
            ],
            nb_map_items = nb_map_items,
            nb_threads = nb_threads,
            item_map = Tensor[ tuple( self.batch_axes ) + ( num_map_item, num_thread ) ](),
            cell = self,
            res = res
        )

        return res

    def _nb_map_items( self ):
        nb_cuts, nb_dims = int( self.nb_cuts.value ), int( self.nb_dims.value )
        res = 0
        res += nb_cuts * ( nb_dims >= 2 )
        res += nb_cuts * nb_cuts * ( nb_dims >= 3 )
        return res

    def _measure_bytes_per_thread( self ):
        # a thread's scratch is its column of `item_map`: `nb_map_items` scalars. `nb_map_items`
        # grows with `nb_cuts` (~nb_cuts^2 in 3D), which is why the per-thread room -- hence how
        # many threads fit in RAM -- depends on the cut-count. Rough double-sized (8-byte) bound.
        return self._nb_map_items() * 8

    def _init_capacities( self ):
        # a hypercube has 2^d vertices, d*2^(d-1) edges and 2d cuts. We allocate ~2x that (a floor
        # of 8, the value hard-coded for 2D) as headroom for the cuts a cell takes over its life,
        # not just its initial shape. A flat 8 was too small in 3D (12 edges) -- an under-provision
        # is now caught (recorded + clamped by ShapeVarView, writes bounded) rather than corrupting,
        # but we still size it right so the call goes through in one shot.
        d = int( self.nb_dims.value )
        return {
            "cell.nb_vertices": max( 8, 2 * 2 ** d ),
            "cell.nb_edges":    max( 8, 2 * d * 2 ** ( d - 1 ) ),
            "cell.nb_cuts":     max( 8, 2 * ( 2 * d ) ),
        }

    def _output_attribute_exceptions( self ):
        if int( self.nb_dims.value ) <= 2:
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

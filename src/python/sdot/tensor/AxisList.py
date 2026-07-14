import numpy

from ..util.aggregate import get_attribute
from .AbstractAxis import AbstractAxis


class AxisList( AbstractAxis ):
    """A *family* of axes indexed by a loop axis, meant to be UNROLLED.

    `AxisList[ loop_axis, expr ]`: `loop_axis` is the axis to unroll over
    (e.g. `dim`), `expr` the affine extent of each member (e.g. `extent`, with
    `extent : ShapeVar[ "dim" ]` holding one count per loop index).

    Used in a `Tensor` declaration with a trailing `...` (`Tensor[ "img_pos..." ]`)
    it expands into `nb_dims` separate static axes, giving the tensor a DYNAMIC
    rank. The count `nb_dims` is unknown at declaration time -- hence the split
    from `Axis` (a single, ragged-or-not, dimension needs no unrolling)."""

    def _init_axis( self, parent_inst, template_args ):
        assert len( template_args ) == 2
        self.loop_axis = get_attribute( template_args[ 0 ], parent_inst )
        self._parse_expr( parent_inst, template_args[ 1 ] )

    def max_list( self ):
        # one extent per loop index: `offset + sum( coeff * shape_var[k] )`, where
        # the loop count is the loop axis' extent (`nb_dims`) and each coeff's
        # ShapeVar is a rank-1 vector of that length.
        res = numpy.full( self.loop_axis.max, self.offset, dtype = int )
        for shape_var, m in self.coeffs.items():
            res = res + m * numpy.asarray( shape_var.value, dtype = int )
        return [ int( x ) for x in res ]

    def capacity_list( self, capacity_of ):
        # an unrolled AxisList is dense: it holds no reservation, so its extents ARE its counts
        # (each of its ShapeVars is a vector, which a scalar capacity could not describe anyway).
        return self.max_list()

    def register_in( self, tensor, index, unroll ):
        assert unroll, "an AxisList must be unrolled in a tensor ('img_pos...')"

        # Unrolled at `index`, this family expands into `count` static axes; the
        # span (start, count) is filled by `Tensor.set` once a value is observed.
        # The loop axis' ShapeVar is solved from the unroll count; each member's
        # ShapeVar(s) from the vector of per-index sizes.
        # `allocated` makes no difference here: an unrolled tensor is dense, so its allocated
        # sizes are its logical ones.
        for shape_var in self.loop_axis.coeffs:
            def resolve_count( t, allocated, axis = self.loop_axis, shape_var = shape_var, index = index ):
                span = t._unroll_spans.get( index )
                if span is None:
                    return None
                return axis.solve_single( shape_var, numpy.array( span[ 1 ], dtype = int ) )
            shape_var.add_usage( tensor, resolve_count )

        for shape_var in self.coeffs:
            def resolve_vec( t, allocated, axis = self, shape_var = shape_var, index = index ):
                span = t._unroll_spans.get( index )
                if span is None or t._sizes is None:
                    return None
                start, count = span
                vals = [ axis.solve_single( shape_var, t._sizes[ start + k ] ) for k in range( count ) ]
                if any( v is None for v in vals ):
                    return None
                return numpy.array( vals, dtype = int )
            shape_var.add_usage( tensor, resolve_vec )

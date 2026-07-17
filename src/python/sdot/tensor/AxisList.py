import numpy

from ..util.Attribute import resolve_attribute
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

    def _init_axis( self, args, scope ):
        assert len( args ) == 2, "an AxisList takes a loop axis and an extent expression"
        self.loop_axis = resolve_attribute( args[ 0 ], scope, AbstractAxis )
        self._parse_expr( args[ 1 ], scope )

    def max_list( self ):
        # one extent per loop index: `offset + sum( coeff * shape_var[k] )`, where
        # the loop count is the loop axis' extent (`nb_dims`) and each coeff's
        # ShapeVar is a rank-1 vector of that length.
        res = numpy.full( self.loop_axis.max, self.offset, dtype = int )
        for shape_var, m in self.coeffs.items():
            res = res + m * numpy.asarray( shape_var.raw, dtype = int )
        return [ int( x ) for x in res ]

    def capacity_list( self, capacity_of ):
        # an unrolled AxisList is dense: it holds no reservation, so its extents ARE its counts
        # (each of its ShapeVars is a vector, which a scalar capacity could not describe anyway).
        return self.max_list()

    def register_in( self, tensor, index, unroll ):
        assert unroll, "an AxisList must be unrolled in a tensor ('img_pos...')"

        # Unrolled at `index`, this family expands into `count` static axes spanning the buffer's
        # dimensions `[ start, start + count )` (`tensor._unroll_span`). The resolvers serve the
        # ALLOCATED capacity: the loop axis' ShapeVar from the span count, each member's from the
        # buffer sizes over the span. An unrolled tensor is dense, so allocated == logical here; the
        # LOGICAL counts are pushed at set time by `observe_span`.
        for shape_var in self.loop_axis.coeffs:
            def resolve_count( t, axis = self.loop_axis, shape_var = shape_var, index = index ):
                span = t._unroll_span( index )
                if span is None:
                    return None
                return axis.solve_single( shape_var, numpy.array( span[ 1 ], dtype = int ) )
            shape_var.add_usage( tensor, resolve_count )

        for shape_var in self.coeffs:
            def resolve_vec( t, axis = self, shape_var = shape_var, index = index ):
                span  = t._unroll_span( index )
                sizes = t.allocated_sizes
                if span is None or sizes is None:
                    return None
                start, count = span
                vals = [ axis.solve_single( shape_var, sizes[ start + k ] ) for k in range( count ) ]
                if any( v is None for v in vals ):
                    return None
                return numpy.array( vals, dtype = int )
            shape_var.add_usage( tensor, resolve_vec )

    def observe_span( self, sizes ):
        """Push the LOGICAL sizes of the unrolled span onto our ShapeVars: the loop axis' from how
        many dimensions we span (`len( sizes )`), each member's from the per-index vector. Mirror of
        the two resolvers above, but in PUSH -- so an unrolled tensor needs no `_sizes` cache."""
        for shape_var in self.loop_axis.coeffs:
            solved = self.loop_axis.solve_single( shape_var, numpy.array( len( sizes ), dtype = int ) )
            if solved is not None:
                shape_var.observe( solved )

        for shape_var in self.coeffs:
            vals = [ self.solve_single( shape_var, s ) for s in sizes ]
            if all( v is not None for v in vals ):
                shape_var.observe( numpy.array( vals, dtype = int ) )

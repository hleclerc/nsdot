from ..util.aggregate import get_attribute
from ..util.Attribute import Attribute


class AbstractAxis( Attribute ):
    """Common base for `Axis` and `AxisList`.

    Both name a tensor dimension whose extent is an affine expression of
    `ShapeVar`s. The shared part -- factored here -- is the *parsing* of that
    expression into `coeffs` (a `{ ShapeVar: int }` map) and an integer
    `offset`, plus the single-variable inversion used to solve a `ShapeVar`
    from an observed size.

    They differ in what is known at declaration time (virtual `register_in`):
    an `Axis` is a single dimension (ragged when one of its `ShapeVar`s has
    rank > 0); an `AxisList` is a *family* of dimensions to be unrolled, whose
    count (`nb_dims`) is unknown when the class is declared.
    """

    def __init__( self, parent_inst = None, /, template_args = [], template_kwargs = {} ) -> None:
        from .ShapeVar import ShapeVar
        self.coeffs: dict[ ShapeVar, int ] = {}
        self.offset = 0
        self._init_axis( parent_inst, template_args )

    # ---- shared affine parser ----
    def _parse_expr( self, parent_inst, expr ):
        """Parse an affine expression like "2 * nb_dims + 3 * nb_xs + 1" into
        `coeffs` / `offset`. Spaces are dropped and subtraction is turned into
        the addition of a negative term."""
        expr = expr.replace( " ", "" ).replace( "-", "+-" )
        for term in ( t for t in expr.split( "+" ) if t ):
            if term.lstrip( "-" ).isdigit():
                self.offset += int( term )
            elif "*" in term:
                coeff_str, var_name = term.split( "*", 1 )
                self._add_coeff( parent_inst, var_name, int( coeff_str ) )
            else:
                self._add_coeff( parent_inst, term.lstrip( "+-" ),
                                 -1 if term[ 0 ] == "-" else 1 )

    def _add_coeff( self, parent_inst, var_name, coeff ):
        from .ShapeVar import ShapeVar
        shape_var = get_attribute( var_name, parent_inst )
        assert isinstance( shape_var, ShapeVar )
        self.coeffs[ shape_var ] = self.coeffs.get( shape_var, 0 ) + coeff

    def solve_single( self, shape_var, size ):
        """Invert the single-variable affine `size = coeff * shape_var + offset`.

        Returns `(size - offset) // coeff` (a scalar or a per-segment array,
        following the shape of `size`), or `None` if `shape_var` is not the sole
        variable of this axis (the multi-variable case is a TODO linear solve)."""
        if list( self.coeffs.keys() ) != [ shape_var ]:
            return None
        return ( size - self.offset ) // self.coeffs[ shape_var ]

    def set( self, value ):
        raise RuntimeError( "An axis cannot be set" )

    # ---- extents (virtual) ----
    def max_list( self ):
        """The member's extents as a list to be concatenated into a tensor shape:
        one entry for an `Axis`, `nb_dims` entries for an unrolled `AxisList`."""
        raise NotImplementedError

    # ---- usage registration (virtual) ----
    def register_in( self, tensor, index, unroll ):
        """Record on our `ShapeVar`s that `tensor`'s dimension `index` constrains
        them, so they can be solved from that tensor's observed sizes. `unroll`
        (the trailing `...` in the declaration) is only valid for an `AxisList`."""
        assert not unroll, "only an AxisList member ('name...') can be unrolled"
        self._register_dense( tensor, index )

    def _register_dense( self, tensor, index ):
        """One declared axis <-> its own array dimension (`tensor._spec_dims[ index ]`,
        which accounts for an unrolled AxisList sibling spanning several dimensions):
        each of our `ShapeVar`s is solved from that size (a scalar when dense, a
        per-segment array when ragged)."""
        for shape_var in self.coeffs:
            def resolve( t, axis = self, index = index, shape_var = shape_var ):
                if t._sizes is None:
                    return None
                return axis.solve_single( shape_var, t._sizes[ t._spec_dims[ index ] ] )
            shape_var.add_usage( tensor, resolve )

    # ---- virtual ----
    def _init_axis( self, parent_inst, template_args ):
        raise NotImplementedError

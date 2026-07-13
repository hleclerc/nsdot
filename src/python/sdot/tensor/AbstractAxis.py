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

    @classmethod
    def make_CallArg( cls, caa, io_category, name, value, ctor_args, schema = None ):
        # An axis is not a buffer, but it is resolvable: a `CallArg_Axis` node that carries its
        # affine extent and computes it on demand (see `CallArg_Tensor.resolve_shape`).
        from ..drivers.CallArg_Axis import CallArg_Axis
        return CallArg_Axis( caa, io_category, name, schema.args[ 0 ] )

    def __init__( self, parent_inst = None, /, template_args = [], template_kwargs = {} ) -> None:
        from .ShapeVar import ShapeVar
        self.coeffs: dict[ ShapeVar, int ] = {}
        self.offset = 0
        self._init_axis( parent_inst, template_args )

    # ---- shared affine parser ----
    @staticmethod
    def parse_affine( expr ):
        """Pure parse of an affine expression like "2 * nb_dims + 3 * nb_xs + 1" into
        `( { var_name: coeff }, offset )`. No `ShapeVar` resolution -- names stay strings, so
        this is usable without a parent instance (e.g. shape resolution in `CallArg_Tensor`).
        Spaces are dropped and subtraction becomes the addition of a negative term."""
        coeffs = {}
        offset = 0
        expr = expr.replace( " ", "" ).replace( "-", "+-" )
        for term in ( t for t in expr.split( "+" ) if t ):
            if term.lstrip( "-" ).isdigit():
                offset += int( term )
            elif "*" in term:
                coeff_str, var_name = term.split( "*", 1 )
                coeffs[ var_name ] = coeffs.get( var_name, 0 ) + int( coeff_str )
            else:
                name = term.lstrip( "+-" )
                coeffs[ name ] = coeffs.get( name, 0 ) + ( -1 if term[ 0 ] == "-" else 1 )
        return coeffs, offset

    def _parse_expr( self, parent_inst, expr ):
        """Instance-side parse: fill `coeffs` (resolving names to `ShapeVar`s via
        `get_attribute`) and `offset` from the affine expression."""
        coeffs, offset = self.parse_affine( expr )
        self.offset += offset
        for var_name, coeff in coeffs.items():
            self._add_coeff( parent_inst, var_name, coeff )

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

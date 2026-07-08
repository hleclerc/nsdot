from typing_extensions import overload

from ..util.Parametrized import Parametrized
from ..util.aggregate import get_attribute
from ..util.Attribute import Attribute
# from .ShapeExpr import ShapeExpr


class Axis( Attribute ):
    """A named tensor dimension whose extent is an affine expression of `ShapeVar`s.

    A single `ShapeVar` can drive several axes, which is why `ShapeVar` and
    `Axis` are distinct (e.g. `nb_dims` drives both `nvec = nb_dims + 1` and
    `dim = nb_dims`).

    An `Axis` carries no per-instance state: its concrete extent is a pure
    function of the `ShapeVar` cells. As a descriptor, `c.dim` reads as the
    evaluated extent (an `int`, or `None` while still unsolved).

    The axis can be RAGGED if `extent` depends on other axes: each row/col/...
    may have a different length, so there is no single extent.
    """

    def __init__( self, parent_inst = None, /, template_args = [], template_kwargs = {} ) -> None:
        from .ShapeVar import ShapeVar
        self.coeffs: dict[ ShapeVar, int ] = {}
        self.offset = 0

        assert len( template_args ) == 1
        expr = template_args[ 0 ]
        # Ex: "2 * nb_dims + 3 * nb_xs + 1"

        # Normalize: remove spaces and convert subtraction to addition
        expr = expr.replace(" ", "").replace("-", "+-")
        terms = [t for t in expr.split("+") if t]

        for term in terms:
            # Check if term is purely numeric (constant)
            if term.lstrip("-").isdigit():
                self.offset += int(term)
            else:
                # Parse "coeff * var_name" or just "var_name"
                if "*" in term:
                    coeff_str, var_name = term.split("*", 1)
                    coeff = int(coeff_str)
                    var_name = var_name.strip()
                else:
                    coeff = 1 if term[0] != "-" else -1
                    var_name = term.lstrip("+-")

                # Get the ShapeVar attribute
                shape_var = get_attribute(var_name, parent_inst)
                assert isinstance(shape_var, ShapeVar)
                self.coeffs[shape_var] = self.coeffs.get(shape_var, 0) + coeff

    def set( self, value ):
        raise RuntimeError( "An axis cannot be set" )

    # def __init__( self, extent: ShapeExpr | int, name = None ) -> None:
    #     if type( extent ) == int:
    #         from .AffineShapeExpr import AffineShapeExpr
    #         extent = AffineShapeExpr( terms = {}, offset = extent )
    #     assert isinstance( extent, ShapeExpr )
    #     self.extent = extent.to_affine()
    #     self.name = name # if None, set by `Attribute.__set_name__` from the field name

    # @overload
    # def __get__( self, obj: None, objtype = None ) -> 'Axis': ...
    # @overload
    # def __get__( self, obj: object, objtype = None ) -> int | None: ...

    # def __get__( self, obj, objtype = None ):
    #     if obj is None:
    #         return self
    #     return self.extent.eval( obj._bindings )

    # def __repr__( self ):
    #     name = self.name or "axis"
    #     return f"{ name }[ { self.extent } ]"

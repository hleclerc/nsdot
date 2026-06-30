from typing_extensions import overload

from ..util.Attribute import Attribute
from .ShapeExpr import ShapeExpr


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

    def __init__( self, extent: ShapeExpr | int, name = None ) -> None:
        if type( extent ) == int:
            from .AffineShapeExpr import AffineShapeExpr
            extent = AffineShapeExpr( terms = {}, offset = extent )
        assert isinstance( extent, ShapeExpr )
        self.extent = extent.to_affine()
        self.name = name # if None, set by `Attribute.__set_name__` from the field name

    @overload
    def __get__( self, obj: None, objtype = None ) -> 'Axis': ...
    @overload
    def __get__( self, obj: object, objtype = None ) -> int | None: ...

    def __get__( self, obj, objtype = None ):
        if obj is None:
            return self
        return self.extent.eval( obj._bindings )

    def __repr__( self ):
        name = self.name or "axis"
        return f"{ name }[ { self.extent } ]"

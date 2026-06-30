from .ShapeExpr import ShapeExpr

class Axis:
    """A named tensor dimension whose extent is an affine expression of `ShapeVar`s.

    A single `SVar` can drive several axes, which is why `SVar` and `Axis`
    are distinct (e.g. `nb_dims` drives both `nvec = nb_dims + 1` and `dim = nb_dims`).

    The axis can be RAGGED if `extent` depends on othe axes: each row/col/... may have
    a different length, so there is no single extent.
    """

    def __init__( self, extent: ShapeExpr | int, name = None ) -> None:
        if type( extent ) == int:
            from .AffineShapeExpr import AffineShapeExpr
            extent = AffineShapeExpr( terms = {}, offset = extent )
        assert isinstance( extent, ShapeExpr )
        self.extent = extent.to_affine()
        self.name = name # if None, set by `@aggregate` from the field name

    def __repr__( self ):
        name = self.name or "axis"
        return f"{ name }[ { self.extent } ]"

    def _copy( self, copy_map ):
        return Axis( self.extent.copy( copy_map ), self.name )

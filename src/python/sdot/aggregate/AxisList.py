from .AffineExpr import to_affine
from .VariableAxesPlaceholder import VariableAxesPlaceholder


class AxisList:
    """A symbolic-length run of axes, whose extents are an affine expression.

    `extent` is an affine expression of `ShapeVar`s, typically built on a rank-1
    `ShapeVar` so that there is one axis per element:

        img_axes = AxisList( img_shape )         # one axis per img_shape element
        num_knot = AxisList( nb_intervals + 1 )  # extents = nb_intervals[ i ] + 1

    This is how axes that cannot be named one by one are declared. Unlike a vector
    `DynamicShapeVar` passed to a single `Axis` (which yields one ragged axis), an
    `AxisList` yields several distinct axes.
    """

    def __init__( self, extent ) -> None:
        self.extent = to_affine( extent )
        self.name   = None                     # set by `@aggregate` from the field name

    def __iter__( self ):
        # called by Python when the list is unpacked with `*` in a Tensor(...) decl;
        # we yield a single placeholder standing for the whole (symbolic) run of axes
        yield VariableAxesPlaceholder( self )

    def __getitem__( self, index ):
        # `axis_list[ i ]` -> the single (still symbolic) axis taken at `i`
        return VariableAxesPlaceholder( self, index = index )

    def __repr__( self ):
        name = self.name or "axis_list"
        return f"{ name }[ { self.extent } ]"

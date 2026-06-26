from .DynamicShapeVar import DynamicShapeVar
from .AffineExpr import to_affine


class Axis:
    """A named tensor dimension whose extent is an affine expression of `ShapeVar`s.

    A single `ShapeVar` can drive several axes, which is why `ShapeVar` and `Axis`
    are distinct (e.g. `nb_dims` drives both `nvec = nb_dims + 1` and `dim = nb_dims`).

    If a *vector* `DynamicShapeVar` is passed, the axis is RAGGED: each row may have
    a different length, so there is no single affine extent.
    """

    def __init__( self, extent ) -> None:
        self.name = None                       # set by `@aggregate` from the field name

        if isinstance( extent, DynamicShapeVar ) and extent.rank > 0:
            self.ragged_sizes = extent          # per-row lengths
            self.ragged       = True
            self.extent       = None
        else:
            self.ragged_sizes = None
            self.ragged       = False
            # a scalar DynamicShapeVar contributes its capacity as the (max) extent
            self.extent       = to_affine( extent.capacity if isinstance( extent, DynamicShapeVar ) else extent )

    def __repr__( self ):
        name = self.name or "axis"
        if self.ragged:
            return f"{ name }[ ragged={ self.ragged_sizes } ]"
        return f"{ name }[ { self.extent } ]"

    def direct_solve( self, name, shape, num_in_shape, aggregate, forbidden_new_values ):
        if self.extent is not None:
            return self.extent.direct_solve( name, shape[ num_in_shape ], aggregate, forbidden_new_values )
        return None

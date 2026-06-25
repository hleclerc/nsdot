from .AffineExpr import to_affine


class DynamicShapeVar:
    """A runtime-mutable count bounded by a capacity.

    `capacity` is an affine expression of `ShapeVar`s (typically a plain capacity
    `ShapeVar`). Assigning past the capacity must trigger a resize of the dependent
    tensors.

    - rank 0 (`shape=None`): a single dynamic count.
    - rank >= 1 (`shape=[ ... ]`): a vector of counts, one per element of the given
      axes. Passing such a vector to a single `Axis` yields one RAGGED axis (each
      row may have a different length).

    Note the asymmetry with `AxisList`: a vector `DynamicShapeVar` makes a ragged
    axis, whereas a vector `ShapeVar` makes several distinct axes.
    """

    def __init__( self, capacity, shape = None ) -> None:
        self.capacity = to_affine( capacity )
        # `shape` elements are kept raw: they may be `Axis`es (e.g. `shape=[ dim ]`)
        self.shape    = None if shape is None else list( shape )
        self.name     = None                   # set by `@aggregate` from the field name

    @property
    def rank( self ):
        return 0 if self.shape is None else len( self.shape )

    def __repr__( self ):
        name = self.name or "dynamic_shape_var"
        if self.shape is None:
            return f"{ name }[ capacity={ self.capacity } ]"
        return f"{ name }[ capacity={ self.capacity }, shape={ self.shape } ]"

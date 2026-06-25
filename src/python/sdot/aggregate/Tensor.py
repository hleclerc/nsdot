class Tensor:
    """A tensor field, declared by its list of axes.

    Each entry is an `Axis`, or a `VariableAxesPlaceholder` coming from `*axis_list`
    (a symbolic-length run of axes). The actual shape is solved/checked from the
    `ShapeVar`s once the tensors are bound.
    """

    def __init__( self, *axes ) -> None:
        self.name = None                       # set by `@aggregate` from the field name
        self.axes = list( axes )

    def __repr__( self ):
        name = self.name or "tensor"
        return f"{ name }( { ', '.join( repr( a ) for a in self.axes ) } )"

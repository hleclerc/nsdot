class TensorList:
    """An indexed list of tensors, declared by its list of axes.

    Like `Tensor`, but meant for collections indexed along a leading axis, where a
    trailing axis may come from an indexed `AxisList` (e.g. `num_knot[ dim ]`),
    giving a ragged per-element extent.
    """

    def __init__( self, *axes ) -> None:
        self.name = None                       # set by `@aggregate` from the field name
        self.axes = list( axes )

    def __repr__( self ):
        name = self.name or "tensor_list"
        return f"{ name }( { ', '.join( repr( a ) for a in self.axes ) } )"

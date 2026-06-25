class VariableAxesPlaceholder:
    """Stands for (part of) the symbolic-length expansion of an `AxisList`.

    When an `AxisList` is unpacked with `*` inside a `Tensor( ... )` declaration,
    Python needs a concrete iteration at class-definition time, but the number of
    axes is itself unknown (it depends on a `ShapeVar`). So the `AxisList` yields
    exactly one of these placeholders:

    - `index is None` (from `*axis_list`): the whole run of axes.
    - `index is not None` (from `axis_list[ i ]`): the single axis taken at `i`
      (used for indexed `TensorList`s, e.g. `num_knot[ dim ]`).
    """

    def __init__( self, axis_list, index = None ) -> None:
        self.axis_list = axis_list
        self.index     = index

    def __repr__( self ):
        if self.index is None:
            return f"*{ self.axis_list!r }"
        return f"{ self.axis_list!r }[ { self.index!r } ]"

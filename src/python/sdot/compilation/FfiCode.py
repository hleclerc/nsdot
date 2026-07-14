

class FfiCode:
    """A C++ body, plus what the CALL must know to run it -- today: its batch axes.

    `batch_axes` is what a `vmap` adds: one named axis per mapping (`vmap_0`, `vmap_1`, ...).
    It belongs here and not to the arguments because it is what changes the KERNEL: it is the
    shape of `global_batch_indices`, hence the arity of the `batch_index` the body is handed.
    WHICH arguments carry which axis is another matter (a mapping may leave an argument out) and
    belongs to the call's arguments.

    `fwd_code` does not change when an axis is added -- that is the whole point (a batch index is
    an ordinary index, and an empty one is a no-op). What changes is the generated source around
    it, so the derived code compiles to a target of its own, for free (the target name is a hash
    of the source).
    """

    def __init__( self, fwd_code, bwd_code = "", name = "", batch_axes = () ) -> None:
        self.fwd_code = fwd_code
        self.bwd_code = bwd_code
        self.name = name
        self.batch_axes = tuple( batch_axes )

    def with_batch_axis( self ):
        """The same code, mapped over one more axis: what a `vmap` runs. The name is derived from
        how many axes are already there, so a nested `vmap` gets a fresh one, deterministically."""
        name = f"vmap_{ len( self.batch_axes ) }"
        return name, FfiCode( self.fwd_code, self.bwd_code, self.name,
                              self.batch_axes + ( name, ) )

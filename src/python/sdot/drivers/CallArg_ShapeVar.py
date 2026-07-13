from .CallArg_Tensor import CallArg_Tensor

class CallArg_ShapeVar( CallArg_Tensor ):
    """A `ShapeVar` crossing the FFI as a (rank-0 here) integer buffer.

    Like any tensor it lives in `caa.tensors`, but it carries how it is seeded:
    * `reserved`   : the reservation from `max_of_<name>` -- a capacity; runtime value starts
                     at 0 and is written by the kernel (an output count).
    * `prescribed` : a fixed init value from `<name>` (e.g. `nb_dims = 2`).

    `capacity` (used to size the tensors that depend on this ShapeVar) is the reservation when
    given, else the prescribed value.
    """

    def __init__( self, call_args_analysis, io_category, name, reserved = None, prescribed = None, schema = None ) -> None:
        super().__init__( call_args_analysis, io_category, name, value = None, schema = schema )

        self.reserved = reserved
        self.prescribed = prescribed
        self.shape = []  # rank 0 (ragged rank > 0 to come)

    @property
    def capacity( self ):
        return self.reserved if self.reserved is not None else self.prescribed

    def resolve_shape( self, owner ) -> None:
        # its own rank drives the shape, not the aggregate's axes.
        pass

    # -- Jax FFI code generation (driver-specific; a torch_* counterpart will come) --
    # A rank-0 ShapeVar is bound as a 1-element S32 buffer (xla FFI rank-0 buffers are
    # avoided for now); the ShapeVarView wraps its data pointer so `cell.<name> = v` writes it.
    def jax_ffi_ret_type( self ):
        return "ffi::BufferR1<ffi::S32>"

    def jax_cpp_member( self ):
        return f"ShapeVarView<std::int32_t> { self.name };"

    def jax_cpp_init( self ):
        return f"ShapeVarView<std::int32_t>{{ { self.name }->typed_data() }}"

    def jax_out_spec( self ):
        import jax
        import jax.numpy as jnp
        return jax.ShapeDtypeStruct( ( 1, ), jnp.int32 )

class CallArg:
    """One node of the lowering tree: how a Python attribute reaches the C++ kernel.

    Codegen is split by concern, and a node opts in by simply DEFINING the method (the parent
    filters on `hasattr`, see `CallArg_Aggregate._fields`): `cpp_*` emits the driver-agnostic
    C++ (the struct is the same for Jax or Torch), `jax_*` carries the Jax FFI ABI. A node with
    no buffer (a `CtShapeVar`) just has no `jax_*` buffer methods.

    As a member of an aggregate, a node is ONE template parameter of it -- its own type, spelled
    out at instantiation (`TensorView<...>`, `NoneTensor<...>`, `ShapeVarView<...>`, `Ct<SI,2>`).
    So what a member IS is decided per call, and the C++ body reads its scalar type, rank, axis
    names or compile-time value straight off it.
    """

    def __init__( self, io_category, name ) -> None:
        self.io_category = io_category
        self.name = name

    def cpp_tpl_name( self ):
        """Our template parameter, named after the field: unique within the struct that holds
        us, which is the only scope it lives in."""
        return f"T_{ self.name }"

    def jax_data_ptr( self ):
        """Pointer to this buffer's data: an FFI arg is passed by value, a result is
        pointer-like."""
        if self.io_category.is_output:
            return f"{ self.ffi_name }->typed_data()"
        return f"{ self.ffi_name }.typed_data()"

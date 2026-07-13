from .CallArg_Tensor import CallArg_Tensor

class CallArg_ShapeVar( CallArg_Tensor ):
    """A `ShapeVar` crossing the FFI: a tensor of `std::int32_t` counts.

    A `ShapeVar` is rank 0 for a plain scalar count, rank > 0 for ragged ones (its `dep_axes`,
    carried by `schema.args`). Shape resolution and the buffer machinery are therefore reused
    from `CallArg_Tensor` (no override); what is specific here is:

    * `reserved`   : the reservation from `max_of_<name>` -- a capacity; the runtime count
                     starts at 0 and is written by the kernel (an output count).
    * `prescribed` : a fixed init value from `<name>` (e.g. `nb_dims = 2`).
    * `capacity`   : the reservation when given, else the prescribed value; used both to size
                     the tensors that depend on this ShapeVar and as the C++ `max` bound.

    In C++ the `TensorView` of counts is wrapped in a `ShapeVarView` holding that `max`, so
    assignments can be capacity-checked. Rank-0 counts are still bound as a 1-element FFI
    buffer (xla FFI rank-0 buffers are avoided), with a rank-0 `TensorView` over its pointer.
    """

    def __init__( self, call_args_analysis, io_category, name, reserved = None, prescribed = None, schema = None ) -> None:
        super().__init__( call_args_analysis, io_category, name, value = None, schema = schema, dtype = None )

        self.reserved = reserved
        self.prescribed = prescribed

    @property
    def capacity( self ):
        return self.reserved if self.reserved is not None else self.prescribed

    # -- driver-agnostic C++ (the struct is the same for every driver) --
    def _cpp_view( self, ptr_expr ):
        # counts use unnamed axes (positional): a rank-0 view is `tensor_view( ptr, tuple() )`.
        return f"tensor_view( { ptr_expr }, { self._cpp_shape_tuple() } )"

    # the scalar type is fixed (counts are `std::int32_t`): only the shape type is a parameter.
    def cpp_tpl_names( self, prefix = "" ):
        return [ f"Shape_{ prefix }{ self.name }" ]

    def cpp_tpl_params( self, prefix = "" ):
        return [ "class " + n for n in self.cpp_tpl_names( prefix ) ]

    def cpp_tpl_args( self ):
        return [ self._cpp_shape_type() ]

    def cpp_member( self, prefix = "" ):
        shape, = self.cpp_tpl_names( prefix )
        return f"ShapeVarView<TensorView<std::int32_t, { shape }, { self._cpp_memory_space() }>> { self.name };"

    def cpp_seed_member( self, var_name ):
        # Jax has no mutable inputs: outputs are seeded here before the body runs. A prescribed
        # value seeds its buffer to that value; otherwise (a reservation, or a pure output the
        # kernel fills) the count starts empty (0). A scalar broadcasts over all elements, so
        # this is correct at any rank.
        value = int( self.prescribed ) if self.prescribed is not None else 0
        return f"{ var_name }.{ self.name } = { value };"

    # -- Jax FFI ABI --
    def _jax_buffer_shape( self ):
        # rank-0 count -> a 1-element buffer (avoid xla FFI rank-0 buffers).
        return self.shape if self.shape else [ 1 ]

    def jax_ffi_ret_type( self ):
        return f"ffi::BufferR{ len( self._jax_buffer_shape() ) }<ffi::S32>"

    def jax_cpp_init( self ):
        # `max` is the capacity bound; -1 marks "unbounded" (a pure output with no reservation).
        max_bound = int( self.capacity ) if self.capacity is not None else -1
        return f"make_shape_var_view( { self._cpp_view( self.ffi_name + '->typed_data()' ) }, SI( { max_bound } ) )"

    def jax_out_spec( self ):
        import jax
        import jax.numpy as jnp
        return jax.ShapeDtypeStruct( tuple( int( s ) for s in self._jax_buffer_shape() ), jnp.int32 )

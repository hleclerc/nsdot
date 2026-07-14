from ..tensor.Dtype import Dtype
from .CallArg import CallArg

class CallArg_ShapeVar( CallArg ):
    """A `ShapeVar` crossing the FFI: a buffer of `std::int32_t` COUNTS, plus a `max` bound.

    The count is a device value -- read by the kernel when the var has one, written by it when
    the var is an output. It is rank 0 for a plain count, rank > 0 for a ragged one (one count
    per segment, along its `dep_axes`). A rank-0 count is still bound as a 1-element buffer
    (xla FFI rank-0 buffers are avoided), viewed as a rank-0 `TensorView` over its pointer.

    What the count is NOT is a size: what sizes the buffers depending on this var is its
    CAPACITY, a decision made by the CALL (see `CallArgsAnalysis`). It crosses as the `max`
    bound of the `ShapeVarView`, so a written count can be capacity-checked on the C++ side.

    An unbound var wraps a `NoneTensor` instead of a view: there is no count anywhere, and the
    type says so -- writing it does not compile, rather than corrupting a null pointer.
    """

    def __init__( self, call_args_analysis, path, name, inst ) -> None:
        super().__init__( call_args_analysis.io_category( path, _has_count( inst ) ), name )

        self.inst = inst
        # one count per cell of the ragged structure this var varies along (none -> a scalar).
        self.shape = [ int( s ) for axis in inst.dep_axes
                       for s in axis.capacity_list( lambda sv: call_args_analysis.capacity_of( sv, path ) ) ]

        # the bound a written count is checked against; -1 marks "unbounded" (nothing sizes
        # itself on this var, so this call had no reason to be given a capacity for it).
        try:
            self.max_bound = call_args_analysis.capacity_of( inst, path )
        except ValueError:
            self.max_bound = -1

        if self.io_category.is_bound:
            call_args_analysis.register_tensor( self )

    # -- driver-agnostic C++ (the same for every driver) --
    def _cpp_shape_tuple( self ):
        return "tuple( " + ", ".join( f"SI( { int( s ) } )" for s in self.shape ) + " )"

    def _cpp_shape_type( self ):
        return "Tuple<" + ", ".join( "SI" for _ in self.shape ) + ">"

    def _cpp_counts_type( self ):
        # counts use unnamed axes (they are positional).
        if not self.io_category.is_bound:
            return f"NoneTensor<std::int32_t, { self._cpp_shape_type() }, Tuple<>>"
        return f"TensorView<std::int32_t, { self._cpp_shape_type() }, CpuHostMemorySpace>"

    def cpp_type( self ):
        return f"ShapeVarView<{ self._cpp_counts_type() }>"

    def cpp_view( self ):
        if not self.io_category.is_bound:
            return f"{ self.cpp_type() }{{ { self._cpp_counts_type() }{{}}, SI( { self.max_bound } ) }}"
        view = f"tensor_view( { self.jax_data_ptr() }, { self._cpp_shape_tuple() } )"
        return f"make_shape_var_view( { view }, SI( { self.max_bound } ) )"

    # -- as a member of an aggregate --
    def cpp_tpl_param( self ):
        return f"class { self.cpp_tpl_name() }"

    def cpp_tpl_arg( self ):
        return self.cpp_type()

    def cpp_member( self ):
        return f"{ self.cpp_tpl_name() } { self.name };"

    def cpp_root_decl( self, var_name ):
        return f"    auto { var_name } = { self.cpp_view() };"

    def cpp_struct_defs( self ):
        return {}

    # -- seeding: what an output must hold before the body runs --
    def cpp_seed_member( self, owner_name ):
        # a pure output starts empty: the kernel is what fills it. A scalar broadcasts over all
        # cells, so this is correct at any rank.
        if not self.io_category.is_output:
            return ""
        return f"{ owner_name }.{ self.name } = 0;"

    def cpp_seed_root( self, var_name ):
        if not self.io_category.is_output:
            return ""
        return f"    { var_name } = 0;"

    # -- Jax FFI ABI --
    def _jax_buffer_shape( self ):
        # rank-0 count -> a 1-element buffer (avoid xla FFI rank-0 buffers).
        return self.shape if self.shape else [ 1 ]

    def jax_ffi_type( self ):
        return f"ffi::BufferR{ len( self._jax_buffer_shape() ) }<ffi::S32>"

    def jax_cpp_init( self ):
        return self.cpp_view()

    def jax_input_array( self ):
        from ..drivers.driver import driver
        return driver.array( self.inst.value, dtype = Dtype.si( 32 ) ).reshape( self._jax_buffer_shape() )

    def jax_out_spec( self ):
        import jax
        import jax.numpy as jnp
        return jax.ShapeDtypeStruct( tuple( int( s ) for s in self._jax_buffer_shape() ), jnp.int32 )

    def jax_write_back( self, array ):
        # the count keeps the rank it was declared with (a scalar count came back as a
        # 1-element buffer). Nothing else moves: what the kernel wrote is a count, not a size.
        self.inst.set_count( array.reshape( self.shape ) )


def _has_count( inst ):
    try:
        inst.value
        return True
    except NotImplementedError:
        return False

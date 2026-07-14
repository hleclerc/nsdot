from .CallArg import CallArg

class CallArg_Tensor( CallArg ):
    """A tensor attribute, and how it reaches the kernel.

    `inst` is the `Tensor` itself, so everything is read off it -- there is nothing to resolve
    from siblings. The shape depends on the direction, and that is the whole point:

    * INPUT   -> `inst.capacity`: the size the data ACTUALLY has (read off its buffer). An
                 output that wants to grow must not force us to inflate the input.
    * OUTPUT  -> the size THIS CALL asks for: the axes evaluated on the capacities the call was
                 given (see `CallArgsAnalysis`). Known to Python, as an XLA shape must be.
    * UNBOUND -> no buffer, and no `TensorView` either: the attribute lowers to a `NoneTensor`,
                 a distinct TYPE carrying the declared TF/Shape/AxisNames and no data. The
                 kernel discriminates at compile time; there is nothing to test at runtime.

    Codegen splits by concern: `cpp_*` emits the driver-agnostic C++ (identical for Jax or
    Torch), `jax_*` carries the Jax FFI ABI (buffer types, data pointer, result specs).
    """

    def __init__( self, call_args_analysis, path, name, inst ) -> None:
        super().__init__( call_args_analysis.io_category( path, inst.raw is not None ), name )

        self.inst = inst
        self.dtype = inst.dtype
        self.axis_names = [ axis.name for axis, _ in inst.specs ]

        if self.io_category.is_output:
            self.shape = [ int( s ) for s in call_args_analysis.output_shape( inst, path ) ]
        elif self.io_category.is_input:
            self.shape = [ int( s ) for s in inst.capacity ]
        else:
            self.shape = [ 0 ] * inst.rank

        if self.io_category.is_bound:
            call_args_analysis.register_tensor( self )
            for axis_name in self.axis_names:
                call_args_analysis.register_axis( axis_name )

    # -- driver-agnostic C++ (the same for every driver) --
    def _cpp_scalar( self ):
        import numpy
        dt = numpy.dtype( self.dtype.driver_version )
        return { ( "f", 4 ): "float", ( "f", 8 ): "double",
                 ( "i", 4 ): "std::int32_t", ( "i", 8 ): "std::int64_t",
                 ( "u", 4 ): "std::uint32_t", ( "u", 8 ): "std::uint64_t" }[ ( dt.kind, dt.itemsize ) ]

    def _cpp_memory_space( self ):
        # `tensor_view` builds CPU-host views for now; a device-dependent memory space will
        # become part of this member's type, like everything else about it.
        return "CpuHostMemorySpace"

    def _cpp_shape_tuple( self ):
        return "tuple( " + ", ".join( f"SI( { int( s ) } )" for s in self.shape ) + " )"

    def _cpp_axis_tuple( self ):
        return "tuple( " + ", ".join( self.axis_names ) + " )"

    def _cpp_shape_type( self ):
        # the *type* of the shape tuple: only the rank (extents are runtime `SI`s) -- a
        # statically known extent would show up here as a `Ct<SI,n>`.
        return "Tuple<" + ", ".join( "SI" for _ in self.shape ) + ">"

    def _cpp_axis_names_type( self ):
        # `DEFINE_AXIS( num_vertex )` declares the type `_num_vertex` (and the value `num_vertex`).
        return "Tuple<" + ", ".join( "_" + n for n in self.axis_names ) + ">"

    def cpp_type( self ):
        """This member's C++ type. An unbound attribute is not a degenerate view: it is a
        `NoneTensor`, so its absence is a compile-time fact."""
        if not self.io_category.is_bound:
            return ( f"NoneTensor<{ self._cpp_scalar() }, { self._cpp_shape_type() }, "
                     f"{ self._cpp_axis_names_type() }>" )
        return ( f"TensorView<{ self._cpp_scalar() }, { self._cpp_shape_type() }, "
                 f"{ self._cpp_memory_space() }, { self._cpp_axis_names_type() }>" )

    def cpp_view( self ):
        # a `NoneTensor` has nothing to view: it value-initializes.
        if not self.io_category.is_bound:
            return f"{ self.cpp_type() }{{}}"
        ptr = self.jax_data_ptr()
        return f"tensor_view( { ptr }, { self._cpp_shape_tuple() }, { self._cpp_axis_tuple() } )"

    # -- as a member of an aggregate: one type parameter, spelled out at instantiation --
    def cpp_tpl_param( self ):
        return f"class { self.cpp_tpl_name() }"

    def cpp_tpl_arg( self ):
        return self.cpp_type()

    def cpp_member( self ):
        return f"{ self.cpp_tpl_name() } { self.name };"

    # -- as a ROOT argument (a tensor needs no wrapper aggregate to be passed) --
    def cpp_root_decl( self, var_name ):
        return f"    auto { var_name } = { self.cpp_view() };"

    def cpp_struct_defs( self ):
        return {}

    # -- Jax FFI ABI --
    def _jax_ffi_elem( self ):
        import numpy
        dt = numpy.dtype( self.dtype.driver_version )
        return { ( "f", 4 ): "ffi::F32", ( "f", 8 ): "ffi::F64",
                 ( "i", 4 ): "ffi::S32", ( "i", 8 ): "ffi::S64",
                 ( "u", 4 ): "ffi::U32", ( "u", 8 ): "ffi::U64" }[ ( dt.kind, dt.itemsize ) ]

    def jax_ffi_type( self ):
        return f"ffi::BufferR{ len( self._jax_buffer_shape() ) }<{ self._jax_ffi_elem() }>"

    def _jax_buffer_shape( self ):
        return self.shape

    def jax_cpp_init( self ):
        return self.cpp_view()

    def jax_input_array( self ):
        return self.inst.raw

    def jax_out_spec( self ):
        import jax
        return jax.ShapeDtypeStruct( tuple( int( s ) for s in self._jax_buffer_shape() ), self.dtype.driver_version )

    def jax_write_back( self, array ):
        self.inst.set_raw( array )

from .CallArg import CallArg

class CallArg_Tensor( CallArg ):
    """A tensor buffer crossing the FFI boundary.

    `value` is the concrete input tensor when known (input), else `None` (an output to build).
    `schema` is the `Parametrized` declaration (`Tensor["num_vertex","dim"]`); its `args` are
    the declared axis names, used to resolve the `shape` and to name the C++ `TensorView`
    axes. `dtype` is the element `Dtype`. Registers itself in `caa.tensors`.

    Codegen splits by concern: `cpp_*` methods emit the driver-agnostic C++ struct (identical
    for Jax or Torch), while `jax_*` methods carry the Jax FFI ABI (buffer types, data
    pointer, result specs). A `torch_*` counterpart will reuse the same `cpp_*`.

    `shape`:
    * `value` known  -> `value.shape`.
    * output (`value is None`) -> resolved from the declared axes against the sibling
      `ShapeVar` capacities, in `resolve_shape` (needs the owning aggregate).
    """

    name : str
    shape : list

    def __init__( self, call_args_analysis, io_category, name, value = None, schema = None, dtype = None ) -> None:
        super().__init__( io_category )

        self.name = name
        self.value = value
        self.schema = schema
        self.dtype = dtype
        self.shape = list( value.shape ) if value is not None else None

        call_args_analysis.register_tensor( self )

    def resolve_shape( self, owner ) -> None:
        if self.shape is not None:
            return
        axis_names = self.schema.args if self.schema is not None else []
        self.shape = [ owner.attributes[ a ].extent( owner.attributes ) for a in axis_names ]

    # -- driver-agnostic C++ (the struct is the same for every driver) --
    def _cpp_scalar( self ):
        import numpy
        dt = numpy.dtype( self.dtype.driver_version )
        return { ( "f", 4 ): "float", ( "f", 8 ): "double",
                 ( "i", 4 ): "std::int32_t", ( "i", 8 ): "std::int64_t",
                 ( "u", 4 ): "std::uint32_t", ( "u", 8 ): "std::uint64_t" }[ ( dt.kind, dt.itemsize ) ]

    def _cpp_memory_space( self ):
        # `tensor_view` builds CPU-host views for now; a device-dependent memory space will
        # become one more template parameter of the owning struct.
        return "CpuHostMemorySpace"

    def _cpp_shape_tuple( self ):
        return "tuple( " + ", ".join( f"SI( { int( s ) } )" for s in self.shape ) + " )"

    def _cpp_axis_tuple( self ):
        names = self.schema.args if self.schema is not None else []
        return "tuple( " + ", ".join( names ) + " )"

    def _cpp_shape_type( self ):
        # the *type* of the shape tuple: only the rank (extents are runtime `SI`s) -- a
        # statically known extent would show up here as a `Ct<SI,n>`.
        return "Tuple<" + ", ".join( "SI" for _ in self.shape ) + ">"

    def _cpp_axis_names_type( self ):
        # `DEFINE_AXIS( num_vertex )` declares the type `_num_vertex` (and the value `num_vertex`).
        names = self.schema.args if self.schema is not None else []
        return "Tuple<" + ", ".join( "_" + n for n in names ) + ">"

    # -- template parameters of the owning struct --
    def cpp_tpl_names( self, prefix = "" ):
        """What varies from one instance of the aggregate to another: the scalar type and the
        shape type (its rank, hence a shape param rather than a spelled-out `Tuple<SI,SI>`: an
        unrolled `AxisList` gives a rank that depends on the compile-time ShapeVars). `prefix`
        is the attribute path of the owning aggregate when it is itself a nested member."""
        return [ f"TF_{ prefix }{ self.name }", f"Shape_{ prefix }{ self.name }" ]

    def cpp_tpl_params( self, prefix = "" ):
        return [ "class " + n for n in self.cpp_tpl_names( prefix ) ]

    def cpp_tpl_args( self ):
        return [ self._cpp_scalar(), self._cpp_shape_type() ]

    def cpp_member( self, prefix = "" ):
        # spelled structurally (rather than a `decltype` of the view expression): C++ code can
        # read the scalar type, the rank and the axis names straight from the member type.
        tf, shape = self.cpp_tpl_names( prefix )
        return ( f"TensorView<{ tf }, { shape }, { self._cpp_memory_space() }, "
                 f"{ self._cpp_axis_names_type() }> { self.name };" )

    # -- Jax FFI ABI --
    def _jax_ffi_elem( self ):
        import numpy
        dt = numpy.dtype( self.dtype.driver_version )
        return { ( "f", 4 ): "ffi::F32", ( "f", 8 ): "ffi::F64",
                 ( "i", 4 ): "ffi::S32", ( "i", 8 ): "ffi::S64",
                 ( "u", 4 ): "ffi::U32", ( "u", 8 ): "ffi::U64" }[ ( dt.kind, dt.itemsize ) ]

    def jax_ffi_ret_type( self ):
        return f"ffi::BufferR{ len( self.shape ) }<{ self._jax_ffi_elem() }>"

    def jax_cpp_init( self ):
        return f"tensor_view( { self.ffi_name }->typed_data(), { self._cpp_shape_tuple() }, { self._cpp_axis_tuple() } )"

    def jax_out_spec( self ):
        import jax
        return jax.ShapeDtypeStruct( tuple( int( s ) for s in self.shape ), self.dtype.driver_version )

    def jax_value( self, buffer_to_array ):
        return buffer_to_array[ self ]

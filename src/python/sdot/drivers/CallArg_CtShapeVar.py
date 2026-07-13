from .CallArg import CallArg

class CallArg_CtShapeVar( CallArg ):
    """A compile-time `ShapeVar`: its value is carried by the C++ type, NOT an FFI buffer.

    Kept in the aggregate's `attributes` (so axes can read its `capacity`), but deliberately
    absent from `caa.tensors` (no buffer). It emits a `Ct<SI, value>` struct member -- the
    value lives in the type, so C++ code can tell it is compile-time known. Because the value
    is part of the generated source, it is also part of the library-name hash.
    """

    name : str

    def __init__( self, call_args_analysis, io_category, name, value ) -> None:
        super().__init__( io_category )

        self.name = name
        self.value = value

    @property
    def capacity( self ):
        return self.value

    # -- driver-agnostic C++ (the struct is the same for every driver) --
    # the value is a *non-type* parameter of the owning struct: two instances of the same
    # aggregate with e.g. `nb_dims = 2` and `nb_dims = 3` are two instantiations of the same
    # template, and `Ct<SI, ct_nb_dims>` says exactly what the member is.
    def cpp_tpl_names( self, prefix = "" ):
        return [ f"ct_{ prefix }{ self.name }" ]

    def cpp_tpl_params( self, prefix = "" ):
        return [ "SI " + n for n in self.cpp_tpl_names( prefix ) ]

    def cpp_tpl_args( self ):
        return [ str( int( self.value ) ) ]

    def cpp_member( self, prefix = "" ):
        name, = self.cpp_tpl_names( prefix )
        return f"Ct<SI, { name }> { self.name };"

    # -- Jax FFI ABI --
    def jax_cpp_init( self ):
        return f"Ct<SI, { int( self.value ) }>{{}}"

    def jax_value( self, buffer_to_array ):
        # compile-time known: reconstructed straight from the Python-side value.
        return self.value

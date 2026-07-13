from ..util.annotations import annotations
from .CallArg import CallArg

class CallArg_Aggregate( CallArg ):
    """Default decomposition of an aggregate: one child `CallArg` per declared field.

    Reached when a type provides no `make_CallArg` of its own (a plain `@aggregate` class).
    The same `ctor_args` is handed to every child (each picks its key by name).

    Once every child exists, a second pass (`resolve_shape`) lets each child fetch what it
    needs from its siblings in `attributes` (e.g. a tensor asking its axes for their extents).
    """

    attributes : dict[ str, CallArg ]

    def __init__( self, call_args_analysis, io_category, name, cls, value, ctor_args ) -> None:
        super().__init__( io_category )

        self.attributes = {}
        for name_attr, klass_attr in annotations( cls ).items():
            klass_value = None
            if value is not None:
                klass_value = getattr( value, name_attr )
            ca = call_args_analysis.make_CallArg( io_category, name_attr, klass_attr, klass_value, ctor_args )
            if ca is not None:
                self.attributes[ name_attr ] = ca

        # sibling info is now complete: resolve shapes that need it.
        for ca in self.attributes.values():
            ca.resolve_shape( self )

    # -- Jax FFI code generation (driver-specific) --
    def jax_buffers( self ):
        """Child CallArgs that cross the FFI as buffers (those able to emit Jax C++)."""
        return [ ca for ca in self.attributes.values() if hasattr( ca, "jax_cpp_member" ) ]

    def jax_struct_def( self, type_name ):
        members = "\n".join( "    " + b.jax_cpp_member() for b in self.jax_buffers() )
        return f"struct { type_name } {{\n{ members }\n}};"

    def jax_struct_init( self, var_name, type_name ):
        inits = ",\n".join( "        " + b.jax_cpp_init() for b in self.jax_buffers() )
        return f"    { type_name } { var_name }{{\n{ inits }\n    }};"

    def jax_reconstruct( self, buffer_to_array ):
        from types import SimpleNamespace
        return SimpleNamespace( **{ b.name: buffer_to_array[ b ] for b in self.jax_buffers() } )

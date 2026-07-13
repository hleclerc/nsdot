from ..util.annotations import annotations
from .CallArg import CallArg

class CallArg_Aggregate( CallArg ):
    """Default decomposition of an aggregate: one child `CallArg` per declared field.

    Reached when a type provides no `make_CallArg` of its own (a plain `@aggregate` class). A
    field whose type is itself a plain `@aggregate` lands here too: aggregates nest.

    Each aggregate opens its own `ctor_args` scope (`enter`) and hands it to every child, which
    picks its initializer by name; a nested aggregate is thus initialized by the mapping bearing
    its name, falling back to the enclosing levels.

    Once every child exists, a second pass (`resolve_shape`) lets each child fetch what it
    needs from its siblings in `attributes` (e.g. a tensor asking its axes for their extents).

    In C++ the aggregate becomes a *template* named after the Python class (`type_name`), with
    one definition per class and one instantiation per use (`cpp_struct_type`): the same `Cell`
    can then appear twice in a call with, say, `dim = 2` and `dim = 3`. The parameters are
    exactly what varies from one instance to another -- each child says so via `cpp_tpl_params`
    (declarations), `cpp_tpl_names` (the names to pass along) and `cpp_tpl_args` (the concrete
    values) -- and the members stay structural (`Ct<SI, ct_nb_dims>`, `TensorView<TF_x,
    Shape_x, ...>`), so the C++ body can read a member's scalar type, rank, axis names or
    compile-time value straight off its type.

    A nested aggregate forwards its own template parameters to its parent, under a `prefix`
    made of the attribute path (`TF_a_vertex_positions` vs `TF_b_vertex_positions`): two `Cell`
    fields of the same `Mesh` stay distinguishable.
    """

    attributes : dict[ str, CallArg ]

    def __init__( self, call_args_analysis, io_category, name, cls, value, ctor_args ) -> None:
        super().__init__( io_category )

        self.type_name = call_args_analysis.cpp_type_name( cls )
        self.name = name
        self.cls = cls

        # the initializers, as seen from inside this aggregate
        ctor_args = ctor_args.enter( name )

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

    def _fields( self, method ):
        """Child CallArgs implementing `method` (polymorphic filter, no type tests): struct
        members, brace-init entries and reconstructable values are each such a subset."""
        return [ ca for ca in self.attributes.values() if hasattr( ca, method ) ]

    def _concat( self, method, *args ):
        """Concatenation of the lists returned by `method` on each child that has it."""
        return [ item for c in self._fields( method ) for item in getattr( c, method )( *args ) ]

    # -- driver-agnostic C++ (the struct is the same for every driver) --
    def cpp_struct_defs( self ):
        """`{ type_name: template definition }`, nested aggregates FIRST (a class must be
        defined before the parent that holds one). Keyed by `type_name`, so a class used
        several times in the call is emitted once."""
        res = {}
        for c in self._fields( "cpp_struct_defs" ):
            res.update( c.cpp_struct_defs() )
        res[ self.type_name ] = self.cpp_struct_def()
        return res

    def cpp_struct_def( self ):
        """The template definition. Inside it, the parameters carry no prefix: they are named
        after the fields of this very class."""
        members = "\n".join( "    " + c.cpp_member() for c in self._fields( "cpp_member" ) )
        params = self._concat( "cpp_tpl_params", "" )
        prefix = f"template<{ ', '.join( params ) }>\n" if params else ""
        return f"{ prefix }struct { self.type_name } {{\n{ members }\n}};"

    def cpp_struct_type( self ):
        """The instantiation, with the concrete values: `Cell<double, Tuple<SI,SI>, Tuple<>, 2>`."""
        args = self._concat( "cpp_tpl_args" )
        return f"{ self.type_name }<{ ', '.join( args ) }>" if args else self.type_name

    # -- as a *member* of another aggregate --
    def cpp_tpl_params( self, prefix = "" ):
        return self._concat( "cpp_tpl_params", self._prefix( prefix ) )

    def cpp_tpl_names( self, prefix = "" ):
        return self._concat( "cpp_tpl_names", self._prefix( prefix ) )

    def cpp_tpl_args( self ):
        return self._concat( "cpp_tpl_args" )

    def cpp_member( self, prefix = "" ):
        # the parent hands its own parameters down to this class's template.
        names = self.cpp_tpl_names( prefix )
        args = f"<{ ', '.join( names ) }>" if names else ""
        return f"{ self.type_name }{ args } { self.name };"

    def _prefix( self, prefix ):
        return f"{ prefix }{ self.name }_"

    # -- seeding --
    def cpp_seed( self, var_name ):
        """Statements seeding this aggregate's outputs before the body runs (driver-agnostic).
        `var_name` is the expression designating THIS aggregate; each child appends its own
        member name to it, so a nested aggregate seeds through `cell.sub.nb_vertices`."""
        seeds = [ c.cpp_seed_member( var_name ) for c in self._fields( "cpp_seed_member" ) ]
        return "\n".join( s for s in seeds if s )

    def cpp_seed_member( self, owner_name ):
        return self.cpp_seed( f"{ owner_name }.{ self.name }" )

    # -- Jax FFI ABI --
    def jax_struct_init( self, var_name, type_name ):
        inits = ",\n".join( "        " + c.jax_cpp_init() for c in self._fields( "jax_cpp_init" ) )
        return f"    { type_name } { var_name }{{\n{ inits }\n    }};"

    def jax_cpp_init( self ):
        # a nested aggregate brace-inits in place, from its own instantiation type.
        inits = ", ".join( c.jax_cpp_init() for c in self._fields( "jax_cpp_init" ) )
        return f"{ self.cpp_struct_type() }{{ { inits } }}"

    def jax_reconstruct( self, buffer_to_array ):
        # build the real output object and set each computed field on it (an output aggregate
        # is a legitimate instance -- unlike the analysis phase, which never instantiates).
        inst = self.cls()
        for c in self._fields( "jax_value" ):
            setattr( inst, c.name, c.jax_value( buffer_to_array ) )
        return inst

    def jax_value( self, buffer_to_array ):
        # a nested aggregate reconstructs into its own instance.
        return self.jax_reconstruct( buffer_to_array )

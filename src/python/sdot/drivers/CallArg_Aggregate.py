from .CallArg import CallArg

class CallArg_Aggregate( CallArg ):
    """Default lowering of an aggregate instance: one child `CallArg` per declared field.

    Reached when a type provides no `make_CallArg` of its own (a plain `@aggregate`). A field
    whose type is itself an `@aggregate` lands here too: aggregates nest. A field that is a
    mere declaration (an `Axis`) lowers to nothing and is skipped.

    In C++ the aggregate becomes a *template* named after the Python class (`type_name`), with
    one definition per class and one instantiation per use (`cpp_struct_type`). Each member is
    exactly ONE type parameter, whose argument is that member's own type, spelled out at
    instantiation:

        template<class T_vertex_positions, class T_nb_vertices, class T_nb_dims>
        struct Cell { T_vertex_positions vertex_positions; ... };

        Cell<TensorView<double, Tuple<SI,SI>, ...>, ShapeVarView<...>, Ct<SI,2>>

    So *what a member is* is decided per call -- a `TensorView`, a `NoneTensor` when the
    attribute is unbound, a `ZeroTensor` when it is symbolically zero -- while the definition
    stays one template. The same `Cell` can then appear twice in a call with `dim = 2` and
    `dim = 3`, or with one of its tensors bound and the other not, and the C++ body still reads
    a member's scalar type, rank, axis names or compile-time value straight off its type.

    A nested aggregate is no exception: it is one type parameter too, its argument being its own
    instantiation. Nothing has to be forwarded or prefixed.
    """

    attributes : dict[ str, CallArg ]

    def __init__( self, call_args_analysis, path, name, inst ) -> None:
        # an aggregate is not itself a buffer; its category is the one its children inherit by
        # path (naming an aggregate in `output_attributes` covers everything below it).
        super().__init__( call_args_analysis.io_category( path, True ), name )

        self.type_name = call_args_analysis.cpp_type_name( type( inst ) )
        self.inst = inst

        self.attributes = {}
        for name_attr, attr in call_args_analysis.attributes_of( inst ).items():
            ca = call_args_analysis.make_CallArg( f"{ path }.{ name_attr }", name_attr, attr )
            if ca is not None:
                self.attributes[ name_attr ] = ca

    def _fields( self, method ):
        """Child CallArgs implementing `method` (polymorphic filter, no type tests): struct
        members, brace-init entries and writable-back values are each such a subset."""
        return [ ca for ca in self.attributes.values() if hasattr( ca, method ) ]

    # -- driver-agnostic C++ (the same for every driver) --
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
        fields = self._fields( "cpp_member" )
        members = "\n".join( "    " + c.cpp_member() for c in fields )
        params = ", ".join( c.cpp_tpl_param() for c in fields )
        prefix = f"template<{ params }>\n" if params else ""
        return f"{ prefix }struct { self.type_name } {{\n{ members }\n}};"

    def cpp_struct_type( self ):
        """The instantiation: every member's type, spelled out."""
        args = ", ".join( c.cpp_tpl_arg() for c in self._fields( "cpp_member" ) )
        return f"{ self.type_name }<{ args }>" if args else self.type_name

    # -- as a member of another aggregate --
    def cpp_type( self ):
        return self.cpp_struct_type()

    def cpp_tpl_param( self ):
        return f"class { self.cpp_tpl_name() }"

    def cpp_tpl_arg( self ):
        return self.cpp_struct_type()

    def cpp_member( self ):
        return f"{ self.cpp_tpl_name() } { self.name };"

    # -- as a ROOT argument --
    def cpp_root_decl( self, var_name ):
        """Each root argument gets an alias to its own instantiation, then brace-inits its
        members from the bound buffers."""
        type_name = "Sdot_" + var_name
        inits = ",\n".join( "        " + c.jax_cpp_init() for c in self._fields( "jax_cpp_init" ) )
        return ( f"    using { type_name } = { self.cpp_struct_type() };\n"
                 f"    { type_name } { var_name }{{\n{ inits }\n    }};" )

    # -- seeding: what an output must hold before the body runs --
    def cpp_seed_root( self, var_name ):
        seeds = [ c.cpp_seed_member( var_name ) for c in self._fields( "cpp_seed_member" ) ]
        return "\n".join( "    " + s for s in seeds if s )

    def cpp_seed_member( self, owner_name ):
        seeds = [ c.cpp_seed_member( f"{ owner_name }.{ self.name }" ) for c in self._fields( "cpp_seed_member" ) ]
        return "\n".join( s for s in seeds if s )

    # -- Jax FFI ABI --
    def jax_cpp_init( self ):
        # a nested aggregate brace-inits in place, from its own instantiation type.
        inits = ", ".join( c.jax_cpp_init() for c in self._fields( "jax_cpp_init" ) )
        return f"{ self.cpp_struct_type() }{{ { inits } }}"

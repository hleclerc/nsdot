from .CallArg import CallArg

class CallArg_Aggregate( CallArg ):
    """Default lowering of an aggregate instance: one child `CallArg` per declared field.

    Reached when a type provides no `make_CallArg` of its own (a plain `@aggregate`). A field
    whose type is itself an `@aggregate` lands here too: aggregates nest. A field that is a
    mere declaration (an `Axis`) lowers to nothing and is skipped.

    In C++ the aggregate becomes a *template* named after the Python class (`type_name`), one
    definition per class, and each member is exactly ONE type parameter:

        template<class T_vertex_positions, class T_nb_vertices, class T_nb_dims>
        struct Cell { T_vertex_positions vertex_positions; ... };

        auto cell = Cell{ tensor_view( ... ), make_shape_var_view( ... ), Ct<SI,2>{} };

    So *what a member is* is decided per call -- a `TensorView`, a `NoneTensor` when the
    attribute is unbound, a `ZeroTensor` when it is symbolically zero -- while the definition
    stays one template. The same `Cell` can then appear twice in a call with `dim = 2` and
    `dim = 3`, or with one of its tensors bound and the other not, and the C++ body still reads
    a member's scalar type, rank, axis names or compile-time value straight off its type.

    We never SPELL an instantiation, though: C++20 deduces it from the members we build it with
    (aggregate CTAD), here and inside `make_available` -- where the deduced type differs anyway,
    the kernel's views living in another memory space. A nested aggregate is no exception: it is
    one type parameter too, deduced in turn from its own members. Nothing is forwarded, prefixed,
    or written twice.
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
        body = "\n\n".join( [ members, self._cpp_transfer_cost( fields ), self._cpp_make_available( fields ) ] )
        return f"{ prefix }struct { self.type_name } {{\n{ body }\n}};"

    # -- as an argument of `run_parallel` (see support/kernels/run_parallel.h) --
    # An aggregate is handed to a kernel as one argument, but it is NOT made available as one
    # block: each member carries its own io category, which Python decided (`cpp_io_list`), so an
    # input is never copied back and an output never copied in. That is also why the category the
    # caller passes for the aggregate as a whole is ignored -- there is nothing it could add.
    #
    # A nested aggregate needs no special case: it is a member, and this is its `make_available`.
    def _cpp_make_available( self, fields ):
        opens = "\n".join(
            f"        return sdot::make_available( queue, { c.cpp_io_list() }, { c.name }, "
            f"[&]( auto &&a_{ c.name } ) {{" for c in fields
        )
        values = ", ".join( f"a_{ c.name }" for c in fields )
        # the members reaching the kernel have other TYPES than ours (a kernel memory space), so
        # what `cont` receives is another instantiation of the same template -- deduced from the
        # members it is built with (C++20 aggregate CTAD), never spelled out. Qualified `::`,
        # because inside the class our own name means the CURRENT instantiation (the injected
        # class name), which is exactly the one we are NOT rebuilding.
        rebuilt = f"::{ self.type_name }{{ { values } }}"
        closes = "        " + "} );" * len( fields )
        return ( "    auto make_available( auto &&queue, auto /*io_category*/, auto &&cont ) const {\n"
                 f"{ opens }\n"
                 f"            return cont( { rebuilt } );\n"
                 f"{ closes }\n"
                 "    }" )

    def _cpp_transfer_cost( self, fields ):
        costs = [ f"sdot::transfer_cost( queue, { c.cpp_io_list() }, { c.name } )" for c in fields ]
        total = "\n             + ".join( costs + [ "Ct<double,0.0>()" ] )
        return ( "    auto transfer_cost( const auto &queue, auto /*io_category*/ ) const {\n"
                 f"        return { total };\n"
                 "    }" )

    # -- as a member of another aggregate --
    def cpp_tpl_param( self ):
        return f"class { self.cpp_tpl_name() }"

    def cpp_member( self ):
        return f"{ self.cpp_tpl_name() } { self.name };"

    # -- as a ROOT argument --
    def cpp_root_decl( self, var_name ):
        """A root argument brace-inits from the buffers bound to it. Its type is DEDUCED from
        them (C++20 aggregate CTAD), and so, recursively, is the type of any aggregate it holds:
        no instantiation is ever spelled out."""
        inits = ",\n".join( "        " + c.jax_cpp_init() for c in self._fields( "jax_cpp_init" ) )
        return f"    auto { var_name } = { self.type_name }{{\n{ inits }\n    }};"

    # -- seeding: what an output must hold before the body runs --
    def cpp_seed_root( self, var_name ):
        seeds = [ c.cpp_seed_member( var_name ) for c in self._fields( "cpp_seed_member" ) ]
        return "\n".join( "    " + s for s in seeds if s )

    def cpp_seed_member( self, owner_name ):
        seeds = [ c.cpp_seed_member( f"{ owner_name }.{ self.name }" ) for c in self._fields( "cpp_seed_member" ) ]
        return "\n".join( s for s in seeds if s )

    # -- Jax FFI ABI --
    def jax_cpp_init( self ):
        # a nested aggregate brace-inits in place; its type is deduced from what it holds.
        inits = ", ".join( c.jax_cpp_init() for c in self._fields( "jax_cpp_init" ) )
        return f"{ self.type_name }{{ { inits } }}"

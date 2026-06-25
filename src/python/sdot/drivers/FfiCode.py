from textwrap import dedent, indent


def select_for( value, context: set ):
    """Resolve a per-context codegen value.

    `value` is either a plain value (applies everywhere) or a dict whose keys are selectors: a
    set of tags joined by '/' or ',' (e.g. "metal", "fwd/metal", "fwd,metal"); "*" or "" is the
    universal fallback. A selector matches when all its tags are in `context` (the active tags,
    typically { pass, target }). The most specific match wins; an equally-specific conflict is an
    error. Returns None when nothing matches (callers substitute their own default).
    """
    if not isinstance( value, dict ):
        return value

    best, best_score, ambiguous = None, -1, False
    for key, v in value.items():
        tags = set() if key in ( "*", "" ) else { t.strip() for t in key.replace( ",", "/" ).split( "/" ) if t.strip() }
        if tags <= context:
            score = len( tags )
            if score > best_score:
                best, best_score, ambiguous = v, score, False
            elif score == best_score and score > 0:
                ambiguous = True
    if ambiguous:
        raise ValueError( f"ambiguous codegen selector for context { context } among { list( value ) }" )
    return best


class FfiCode:
    """ Abstract base for all FfiCode variants. Subclasses generate C++ on demand. """

    def with_prepended_batch_axis( self, name: str ) -> 'FfiCode':
        raise NotImplementedError( "vmap requires FfiCodeParallel or a custom with_prepended_batch_axis implementation" )

    def header_for( self, pass_name: str ) -> str:
        raise NotImplementedError

    def includes_for( self, pass_name: str ) -> list[ str ]:
        raise NotImplementedError

    def signature( self ) -> str:
        raise NotImplementedError

    @property
    def has_grad_code( self ) -> bool:
        raise NotImplementedError

    def code_for( self, pass_name: str, fai = None ) -> str:
        raise NotImplementedError

    @property
    def name( self ) -> str:
        raise NotImplementedError

    def metal_source( self, pass_name: str, fai, module_name: str ) -> "tuple[ list[ str ], str ]":
        """(Metal codegen) Return ( header_lines, handler_body ) for the binding's `pass_name`.

        Uniform entry point used by the Metal dylib builder — each FfiCode decides internally
        what to emit (no isinstance in the driver). The default reuses the device-agnostic
        header/code, which is exactly what a hand-written FfiCodeCustom wants. Subclasses with a
        templated body (e.g. FfiCodeParallel) override this to generate MSL instead.
        """
        header = self.header_for( pass_name )
        header_lines = [ dedent( header ) ] if header else []
        return header_lines, self.code_for( pass_name, fai )


class FfiCodeCustom( FfiCode ):
    """ FfiCode with manually written C++ strings — no automatic vmap support. """

    def __init__( self, fwd_code: str = "", bwd_code: str = "",
                  includes: "list | dict[ str, list ]" = None,
                  header: "str | dict[ str, str ]" = None,
                  name: str = "" ) -> None:
        self._fwd_code = fwd_code
        self._bwd_code = bwd_code
        self._includes = { "*": includes } if isinstance( includes, list ) else ( includes or {} )
        self._header   = { "*": header }   if isinstance( header,   str  ) else ( header   or {} )
        self._name     = name

    def header_for( self, pass_name: str ) -> str:
        return self._header.get( "*", "" ) + self._header.get( pass_name, "" )

    def includes_for( self, pass_name: str ) -> list[ str ]:
        return self._includes.get( "*", [] ) + self._includes.get( pass_name, [] )

    def signature( self ) -> str:
        parts = [ repr( self._fwd_code ), repr( self._bwd_code ), repr( self._header ), repr( self._includes ) ]
        if self._name:
            parts.insert( 0, self._name )
        return "__".join( parts )

    def code_for( self, pass_name: str, fai = None ) -> str:
        # fwd_code / bwd_code may be plain strings or per-context dicts ( see select_for ).
        from .driver import driver
        src = self._fwd_code if pass_name == "fwd" else self._bwd_code if pass_name == "bwd" else ""
        return select_for( src, { pass_name, driver.device.codegen_target } ) or ""

    @property
    def has_grad_code( self ) -> bool:
        return bool( self._bwd_code )

    @property
    def name( self ) -> str:
        return self._name


class FfiCodeParallel( FfiCode ):
    """
    Trivially parallel FfiCode: iterates in parallel over the batch axes of the objects named
    in `parallel_over`, one independent call per element — no synchronisation, auto-dispatched
    to GPU.

    `parallel_over` is a list of attribute paths into the `Parameters` aggregate (e.g. "cell",
    "power_diagram.cell"). The iteration space (sizes + ordering) is resolved at codegen time
    from those objects' named `batch_axes` by `BatchPlan` — this FfiCode stays a pure codegen
    description and never hard-codes C++ size expressions.

    vmap is transparent: it adds a named batch axis on the *objects* (see the JaxDriver batch
    rule), which `BatchPlan` then picks up — so `with_prepended_batch_axis` is a no-op here.
    """

    def __init__( self, parallel_over: list[ str ], fwd_body: str, bwd_body: str = "",
                  per_thread_args: "list[ str ] | dict[ str, list[ str ] ]" = None,
                  per_thread: "str | dict[ str, str ]" = None,
                  includes: "list | dict[ str, list ]" = None,
                  name: str = "" ) -> None:
        self._per_thread_args = { "*": per_thread_args } if isinstance( per_thread_args, list ) else ( per_thread_args or {} )
        self._per_thread      = { "*": per_thread } if isinstance( per_thread, str ) else ( per_thread or {} )
        self._parallel_over   = list( parallel_over )
        self._includes        = { "*": includes } if isinstance( includes, list ) else ( includes or {} )
        self._fwd_body        = fwd_body
        self._bwd_body        = bwd_body
        self._name            = name

    def body_for( self, pass_name: str, target: str ) -> str:
        """The per-element body for `pass_name`, resolved for codegen `target` (see select_for).
        The same string feeds the CPU/CUDA functor or, on Metal, the generated MSL kernel."""
        src = self._fwd_body if pass_name == "fwd" else self._bwd_body
        return select_for( src, { pass_name, target } ) or ""

    def metal_source( self, pass_name: str, fai, module_name: str ) -> "tuple[ list[ str ], str ]":
        # Emit an MSL kernel from the same per-element body (no host functor / run_parallel).
        # The CallArgs own the tensor -> MSL mapping (struct + buffers); we hand them the body.
        from ..compilation.BatchPlan import BatchPlan
        if pass_name != "fwd":
            raise NotImplementedError( "metal backward for FfiCodeParallel is not implemented yet" )
        plan = BatchPlan( fai, self._parallel_over )
        return [], fai.metal_forward_source( module_name, self.body_for( "fwd", "metal" ), plan.size_exprs )

    def with_prepended_batch_axis( self, name: str ) -> 'FfiCodeParallel':
        # vmap axes live on the objects (BatchPlan reads them); nothing to change on the code.
        return self

    def header_for( self, pass_name: str ) -> str:
        from .driver import driver
        target = driver.device.codegen_target

        lines = []
        # Templated on the batch-axis name tags `Ax...` (supplied by `code_for` from the BatchPlan,
        # in iteration order): the positional index from `cartesian_product_ranges` is zipped into a
        # named `batch_index` ( Tuple<AxisIndex<Ax>...> ) that every batched member is indexed with.
        lines.append( "template<class... Ax>" )
        lines.append( f"struct Parallel_{ pass_name } {{" )

        # per_thread
        if pt := self.per_thread_for( pass_name ):
            lines.append( "    template<class A,class B,class C,class D>" )
            lines.append( "    HD void per_thread( const A &thread_info, const B &batch_indices, C &&cont, D &&p ) const {" )
            lines.append( indent( dedent( pt ), '        ' )  )
            lines.append( "    }" )

        pa = self.per_thread_args_for( pass_name )

        # operator()
        args = [ "BI _batch_index", "P &&p" ] + [ f"T_{ a } { a }" for a in pa ]
        prms = [ "class BI", "class P" ] + [ f"class T_{ a }" for a in pa ]
        lines.append( f"    template<{ ', '.join( prms ) }> HD void operator()( { ', '.join( args ) } ) const {{" )
        lines.append( "        auto batch_index = named_batch_index<Ax...>( _batch_index );" )
        body = self.body_for( pass_name, target )
        if body:
            lines.append( indent( dedent( body ), '        ' ) )
        lines.append( "    }" )

        lines.append( "};" )
        return "\n".join( lines )

    def includes_for( self, pass_name: str ) -> list[ str ]:
        # the generated functor zips the positional index into a named one ( named_batch_index )
        return [ "sdot/support/containers/BatchOf.h" ] + self._includes.get( "*", [] ) + self._includes.get( pass_name, [] )

    def per_thread_args_for( self, pass_name: str ) -> list[ str ]:
        if pass_name in self._per_thread_args:
            return self._per_thread_args[ pass_name ]
        if "*" in self._per_thread_args:
            return self._per_thread_args[ "*" ]
        return []

    def per_thread_for( self, pass_name: str ) -> str:
        if pass_name in self._per_thread:
            return self._per_thread[ pass_name ]
        if "*" in self._per_thread:
            return self._per_thread[ "*" ]
        return ""

    def signature( self ) -> str:
        parts = [ repr( self._fwd_body ), repr( self._bwd_body ), repr( self._parallel_over ), repr( self._includes ), repr( self._per_thread ),  ]
        if self._name:
            parts.insert( 0, self._name )
        return "__".join( parts )

    def code_for( self, pass_name: str, fai = None ) -> str:
        from ..compilation.BatchPlan import BatchPlan
        plan        = BatchPlan( fai, self._parallel_over )
        batch_sizes = ", ".join( plan.size_exprs )
        ax_tags     = ", ".join( f"ax_{ name }" for name in plan.axes )
        return f"run_parallel( cartesian_product_ranges( tuple( { batch_sizes } ) ), Parallel_{ pass_name }<{ ax_tags }>(), p );"

    @property
    def has_grad_code( self ) -> bool:
        return bool( self._bwd_body )

    @property
    def name( self ) -> str:
        return self._name

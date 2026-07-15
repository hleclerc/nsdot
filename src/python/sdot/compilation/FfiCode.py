class AbstractFfiCode:
    """The C++ a call runs, behind one question: `code_for( "fwd" | "bwd", call_args_analysis )`.

    A subclass may hand the body back VERBATIM (`FfiCode`) or GENERATE the scaffold around it from
    what the call turns out to be (`FfiCodeParallel` wraps the body in a `run_parallel` over the
    call's arguments). Either way the rest of the pipeline only ever asks `code_for`, so it never
    has to know which -- nor how much of the source was written by hand."""

    def code_for( self, code_type: str, call_args_analysis ) -> str:
        raise NotImplementedError


class FfiCode( AbstractFfiCode ):
    """A C++ body, plus what the CALL must know to run it -- today: its batch axes and includes.

    `batch_axes` is what a `vmap` adds: one named axis per mapping (`vmap_0`, `vmap_1`, ...).
    It belongs here and not to the arguments because it is what changes the KERNEL: it is the
    shape of `global_batch_indices`, hence the arity of the `batch_index` the body is handed.
    WHICH arguments carry which axis is another matter (a mapping may leave an argument out) and
    belongs to the call's arguments.

    `includes` are extra headers the body needs (the free functions it calls, e.g.
    `sdot/Cell/init_full.h`). They are emitted after the runtime's own includes, so the body can
    lean on everything the generated source already brings in.

    `fwd_code` does not change when an axis is added -- that is the whole point (a batch index is
    an ordinary index, and an empty one is a no-op). What changes is the generated source around
    it, so the derived code compiles to a target of its own, for free (the target name is a hash
    of the source).
    """

    def __init__( self, fwd_code, bwd_code = "", name = "", batch_axes = (), includes = () ) -> None:
        self._code = dict( fwd = fwd_code, bwd = bwd_code )
        self.batch_axes = tuple( batch_axes )
        self.includes = tuple( includes )
        self.name = name

    def code_for( self, code_type: str, call_args_analysis ) -> str:
        return self._code[ code_type ]

    def has_code_for( self, code_type: str ) -> bool:
        """Whether there is a body for this direction. A `bwd` is what makes a call differentiable."""
        return bool( self._code.get( code_type ) )

    def for_backward( self ):
        """The backward as an ordinary FORWARD call: a code object of the SAME kind whose forward
        body is our backward one. Run through the normal path, so a `FfiCodeParallel` scaffolds
        the backward exactly as it does the forward (see `_call_backward`)."""
        return type( self )( self._code[ "bwd" ], name = ( self.name or "sdot" ) + "_bwd",
                             includes = self.includes )

    def with_batch_axis( self ):
        """The same code, mapped over one more axis: what a `vmap` runs. The name is derived from
        how many axes are already there, so a nested `vmap` gets a fresh one, deterministically.
        `type( self )` keeps the subclass, so a `FfiCodeParallel` stays one under `vmap`."""
        name = f"vmap_{ len( self.batch_axes ) }"
        return name, type( self )( self._code[ "fwd" ], self._code[ "bwd" ], self.name,
                                   self.batch_axes + ( name, ), self.includes )


class FfiCodeParallel( FfiCode ):
    """A body run over EVERY argument of the call, in parallel: `code_for` here is not the body
    verbatim but a `run_parallel` scaffold GENERATED around it, from the arguments the call turns
    out to have.

    So the caller writes only what happens per item -- `cell( batch_index ).init_full();` -- and
    the boilerplate is filled in from `call_args_analysis`: the lambda's parameters (one per
    argument, plus the `batch_index`), and the `<arg>_io, <arg>` pairs `run_parallel` maps over
    (an io policy or tag, then the value). Add an argument to the call and it appears in both,
    without the body changing.
    """

    def code_for( self, code_type: str, call_args_analysis ) -> str:
        body = self._code[ code_type ]
        names = list( call_args_analysis.args )
        params = ", ".join( [ "auto batch_index" ] + [ f"auto { n }" for n in names ] )
        mapped = ", ".join( call_args_analysis.args[ n ].cpp_run_parallel_pair() for n in names )
        return ( "run_parallel(\n"
                 "    queue,\n"
                 "    global_batch_indices,\n"
                 f"    []( { params } ) {{\n"
                 f"        { body }\n"
                 "    },\n"
                 f"    { mapped }\n"
                 ");" )

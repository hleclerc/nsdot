

class Parametrized:
    def __init__( self, cls, *args ) -> None:
        self.args = []
        self.kwargs = {}
        self.cls = cls

        for arg in args:
            if isinstance( arg, dict ):
                self.kwargs.update( arg )
            elif isinstance( arg, tuple ) and len( arg ) == 2 and isinstance( arg[ 0 ], str ):
                self.kwargs[ arg[ 0 ] ] = arg[ 1 ]
            else:
                self.args.append( arg )

    def __call__( self, *args, **kwargs ):
        merged_kwargs = { **self.kwargs, **kwargs }
        return self.cls( *args, template_args = self.args, template_kwargs = merged_kwargs )

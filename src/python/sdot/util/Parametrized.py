class Parametrized:
    def __init__( self, cls, *args, **kwargs ) -> None:
        self.kwargs = kwargs
        self.args = args
        self.cls = cls

    def __call__( self, *args, **kwargs ):
        return self.cls( *args, **kwargs, template_args = self.args, template_kwargs = self.kwargs )

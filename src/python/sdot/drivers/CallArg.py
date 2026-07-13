

class CallArg:
    """  """

    def __init__( self, io_category ) -> None:
        self.io_category = io_category

    def resolve_shape( self, owner ) -> None:
        """Second pass, once every sibling `CallArg` of `owner` (a `CallArg_Aggregate`) is
        built: resolve any shape that needs sibling info. Default: nothing to do."""
        pass

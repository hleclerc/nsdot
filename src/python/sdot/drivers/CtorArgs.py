class CtorArgs:
    """The initializers of a `Return( ... )`, seen from one point of the aggregate tree.

    A *scope chain*, one level per nested aggregate: a field looks its initializer up in the
    scope of the aggregate holding it, then outwards. A nested aggregate opens its scope with a
    plain `dict` under its own name, so an initializer can address one nested field precisely,
    while what is left at an outer level reaches every field below it:

        Return( Pair, nb_dims = 2, left = { "max_of_nb_vertices": 8 } )

    reaches `left.nb_dims` and `right.nb_dims` (outer scope), and `left.max_of_nb_vertices`
    only (inner scope). Nesting is free (`{ "sub": { ... } }`). The Return's target class is
    the root scope: its own name (the `driver.call` kwarg) never appears in the keys.

    Lookups take an optional `prefix`, so a reader keeps its own naming convention within a
    scope (`ShapeVar` reserves with `max_of_` + its name).
    """

    _MISSING = object()

    def __init__( self, args : dict, parent = None, at_root = True ) -> None:
        self.at_root = at_root
        self.parent = parent
        self.args = args

    def enter( self, name ):
        """The scope INSIDE the aggregate field `name`: the mapping it is given, backed by the
        enclosing scopes. The root aggregate IS the Return target, so it does not open a scope
        of its own -- it already sits in the Return's."""
        if self.at_root:
            return CtorArgs( self.args, self.parent, at_root = False )
        nested = self.args.get( name )
        # only a mapping opens a scope (a prescribed value is a number / an array).
        nested = nested if hasattr( nested, "keys" ) else {}
        return CtorArgs( nested, parent = self, at_root = False )

    def find( self, name, prefix = "", default = None ):
        value = self.args.get( prefix + name, self._MISSING )
        if value is not self._MISSING:
            return value
        if self.parent is None:
            return default
        return self.parent.find( name, prefix, default )

    def has( self, name, prefix = "" ):
        return self.find( name, prefix, self._MISSING ) is not self._MISSING

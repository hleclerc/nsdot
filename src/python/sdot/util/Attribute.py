from .Parametrized import Parametrized


class Attribute:
    """Base protocol for `@aggregate` field declarations.

    This is the *only* thing `@aggregate` knows about: it operates on classes
    whose fields are `Attribute`s, with no knowledge of any concrete field type
    (`ShapeVar`, `Tensor`, ... are just examples).

    A field annotation is a `Parametrized` (`Attribute[...]`) that acts as the
    class-level *schema*. `get_attribute` calls it once per parent instance to
    build a fresh per-instance `Attribute` that holds that instance's state, kept
    in `self._attributes`. `@aggregate` installs one data descriptor per field:
    `c.field` returns `get` (the read view), `c.field = value` routes to `set`.
    """

    def __class_getitem__( cls, item ):
        if isinstance( item, tuple ):
            return Parametrized( cls, *item )
        else:
            return Parametrized( cls, item )

    def set( self, value ):
        raise NotImplementedError

    def get( self ):
        """Per-instance read view exposed by `c.field`; default: the object itself."""
        return self

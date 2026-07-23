"""ComputedAttribute: a marker and cache-invalidator for dependent attributes.

Usage in an Aggregate:
    class MyBox(Aggregate):
        width : Tensor
        height : Tensor
        area : ComputedAttribute[ Tensor, ("width", "height") ]

The first argument to ComputedAttribute is the actual type to instantiate (Tensor, ShapeVar, etc.);
the remaining arguments are the dependency names. When any dependency is modified,
the ComputedAttribute is invalidated.

The ComputedAttribute is mainly used to track invalidation; the actual value computation
is up to you (via @property, methods, or direct attribute access).
"""

from .Attribute import Attribute


class ComputedAttribute( Attribute ):
    """An Attribute that tracks cache invalidation based on dependencies.

    This is a minimal Attribute that exists primarily to track when dependent
    fields change. The actual value and caching logic are external (e.g., in a
    @property on the class).

    Example:
        area : ComputedAttribute[ Tensor, ("width", "height") ]

    The Aggregate creates a ComputedAttribute in __dict__, tracks that it depends
    on width and height, and invalidates it when either changes. The actual compute
    logic is elsewhere (in a @property, method, etc.).
    """

    def __init__( self, **kwargs ):
        super().__init__( **kwargs )
        self._cache_valid = True  # Start valid until dependencies change

    def set( self, value ):
        # ComputedAttributes are read-only
        raise TypeError( f"ComputedAttribute '{self.name}' is read-only; it is computed from dependencies" )

    def get( self ):
        # Return self (the attribute object), not a value; caching is external
        return self

    def invalidate( self ):
        """Mark this computed attribute as invalid; external logic will recompute on next access."""
        self._cache_valid = False

    def __class_getitem__( cls, item ):
        """Allow ComputedAttribute[Type, ("dep1", "dep2")] syntax."""
        from .Parametrized import Parametrized

        if isinstance( item, tuple ):
            return Parametrized( cls, *item )
        else:
            return Parametrized( cls, item )

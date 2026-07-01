from ..util.Parametrized import Parametrized
from ..util.Attribute import Attribute
import inspect

def aggregate( cls ):
    """
    Class decorator for classes whose fields are `Attribute` declarations.

    `aggregate` works one level of abstraction above any concrete field type: it
    knows nothing about `ShapeVar` / `Axis` / `Tensor` (mere examples), only the
    `Attribute` protocol. Two levels are at play:

      - the *declaration* (an `Attribute`): a class-level descriptor, shared by
        all instances -- the immutable schema and the typed accessor.
      - the *instance object*: the per-instance state, obtained from
        `decl.instantiate(...)` and stored in `self._bindings` (`decl -> inst`).
        It references its declaration; nothing is copied.

    Per-field state is reached through `_bindings`, so the descriptors always
    fire (no data/non-data shadowing) and the instance namespace stays clean.

    Each field can be *injected* at construction by passing its name as a kwarg;
    the declaration decides what the injection means (e.g. a `ShapeVar` shares a
    cell, enabling several objects to agree on the same value):

        n = ShapeVar()
        a = Cell( nb_dims = n )   # a and b share nb_dims
        b = Cell( nb_dims = n )

    Generated: an `__init__` that builds `self._bindings` (honoring injections)
    before delegating to the user-defined `__init__`.
    """

    # collect the field declarations in MRO order (parents first); a field
    # redefined in a subclass overrides the parent's while keeping its position.


    # ------------------ __init__ ------------------
    orig_init = cls.__init__

    def __init__( self, *args, **kwargs ):
        for klass in reversed( cls.__mro__ ):
            for name_attr, type_attr in getattr( klass, '__annotations__', {} ).items():
                sc = type_attr.cls if isinstance( type_attr, Parametrized ) else type_attr
                if inspect.isclass( sc ) and issubclass( sc, Attribute ):
                    setattr( self, name_attr, type_attr( self ) )
    cls.__init__ = __init__

    return cls

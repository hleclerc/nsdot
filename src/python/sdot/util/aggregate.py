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

    # ------------------ __init__ ------------------
    def __base_init__( self, **kwargs ):
        # get references
        if len( kwargs ):
            for name_attr, type_attr in annotations( cls ).items():
                if name_attr in kwargs:
                    value = kwargs.pop( name_attr )
                    type = type_attr.cls if isinstance( type_attr, Parametrized ) else type_attr
                    if inspect.isclass( type ) and isinstance( value, type ):
                        setattr( self, name_attr, value )

        # initialize new attributes
        for name_attr in annotations( cls ).keys():
            get_attribute( name_attr, self )

        # assign values
        if len( kwargs ):
            raise NotImplementedError

    cls.__base_init__ = __base_init__
    cls.__init__ = __base_init__

    # ------------------ __setattr__ ------------------
    def __setattr__( self, name, value ):
        if name not in self.__dict__:
            self.__dict__[ name ] = value
            return
        attr = self.__dict__[ name ]
        if isinstance( attr, Attribute ):
            attr.set( value )
        else:
            self.__dict__[ name ] = value
    cls.__setattr__ = __setattr__


    return cls


def annotations( cls ):
    res = {}
    for klass in reversed( cls.__mro__ ):
        for name, attr in getattr( klass, '__annotations__', {} ).items():
            res[ name ] = attr
    return res


def get_attribute( name, parent_inst ):
    # already in attributes ?
    res = getattr( parent_inst, name, None )
    if res is not None:
        return res

    # in annotation ?
    dct = getattr( parent_inst.__class__, '__annotations__', {} )
    if name in dct:
        type_attr = dct[ name ]
        sc = type_attr.cls if isinstance( type_attr, Parametrized ) else type_attr
        if inspect.isclass( sc ) and issubclass( sc, Attribute ):
            res = type_attr( parent_inst )
        else:
            res = type_attr()

        setattr( parent_inst, name, res )
        return res

    #
    raise ValueError( f"There's no atttribue '{ name }' in '{ type( parent_inst ) }'" )

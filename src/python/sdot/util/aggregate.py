from .Parametrized import Parametrized
from .annotations import annotations
from .Attribute import Attribute
import inspect


def aggregate( cls ):
    """
    Class decorator for classes whose fields are `Attribute` declarations.

    `aggregate` works one level of abstraction above any concrete field type: it
    knows nothing about `ShapeVar` / `Axis` / `Tensor` (mere examples), only the
    `Attribute` protocol. Each annotation is a `Parametrized` schema; a fresh
    per-instance `Attribute` is built lazily by `get_attribute` and kept in
    `self._attributes`.

    Each field can be *injected* at construction by passing its name as a kwarg;
    an `Attribute` value is shared (several instances then agree on the same
    value, e.g. a `ShapeVar` solved from the union of their tensors), any other
    value prescribes it:

        n = ShapeVar()
        a = Cell( nb_dims = n )   # a and b share nb_dims
        b = Cell( nb_dims = n )

    Generated: an `__init__` that honors injections and instantiates every field,
    and one data descriptor per field routing `c.field` to `get` and
    `c.field = value` to `set`.
    """

    # ------------------ __init__ ------------------
    def __base_init__( self, **kwargs ):
        self._attributes = {}

        # injections: share the passed `Attribute`, prescribe any other value
        for name, type_attr in annotations( cls ).items():
            if name in kwargs:
                sc = _field_cls( type_attr )
                if inspect.isclass( sc ) and isinstance( kwargs[ name ], sc ):
                    self._attributes[ name ] = kwargs.pop( name )

        # instantiate the remaining fields
        for name in annotations( cls ).keys():
            get_attribute( name, self )

        # assignation
        for name, value in kwargs.items():
            get_attribute( name, self ).set( value )

    cls.__init__ = __base_init__

    # ------------------ per-field descriptors ------------------
    for name, type_attr in annotations( cls ).items():
        if _is_attribute_field( type_attr ):
            setattr( cls, name, FieldDescriptor( name ) )

    return cls


class FieldDescriptor:
    """Data descriptor generated per `@aggregate` field.

    Reads return the per-instance read view (`attr.get()`); writes route to
    `attr.set(value)`. Class access (`Cls.field`) returns the descriptor itself,
    a handle on the schema.
    """
    def __init__( self, name ):
        self.name = name

    def __get__( self, obj, objtype = None ):
        if obj is None:
            return self
        return get_attribute( self.name, obj ).get()

    def __set__( self, obj, value ):
        get_attribute( self.name, obj ).set( value )




def _field_cls( type_attr ):
    return type_attr.cls if isinstance( type_attr, Parametrized ) else type_attr


def _is_attribute_field( type_attr ):
    sc = _field_cls( type_attr )
    return inspect.isclass( sc ) and issubclass( sc, Attribute )


def get_attribute( name, parent_inst ):
    # already instantiated ?
    attrs = parent_inst._attributes
    if name in attrs:
        return attrs[ name ]

    # in annotation ?
    dct = annotations( parent_inst.__class__ )
    if name in dct:
        type_attr = dct[ name ]
        if _is_attribute_field( type_attr ):
            res = type_attr( parent_inst )
        else:
            res = type_attr()

        attrs[ name ] = res
        return res

    raise ValueError( f"There's no attribute '{ name }' in '{ type( parent_inst ) }'" )

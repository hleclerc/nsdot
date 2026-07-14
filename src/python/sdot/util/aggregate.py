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

    A field whose type is itself an `@aggregate` NESTS: it is built with the same
    kwargs, so what is written at the outer level reaches every field below it. A
    plain mapping under a field's name opens a scope for that field alone, which
    is how two nested aggregates get different values:

        Pair( nb_dims = 2, left = { "dtype": ... }, right = { "nb_dims": 3 } )

    A kwarg either SHARES an `Attribute` (the same object lands in both aggregates)
    or PRESCRIBES a value. Nothing else: notably, a capacity is NOT set here -- it
    is a decision about one allocation, so it is given to the call that allocates
    (`driver.call( ..., capacities = { ... } )`). A key matching no field here is
    not an error as long as some nested aggregate could consume it.

    Generated: an `__init__` that honors injections and instantiates every field,
    and one data descriptor per field routing `c.field` to `get` and
    `c.field = value` to `set`.
    """

    # ------------------ __init__ ------------------
    def __base_init__( self, **kwargs ):
        self._attributes = {}
        anns = annotations( cls )

        # a mapping under a FIELD's name scopes that field; everything else is visible to this
        # class AND to every aggregate nested below it.
        scoped = { n: v for n, v in kwargs.items() if n in anns and isinstance( v, dict ) }
        shared = { n: v for n, v in kwargs.items() if n not in scoped }

        # injections: share the passed `Attribute` (same object) rather than assign it
        for name, type_attr in anns.items():
            sc = _field_cls( type_attr )
            value = shared.get( name )
            if value is not None and inspect.isclass( sc ) and isinstance( value, sc ):
                self._attributes[ name ] = value

        # instantiate the fields; a nested aggregate inherits our scope, refined by its own
        for name, type_attr in anns.items():
            if name in self._attributes:
                continue
            if _is_aggregate( type_attr ):
                self._attributes[ name ] = type_attr( **{ **shared, **scoped.get( name, {} ) } )
                self._attributes[ name ].name = name
            else:
                get_attribute( name, self )

        # prescriptions
        for key, value in shared.items():
            if key in anns:
                if self._attributes[ key ] is not value:   # an injection is already in place
                    get_attribute( key, self ).set( value )
            elif not any( _is_aggregate( t ) for t in anns.values() ):
                raise TypeError( f"'{ cls.__name__ }' has no field '{ key }' to initialize" )

    cls.__init__ = __base_init__
    cls._is_sdot_aggregate = True

    # the scope protocol (see `resolve_attribute`): what turns the NAME an attribute reads in a
    # declaration (`Tensor[ "num_vertex" ]`) into the very object this instance holds.
    cls.get_attribute = lambda self, name: get_attribute( name, self )

    # ------------------ per-field descriptors ------------------
    for name, type_attr in annotations( cls ).items():
        if _is_attribute_field( type_attr ):
            setattr( cls, name, FieldDescriptor( name ) )
        elif _is_aggregate( type_attr ):
            setattr( cls, name, NestedDescriptor( name ) )

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


class NestedDescriptor:
    """Generated per field whose type is itself an `@aggregate`.

    Such a field has no read view to speak of: `p.left` IS the nested instance (the one built
    by our constructor and kept in `_attributes`, so that `p.left.nb_dims` reaches the very
    `ShapeVar` the kernel wrote)."""
    def __init__( self, name ):
        self.name = name

    def __get__( self, obj, objtype = None ):
        if obj is None:
            return self
        return get_attribute( self.name, obj )




def _field_cls( type_attr ):
    return type_attr.cls if isinstance( type_attr, Parametrized ) else type_attr


def _is_attribute_field( type_attr ):
    sc = _field_cls( type_attr )
    return inspect.isclass( sc ) and issubclass( sc, Attribute )


def _is_aggregate( type_attr ):
    sc = _field_cls( type_attr )
    return inspect.isclass( sc ) and getattr( sc, "_is_sdot_aggregate", False )


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
            # the parent is not a ctor argument, it is the SCOPE the declaration's names are
            # resolved in -- an `Attribute` is built the same way with or without one.
            res = type_attr( scope = parent_inst )
        else:
            res = type_attr()

        # the field name is the aggregate's to give: an `Attribute` does not know it (and may
        # outlive its parent -- a borrowed `Axis` still has to be nameable in the C++ code).
        res.name = name

        attrs[ name ] = res
        return res

    raise ValueError( f"There's no attribute '{ name }' in '{ type( parent_inst ) }'" )

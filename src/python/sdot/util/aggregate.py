from .Parametrized import Parametrized
from .annotations import annotations
from .Attribute import Attribute
import inspect


class Aggregate:
    """
    Base class for aggregates: classes whose fields are `Attribute` declarations.

    `Aggregate` works one level of abstraction above any concrete field type: it
    knows nothing about `ShapeVar` / `Axis` / `Tensor` (mere examples), only the
    `Attribute` protocol. Each annotation is a `Parametrized` schema; a fresh
    per-instance `Attribute` is built lazily by `get_attribute` and kept, under
    its field name, in the instance `__dict__`.

    Subclass it -- `class Cell( Aggregate ): ...` -- rather than decorating: the
    machinery is REAL methods (`__base_init__`, `apply_batch_axes`, `get_attribute`)
    and a real `__init__`, so a type checker / IDE sees them (no `TYPE_CHECKING`
    stubs to hand-maintain), and a subclass with its own construction just OVERRIDES
    `__init__` and calls `super().__base_init__( ... )` where it wants the fields set
    up (see `tests/python/test_Cell.py`) -- plain MRO, no bytecode sniffing. Per-field
    SET-only descriptors are installed by `__init_subclass__` at class-creation time.

    Each field can be *injected* at construction by passing its name as a kwarg;
    an `Attribute` value is shared (several instances then agree on the same
    value, e.g. a `ShapeVar` solved from the union of their tensors), any other
    value prescribes it:

        n = ShapeVar()
        a = Cell( nb_dims = n )   # a and b share nb_dims
        b = Cell( nb_dims = n )

    A field whose type is itself an `Aggregate` NESTS: it is built with the same
    kwargs, so what is written at the outer level reaches every field below it. A
    plain mapping under a field's name opens a scope for that field alone, which
    is how two nested aggregates get different values:

        Pair( nb_dims = 2, left = { "dtype": ... }, right = { "nb_dims": 3 } )

    A kwarg either SHARES an `Attribute` (the same object lands in both aggregates)
    or PRESCRIBES a value. Nothing else: notably, a capacity is NOT set here -- it
    is a decision about one allocation, so it is given to the call that allocates
    (`driver.call( ..., capacities = { ... } )`). A key matching no field here is
    not an error as long as some nested aggregate could consume it.

    Reads: `c.field` is a plain read of the instance `__dict__` -- so it hands back
    the `Attribute` OBJECT itself (a `ShapeVar`, an `Axis`, ...), value-on-read
    being the concrete type's business (`c.nb_dims.value`), not the aggregate's;
    `c.field = value` routes to `set` via the per-field descriptor.
    """

    # convention: every aggregate carries `batch_axes`, the (possibly empty) list of axes it is
    # batched over -- read uniformly by the methods (an auxiliary output is `Tensor[ *batch_axes ]`)
    # and by `CallArgsAnalysis` (which folds them into `global_batch_indices`). A plain aggregate has
    # none; `apply_batch_axes` sets them. Left UNANNOTATED so it is not a field (no descriptor, and
    # `annotations()` -- hence the C++ lowering -- never sees it). The class-level `[]` is a mere
    # default: every instance reassigns `self.batch_axes` in `__base_init__`, so the shared list is
    # never mutated in place.
    batch_axes = []

    def __init_subclass__( cls, **kwargs ):
        super().__init_subclass__( **kwargs )

        # one SET-only data descriptor per leaf `Attribute` field, to route its writes to `.set()`.
        # A nested aggregate has no `set` (you reach into it: `p.left.nb_dims = ...`), so it needs no
        # descriptor at all -- `p.left` is just the instance our constructor stored in `__dict__`.
        for name, type_attr in annotations( cls ).items():
            if _is_attribute_field( type_attr ):
                setattr( cls, name, FieldDescriptor( name ) )

    # ------------------ construction ------------------
    def __base_init__( self, **kwargs ):
        cls = type( self )
        anns = annotations( cls )

        # batching is a reserved construction option, not a field: `batch_axes = [ ax, ... ]` adds
        # those axes on the LEFT of every tensor we declare. Popped here so it never reaches the
        # field logic; the tensors are (re)built with the axes prepended at the end, once the plain
        # fields exist -- the very same path a method (`init_as_hypercube( batch_axes = ... )`) takes
        # to batch a cell after construction (see `apply_batch_axes`).
        batch_axes = kwargs.pop( "batch_axes", None )
        self.batch_axes = []

        # a mapping under a FIELD's name scopes that field; everything else is visible to this
        # class AND to every aggregate nested below it.
        scoped = { n: v for n, v in kwargs.items() if n in anns and isinstance( v, dict ) }
        shared = { n: v for n, v in kwargs.items() if n not in scoped }

        # injections: share the passed `Attribute` (same object) rather than assign it. Stored
        # straight into `__dict__` (a raw write, past the set-only descriptor which would else
        # `.set()` it) -- the same slot every read then hands back. A `Tensor` is excluded: its
        # axes are OUR schema's (`Tensor[ "num_vertex", "dim" ]`, resolved in OUR scope), never
        # the ones the caller's tensor happened to carry (e.g. from another aggregate's fields, or
        # an anonymous axis from `append_axis`) -- so it always goes through the schema-built path
        # below and adopts the value via `.set()` (buffer + `_shape` only) instead of by identity.
        for name, type_attr in anns.items():
            sc = _field_cls( type_attr )
            value = shared.get( name )
            if value is not None and inspect.isclass( sc ) and isinstance( value, sc ) and not _is_tensor_field( type_attr ):
                self.__dict__[ name ] = value

        # instantiate the fields; a nested aggregate inherits our scope, refined by its own
        for name, type_attr in anns.items():
            if name in self.__dict__:
                continue
            if _is_aggregate( type_attr ):
                nested = type_attr( **{ **shared, **scoped.get( name, {} ) } )
                nested.name = name
                self.__dict__[ name ] = nested
            else:
                get_attribute( name, self )

        # prescriptions
        for key, value in shared.items():
            if key in anns:
                if self.__dict__[ key ] is not value:   # an injection is already in place
                    get_attribute( key, self ).set( value )
            elif not any( _is_aggregate( t ) for t in anns.values() ):
                raise TypeError( f"'{ cls.__name__ }' has no field '{ key }' to initialize" )

        if batch_axes:
            self.apply_batch_axes( batch_axes )

    # By default `Aggregate()` builds every field. A subclass with its own construction just defines
    # `__init__` (plain override) and calls `self.__base_init__( ... )` where it wants the fields set
    # up (see `tests/python/test_Cell.py`).
    __init__ = __base_init__

    # ------------------ batching ------------------
    def apply_batch_axes( self, batch_axes ):
        """Batch this aggregate over `batch_axes`: (re)build every tensor field with those axes
        PREPENDED. Reachable at construction (`batch_axes = ...`, via `__base_init__`) or later,
        before the tensors are written (`c.init_as_hypercube( batch_axes = ... )`).

        The annotations -- the shared "scalar" schema -- are untouched; only the per-instance tensors
        gain the leading axes. Rebuilding (rather than mutating `specs` in place) is what keeps the
        dimension indices right: a fresh tensor registers each axis at its true position, and the
        unbatched one it replaces dies, taking its now-stale registrations with it. A nested
        aggregate is batched over the SAME axis objects -- joined iteration, no name to collide."""
        self.batch_axes = list( batch_axes )
        for name, type_attr in annotations( type( self ) ).items():
            if _is_aggregate( type_attr ):
                get_attribute( name, self ).apply_batch_axes( self.batch_axes )
            elif _is_tensor_field( type_attr ):
                attr = _batched_schema( type_attr, self.batch_axes )( scope = self )
                attr.name = name
                self.__dict__[ name ] = attr

    # the scope protocol (see `resolve_attribute`): what turns the NAME an attribute reads in a
    # declaration (`Tensor[ "num_vertex" ]`) into the very object this instance holds.
    def get_attribute( self, name ):
        return get_attribute( name, self )


class FieldDescriptor:
    """SET-only data descriptor installed per `Aggregate` field.

    It intercepts WRITES only (`c.field = value` -> `attr.set(value)`); it has no `__get__`, so a
    read falls through to the instance `__dict__`, where `get_attribute` keeps the per-instance
    `Attribute`. `c.field` therefore returns that OBJECT (a `ShapeVar`, an `Axis`, ...): reading a
    value off it (`c.nb_dims.value`) is the concrete type's affair, not something the aggregate
    imposes. Class access (`Cls.field`) returns the descriptor itself, a handle on the schema.

    Having `__set__` makes it a DATA descriptor -- which would normally shadow the instance dict,
    but only through its `__get__`; with none, CPython's lookup skips straight to `__dict__`.
    """
    def __init__( self, name ):
        self.name = name

    def __set__( self, obj, value ):
        get_attribute( self.name, obj ).set( value )


def _field_cls( type_attr ):
    return type_attr.cls if isinstance( type_attr, Parametrized ) else type_attr


def _is_attribute_field( type_attr ):
    sc = _field_cls( type_attr )
    return inspect.isclass( sc ) and issubclass( sc, Attribute )


def _is_aggregate( type_attr ):
    sc = _field_cls( type_attr )
    return inspect.isclass( sc ) and issubclass( sc, Aggregate )


def _is_tensor_field( type_attr ):
    from ..tensor.Tensor import Tensor
    sc = _field_cls( type_attr )
    return inspect.isclass( sc ) and issubclass( sc, Tensor )


def _batched_schema( type_attr, axes ):
    """`type_attr` (a Tensor field schema) with `axes` prepended to its declared axes. A bare
    `Tensor` becomes `Tensor[ *axes ]`; a `Tensor[ "num_vertex", "dim" ]` becomes
    `Tensor[ *axes, "num_vertex", "dim" ]`, its dtype/device kwargs preserved."""
    from ..tensor.Tensor import Tensor
    if isinstance( type_attr, Parametrized ):
        return Parametrized( Tensor, *axes, *type_attr.args, dict( type_attr.kwargs ) )
    return Parametrized( Tensor, *axes )


def get_attribute( name, parent_inst ):
    # already instantiated ? (the per-instance `Attribute` lives under its field name in `__dict__`)
    attrs = parent_inst.__dict__
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

        # a raw write, past the set-only `FieldDescriptor` (which would `.set()` the value instead).
        attrs[ name ] = res
        return res

    raise ValueError( f"There's no attribute '{ name }' in '{ type( parent_inst ) }'" )

from .CallArg_Aggregate import CallArg_Aggregate
from .IoCategory import IoCategory
from .CallArg import CallArg


class CallArgsAnalysis:
    """Decompose the kwargs of `driver.call` into a tree of `CallArg`.

    A `CallArg` *describes* a buffer to bind (or an object to (re)construct); the aggregates
    are never instantiated. `value is None` means "object to build" (a `Return` output);
    `value is not None` means "concrete input object whose data we read".

    Dispatch is polymorphic and carries no type knowledge: the schema/type is asked to
    decompose itself via `make_CallArg`. A type that does not know how (a plain `@aggregate`
    class) falls back to the default -- walking its fields as a `CallArg_Aggregate`. The
    attribute `name` is passed along so a field type can look itself up in `ctor_args` (e.g.
    `ShapeVar` resolves its own `max_of_<name>` / `<name>`).
    """

    tensors: list
    axes: list
    args: dict

    def __init__( self, args : dict ) -> None:
        io_category = IoCategory.pure_input()
        self.tensors = []
        self.axes = []
        self.args = {}
        for name, arg in args.items():
            self.args[ name ] = self.make_CallArg( io_category, name, type( arg ), arg, {} )

    def make_CallArg( self, io_category: IoCategory, name, klass, value, ctor_args ) -> CallArg:
        make = getattr( klass, "make_CallArg", None )
        if make is None:
            # default decomposition: a plain aggregate, walked field by field.
            return CallArg_Aggregate( self, io_category, name, klass, value, ctor_args )
        return make( self, io_category, name, value, ctor_args )

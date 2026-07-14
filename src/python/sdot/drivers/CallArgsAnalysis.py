from ..util.aggregate import get_attribute
from ..util.annotations import annotations
from .CallArg_Aggregate import CallArg_Aggregate
from .CallArg_Errors import CallArg_Errors, ERRORS_VAR_NAME
from .IoCategory import IoCategory
from .CallArg import CallArg


# namespace of the FFI buffer parameters, kept apart from the call's argument names
_FFI_PREFIX = "ffi_"


class CallArgsAnalysis:
    """Lower the kwargs of `driver.call` into a tree of `CallArg`.

    The objects are built by the CALLER (`cell = Cell( nb_dims = 2 )`); this class never
    constructs anything. It walks them and says how each attribute reaches the kernel: which
    buffers to bind and in what order, what C++ struct to emit, what to allocate, and where the
    results go back. Prescribing and sharing belong to the aggregate's constructor, not here.

    Dispatch carries no type knowledge: an attribute is asked to lower ITSELF via `make_CallArg`
    (`Tensor`, `ShapeVar` and `CtShapeVar` each answer differently; an `Axis` answers `None` --
    it is a declaration, not data). An object that does not know how (a plain `@aggregate`)
    falls back to the default: walk its fields as a `CallArg_Aggregate`.

    `path` is the dotted attribute path from the call kwarg (`"cell.vertex_positions"`): it is
    what `output_attributes` and `capacities` name, and naming an aggregate covers everything
    below it.

    CAPACITIES live here, and nowhere else. Sizing an output is a decision about ONE allocation
    -- so it is a parameter of the call, not state on the object (which would let an object
    claim a capacity of 64 while holding a buffer of 8). `capacity_of` resolves one, in order:

    1. what this call was given (`capacities = { "cell.nb_vertices": 8 }`);
    2. else what is ALREADY ALLOCATED for that var, read off the buffers of the tensors that
       use it -- a fact, not a decision, so a chained call need not restate it;
    3. else a count Python actually holds (a `CtShapeVar`, a prescribed value);
    4. else there is nothing to allocate from, and we say which attribute is missing.

    The DEVICE is here too: it is not a runtime parameter of the kernel but part of the type of
    everything the kernel touches (a buffer's memory space is in its `TensorView` type), so every
    node needs it while it emits its C++.
    """

    tensors: list   # the buffers to bind, in FFI order
    args: dict

    def __init__( self, args : dict, device, output_attributes = (), capacities = {} ) -> None:
        self.device = device
        self.output_paths = list( output_attributes )
        self.declared_outputs = set()
        self.type_names = {}
        self.axis_names = []
        self.batch_axes = []
        self.error_vars = []
        self.tensors = []
        self.args = {}

        # paths resolve against the objects, so a capacity keyed by path becomes one keyed by
        # the ShapeVar itself: two aggregates sharing a ShapeVar cannot disagree about it. The
        # var gets to refuse a capacity that contradicts what it is (a `CtShapeVar` does).
        self.capacities = {}
        for path, capacity in capacities.items():
            shape_var = self._attribute_at( args, path )
            shape_var.accept_capacity( capacity )
            self.capacities[ shape_var ] = int( capacity )

        for name, inst in args.items():
            self.args[ name ] = self.make_CallArg( name, name, inst )

        # what the kernel writes when something goes wrong -- last, so that everything that can
        # fail has been given its id by then. It is a buffer of the call, not of an argument: no
        # object of the caller's has anything to do with it.
        self.errors = CallArg_Errors( self )

        # a path that reached nothing is a typo, and a silent one would turn an output into an
        # unbound attribute (the kernel would write into the void).
        unused = [ p for p in self.output_paths if p not in self.declared_outputs ]
        if unused:
            raise ValueError( f"output_attributes: no such attribute: { ', '.join( unused ) }" )

    # -- errors --
    def register_error_var( self, shape_var_ca ) -> int:
        """Give a value that can fail its id in the error buffer -- what a record points back to."""
        self.error_vars.append( shape_var_ca )
        return len( self.error_vars ) - 1

    def capacity_overflows( self ):
        """The capacities this call turned out to be too small for: `[ ( path, wanted, capacity ) ]`,
        empty if the call went through, `None` if we cannot know here (a traced buffer -- see
        `CallArg_Errors.capacity_overflows`)."""
        return self.errors.capacity_overflows( self.error_vars )

    # -- what a `vmap` derives --
    def batched( self, axis_name, axis_size, batched_inputs ):
        """The same lowering, of the same objects, with one more axis: what a `vmap` calls.

        A COPY, not a mutation: the caller's own lowering must keep saying what its tensors are,
        since it is the one that writes the results back onto them (and it does so with the shapes
        the outer trace sees, batch axis stripped). The Python objects are shared -- only the
        lowering nodes are cloned.

        Every OUTPUT gains the axis (the kernel writes one per item); an input gains it only if the
        mapping gave it one (`batched_inputs`, the buffers the framework says it mapped). A value
        left out is not broadcast: it is shared by every item, and the kernel simply lets the batch
        index pass through it."""
        import copy

        mapping = {}
        res = copy.copy( self )
        res.args = { n: c._clone( mapping ) for n, c in self.args.items() }
        res.errors = self.errors._clone( mapping )
        res.error_vars = [ mapping[ id( v ) ] for v in self.error_vars ]
        res.tensors = [ mapping[ id( t ) ] for t in self.tensors ]
        res.axis_names = list( self.axis_names )
        res.batch_axes = list( self.batch_axes ) + [ axis_name ]
        res.register_axis( axis_name )

        for tensor in res.tensors:
            if not tensor.takes_batch_axis():
                continue
            if tensor.io_category.is_output or tensor.ffi_name in batched_inputs:
                tensor.add_batch_axis( axis_name, axis_size )

        return res

    def batch_dim_expr( self, axis_name ):
        """Where the size of a batch axis is READ at run time: an extent of the first buffer that
        carries it. Like any extent, it is not a literal in the source -- one compiled kernel has
        to serve every batch size."""
        for tensor in self.tensors:
            expr = tensor.batch_dim_expr( axis_name )
            if expr is not None:
                return expr
        raise ValueError( f"no buffer carries the batch axis '{ axis_name }'" )

    @staticmethod
    def _attribute_at( args, path ):
        """The `Attribute` object a dotted path designates (`"cell.nb_vertices"`)."""
        names = path.split( "." )
        if names[ 0 ] not in args:
            raise ValueError( f"'{ path }': no argument named '{ names[ 0 ] }' in this call" )
        inst = args[ names[ 0 ] ]
        for name in names[ 1: ]:
            inst = get_attribute( name, inst )
        return inst

    @property
    def cpp_memory_space( self ):
        """Where the buffers of this call already live -- part of the type of every view we emit."""
        return self.device.cpp_memory_space

    def capacity_of( self, shape_var, path ):
        """How much to allocate for `shape_var` in this call (see the class docstring)."""
        if shape_var in self.capacities:
            return self.capacities[ shape_var ]

        allocated = shape_var.allocated_capacity()
        if allocated is not None:
            return allocated

        static = shape_var.static_count()
        if static is not None:
            return static

        raise ValueError(
            f"'{ path }' has to be allocated, but '{ shape_var.name }' has no capacity: give it "
            f"to the call ( capacities = {{ '...{ shape_var.name }': n }} )"
        )

    def output_shape( self, tensor, path ):
        """The extents to allocate `tensor` with: its axes, evaluated on the capacities."""
        res = []
        for axis, _ in tensor.specs:
            res += axis.capacity_list( lambda sv: self.capacity_of( sv, path ) )
        return res

    def io_category( self, path, has_value ) -> IoCategory:
        """Declared as an output, or else OBSERVED: holds data -> input, empty -> unbound."""
        for p in self.output_paths:
            if path == p or path.startswith( p + "." ):
                self.declared_outputs.add( p )
                return IoCategory.OUTPUT
        return IoCategory.INPUT if has_value else IoCategory.UNBOUND

    def is_exact_output( self, path ) -> bool:
        """Whether `path` was named ITSELF, as opposed to being covered by an ancestor.

        Naming an aggregate is a convenience -- "the outputs are in there" -- so an attribute
        that cannot be one (a compile-time `CtShapeVar`) is simply skipped. Naming an attribute
        is an assertion about that attribute, and an impossible one must be told."""
        return path in self.output_paths

    def make_CallArg( self, path, name, inst ) -> CallArg | None:
        make = getattr( type( inst ), "make_CallArg", None )
        if make is None:
            # default lowering: a plain aggregate, walked field by field.
            return CallArg_Aggregate( self, path, name, inst )
        return make( self, path, name, inst )

    def attributes_of( self, inst ):
        """The declared attributes of an aggregate instance, as `Attribute` OBJECTS: `getattr`
        would hand back the read view (a ShapeVar's count, not the ShapeVar itself)."""
        return { name: get_attribute( name, inst ) for name in annotations( type( inst ) ) }

    def register_tensor( self, tensor ):
        """Register a buffer to bind, and give it a unique `ffi_name`.

        The FFI buffers share one flat namespace (the handler's parameter list) with each
        other AND with the call's arguments, whereas attribute names are only unique within
        their aggregate: two `Cell`s both bring a `vertex_positions`, and a bare tensor
        argument would collide with its own buffer (`auto res = tensor_view( res->... )`).
        Hence a namespace of their own, plus disambiguation. Only the buffer is renamed: the
        struct member keeps the attribute name the C++ body uses."""
        name = _FFI_PREFIX + tensor.name
        while any( t.ffi_name == name for t in self.tensors ):
            name += "_"
        tensor.ffi_name = name
        self.tensors.append( tensor )

    def register_axis( self, name ):
        """An axis name used by a bound tensor: it needs a `DEFINE_AXIS` in the source.

        Collected from the TENSORS rather than from the aggregates' declarations, because a
        tensor may borrow an axis from an object that is not itself an argument of this call
        (`Tensor[ cell.num_vertex ]`)."""
        if name not in self.axis_names:
            self.axis_names.append( name )

    def cpp_type_name( self, cls ):
        """C++ name of the struct template generated for the `@aggregate` class `cls`: its
        Python name, made unique should two distinct classes share it in the same call."""
        if cls not in self.type_names:
            name = cls.__name__
            while name in self.type_names.values():
                name += "_"
            self.type_names[ cls ] = name
        return self.type_names[ cls ]

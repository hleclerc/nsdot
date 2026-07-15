# from typing_extensions import Sequence, overload
from ..util.Attribute import Attribute, resolve_attribute
from numpy._typing import ArrayLike
from typing import TYPE_CHECKING, Iterator
import weakref
import numpy


class ShapeVar( Attribute ):
    """An integer (possibly rank > 0) variable used to build shapes.

    Rank 0 is a plain scalar count; rank > 0 holds the per-row/per-segment counts
    of a RAGGED structure (`dep_axes` records which axes it varies along).

    Axis extents are affine *expressions* over `ShapeVar`s (e.g. `nb_dims + 1`).
    A `ShapeVar` is either prescribed (`c.nb_dims = 1222`) or solved from the
    shapes observed on the declared tensors that use it; a prescribed value wins.
    `c.nb_dims` reads back the value.

    Sharing: pass a `ShapeVar` to several constructors (`Cell( nb_dims = n )`);
    they then reference the same object, so the value is solved from the union of
    their tensors.

    The value a `ShapeVar` holds is a COUNT: how many items are used. A kernel
    reads it and/or writes it, so after a call it lives in a DEVICE buffer -- under
    `jit`, Python does not know it, and it can therefore never size anything.

    What sizes a buffer is a CAPACITY, and a capacity is deliberately NOT state
    kept here: it is a decision about ONE allocation, so it belongs to the call
    that allocates (`driver.call( ..., capacities = { "cell.nb_vertices": 8 } )`).
    Storing it on the object would open a window where the object contradicts
    itself -- a capacity of 64 next to a buffer of 8. What we can offer is what we
    KNOW: `allocated_capacity` (read off the buffers our tensors already have) and
    `static_count` (a count Python actually holds). See `CallArgsAnalysis`.
    """

    if TYPE_CHECKING:
        def __set__( self, obj, value: int ) -> None: ...
        def __iter__( self ) -> Iterator: ...

    @classmethod
    def make_CallArg( cls, caa, path, name, inst ):
        from ..drivers.CallArg_ShapeVar import CallArg_ShapeVar
        return CallArg_ShapeVar( caa, path, name, inst )

    def __init__( self, value = None, /, *, template_args = (), template_kwargs = {}, scope = None ) -> None:
        from .AbstractAxis import AbstractAxis

        # (weakref(tensor), resolver): each `resolver(tensor)` returns our solved
        # value from that tensor's observed sizes, or None if it cannot yet.
        # Weak so a shared ShapeVar does not keep dropped instances' tensors alive.
        self.usages = []
        self.dep_axes = [ resolve_attribute( d, scope, AbstractAxis ) for d in template_args ]

        self.prescribed_value = None
        self._count = None     # count produced by a kernel: a driver tensor, possibly traced

        if value is not None:
            self.set( value )

    def add_usage( self, tensor, resolver ):
        self.usages.append( ( weakref.ref( tensor ), resolver ) )

    def set_count( self, value ):
        """Rebind the count to what a kernel produced (a driver tensor). Nothing else moves:
        the buffers keep the size they were allocated with."""
        self._count = value

    def accept_capacity( self, capacity ):
        """Called when a call is given a capacity for us: our chance to refuse one that would
        contradict what we are (see `CtShapeVar`). Nothing is stored -- a capacity belongs to
        the call that allocates."""
        pass

    def set( self, value ):
        if isinstance( value, ShapeVar ):
            value = value.value
        self.prescribed_value = numpy.array( value, dtype = int )

    def get( self ) -> ArrayLike:
        return self.value

    def _solve( self, allocated = False ):
        """Solve us from the tensors that use us: the first usage able to invert one of their
        sizes (single-var affine inversion, unroll count, ...); `None` if none can.

        `allocated` picks WHICH size: the LOGICAL one (how much is used -> our count) or the
        ALLOCATED one (how much the buffer holds -> the capacity it was allocated with)."""
        for tensor_ref, resolver in self.usages:
            tensor = tensor_ref()
            if tensor is None:
                continue
            solved = resolver( tensor, allocated )
            if solved is not None:
                return numpy.asarray( solved )
        return None

    @property
    def value( self ) -> ArrayLike:
        """The COUNT. A kernel-written count wins, being the freshest truth -- and it is then a
        DEVICE value: never size a buffer with it. `None` when we are UNRESOLVED: neither
        prescribed nor constrained by a tensor, so there is no count to hand back yet."""
        if self._count is not None:
            return self._count

        if self.prescribed_value is not None:
            return self.prescribed_value

        return self._solve()

    @value.setter
    def value( self, value ):
        self.set( value )

    def allocated_capacity( self ):
        """The capacity our tensors were ALLOCATED with, read off their buffers -- a fact, not
        a decision, so a chained call needs no restating of a capacity already materialized.
        `None` when we have no allocated tensor to read it from."""
        solved = self._solve( allocated = True )
        return int( numpy.max( solved ) ) if solved is not None else None

    def static_count( self ):
        """Our count when PYTHON holds it (prescribed, or solved from the data of a tensor we
        were given). `None` when it only lives on the device -- where it cannot size anything."""
        if self.prescribed_value is not None:
            return int( numpy.max( self.prescribed_value ) )
        solved = self._solve()
        return int( numpy.max( solved ) ) if solved is not None else None

# from typing_extensions import Sequence, overload
from ..util.Parametrized import Parametrized
from ..util.aggregate import get_attribute
from ..util.Attribute import Attribute
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
    """

    if TYPE_CHECKING:
        def __set__( self, obj, value: int ) -> None: ...
        def __iter__( self ) -> Iterator: ...

    @classmethod
    def make_CallArg( cls, caa, io_category, name, value, ctor_args, schema = None ):
        # `max_of_<name>` reserves a capacity (runtime value 0, written by the kernel);
        # `<name>` prescribes a fixed init value. This lookup is ShapeVar's own business.
        from ..drivers.CallArg_ShapeVar import CallArg_ShapeVar
        return CallArg_ShapeVar(
            caa, io_category, name,
            reserved = ctor_args.get( "max_of_" + name ),
            prescribed = ctor_args.get( name ),
            schema = schema,
        )

    def __init__( self, parent_inst = None, /, template_args = [], template_kwargs = {} ) -> None:
        from .AbstractAxis import AbstractAxis

        # (weakref(tensor), resolver): each `resolver(tensor)` returns our solved
        # value from that tensor's observed sizes, or None if it cannot yet.
        # Weak so a shared ShapeVar does not keep dropped instances' tensors alive.
        self.usages = []
        self.dep_axes = []
        for dep_axis in template_args:
            axis = get_attribute( dep_axis, parent_inst )
            assert isinstance( axis, AbstractAxis )
            self.dep_axes.append( axis )

        self.prescribed_value = None

    def add_usage( self, tensor, resolver ):
        self.usages.append( ( weakref.ref( tensor ), resolver ) )

    def set( self, value ):
        if isinstance( value, ShapeVar ):
            value = value.value
        self.prescribed_value = numpy.array( value, dtype = int )

    def get( self ) -> ArrayLike:
        return self.value

    @property
    def value( self ) -> ArrayLike:
        if self.prescribed_value is not None:
            return self.prescribed_value

        # else solve from the first tensor able to constrain us: each usage knows
        # how (single-var affine inversion of an observed size, unroll count, ...).
        for tensor_ref, resolver in self.usages:
            tensor = tensor_ref()
            if tensor is None:
                continue
            solved = resolver( tensor )
            if solved is not None:
                return numpy.asarray( solved )

        raise NotImplementedError( "ShapeVar is neither prescribed nor constrained by a tensor" )

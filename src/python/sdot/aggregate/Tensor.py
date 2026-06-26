from ..drivers.driver import driver
from typing import TYPE_CHECKING
from ..drivers.Dtype import Dtype


if TYPE_CHECKING:
    # For typing only, we pretend it is a numpy `ndarray` so editors offer array autocompletion
    # (`.shape`, `.reshape`, `.sum`, ...) on the fields once they are bound. The
    # `*axes` constructor keeps `Tensor( nvec, dim )` declarations from type-erroring.
    #
    # It is also a descriptor: `__get__` keeps the `ndarray` view (read access), while
    # `__set__` widens assignment to any `ArrayLike`, so `c.frame = [[0,0],[1,0],[0,1]]`
    # (or a numpy array) type-checks even though the field is declared as a `Tensor`.
    from numpy.typing import ArrayLike
    from numpy import ndarray

    class Tensor( ndarray ):
        def __new__( cls, *axes, dtype = None ) -> "Tensor": ...
        def __get__( self, obj, objtype = None ) -> "Tensor": ...
        def __set__( self, obj, value: ArrayLike ) -> None: ...
else:
    class Tensor:
        """A tensor field, declared by its list of axes.

        Each entry is an `Axis`, or a `VariableAxesPlaceholder` coming from `*axis_list`
        (a symbolic-length run of axes). The actual shape is solved/checked from the
        `ShapeVar`s once the tensors are bound.
        """

        def __init__( self, *axes, dtype = None ) -> None:
            self.dtype = Dtype.factory( dtype )
            self.axes = list( axes )
            self.name = None # set by `@aggregate` from the field name

        def __repr__( self ):
            name = self.name or "tensor"
            return f"{ name }( { ', '.join( repr( a ) for a in self.axes ) } )"

        def coerce( self, value ):
            if value is None:
                return None
            return driver.array( value, dtype = self.dtype )

        def direct_solve( self, name: str, shape, aggregate, forbidden_new_values ):
            if len( self.axes ) != len( shape ):
                raise NotImplementedError
            num_in_shape = 0
            for axis in self.axes:
                res = axis.direct_solve( name, shape, num_in_shape, aggregate, forbidden_new_values )
                if res is not None:
                    return res

                num_in_shape += 1

            return None

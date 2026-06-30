from ..drivers.driver import driver
from ..drivers.Dtype import Dtype
from typing import TYPE_CHECKING

from .VariableAxesPlaceholder import VariableAxesPlaceholder


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

        def _segments( self, shape ):
            """Map each declared axis to the slice of `shape` it occupies.

            Yields `( axis, sizes )` where `sizes` is a tuple of the concrete extents
            taken by `axis`. A regular `Axis` takes exactly one; a `*axis_list` run
            (a `VariableAxesPlaceholder`) takes a variable number, deduced from the
            remaining (non-fixed) shape positions.
            """
            runs = [ a for a in self.axes if isinstance( a, VariableAxesPlaceholder ) ]
            if len( runs ) > 1:
                raise NotImplementedError( "only one variable-length axis run is supported" )

            if runs:
                run_len = len( shape ) - ( len( self.axes ) - 1 )
                if run_len < 0:
                    raise ValueError( f"shape { tuple( shape ) } has fewer dims than the declared axes of { self.name }" )
            elif len( self.axes ) != len( shape ):
                raise ValueError( f"shape { tuple( shape ) } does not match the { len( self.axes ) } declared axes of { self.name }" )

            i = 0
            for axis in self.axes:
                width = run_len if isinstance( axis, VariableAxesPlaceholder ) else 1
                yield axis, tuple( shape[ i : i + width ] )
                i += width

        def direct_solve( self, name: str, value, aggregate, forbidden_new_values ):
            for axis, sizes in self._segments( value.shape ):
                if isinstance( axis, VariableAxesPlaceholder ):
                    res = axis.axis_list.direct_solve( name, sizes, aggregate, forbidden_new_values )
                else:
                    res = axis.direct_solve( name, sizes[ 0 ], aggregate, forbidden_new_values )
                if res is not None:
                    return res

            return None

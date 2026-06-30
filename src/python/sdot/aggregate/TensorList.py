from .VariableAxesPlaceholder import VariableAxesPlaceholder
from ..drivers.driver import driver
from typing import TYPE_CHECKING


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

    class TensorList( ndarray ):
        def __new__( cls, *axes, dtype = None ) -> "TensorList": ...
        def __get__( self, obj, objtype = None ) -> "TensorList": ...
        def __set__( self, obj, value: ArrayLike ) -> None: ...
else:
    class TensorList:
        """An indexed list of tensors, declared by its list of axes.

        Like `Tensor`, but meant for collections indexed along a leading axis, where a
        trailing axis may come from an indexed `AxisList` (e.g. `num_knot[ dim ]`),
        giving a ragged per-element extent. The bound value is therefore a Python list
        of arrays (one per leading index), each with its own length.
        """

        def __init__( self, *axes ) -> None:
            self.name = None                       # set by `@aggregate` from the field name
            self.axes = list( axes )

        def __repr__( self ):
            name = self.name or "tensor_list"
            return f"{ name }( { ', '.join( repr( a ) for a in self.axes ) } )"

        def coerce( self, value ):
            # ragged: keep one array per leading index rather than a single dense tensor
            if value is None:
                return None
            return [ driver.array( v ) for v in value ]

        def direct_solve( self, name, value, aggregate, forbidden_new_values ):
            lead, *rest = self.axes

            # 1) the leading axis extent is simply the number of elements in the list
            res = lead.direct_solve( name, len( value ), aggregate, forbidden_new_values )
            if res is not None:
                return res

            # 2) the per-element axes, currently a single indexed `AxisList` placeholder
            #    (e.g. `num_knot[ dim ]`): each element brings its own extent.
            if len( rest ) == 1 and isinstance( rest[ 0 ], VariableAxesPlaceholder ):
                return rest[ 0 ].axis_list.direct_solve_indexed( name, value, aggregate, forbidden_new_values )

            raise NotImplementedError( "unsupported TensorList axes layout" )

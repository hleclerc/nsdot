from typing_extensions import Sequence
from typing import TYPE_CHECKING
from weakref import ref

from .AffineShapeExpr import AffineShapeExpr
from .ShapeExpr import ShapeExpr
from ..util.find import find

from ..util.Reassignable import Reassignable

if TYPE_CHECKING:
    from .Axis import Axis


class ShapeVar( ShapeExpr, Reassignable ):
    """A free (symbolic) integer variable that can be used to make shapes.

    Axis extents are *expressions* built on top of `SVar`s (e.g. `nb_dims + 1`).
    A `SVar` is either prescribed or solved from the shapes of the declared
    tensors.

    - rank 0 (`shape=None`): a scalar unknown.
    - rank >= 1 (`shape=[ ... ]`): a vector of unknowns, indexed by the given axes
      (e.g. `SVar( [ dim ] )` is one unknown per `dim` element). It is then
      either expanded into several axes through an `AxisList`, or used in an affine
      expression elementwise (`nb_intervals + 1`).

    `shape` can be given positionally or by keyword; its elements are kept raw
    (they are typically `Axis`es, not affine expressions).
    """

    def __init__( self, shape : None | Sequence[ 'Axis' ] = None, name = None, prescribed_value = None ) -> None:
        if shape is None:
            shape = []

        self.prescribed_value = prescribed_value
        self.shape = list( shape )
        self.name = name # if None, set by `@aggregate` from the field name

        self.usage = []

    def reassign( self, value ):
        self.prescribed_value = int( value )

    @property
    def value( self ):
        forbidden_shape_vars = []
        for tensor in self.usage:
            res = tensor().shape_var_value( self, forbidden_shape_vars )
            if res is not None:
                return res
        return None

    @property
    def rank( self ):
        return 0 if self.shape is None else len( self.shape )

    def __repr__( self ):
        return f"{ self.name or "shape_var" }[ shape={ self.shape } ]"

    def to_affine( self ) -> AffineShapeExpr:
        return AffineShapeExpr( terms = { self: 1 }, offset = 0 )

    def _copy( self, copy_map ):
        return ShapeVar( [ s.copy( copy_map ) for s in self.shape ], self.name, self.prescribed_value )

    def register_tensor( self, tensor ):
        if find( self.usage, lambda r : r() == tensor ) is None:
            self.usage.append( ref( tensor ) )

    def unregister_tensor( self, tensor ):
        self.usage = [ r for r in self.usage if r() is not None and r() is not tensor ]

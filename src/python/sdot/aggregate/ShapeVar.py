from .AffineOperand import AffineOperand
from .AffineExpr import AffineExpr


class ShapeVar( AffineOperand ):
    """A free (symbolic) integer variable, NOT a size by itself.

    Axis extents are *expressions* built on top of `ShapeVar`s (e.g. `nb_dims + 1`).
    A `ShapeVar` is either prescribed or solved from the shapes of the declared
    tensors.

    - rank 0 (`shape=None`): a scalar unknown.
    - rank >= 1 (`shape=[ ... ]`): a vector of unknowns, indexed by the given axes
      (e.g. `ShapeVar( [ dim ] )` is one unknown per `dim` element). It is then
      either expanded into several axes through an `AxisList`, or used in an affine
      expression elementwise (`nb_intervals + 1`).

    `shape` can be given positionally or by keyword; its elements are kept raw
    (they are typically `Axis`es, not affine expressions).
    """

    def __init__( self, shape = None ) -> None:
        self.shape = None if shape is None else list( shape )
        self.name  = None                      # set by `@aggregate` from the field name

    def _as_affine( self ):
        # works for any rank: a vector ShapeVar enters affine expressions elementwise
        return AffineExpr( terms = { self: 1 } )

    @property
    def rank( self ):
        return 0 if self.shape is None else len( self.shape )

    def __repr__( self ):
        name = self.name or "shape_var"
        if self.shape is None:
            return name
        return f"{ name }[ shape={ self.shape } ]"

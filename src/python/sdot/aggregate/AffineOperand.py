class AffineOperand:
    """Anything that can take part in an affine expression of `ShapeVar`s.

    The operators below are triggered *while the body of an `@aggregate` class is
    being parsed* (e.g. `2 * nb_dims + 1`). They always return an `AffineExpr`, or
    `NotImplemented` when the other operand is not linear (letting Python try the
    reflected operator / raise).

    `AffineExpr` is imported lazily inside the methods to avoid an import cycle
    (`AffineExpr` subclasses `AffineOperand`).
    """

    def _as_affine( self ) -> "AffineExpr":
        raise NotImplementedError

    def __add__( self, other ):
        from .AffineExpr import _to_affine, _add
        o = _to_affine( other )
        if o is None:
            return NotImplemented
        return _add( self._as_affine(), o )

    __radd__ = __add__

    def __sub__( self, other ):
        from .AffineExpr import _to_affine, _add, _scale
        o = _to_affine( other )
        if o is None:
            return NotImplemented
        return _add( self._as_affine(), _scale( o, -1 ) )

    def __rsub__( self, other ):
        from .AffineExpr import _to_affine, _add, _scale
        o = _to_affine( other )
        if o is None:
            return NotImplemented
        return _add( o, _scale( self._as_affine(), -1 ) )

    def __mul__( self, other ):
        # only scalar (integer) multiplication stays affine
        from .AffineExpr import _scale
        if not isinstance( other, int ):
            return NotImplemented
        return _scale( self._as_affine(), other )

    __rmul__ = __mul__

    def __neg__( self ):
        from .AffineExpr import _scale
        return _scale( self._as_affine(), -1 )

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .AffineShapeExpr import AffineShapeExpr


class ShapeExpr:
    """
    Expression that can be used to define a ShapeExtent

    A ShapeExpr must define a method .to_affine that return a AffineShapeExpr
    """

    def __add__( self, that ):
        from .AffineShapeExpr import AffineShapeExpr
        if isinstance( that, int ):
            that = AffineShapeExpr( terms = {}, offset = that )
        assert isinstance( that, ShapeExpr )

        a = self.to_affine()
        b = that.to_affine()

        terms = dict( a.terms )
        for var, coeff in b.terms.items():
            terms[ var ] = terms.get( var, 0 ) + coeff
        return AffineShapeExpr( terms = terms, offset = a.offset + b.offset )

    __radd__ = __add__

    def __mul__( self, that ):
        from .AffineShapeExpr import AffineShapeExpr
        if isinstance( that, int ):
            that = AffineShapeExpr( terms = {}, offset = that )
        assert isinstance( that, ShapeExpr )

        a = self.to_affine()
        b = that.to_affine()

        if len( a.terms ) and len( b.terms ):
            raise ValueError( "Scaling is supported for integers" )
        if len( b.terms ):
            a, b = b, a

        k = b.offset

        return AffineShapeExpr(
            terms = { var: coeff * k for var, coeff in a.terms.items() },
            offset = a.offset * k,
        )

    __rmul__ = __mul__

    def to_affine( self ) -> 'AffineShapeExpr':
        raise NotImplementedError

from typing import TYPE_CHECKING

from .ShapeExpr import ShapeExpr

if TYPE_CHECKING:
    from .ShapeVar import ShapeVar


class AffineShapeExpr( ShapeExpr ):
    """
    """

    def __init__( self, terms: dict[ 'ShapeVar', int ], offset: int ) -> None:
        self.offset = offset
        self.terms = terms

    def to_affine( self ):
        return self

    def copy( self, copy_map ):
        return AffineShapeExpr(
            terms = { var.copy( copy_map ): coeff for var, coeff in self.terms.items() },
            offset = self.offset,
        )

    def solve_shape_var( self, shape_var, extent ):
        for n, ( term, coeff ) in enumerate( self.terms.items() ):
            if term == shape_var:
                res = extent - self.offset
                if len( self.terms ) > 1:
                    raise NotImplementedError
                return res // coeff
        return None

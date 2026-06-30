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

    def eval( self, bindings ):
        """Evaluate the affine extent given an instance's `_bindings` (decl -> inst).

        Returns `None` if any `ShapeVar` term is still unsolved.
        """
        total = self.offset
        for shape_var, coeff in self.terms.items():
            value = bindings[ shape_var ].value
            if value is None:
                return None
            total += value * coeff
        return total

    def solve_shape_var( self, shape_var, extent ):
        for n, ( term, coeff ) in enumerate( self.terms.items() ):
            if term == shape_var:
                res = extent - self.offset
                if len( self.terms ) > 1:
                    raise NotImplementedError
                return res // coeff
        return None

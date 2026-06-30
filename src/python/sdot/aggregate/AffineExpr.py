from types import NotImplementedType

from .AffineOperand import AffineOperand


class AffineExpr( AffineOperand ):
    """Affine expression `constant + Σ coeff * shape_var`.

    `ShapeVar`s (rank 0) are used as keys; coefficients and the constant are
    integers. This is the form taken by an `Axis` extent.
    """

    def __init__( self, terms = None, constant = 0 ) -> None:
        self.constant = constant
        self.terms = {}                     # dict[ ShapeVar, int ], coeff != 0
        if terms:
            for var, coeff in terms.items():
                if coeff:
                    self.terms[ var ] = self.terms.get( var, 0 ) + coeff

    def _as_affine( self ):
        return self

    def direct_solve( self, name, size, aggregate, forbidden_new_values ):
        for term, coeff in self.terms.items():
            if term.name == name:
                if len( self.terms ) > 1:
                    # get the other values
                    raise NotImplementedError
                if ( size - self.constant ) % coeff:
                    raise ValueError( f"shape size ({ size }) - offset ({ self.constant }) is not divisible by coefficient ({ coeff })" )
                return ( size - self.constant ) // coeff
        return None

    @property
    def is_constant( self ):
        return not self.terms

    def __repr__( self ):
        parts = []
        for var, coeff in self.terms.items():
            name = getattr( var, "name", None ) or "shape_var"
            if coeff == 1:
                parts.append( name )
            elif coeff == -1:
                parts.append( f"-{ name }" )
            else:
                parts.append( f"{ coeff } * { name }" )
        if self.constant or not parts:
            parts.append( str( self.constant ) )

        out = parts[ 0 ]
        for p in parts[ 1: ]:
            out += f" - { p[ 1: ] }" if p.startswith( "-" ) else f" + { p }"
        return out


def to_affine( x ) -> AffineExpr:
    """Convert `x` (int, `ShapeVar`, `AffineExpr`, ...) into an `AffineExpr`."""
    o = _to_affine( x )
    if o is None:
        raise TypeError( f"{ x!r } cannot be turned into an affine expression" )
    return o


def extent_affine( x ) -> AffineExpr:
    """Affine extent of a *shape element* `x`.

    Shape elements (the entries of `ShapeVar( shape=[ ... ] )`) are kept raw: they
    may be `Axis`es (e.g. `shape=[ dim ]`) or plain `AffineOperand`s / ints. An
    `Axis` contributes its own extent expression, everything else goes through
    `to_affine`. Imported lazily to avoid an `Axis` <-> `AffineExpr` import cycle.
    """
    from .Axis import Axis
    if isinstance( x, Axis ):
        return x.extent
    return to_affine( x )


def _to_affine( x ):
    if isinstance( x, AffineOperand ):
        return x._as_affine()
    if isinstance( x, int ):
        return AffineExpr( constant = x )
    return None


def _add( a: AffineExpr, b: AffineExpr ) -> AffineExpr:
    terms = dict( a.terms )
    for var, coeff in b.terms.items():
        terms[ var ] = terms.get( var, 0 ) + coeff
    return AffineExpr( terms = terms, constant = a.constant + b.constant )


def _scale( a: AffineExpr, k: int ) -> AffineExpr:
    return AffineExpr(
        terms    = { var: coeff * k for var, coeff in a.terms.items() },
        constant = a.constant * k,
    )

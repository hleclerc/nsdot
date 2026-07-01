from .AffineExpr import to_affine
from .VariableAxesPlaceholder import VariableAxesPlaceholder


class AxisList:
    """A symbolic-length run of axes, whose extents are an affine expression.

    `extent` is an affine expression of `ShapeVar`s, typically built on a rank-1
    `ShapeVar` so that there is one axis per element:

        img_axes = AxisList( img_shape )         # one axis per img_shape element
        num_knot = AxisList( nb_intervals + 1 )  # extents = nb_intervals[ i ] + 1

    This is how axes that cannot be named one by one are declared. Unlike a vector
    `DynamicShapeVar` passed to a single `Axis` (which yields one ragged axis), an
    `AxisList` yields several distinct axes.
    """

    def __init__( self, extent ) -> None:
        self.extent = to_affine( extent )
        self.name   = None                     # set by `@aggregate` from the field name

    def __iter__( self ):
        # called by Python when the list is unpacked with `*` in a Tensor(...) decl;
        # we yield a single placeholder standing for the whole (symbolic) run of axes
        yield VariableAxesPlaceholder( self )

    def __getitem__( self, index ):
        # `axis_list[ i ]` -> the single (still symbolic) axis taken at `i`
        return VariableAxesPlaceholder( self, index = index )

    def __repr__( self ):
        name = self.name or "axis_list"
        return f"{ name }[ { self.extent } ]"

    def base_var( self ):
        """The single rank-1 `ShapeVar` the run of axes is built upon."""
        vars = list( self.extent.terms )
        if len( vars ) != 1:
            raise NotImplementedError( "an AxisList must depend on exactly one ShapeVar" )
        return vars[ 0 ]

    def count_affine( self ):
        """Affine expression giving the *number* of axes in the run.

        That count is the length of the underlying rank-1 `ShapeVar`, i.e. the
        extent of its (single) declaring axis (e.g. `img_shape = ShapeVar( [ nb_dims ] )`
        has `nb_dims` elements).
        """
        from .AffineExpr import extent_affine
        var = self.base_var()
        if not var.shape:
            raise NotImplementedError( "an AxisList must be built on a rank-1 ShapeVar" )
        return extent_affine( var.shape[ 0 ] )

    def direct_solve( self, name, sizes, aggregate, forbidden_new_values ):
        """Solve `name` from the concrete `sizes` taken by the whole run of axes."""
        # 1) the number of axes pins down the rank-1 var's length (e.g. `nb_dims`)
        res = self.count_affine().direct_solve( name, len( sizes ), aggregate, forbidden_new_values )
        if res is not None:
            return res

        # 2) elementwise: each extent is `coeff * base_var[ i ] + constant`, so the
        #    rank-1 var itself is recovered by inverting that affine per element.
        var = self.base_var()
        if var.name == name:
            coeff    = self.extent.terms[ var ]
            constant = self.extent.constant
            out = []
            for size in sizes:
                if ( size - constant ) % coeff:
                    raise ValueError( f"axis size ({ size }) - offset ({ constant }) is not divisible by coefficient ({ coeff })" )
                out.append( ( size - constant ) // coeff )
            return out

        return None

    def direct_solve_indexed( self, name, value, aggregate, forbidden_new_values ):
        """Solve `name` from a ragged `TensorList` value indexed by this list.

        Used for `knots = TensorList( dim, num_knot[ dim ] )`: element `d` is a 1-D
        array whose length is this list's extent at index `d`, i.e.
        `nb_intervals[ d ] + 1`. Inverting that affine per element recovers the
        rank-1 base var (`nb_intervals`).
        """
        var = self.base_var()
        if var.name == name:
            coeff    = self.extent.terms[ var ]
            constant = self.extent.constant
            out = []
            for arr in value:
                size = arr.shape[ 0 ]
                if ( size - constant ) % coeff:
                    raise ValueError( f"element size ({ size }) - offset ({ constant }) is not divisible by coefficient ({ coeff })" )
                out.append( ( size - constant ) // coeff )
            return out

        return None

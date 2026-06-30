from ..util.info import info

# re-exported so that `from sdot.aggregate import aggregate, Tensor, ShapeVar, Axis`
# keeps working from a single place
# from .DynamicShapeVar import DynamicShapeVar
# from .TensorList import TensorList
# from .ShapeVar import ShapeVar
# from .AxisList import AxisList
# from .Tensor import Tensor
# from .Axis import Axis
# from .Attribute import Reassignable


def aggregate( cls ):
    """
    Class decorator that generates boilerplate for classes with Tensor field declarations.

    The shape system has two levels:
      - `ShapeVar`: a free (symbolic) integer variable. It is NOT a size itself;
        axis extents are *expressions* built on top of ShapeVars (e.g. `nb_dims + 1`,
        `nb_vertices + 2 * nb_dims + 1`). A ShapeVar is either prescribed or solved
        from the shapes of the declared tensors.
      - `Axis`: a named tensor dimension whose extent is an expression of ShapeVars.

    A single ShapeVar can drive several axes (this is why ShapeVar and Axis are
    distinct): below `nb_dims` drives both `nvec = nb_dims + 1` and `dim = nb_dims`.

    Generated:
      - __init__( self, fields in declaration order )
      - __setattr__ with field coercion
      - one property per unique axis name (dim, nb_points, my_axis, ...)
      - batch_axes = []

    Example:

        @aggregate
        class Cell:
            # Basic case: `nb_dims` is a scalar (rank-0) ShapeVar. From `frame.shape`
            # we can solve for `nb_dims` such that frame.shape == [ nb_dims + 1, nb_dims ],
            # and check that all tensors agree on it before calling a kernel.
            nb_dims      = ShapeVar()
            nvec         = Axis( nb_dims + 1 )   # extent is a symbolic expression
            dim          = Axis( nb_dims )
            frame        = Tensor( nvec, dim )

            # Axes that cannot be named one by one (e.g. an image): `img_shape` is a
            # rank-1 ShapeVar (a vector of unknowns). `AxisList` turns that vector into
            # several distinct axes, one per element.
            img_shape    = ShapeVar( shape=[ nb_dims ] )
            img_axes     = AxisList( img_shape )
            img          = Tensor( *img_axes ) # (`*` is a hack, it actually return a single symbolic item)

            # Capacities and dynamic sizes: `max_of_nb_xs` is a plain ShapeVar used as a
            # capacity. `nb_xs` is a runtime-mutable count bounded by `capacity`; assigning
            # past the capacity raises (so dependent tensors get resized). Note the
            # asymmetry with AxisList above: passing a *vector* DynamicShapeVar to a single
            # `Axis` yields one RAGGED axis (each row may have a different length), not
            # several axes.
            max_of_nb_xs = ShapeVar()
            nb_xs        = DynamicShapeVar( capacity=max_of_nb_xs, shape=[ dim ] )
            x            = Axis( nb_xs )
            workspace    = Tensor( dim, x )      # shape == [ nb_dims, max_of_nb_xs ], rows may vary

            # TensorList allows for indexed lists of Tensors
            nb_intervals = ShapeVar( [ dim ] )
            num_knot     = AxisList( nb_intervals + 1 )
            knots        = TensorList( dim, num_knot[ dim ] )
    """

    # .name
    for klass in reversed( cls.mro() ):
        for name, value in vars( klass ).items():
            if isinstance( value, Reassignable ):
                value.name = name

    # __base_init__
    def __base_init__( self, *arg, **kwarg ):
        copy_map = {}
        for klass in reversed( cls.mro() ):
            for name, value in vars( klass ).items():
                if isinstance( value, Reassignable ):
                    self.__dict__[ name ] = value.copy( copy_map )
    cls.__base_init__ = __base_init__

    if '__init__' not in vars( cls ):
        cls.__init__ = __base_init__


    # # collect the field declarations, in declaration order, parents first
    # shape_vars = {}
    # tensors = {}
    # axes = {}
    # all = {}
    # for klass in reversed( cls.mro() ):
    #     for name, value in vars( klass ).items():
    #         # _FIELD_TYPES = ( Tensor, TensorList, DynamicShapeVar, ShapeVar, AxisList, Axis )
    #         if isinstance( value, ShapeVar ):
    #             shape_vars[ name ] = value
    #             all[ name ] = value
    #         if isinstance( value, ( Tensor, TensorList ) ):
    #             tensors[ name ] = value
    #             all[ name ] = value
    #         if isinstance( value, ( Axis, AxisList ) ):
    #             axes[ name ] = value
    #             all[ name ] = value

    # # give every declaration the name of the field it is bound to (used in reprs
    # # and, later, to generate properties / __init__)
    # for name, decl in all.items():
    #     if getattr( decl, "name", None ) is None:
    #         decl.name = name

    # # __setattr__
    # def __setattr__( self, name, value ):
    #     annotation = all.get( name )
    #     if annotation is not None:
    #         if coerce := getattr( annotation, "coerce", None ):
    #             value = coerce( value )
    #         elif value is not None and not isinstance( value, annotation ):
    #             value = annotation( value )
    #     object.__setattr__( self, name, value )
    # cls.__setattr__ = __setattr__

    # # __getattribute__
    # def __getattribute__( self, name ):
    #     if name in shape_vars:
    #         valued_tensors = []
    #         for tensor_name, decl in tensors.items():
    #             value = getattr( self, tensor_name, None )
    #             if value is not None:
    #                 valued_tensors.append( ( decl, value ) )
    #         return get_shape_var( name, valued_tensors, self )
    #     return object.__getattribute__( self, name )
    # cls.__getattribute__ = __getattribute__

    return cls

# def get_shape_var( name, valued_tensors, aggregate ):
#     """Solve the ShapeVar `name` from the shapes of the currently bound tensors.

#     Triggered when a ShapeVar field is read (`c.nb_dims`) via `__getattribute__`.
#     Each bound field's *value* is passed to `decl.direct_solve( name, value, ... )`
#     (an array for a `Tensor`, a ragged `list` of arrays for a `TensorList`); the
#     first non-`None` answer wins. The answer is a scalar, or a `list` for a rank-1
#     (vector) ShapeVar. The `direct_solve` chain:

#       - `Tensor`: `_segments( value.shape )` cuts the concrete shape into one slice
#         per declared axis (the length of a `*axis_list` run is the leftover dims).
#         A plain `Axis` inverts its affine extent; a run delegates to `AxisList`.
#       - `AxisList`: (1) the *number* of axes in the run pins the rank-1 var's length
#         via `count_affine()`; (2) otherwise solve elementwise (invert the affine per
#         element) when `name` is the base var.
#       - `TensorList`: leading axis -> element count; an indexed axis (`num_knot[ dim ]`)
#         -> `AxisList.direct_solve_indexed` (each element's own length).

#     `direct_solve` returns `None` when *this* decl doesn't constrain `name` (so we
#     move on), and raises on an actual inconsistency. Not done yet: *indirect* solve
#     (no tensor pins `name` directly) and multi-term affines (`AffineExpr.direct_solve`
#     raises `NotImplementedError`).
#     """
#     for decl, value in valued_tensors:
#         res = decl.direct_solve( name, value, aggregate, [ name ] )
#         if res is not None:
#             return res

#     # TODO: indirect solve
#     raise NotImplementedError( f"Unable to find value of shape variable '{ name }'" )

# def batch_version_of( cls, base_version ):
#     return cls

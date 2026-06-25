from ..util.info import info

# re-exported so that `from sdot.aggregate import aggregate, Tensor, ShapeVar, Axis`
# keeps working from a single place
from .DynamicShapeVar import DynamicShapeVar
from .TensorList import TensorList
from .ShapeVar import ShapeVar
from .AxisList import AxisList
from .Tensor import Tensor
from .Axis import Axis

# declaration types that `@aggregate` collects as fields
_FIELD_TYPES = ( Tensor, TensorList, DynamicShapeVar, ShapeVar, AxisList, Axis )


def _is_aggregate_attribute( attr ):
    return isinstance( attr, _FIELD_TYPES )


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

    # collect the field declarations, in declaration order, parents first
    attributes = {}
    for klass in reversed( cls.mro() ):
        for name, value in vars( klass ).items():
            if _is_aggregate_attribute( value ):
                attributes[ name ] = value

    # give every declaration the name of the field it is bound to (used in reprs
    # and, later, to generate properties / __init__)
    for name, decl in attributes.items():
        if getattr( decl, "name", None ) is None:
            decl.name = name

    # TODO: generate __init__, __setattr__, the per-axis properties and batch_axes.
    # info( attributes )

    return cls

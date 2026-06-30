import numpy as np

from sdot.aggregate import aggregate, Tensor, TensorList, ShapeVar, DynamicShapeVar, AxisList, Axis
from . import test


if test( "basic" ):
    # `nb_dims` is a scalar (rank-0) ShapeVar driving two axes. From `frame.shape`
    # we solve `nb_dims` such that frame.shape == [ nb_dims + 1, nb_dims ].
    @aggregate
    class Cell:
        nb_dims      = ShapeVar() # nb_dims est un tenseur entier de rang 0

        nvec         = Axis( nb_dims + 1 ) # le 1er argument donne la taille max selon cet axe. À ce stade c'est une opération symbolique
        dim          = Axis( nb_dims )

        frame        = Tensor( nvec, dim ) # on doit pouvoir trouver `nb_dims` tel que `frame.shape = [ nb_dims + 1, nb_dims ]`

    c = Cell()
    c.frame = [ [ 0, 0 ], [ 1, 0 ], [ 0, 1 ] ]
    assert c.nb_dims == 2


if test( "affine_extent" ):
    # the extent is a non-trivial affine expression (`2 * nb_dims + 1`); solving
    # must invert the coefficient and the offset.
    @aggregate
    class Stencil:
        nb_dims = ShapeVar()
        wide    = Axis( 2 * nb_dims + 1 )
        data    = Tensor( wide )

    s = Stencil()
    s.data = [ 0, 1, 2, 3, 4, 5, 6 ]   # 7 == 2 * 3 + 1
    assert s.nb_dims == 3


if test( "image" ):
    # Axes that cannot be named one by one: `img_shape` is a rank-1 ShapeVar (a
    # vector of unknowns). `AxisList` turns that vector into several axes, one per
    # element, so the *number* of axes gives `nb_dims` and their sizes give
    # `img_shape`.
    @aggregate
    class Img:
        nb_dims   = ShapeVar()
        img_shape = ShapeVar( shape = [ nb_dims ] )
        img_axes  = AxisList( img_shape )
        img       = Tensor( *img_axes )

    i = Img()
    i.img = np.zeros( [ 4, 6, 8 ] )
    assert i.nb_dims == 3
    assert list( i.img_shape ) == [ 4, 6, 8 ]


if test( "capacity" ):
    # `max_of_nb_xs` is a plain ShapeVar used as a capacity. `nb_xs` is a runtime
    # count bounded by it; passing a *vector* DynamicShapeVar to a single `Axis`
    # yields one RAGGED axis whose physical storage is the capacity. So
    # `workspace.shape == [ nb_dims, max_of_nb_xs ]`.
    @aggregate
    class WS:
        nb_dims      = ShapeVar()
        dim          = Axis( nb_dims )
        max_of_nb_xs = ShapeVar()
        nb_xs        = DynamicShapeVar( capacity = max_of_nb_xs, shape = [ dim ] )
        x            = Axis( nb_xs )
        workspace    = Tensor( dim, x )

    w = WS()
    w.workspace = np.zeros( [ 2, 5 ] )
    assert w.nb_dims == 2
    assert w.max_of_nb_xs == 5


if test( "tensor_list" ):
    # `nb_intervals` is a rank-1 ShapeVar (one unknown per `dim`). `knots` is an
    # indexed list of ragged tensors: element `d` has length `nb_intervals[ d ] + 1`.
    @aggregate
    class Knots:
        nb_dims      = ShapeVar()
        dim          = Axis( nb_dims )
        nb_intervals = ShapeVar( [ dim ] )
        num_knot     = AxisList( nb_intervals + 1 )
        knots        = TensorList( dim, num_knot[ dim ] )

    k = Knots()
    k.knots = [ [ 0, 1, 2, 3 ], [ 0, 5 ] ]   # 2 rows, lengths 4 and 2
    assert k.nb_dims == 2
    assert list( k.nb_intervals ) == [ 3, 1 ]

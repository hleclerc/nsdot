from sdot import ShapeVar, Axis, AxisList, Tensor, aggregate, driver
from . import test

if test( "basic_tensor" ):
    @aggregate
    class Cell:
        vertex_positions : Tensor[ "num_vertex", "dim" ]
        vertex_indices   : Tensor[ "num_vertex", "dim", { "dtype": int } ]

        num_vertex       : Axis[ "nb_vertices" ]
        dim              : Axis[ "nb_dims" ]

        nb_vertices      : ShapeVar
        nb_dims          : ShapeVar

        def __init__( self, **kw ) -> None: ...


    nb_dims = ShapeVar()
    c = Cell( nb_dims = nb_dims )

    c.vertex_positions = [ [ 1, 2 ] ]
    # Vérifier que le dtype est bien extracté

    info( c.nb_vertices )

    # `c.nb_vertices` now reads back the solved value (value-on-read)
    assert c.nb_vertices == 1
    assert c.nb_dims == 2



if test( "ragged" ):
    @aggregate
    class Mesh:
        cell_vertices   : Tensor[ "cell", "vtx" ]

        cell            : Axis[ "nb_cells" ]
        vtx             : Axis[ "nb_vtx_per_cell" ]      # ragged: depends on `cell`

        nb_cells        : ShapeVar
        nb_vtx_per_cell : ShapeVar[ "cell" ]             # rank-1 (one count per cell)

        def __init__( self, **kw ) -> None: ...


    m = Mesh()
    m.cell_vertices = [ [ 10, 11 ], [ 12 ] ]   # cell 0 has 2 vertices, cell 1 has 1

    # sizes are read from the nesting only (no data touched)
    assert m.nb_cells == 2
    assert list( m.nb_vtx_per_cell ) == [ 2, 1 ]

    # values assembled into a padded rank-2 buffer (pad = 0), functionally
    import numpy
    raw = numpy.asarray( m.cell_vertices.raw )
    assert raw.shape == ( 2, 2 )
    assert raw.tolist() == [ [ 10, 11 ], [ 12, 0 ] ]

    info( m.nb_vtx_per_cell )


if test( "AxisList" ):
    @aggregate
    class Image:
        values  : Tensor[ "img_pos..." ]
        knots   : Tensor[ "dim", "num_knot" ]

        num_knot: Axis[ "extent + 1" ]          # ragged over `dim` via `extent`
        img_pos : AxisList[ "dim", "extent" ]    # unrolled into `nb_dims` static axes
        dim     : Axis[ "nb_dims" ]

        nb_dims : ShapeVar
        extent  : ShapeVar[ "dim" ]             # rank-1 (one count per dim)

        def __init__( self, **kw ) -> None: ...


    m = Image( values = driver.random( [ 2, 1 ] ), knots = [ [ 0, 1, 2 ], [ 0, 1 ] ] )
    assert list( m.extent ) == [ 2, 1 ]
    assert m.nb_dims == 2


if test( "axis_parsing" ):
    @aggregate
    class Dell:
        x: ShapeVar
        y: ShapeVar

        a1: Axis[ "x" ]
        a2: Axis[ "2 * x + 3" ]
        a3: Axis[ "x - 5" ]
        a4: Axis[ "3 * x + 2 * y - 1" ]
        a5: Axis[ "- x + 10" ]

        def __init__( self, **ka ) -> None: ...

    c = Dell()

    # Test simple variable
    assert len( c.a1.coeffs ) == 1 and c.a1.offset == 0

    # Test coefficient + constant
    assert len( c.a2.coeffs ) == 1 and c.a2.offset == 3
    coeff2 = list(c.a2.coeffs.values())[0]
    assert coeff2 == 2

    # Test subtraction
    assert len( c.a3.coeffs ) == 1 and c.a3.offset == -5

    # Test multiple variables
    assert len( c.a4.coeffs ) == 2 and c.a4.offset == -1

    # Test negative variable
    assert len( c.a5.coeffs ) == 1 and c.a5.offset == 10
    coeff5 = list(c.a5.coeffs.values())[0]
    assert coeff5 == -1

    info( "All axis parsing tests passed" )



# if test( "basic" ):
#     @aggregate
#     class Cell:
#         nb_dims = ShapeVar()
#         num     = Axis( nb_dims + 1 )
#         dim     = Axis( nb_dims )

#         frame   = Tensor( num, dim )

#         def __init__( self ) -> None:
#             self.pouet = 32

#     c = Cell()
#     # frame shape == [ num, dim ] == [ nb_dims + 1, nb_dims ] -> [ 3, 2 ] solves nb_dims = 2
#     c.frame = [ [ 0, 0 ], [ 1, 1 ], [ 2, 2 ] ]
#     assert c.nb_dims == 2
#     assert c.pouet == 32

#     # a prescribed value wins over what the tensors imply
#     c.nb_dims = 1222
#     assert c.nb_dims == 1222


# if test( "shared" ):
#     @aggregate
#     class Celm:
#         nb_dims = ShapeVar()
#         num     = Axis( nb_dims + 1 )
#         dim     = Axis( nb_dims )
#         frame   = Tensor( num, dim )

#     n = ShapeVar()
#     a = Celm( nb_dims = n )        # a and b share the same nb_dims cell
#     b = Celm( nb_dims = n )

#     a.frame = [ [ 0, 0 ], [ 1, 1 ], [ 2, 2 ] ]   # only a is given a value
#     assert a.nb_dims == 2
#     assert b.nb_dims == 2          # b sees it through the shared cell

# from typing import Any

# class MonTypeTemplate:
#     def __init__(self, params: Any):
#         self.params = params

#     # Permet la syntaxe MonTypeTemplate[int] ou MonTypeTemplate["X"]
#     def __class_getitem__(cls, item, **kw ):
#         # En production, vous pouvez retourner un objet proxy ou une instance spécialisée
#         return cls(params=item)

# @aggregate
# class Test:
#     a : MonTypeTemplate[ 132, a = 2 ]

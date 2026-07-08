from sdot import ShapeVar, Axis, Tensor, aggregate
from . import test

if test( "basic" ):
    @aggregate
    class Cell:
        nb_vertices: ShapeVar
        nb_dims    : ShapeVar

        num_vertex : Axis[ "nb_vertices" ]
        dim        : Axis[ "nb_dims" ]

        positions  : Tensor[ "num_vertex", "dim", ("dtype", int) ]

        def __init__( self, **kw ) -> None: ...


    nb_dims = ShapeVar()
    c = Cell( nb_dims = nb_dims )

    c.positions = [ [ 1, 2 ] ]
    # Vérifier que le dtype est bien extracté
    info( c.positions.raw )



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

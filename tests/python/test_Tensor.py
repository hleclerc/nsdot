from sdot import Tensor, ShapeVar, Axis, Node
from . import test

if test( "basic" ):
    class Cell( Node ):
        def __init__( self ) -> None:
            self.nb_dims = ShapeVar()
            self.num     = Axis( extent = self.nb_dims + 1 )
            self.dim     = Axis( extent = self.nb_dims )
            self.frame   = Tensor( self.num, self.dim )

    c = Cell()
    c.frame = [ [ 0 ], [ 1 ] ]

    info( c.frame )
    info( c.nb_dims.value )

from sdot import Cell
from . import test


if test( "pouet" ):
    c = Cell()
    c.frame = [[0,0],[1,0],[0,1]]
    info( c.frame )
    info( c.nb_dims )

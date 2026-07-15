from sdot import Cell
from . import test

if test( "basic" ):
    c = Cell.make_hypercube( 2 )
    info( c )

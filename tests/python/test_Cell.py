from sdot import Cell, driver
from . import test

if test( "pouet" ):
    info( driver.array( [ 1, 2 ] ) )
    info( driver.ftype )
    c = Cell()

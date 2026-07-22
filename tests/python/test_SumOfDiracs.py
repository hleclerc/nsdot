from sdot import SumOfDiracs
from . import test

if test( "basic" ):
    di = SumOfDiracs( positions = [ [ 1 ], [ 2 ], [ 3 ] ] )
    assert di.measure == 3

    dj = SumOfDiracs( positions = [ [ 1 ], [ 2 ], [ 3 ] ], weights = [ 2, 2, 4 ] )
    assert dj.measure == 8

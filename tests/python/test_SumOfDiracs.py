from sdot import SumOfDiracs
from sdot.testing import test

if test( "basic" ):
    di = SumOfDiracs( positions = [ [ 1 ], [ 2 ], [ 3 ] ] )
    assert di.mass == 3

    dj = SumOfDiracs( positions = [ [ 1 ], [ 2 ], [ 3 ] ], weights = [ 2, 2, 4 ] )
    assert dj.mass == 8

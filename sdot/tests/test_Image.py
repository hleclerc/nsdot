from sdot import Image # OtPlan1D, SumOfDiracs,
from loom.testing import test, check_grad

if test( "basic" ):
    # ds = SumOfDiracs( [ 1, 2, 3, 4 ] )
    # op = OtPlan1D( ds, di )
    di = Image( values = [[ 1, 0, 1 ]] )
    assert float( di.mass ) == 2

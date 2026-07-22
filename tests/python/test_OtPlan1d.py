from sdot import Image, OtPlan1d, SumOfDiracs1d
from . import test, check_grad

if test( "basic" ):
    src = SumOfDiracs1d( positions = [ 0, 1, 2 ] )
    dst = Image( values = [ 1, 0, 1 ] )

    otp = OtPlan1d( src, dst )

    info( otp.cost )

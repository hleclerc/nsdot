# import sdot

# a = sdot.SumOfDiracs( [ [ 0, 0 ], [ 1, 0 ], ] )
# p = sdot.SdotPlanNd( a )
# print( p.cost )
from errand import bench, Param

if p := bench( "gro", gro = Param( 100, help = "pouet" ) ):
    print("yo")
    p.results[ "truc" ] = 1232

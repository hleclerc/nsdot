# import dask.array as da  # type: ignore[import-untyped]
from icecream import install
import sdot

install()

# points = da.random.random( ( 100, 2 ), chunks = ( 100, 2 ) ).astype( "float64" ) * da.array( [ 2, 1 ] )
# bsp = sdot.Bsp( points, max_points_per_node = 40, max_points_per_cell = 20 )

# for ind, item in enumerate( bsp.items ):
#     # print( item )
#     item.compute().write_vtk( f"results/out_{ ind }.vtk" )

# for ind, item in enumerate( bsp.items ):
#     item.compute().write_vtk( f"results/out_{ ind }.vtk" )

# ( X - Y )^2 - w1 - ( X - D )^2 + w0
f = sdot.SumOfWeightedDiracs( [ [ 0, .5 ], [ 1, .5 ] ] )
g = sdot.PiecewiseConstantImage( [ [ 1, 1 ] ] )
plan = sdot.plan( f, g )
ic( plan )

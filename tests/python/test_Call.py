# from sdot.aggregate import aggregate, batch_version_of, Tensor, ShapeVar, Axis
# from sdot import driver
# from . import test
# import numpy


# if test( "basic" ):
#     @aggregate
#     class Cell:
#         nb_dims = ShapeVar()
#         dim     = Axis( nb_dims )
#         frame   = Tensor( dim )

#         def determinant( self ):
#             return driver.determinant( self.frame[ ..., 1:, : ] )

#     def BatchOfCells( batch_axes ):
#         return batch_version_of( Cell, batch_axes )

#     c = Cell()
#     c.frame = [ 10, 11 ]
#     info( c.determinant() )

#     b = BatchOfCells( [ "batch_axis" ] )
#     b.frame = [ [ 10, 12 ], [ 11, 13 ] ]
#     info( b.determinant() )



# info( c.frame )
#
# return driver.call(
#     FfiCode(
#         """
#         run_parallel( queue_list, p.batch_axes, [](  ) {

#         } );
#         p.output[ ]
#         """
#     ),
#     output = Return( int )
#     cell = c
# )

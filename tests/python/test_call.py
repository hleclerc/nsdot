from sdot import ShapeVar, Axis, Tensor, Return, aggregate, driver
from . import test

if test( "basic" ):
    @aggregate
    class Cell:
        vertex_positions : Tensor[ "num_vertex", "dim" ]

        num_vertex       : Axis[ "nb_vertices" ]
        dim              : Axis[ "nb_dims" ]

        nb_vertices      : ShapeVar
        nb_dims          : ShapeVar

        def __init__( self, **kw ) -> None: ...


    # "run_parallel( batch_axes, []( auto batch_indices, auto cell ) { cell( batch_indices ).nb_vertices = 0; } )",
    c = driver.call(
        'cell.nb_vertices = 0;',
        cell = Return( Cell, nb_vertices = 8, nb_dims = 2 ), # reservation
        # frame = driver.array( [ [ 0 ] ] )
    )

    # info( c.nb_vertices )

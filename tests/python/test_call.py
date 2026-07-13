from sdot import CtShapeVar, ShapeVar, Axis, Tensor, Return, aggregate, driver
from . import test

if test( "basic" ):
    @aggregate
    class Cell:
        vertex_positions : Tensor[ "num_vertex", "dim" ]

        num_vertex       : Axis[ "nb_vertices" ]
        dim              : Axis[ "nb_dims" ]

        nb_vertices      : ShapeVar
        nb_dims          : CtShapeVar

        def __init__( self, **kw ) -> None: ...


    # "run_parallel( batch_axes, []( auto batch_indices, auto cell ) { cell( batch_indices ).nb_vertices = 0; } )",
    c = driver.call(
        """
        //run_parallel(
        //    queue,
        //    global_batch_sizes,
        cell.nb_vertices = 1;
        cell.vertex_positions( dim = 0, num_vertex = 0 ) = 1;
        cell.vertex_positions( dim = 1, num_vertex = 0 ) = 2;
        //)
        """,
        cell = Return( Cell, max_of_nb_vertices = 8, nb_dims = 2 ),
        # frame = driver.array( [ [ 0 ] ] )
    )

    info( c.vertex_positions )


if test( "two_instances" ):
    # the same aggregate, twice in one call, with different compile-time shape vars: `Cell` is
    # generated as a C++ template, instantiated once per argument.
    @aggregate
    class Cell:
        vertex_positions : Tensor[ "num_vertex", "dim" ]

        num_vertex       : Axis[ "nb_vertices" ]
        dim              : Axis[ "nb_dims" ]

        nb_vertices      : ShapeVar
        nb_dims          : CtShapeVar

        def __init__( self, **kw ) -> None: ...


    r = driver.call(
        """
        flat.nb_vertices = 1;
        flat.vertex_positions( num_vertex = 0, dim = 0 ) = 1;
        flat.vertex_positions( num_vertex = 0, dim = 1 ) = 2;

        volu.nb_vertices = 1;
        volu.vertex_positions( num_vertex = 0, dim = 2 ) = 3;
        """,
        flat = Return( Cell, max_of_nb_vertices = 8, nb_dims = 2 ),
        volu = Return( Cell, max_of_nb_vertices = 4, nb_dims = 3 ),
    )

    info( r[ "flat" ].vertex_positions, r[ "volu" ].vertex_positions )


if test( "nested" ):
    # an aggregate field whose type is itself an aggregate: `Cell` is generated as its own C++
    # template, and `Pair` holds two instantiations of it (and forwards their parameters).
    @aggregate
    class Cell:
        vertex_positions : Tensor[ "num_vertex", "dim" ]

        num_vertex       : Axis[ "nb_vertices" ]
        dim              : Axis[ "nb_dims" ]

        nb_vertices      : ShapeVar
        nb_dims          : CtShapeVar

        def __init__( self, **kw ) -> None: ...


    @aggregate
    class Pair:
        left  : Cell
        right : Cell

        def __init__( self, **kw ) -> None: ...


    # a mapping under a field's name scopes an initializer to that field; what stays at the
    # `Pair` level (`max_of_nb_vertices`) reaches both cells.
    p = driver.call(
        """
        pair.left.nb_vertices = 1;
        pair.left.vertex_positions( num_vertex = 0, dim = 1 ) = 1;

        pair.right.nb_vertices = 1;
        pair.right.vertex_positions( num_vertex = 0, dim = 2 ) = 2;
        """,
        pair = Return(
            Pair,
            max_of_nb_vertices = 8,
            left  = { "nb_dims": 2 },
            right = { "nb_dims": 3 },
        ),
    )

    info( p.left.vertex_positions, p.right.vertex_positions )

# Comme on peut le voir dans @tests/python/test_call.py , on peut proposer des max pour les variables qui déterminent les tailles des tenseurs de sortie. Il est possible que ce qui est proposé ne soit pas suffisant. Dans ce cas, il faudrait le repérer (un test à faire dans @src/cpp/sdot/support/containers/ShapeVarView.h ), et stocker dans une zone mémoire apropriés

from sdot import CtShapeVar, ShapeVar, Axis, Tensor, aggregate, driver, FfiCode
from . import test

# An `@aggregate` instance is built BEFORE the call and passed as a plain kwarg. Inputs and
# outputs are DISJOINT (as in XLA): a kernel never writes what it reads, so there is no
# aliasing, and no reconciling of two sizes under one name. What looks like a mutation is a
# Python-side rebinding, done by the caller between two calls.
#
# Each attribute of a passed object falls in exactly one category:
#
# * named in `output_attributes` -> an OUTPUT: a fresh buffer, allocated at the capacity this
#   call decided, rebound onto the attribute once the call returns.
# * holds a value -> an INPUT, bound at the size its data actually has.
# * empty and not declared -> UNBOUND: nothing crosses the FFI, and the kernel is not handed a
#   degenerate view to test but a `NoneTensor`, a TYPE with no data (an attribute may simply be
#   optional and unused -- see `partial_init`).
#
# Nothing is returned: outputs are written back onto the instances we were given (the
# aggregate is our own Python object -- Jax only ever sees the tensors inside it).
#
# The io category is not just bookkeeping: it is what `make_available` uses to move a member to
# the device and back. An aggregate carries its members' categories, so nothing crosses that has
# no reason to; a bare tensor is tagged in the body (`OutList()`, ...).
#
# A `ShapeVar` holds a COUNT: how many items are used. A kernel reads it or writes it, so it
# lives in a device buffer; under `jit` Python does not know it, and it can therefore never
# size anything (an XLA shape cannot depend on a device value).
#
# What sizes a buffer is a CAPACITY -- and a capacity is NOT state on the object: it is a
# decision about ONE allocation, so it is given to the CALL that allocates. An object only
# ever says what it IS; the call says what it allocates. A capacity already materialized in a
# buffer is read back from it, so a chained call need not restate it.

if test( "basic" ):
    @aggregate
    class Cell:
        vertex_positions : Tensor[ "num_vertex", "dim" ]

        num_vertex       : Axis[ "nb_vertices" ]
        dim              : Axis[ "nb_dims" ]

        nb_vertices      : ShapeVar
        nb_dims          : CtShapeVar

        def __init__( self, **kw ) -> None: ...


    # the ctor only prescribes what the cell IS: `nb_dims` is compile-time known (its count is
    # its size). `nb_vertices` has no count yet -- the kernel is what writes it.
    cell = Cell( nb_dims = 2 )

    # the body runs on a DEVICE. `queue` is the call's execution context (a `sdot::Queue`, chosen
    # by the device -- a typedef, since the memory space a pointer lives in is part of its type),
    # and `global_batch_indices` is what the kernel iterates over: by default a single item, the
    # empty multi-index (a `vmap` is what will give it axes).
    #
    # An object handed to the kernel is an ARGUMENT of `run_parallel`, not a capture: that is
    # what lets `make_available` retype its pointers into the memory space the kernel reads. It
    # is made available member by member, each with the io category Python knows it has -- an
    # input is not copied back, an output not copied in.
    driver.call(
        FfiCode( name = "test_call_basic", fwd_code = """
        run_parallel(
            queue,
            global_batch_indices,
            []( auto batch_index, auto cell ) {
                cell.nb_vertices = 1;
                cell.vertex_positions( dim = 0, num_vertex = 0 ) = 1;
                cell.vertex_positions( dim = 1, num_vertex = 0 ) = 2;
            },
            cell
        );
        """ ),
        cell = cell,
        output_attributes = [ "cell.nb_vertices", "cell.vertex_positions" ],
        capacities = { "cell.nb_vertices": 8 },   # => `vertex_positions` is allocated 8x2
        # frame = driver.array( [ [ 0 ] ] )
    )

    info( cell.vertex_positions )

    # a `Tensor` needs no wrapper aggregate to be an argument: here it borrows `cell`'s axis,
    # hence `cell`'s ShapeVar. That ShapeVar's capacity is not restated: it is READ BACK from
    # `cell.vertex_positions`, which is allocated 8x2 -- so `res` gets 8 too. `cell` is a
    # read-only input of this second call.
    #
    # A bare tensor has no members, hence no io category of its own to carry: it is the body that
    # tags it (`OutList()`), as it would in hand-written C++. An aggregate needs no tag -- each of
    # its members already knows what it is.
    res = Tensor[ cell.num_vertex ]()

    driver.call(
        FfiCode( name = "test_call_basic_res", fwd_code = """
        run_parallel(
            queue,
            global_batch_indices,
            []( auto batch_index, auto cell, auto res ) {
                res( num_vertex = 0 ) = cell.nb_vertices;
            },
            cell, OutList(), res
        );
        """ ),
        cell = cell,
        res = res,
        output_attributes = [ "res" ],
    )

    info( res )


if test( "partial_init" ):
    # not every declared tensor has to be filled: `vertex_indices` is neither given a value nor
    # declared as an output, so nothing is bound for it. It does not become a degenerate
    # `TensorView` to be tested at runtime -- it lowers to a `NoneTensor`, a distinct TYPE with
    # no data, so the kernel discriminates at COMPILE time (and a `static_assert` can forbid
    # touching it outright).
    @aggregate
    class Cell2:
        vertex_positions : Tensor[ "num_vertex", "dim" ]
        vertex_indices   : Tensor[ "num_vertex", "dim", dict( dtype = int ) ]

        num_vertex       : Axis[ "nb_vertices" ]
        dim              : Axis[ "nb_dims" ]

        nb_vertices      : ShapeVar
        nb_dims          : CtShapeVar

        def __init__( self, **kw ) -> None: ...


    cell = Cell2( nb_dims = 2 )

    driver.call(
        FfiCode( name = "test_partial_init", fwd_code = """
        run_parallel(
            queue,
            global_batch_indices,
            []( auto batch_index, auto cell ) {
                cell.nb_vertices = 1;
                cell.vertex_positions( num_vertex = 0, dim = 0 ) = 1;

                static_assert( DECAYED_TYPE_OF( cell.vertex_positions.is_valid() )::value == 1 );
                static_assert( DECAYED_TYPE_OF( cell.vertex_indices  .is_valid() )::value == 0 );
            },
            cell
        );
        """ ),
        cell = cell,
        output_attributes = [ "cell.nb_vertices", "cell.vertex_positions" ],
        capacities = { "cell.nb_vertices": 8 },
    )

    info( cell.vertex_positions )


if test( "two_instances" ):
    # the same aggregate, twice in one call, with different compile-time shape vars: `Cell` is
    # generated as a C++ template, instantiated once per argument.
    @aggregate
    class Cell3:
        vertex_positions : Tensor[ "num_vertex", "dim" ]

        num_vertex       : Axis[ "nb_vertices" ]
        dim              : Axis[ "nb_dims" ]

        nb_vertices      : ShapeVar
        nb_dims          : CtShapeVar

        def __init__( self, **kw ) -> None: ...


    flat = Cell3( nb_dims = 2 )
    volu = Cell3( nb_dims = 3 )

    driver.call(
        FfiCode( name = "two_instances", fwd_code = """
        run_parallel(
            queue,
            global_batch_indices,
            []( auto batch_index, auto flat, auto volu ) {
                flat.nb_vertices = 1;
                flat.vertex_positions( num_vertex = 0, dim = 0 ) = 1;
                flat.vertex_positions( num_vertex = 0, dim = 1 ) = 2;

                volu.nb_vertices = 1;
                volu.vertex_positions( num_vertex = 0, dim = 2 ) = 3;
            },
            flat, volu
        );
        """ ),
        flat = flat,
        volu = volu,
        output_attributes = [
            "flat.nb_vertices", "flat.vertex_positions",
            "volu.nb_vertices", "volu.vertex_positions",
        ],
        capacities = { "flat.nb_vertices": 8, "volu.nb_vertices": 4 },
    )

    info( flat.vertex_positions, volu.vertex_positions )


if test( "nested" ):
    # an aggregate field whose type is itself an aggregate: `Cell` is generated as its own C++
    # template, and `Pair` holds two instantiations of it (and forwards their parameters).
    @aggregate
    class Cell4:
        vertex_positions : Tensor[ "num_vertex", "dim" ]

        num_vertex       : Axis[ "nb_vertices" ]
        dim              : Axis[ "nb_dims" ]

        nb_vertices      : ShapeVar
        nb_dims          : CtShapeVar

        def __init__( self, **kw ) -> None: ...


    @aggregate
    class Pair:
        left  : Cell4
        right : Cell4

        def __init__( self, **kw ) -> None: ...


    # a mapping under a field's name scopes a prescription to that field alone.
    pair = Pair( left = { "nb_dims": 2 }, right = { "nb_dims": 3 } )

    driver.call(
        FfiCode( name = "test_call_nested", fwd_code = """
        run_parallel(
            queue,
            global_batch_indices,
            []( auto batch_index, auto pair ) {
                pair.left.nb_vertices = 1;
                pair.left.vertex_positions( num_vertex = 0, dim = 1 ) = 1;

                pair.right.nb_vertices = 1;
                pair.right.vertex_positions( num_vertex = 0, dim = 2 ) = 2;
            },
            pair
        );
        """ ),
        pair = pair,
        output_attributes = [ "pair" ],   # a whole subtree can be named at once
        capacities = { "pair.left.nb_vertices": 8, "pair.right.nb_vertices": 4 },
    )

    info( pair.left.vertex_positions, pair.right.vertex_positions )

# TODO: a capacity may turn out to be too small. `ShapeVarView::operator=` (see
# src/cpp/sdot/support/containers/ShapeVarView.h) must detect a count > max, record the
# offending ShapeVar in an error buffer, and let Python relaunch with a larger capacity.

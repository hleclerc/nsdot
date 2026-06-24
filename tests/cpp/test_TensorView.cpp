// #include "../../src/cpp/sdot/support/containers/contiguous_strides.h"
// #include "../../src/cpp/sdot/support/hardware/MemorySpace_CpuRam.h"
// #include "../../src/cpp/sdot/support/hardware/Run.h"
#include <sdot/support/containers/TensorView.h>
#include <sdot/support/kernels/run_parallel.h>
#include <sdot/support/containers/Range.h>
// #include "sdot_test_matrix.h"
#include "sdot/support/common_macros.h"
#include "sdot/support/kernels/CpuHostMemorySpace.h"
#include "test_main.h"

using namespace sdot;

// struct Test {
//     auto apply_values( auto &&func ) { return func( a ); }
//     static Test make_variant( int a ) { return { a }; }
//     void display( auto &ds ) const { ds << a; }
//     int a;
// };

// template<class Op,class Mt>
// struct LimitNbThread {
//     auto max_nb_threads( auto&&...args ) const { return mt( FORWARD( args )... ); }
//     auto operator()    ( auto&&...args ) const { return op( FORWARD( args )... ); }
//     Op   op;
//     Mt   mt;
// };
//

/// nom
DEFINE_AXIS( row_ );
DEFINE_AXIS( col_ );
DEFINE_AXIS( dim );


TEST_CASE( "TensorView — indexation (positions + axes nommés)", "" ) {
    using TF = double;
    TF data[] = { 1, 2, 3, 4 }; // 2x2 row-major : [[1,2],[3,4]]

    // axe 0 anonyme, axe 1 nommé `dim`
    auto t = tensor_view( data, tuple( 2, 2 ), tuple( UnnamedAxis{}, dim ) );

    // indexation mixte position / nom
    CHECK( t( 0, dim = 0 ).value() == 1 );
    CHECK( t( 0, dim = 1 ).value() == 2 );
    CHECK( t( 1, dim = 0 ).value() == 3 );
    CHECK( t( 1, 1 ).value() == 4 );          // tout positionnel

    // squeeze direct (1 et 2 args) et row
    CHECK( t.squeeze( 0_c, 1 ).squeeze( dim = 0 ).value() == 3 );
    CHECK( t.row( 0 )( dim = 1 ).value() == 2 );

    // ref() écrit en place
    t( 1, dim = 1 ).ref() = 40;
    CHECK( data[ 3 ] == 40 );
}


TEST_CASE( "TensorView — strides non contigus, rang 3, offset", "" ) {
    // 2x3 row-major : [[1,2,3],[4,5,6]] ; vue transposée 3x2 via strides (8, 24) octets
    double m[] = { 1, 2, 3, 4, 5, 6 };
    auto tT = tensor_view( m, tuple( 3, 2 ), tuple( row_, col_ ), tuple( 8, 24 ) );
    CHECK( tT( 0, 0 ).value() == 1 );
    CHECK( tT( 2, 1 ).value() == 6 );
    CHECK( tT( row_ = 0, col_ = 1 ).value() == 4 ); // indexation par noms, ordre indifférent
    CHECK( tT( col_ = 1, row_ = 2 ).value() == 6 );

    // rang 3 contigu 2x2x2 : c[i][j][k] = 4i + 2j + k
    double c[] = { 0, 1, 2, 3, 4, 5, 6, 7 };
    auto t3 = tensor_view( c, tuple( 2, 2, 2 ) );
    CHECK( t3( 1, 0, 1 ).value() == 5 );
    CHECK( t3( 0, 1, 1 ).value() == 3 );

    // offset : sous-vue à partir de l'index 1 sur l'axe 0
    double v[] = { 10, 20, 30, 40 };
    auto o = tensor_view( v, tuple( 4 ) ).offset( 1 ); // [20,30,40]
    CHECK( o.shape( 0_c ) == 3 );
    CHECK( o( 0 ).value() == 20 );
    CHECK( o( 2 ).value() == 40 );
}

TEST_CASE( "TensorView — fill_with / for_each_scalar", "" ) {
    double d[ 6 ] = {};
    auto t = tensor_view( d, tuple( 2, 3 ) );

    CHECK_REPR( t.indices_col_ordering( 0 ), tuple( 0, 0 ) );
    CHECK_REPR( t.indices_col_ordering( 1 ), tuple( 0, 1 ) );
    CHECK_REPR( t.indices_col_ordering( 2 ), tuple( 0, 2 ) );
    CHECK_REPR( t.indices_col_ordering( 3 ), tuple( 1, 0 ) );
    CHECK_REPR( t.indices_col_ordering( 4 ), tuple( 1, 1 ) );

    t.fill_with( 7 );
    for ( double x : d )
        CHECK( x == 7 );

    double s = 0;
    t.for_each_scalar( [&]( auto e ) { s += e.value(); } );
    CHECK( s == 42 );

    // fill d'une sous-vue (ligne 1 seulement)
    double d2[ 6 ] = {};
    tensor_view( d2, tuple( 2, 3 ) ).row( 1 ).fill_with( 5 );
    CHECK( d2[ 0 ] == 0 );
    CHECK( d2[ 3 ] == 5 );
    CHECK( d2[ 5 ] == 5 );
}

TEST_CASE( "TensorView — fill_with avec contextes (run_parallel)", "" ) {
    auto ql = tuple( CpuQueue() );

    // handle ignoré -> RAII : wait à la destruction du temporaire (synchrone)
    double d[ 6 ] = {};
    auto t = tensor_view( d, tuple( 2, 3 ) );
    t.fill_with( ql, 9 );
    for ( double x : d )
        CHECK( x == 9 );

    // handle gardé -> asynchrone, puis wait() explicite (consomme l'event)
    double e[ 4 ] = {};
    auto u = tensor_view( e, tuple( 4 ) );
    {
        auto h = u.fill_with( ql, 3 );
        h.wait();
    }
    for ( double x : e )
        CHECK( x == 3 );
}

TEST_CASE( "TensorView — fill_with avec contextes (run_parallel) et shapes éclatées", "" ) {
    auto ql = tuple( CpuQueue() );

    double d[ 4*4 ] = {};
    for( PI i = 0; i < 4*4; ++i )
        d[ i ] = i;

    auto t = tensor_view( d, tuple( 2, 4 ), tuple( row_, col_ ), tuple( sizeof( double ) * 2, sizeof( double ) * 4 ) );
    CHECK_REPR( t( 1, 1 ), 6 );
    t.fill_with( ql, 100 );

    for( PI i = 0; i < 4*4; i += 2 ) {
        CHECK_REPR( d[ i + 0 ], 100 );
        CHECK_REPR( d[ i + 1 ], i + 1 );
    }
}

// // Inner body of the nested run_parallel: writes one element of a row.
// struct WriteElem {
//     void operator()( auto j, auto row, double s ) const { row.row( j ) = s; }
// };

// // Outer body: for each row, spawn a *nested* run_parallel over that row's columns.
// // `a` arrives tagged has_already_been_parallelized (added on the pool path), and the
// // tag propagates through a.row(i) (squeeze transform), so the nested run_parallel runs
// // inline on the current worker — no pool round-trip, hence no self-wait deadlock.
// struct FillRows {
//     void operator()( auto i, auto a, double s ) const {
//         auto r = a.row( i );
//         static_assert( DECAYED_TYPE_OF( r )::template has_tag<container_tags::has_already_been_parallelized> );
//         run_parallel( range( r.size() ), WriteElem{}, r, s + double( i ) );
//     }
// };

// // Operation under test. A functor (not a generic lambda): nvcc forbids generic
// // extended __host__ __device__ lambdas, and the matrix runs this on the device too.
// struct PlusEq {
//     HD void operator()( auto a, auto b ) const { a += b; }
// };

// // Element-wise `a += b` over the full matrix: operand-A memory space × operand-B
// // memory space × shape × execution context. Exercises cross-space transfer
// // (make_accessible) and the nested/inline dispatch, all checked against a + b.
// TEST_CASE( "operator+= — element-wise, matrix over memory spaces and contexts", "" ) {
//     sdot_test::check_binary_op_matrix(
//         PlusEq{},
//         [] ( double a, double b ) { return a + b; } );
// }

// TEST_CASE( "run_parallel — nested run_parallel runs inline (no deadlock)", "" ) {
//     double da[ 4 ] = { 0, 0, 0, 0 };
//     auto shape = tuple( 2, 2 );
//     TensorView a( da, shape, contiguous_strides<double>( shape ), MemorySpace_CpuRam{} );

//     // top-level operand a is not tagged -> pool path (and it gets tagged for the body)
//     static_assert( ! DECAYED_TYPE_OF( a )::template has_tag<container_tags::has_already_been_parallelized> );
//     run_parallel( range( shape[ 0_c ] ), FillRows{}, a, 100.0 );

//     CHECK( da[ 0 ] == 100 ); // row 0 -> 100 + 0
//     CHECK( da[ 1 ] == 100 );
//     CHECK( da[ 2 ] == 101 ); // row 1 -> 100 + 1
//     CHECK( da[ 3 ] == 101 );
// }

// #ifdef __CUDACC__
// TEST_CASE( "GPU tensor", "" ) {
//     double data[ 4 ] = { 1, 2, 3, 4 };
//     MemorySpace_GlobalCudaRam gpu_ram;
//     gpu_ram.with_reservation<double>( 4, [&]( auto dev ) {
//         copy( dev, Ptr<double,MemorySpace_CpuRam>( data ), 4 );
//         cudaStreamSynchronize( ExecutionContext_Cuda{}.stream );

//         TensorView dt( dev.raw, tuple( 4 ), tuple( sizeof( double ) ), dev.memory_space );
//         // run_parallel( shape.all_indices(), Doubler{}, dt ); // GlobalCudaRam -> dispatch picks CUDA
//         // cudaStreamSynchronize( ExecutionContext_Cuda{}.stream );

//         // copy( ExecutionContext_Cpu{}, Ptr<double,MemorySpace_CpuRam>( back ), dev, 4 );
//         info( dt[ 0 ].value() );
//     } );

//     // TensorView t( data, tuple( 4 ), tuple( sizeof( double ) ), MemorySpace_CpuRam{} );
//     // info( t );
// }
// #endif // __CUDACC__

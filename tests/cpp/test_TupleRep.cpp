// #include "../../src/cpp/sdot/support/containers/CartesianProduct.h"
// #include "../../src/cpp/sdot/support/containers/Range.h"
#include <sdot/support/containers/TupleRep.h>
#include "main.h"
#include <array>

using namespace sdot;

auto array( auto...values ) {
    return std::array{ values... };
}

TEST_CASE( "AxisValues", "" ) {
    auto t = tuple_rep( 1, 2 );
    info( t[ 0 ] );
    info( t[ 1 ] );
    info( t[ 2 ] );
}

#include <sdot/support/containers/Vector.h>
#include "main.h"

using namespace sdot;

auto array( auto...values ) {
    return std::array{ values... };
}

TEST_CASE( "Vector", "" ) {
    SECTION( "constant" ) {
        Vector<double,3> v( Values(), 1, 2, 3 );
        CHECK_REPR( v, array( 1, 2, 3 ) );
        CHECK_REPR( v.with_pushed_value( 17 ), array( 1, 2, 3, 17 ) );
        CHECK_REPR( v.without_index( 1 ), array( 1, 3 ) );
        CHECK_REPR( v.size(), 3_c );
    }
}

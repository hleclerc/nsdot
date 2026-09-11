// LE BSP EMPAQUETE : l'arbre entier dans UNE arene de `TF`, en-tetes et germes melanges
// en preordre. Ce qu'il teste : est-ce la FORME de l'arbre qui coute, ou la dispersion de ses
// lectures ?
//
//   xmake run pd_packed --help

#include "spatial_accel/AaBspPacked.h"
#include "bench/Bench.h"
#include <cstdio>
#include <string>

using namespace pd;
using namespace pd::bench;

int main( int argc, char **argv ) {
    Args a;
    bool stats = false;
    for ( int i = 1; i < argc; ++i ) {
        const std::string s = argv[ i ];
        auto val = [ & ]() { return i + 1 < argc ? argv[ ++i ] : ""; };
        (void) val;
        if ( parse_commun( a, s, i, argc, argv ) ) continue;
        if ( s == "--stats" ) stats = true;
        else {
            std::printf( "usage: pd_packed [options]\n" );
            usage_commun();
            std::printf( "  --stats         compte boites/coupes par cellule au lieu de chronometrer\n"
                       );
            return s == "--help" || s == "-h" ? 0 : 1;
        }
    }
    finalise( a );

    if ( stats ) {
        std::printf( "=== 2D  ce que le parcours fait par cellule\n" );
        for ( const Cloud<2> &cl : suite_2d( a ) ) {
            if ( cl.absent ) continue;
            std::printf( "  %s\n", cl.nom.c_str() );
            AaBspPacked tr; tr.build( cl.P, cl.W, cl.n, a.leaf );
            stats_une<AaBspPacked, CellSoAT<64>, true>( tr );
        }
        return 0;
    }

    return banc<AaBspPacked, Absent>( a );
}

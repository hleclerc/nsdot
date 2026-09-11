// LA SUR-CELLULE : chaque paquet porte un k-DOP qui contient deja toutes les cellules de ses
// germes, et la cellule part de la plutot que du domaine.
//
//   xmake run pd_hull --help

#include "spatial_accel/AaBspHull.h"
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
        else if ( s == "--hull-rate" ) hull_rate = std::atoi( val() );
        else if ( s == "--no-hull-init" ) hull_init = false;
        else {
            std::printf( "usage: pd_hull [options]\n" );
            usage_commun();
            std::printf( "  --stats         compte boites/coupes par cellule au lieu de chronometrer\n"
                         "  --hull-rate R   germes par paquet                            (12)\n"
                         "  --no-hull-init  partir du domaine et non de la sur-cellule\n"
                       );
            return s == "--help" || s == "-h" ? 0 : 1;
        }
    }
    finalise( a );
    hull_threads = a.threads;

    if ( stats ) {
        std::printf( "=== 2D  ce que le parcours fait par cellule\n" );
        for ( const Cloud<2> &cl : suite_2d( a ) ) {
            if ( cl.absent ) continue;
            std::printf( "  %s\n", cl.nom.c_str() );
            AaBspHull tr; tr.build( cl.P, cl.W, cl.n, a.leaf );
            stats_une<AaBspHull, CellSoAT<64>, true>( tr );
        }
        return 0;
    }

    return banc<AaBspHull, Absent>( a );
}

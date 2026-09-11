// LE BSP NON ALIGNE : coupes obliques choisies par SAH, boites toujours alignees.
// Resultat mesure : perdant, et pour une raison structurelle -- une coupe oblique donne deux
// boites alignees qui SE RECOUVRENT. Voir README, section « LES BSP NON ALIGNES ».
//
//   xmake run pd_obsp --help

#include "spatial_accel/ObBsp.h"
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
            std::printf( "usage: pd_obsp [options]\n" );
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
            ObBsp tr; tr.build( cl.P, cl.W, cl.n, a.leaf );
            stats_une<ObBsp, CellSoAT<64>, true>( tr );
        }
        return 0;
    }

    return banc<ObBsp, Absent>( a );
}

// LE BSP A QUATRE FILS : on separe comme avant, mais on sort directement quatre enfants.
// Resultat mesure : perdant. Voir README, section « LE BSP A QUATRE FILS ».
//
//   xmake run pd_bsp4 --help

#include "spatial_accel/AaBsp4.h"
#include "bench/Bench.h"
#include <cstdio>
#include <string>

using namespace pd;
using namespace pd::bench;

int main( int argc, char **argv ) {
    Args a;
    bool stats = false, lazy = false;
    for ( int i = 1; i < argc; ++i ) {
        const std::string s = argv[ i ];
        auto val = [ & ]() { return i + 1 < argc ? argv[ ++i ] : ""; };
        (void) val;
        if ( parse_commun( a, s, i, argc, argv ) ) continue;
        if ( s == "--stats" ) stats = true;
        else if ( s == "--lazy" ) lazy = true;
        else {
            std::printf( "usage: pd_bsp4 [options]\n" );
            usage_commun();
            std::printf( "  --stats         compte boites/coupes par cellule au lieu de chronometrer\n"
                         "  --lazy          chaque fils teste a SA sortie de pile (bsp4l)\n"
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
            if ( lazy ) { AaBsp4L tr; tr.build( cl.P, cl.W, cl.n, a.leaf );
                          stats_une<AaBsp4L, CellSoAT<64>, true>( tr ); }
            else        { AaBsp4  tr; tr.build( cl.P, cl.W, cl.n, a.leaf );
                          stats_une<AaBsp4,  CellSoAT<64>, true>( tr ); }
        }
        return 0;
    }

    // « bsp4 » teste les QUATRE boites chez le pere ; « bsp4l » teste chaque fils a SA sortie de
    // pile, donc contre une cellule deja retrecie par les fils precedents. Le second fait moins de
    // travail et coute plus cher : c'est le resultat, et c'est pour ca que les deux sont ici.
    return lazy ? banc<AaBsp4L, Absent>( a ) : banc<AaBsp4, Absent>( a );
}

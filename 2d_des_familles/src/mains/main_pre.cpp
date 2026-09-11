// LA PRE-PASSE : couper d'abord contre un germe sur `--pre-rate`, le reste ensuite. Ce qu'elle
// teste : est-ce que retrecir la cellule TOT fait gagner plus que la seconde descente ne coute ?
//
//   xmake run pd_pre --help

#include "spatial_accel/AaBspPre.h"
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
        else if ( s == "--pre-rate" ) pre_rate = std::atoi( val() );
        else if ( s == "--pre-overlap" ) pre_overlap = true;
        else {
            std::printf( "usage: pd_pre [options]\n" );
            usage_commun();
            std::printf( "  --stats         compte boites/coupes par cellule au lieu de chronometrer\n"
                         "  --pre-rate R    couper d abord contre un germe sur R          (16)\n"
                         "  --pre-overlap   recouvrir les deux arbres : chaque germe de S coupe DEUX\n"
                         "                  fois avec le meme plan -- le cas degenere, expres\n"
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
            AaBspPre tr; tr.build( cl.P, cl.W, cl.n, a.leaf );
            stats_une<AaBspPre, CellSoAT<64>, true>( tr );
        }
        return 0;
    }

    return banc<AaBspPre, Absent>( a );
}

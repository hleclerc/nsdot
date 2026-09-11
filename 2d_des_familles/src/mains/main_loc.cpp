// LE PAQUET LOCAL D'ABORD : couper contre les germes du noeud de Bsp qui porte le germe, puis
// descendre l'arbre entier en SAUTANT ce noeud. Ce qu'il teste : est-ce qu'une cellule deja
// retrecie par son voisinage immediat fait mordre le test d'eviction assez tot pour payer les
// coupes du paquet qu'on s'impose ?
//
//   xmake run pd_loc --help

#include "spatial_accel/AaBspLoc.h"
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
        if ( parse_commun( a, s, i, argc, argv ) ) continue;
        if ( s == "--stats" ) stats = true;
        else if ( s == "--rate" ) loc_rate = std::atoi( val() );
        else if ( s == "--phase" ) loc_phase = std::atoi( val() );
        else {
            std::printf( "usage: pd_loc [options]\n" );
            usage_commun();
            std::printf( "  --stats         compte boites/coupes par cellule au lieu de chronometrer\n"
                         "  --rate R        germes vises par paquet                        (64)\n"
                         "                  R = --leaf : le paquet EST la feuille, donc `bsp`\n"
                         "  --phase P       SONDE : 0 les deux phases, 1 le paquet seul (volumes\n"
                         "                  FAUX), 2 les deux phases puis une CERTIFICATION du\n"
                         "                  resultat par un parcours qui ne coupe pas.     (0)\n"
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
            AaBspLoc tr; tr.build( cl.P, cl.W, cl.n, a.leaf );
            stats_une<AaBspLoc, CellSoAT<64>, true>( tr );
        }
        std::printf( "=== 3D  ce que le parcours fait par cellule\n" );
        for ( const Cloud<3> &cl : suite_3d( a ) ) {
            if ( cl.absent ) continue;
            std::printf( "  %s\n", cl.nom.c_str() );
            AaBspLoc3 tr; tr.build( cl.P, cl.W, cl.n, a.leaf );
            stats_une<AaBspLoc3, Cell3T<128>, true>( tr );
        }
        return 0;
    }

    return banc<AaBspLoc, AaBspLoc3>( a );
}

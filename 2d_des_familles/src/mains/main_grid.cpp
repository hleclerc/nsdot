// LA GRILLE REGULIERE -- l'accelerateur sans arbre : le germe trouve sa case par `D` divisions, et
// s'etale par anneaux.
//
// Ce que ce banc doit montrer, et pourquoi il existe : la grille GAGNE sur l'uniforme (pas de
// descente racine -> feuille) et PERD lourdement des que la densite varie. La 3D est la ou l'ecart
// se creuse le plus vite -- un anneau de trop y coute `24 r^2` cases au lieu de `8 r`.
//
//   xmake run pd_grid --help

#include "spatial_accel/Grid.h"
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
        if ( parse_commun( a, s, i, argc, argv ) )
            continue;
        if ( s == "--stats" ) stats = true;
        else {
            std::printf( "usage: pd_grid [options]\n" );
            usage_commun();
            std::printf( "  --leaf L        ici : germes VISES par case (la resolution en decoule)\n"
                         "  --stats         compte cases/coupes par cellule au lieu de chronometrer\n" );
            return s == "--help" || s == "-h" ? 0 : 1;
        }
    }
    finalise( a );

    if ( stats ) {
        if ( a.dims != 3 ) {
            std::printf( "=== 2D  ce que le parcours fait par cellule\n" );
            for ( const Cloud<2> &cl : suite_2d( a ) ) {
                if ( cl.absent ) continue;
                std::printf( "  %s\n", cl.nom.c_str() );
                Grid gr; gr.build( cl.P, cl.W, cl.n, a.leaf );
                // `CellBox = true`, comme le chemin CHRONOMETRE. L'ancien banc comptait la grille
                // avec `false` et tous les autres avec `true`, ce qui rendait « balayees »
                // incomparable d'une ligne a l'autre -- et masquait que le test en `O( 1 )` fait
                // tomber 4546 boites a 719 sur le cas dur.
                stats_une<Grid, CellSoAT<64>, true>( gr );
            }
        }
        if ( a.dims != 2 ) {
            std::printf( "=== 3D  ce que le parcours fait par cellule\n" );
            for ( const Cloud<3> &cl : suite_3d( a ) ) {
                if ( cl.absent ) continue;
                std::printf( "  %s\n", cl.nom.c_str() );
                Grid3 gr; gr.build( cl.P, cl.W, cl.n, a.leaf );
                stats_une<Grid3, Cell3T<128>, false>( gr );
            }
        }
        return 0;
    }

    return banc<Grid, Grid3>( a );
}

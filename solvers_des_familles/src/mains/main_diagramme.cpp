// =====================================================================================
// UN DIAGRAMME CHRONOMETRE, sur la suite : la ligne de base que Newton multiplie par le nombre de
// diagrammes qu'il demande. Arbre a part, tour de chauffe hors chrono, minimum des repetitions.
// Le seul controle est la somme des mesures, et il suffit : une cellule fausse d'un cote et fausse
// a l'envers de l'autre est ce qu'il attrape.
//
//   xmake run diagramme
//   xmake run diagramme --kernel float --threads 8
// =====================================================================================

#include "bench/Dispatch.h"
#include <cmath>
#include <cstdio>
#include <string>

using namespace sf;

namespace {

template<class PD>
int mesure( const Args &a, const Nuage<PD::dim> &nu ) {
    PD pd;
    double t0 = now();
    pd.build( nu.P, nu.W, nu.n, a.leaf );
    const double t_arbre = now() - t0;

    std::vector<TF> res;
    SI deb = pd.measures( res, a.par );                  // chauffe
    double t = 1e300;
    for ( int r = 0; r < a.reps; ++r ) {
        t0 = now();
        deb = pd.measures( res, a.par );
        t = std::min( t, now() - t0 );
    }
    TF somme = 0;
    for ( TF v : res ) somme += v;
    const bool ok = std::fabs( somme - 1 ) < 1e-6 && deb == 0;
    std::printf( "  %-24s n=%-7d %-8s : %8.3f s ( %7.0f ns/germe )  arbre %6.0f ms  somme %.9f%s\n",
                 nu.nom.c_str(), int( nu.n ), nu.W ? "Laguerre" : "Voronoi", t, t / nu.n * 1e9,
                 t_arbre * 1e3, double( somme ), ok ? "" : "   <-- FAUX" );
    if ( deb )
        std::printf( "      ATTENTION : %d cellules ont DEBORDE %d sommets -- relancer avec --maxnv %d.\n",
                     int( deb ), PD::max_nv, 2 * PD::max_nv );
    return ! ok;
}

template<int D>
int deroule( const Args &a ) {
    int bad = 0;
    std::printf( "=== %dD  threads=%d kernel=%s maxnv=%d leaf=%d\n",
                 D, a.par.threads, a.kernel.c_str(), a.nv( D ), int( a.leaf ) );
    for ( const Nuage<D> &nu : a.nuages<D>() ) {
        if ( nu.absent ) {
            std::printf( "  %-28s : ABSENT ( --cases DIR )\n", nu.nom.c_str() );
            continue;
        }
        bad += dispatch<D>( a, [ & ]( auto tag ) {
            return mesure<typename decltype( tag )::type>( a, nu );
        } );
    }
    return bad;
}

} // namespace

int main( int argc, char **argv ) {
    Args a;
    for ( int i = 1; i < argc; ++i ) {
        const std::string s = argv[ i ];
        if ( a.parse( s, i, argc, argv ) ) continue;
        std::printf( "usage: diagramme [options]\n" );
        Args::usage();
        return s == "--help" || s == "-h" ? 0 : 1;
    }
    a.finalise();

    int bad = 0;
    if ( a.dims != 3 ) bad += deroule<2>( a );
    if ( a.dims != 2 ) bad += deroule<3>( a );
    return bad ? 1 : 0;
}

// =====================================================================================
// LE BANC : les mesures des cellules sur GPU, contre le moteur CPU sur le MEME arbre. Pour chaque
// nuage de la suite ( 2D : uniforme, lignes / Voronoi, lignes / aires egales ; 3D : uniforme,
// plans / Voronoi, plans / volumes egaux ) :
//   * le CPU bati l'arbre et mesure ( chauffe, minimum des repetitions ) -- c'est le TEMOIN ;
//   * le GPU recoit l'arbre et mesure, variante par variante ( chauffe, minimum ) ;
//   * on compare cellule par cellule : l'ecart maximal relatif a la cellule moyenne, la somme.
// Le noyau ( `--kernel float | double` ) est le meme des deux cotes.
//
//   xmake run mesures --threads 8
//   xmake run mesures --threads 8 --kernel float --2d
//   xmake run mesures --threads 8 --variante voies --load uniforme -n 1000000
// =====================================================================================

#include "bench/Dispatch.h"
#include "gpu/Mesures.h"
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <string>

using namespace sf;

namespace {

struct Opt {
    std::string variante = "toutes";
    int         reps_gpu = 10;
};

template<class PD>
int mesure( const Args &a, const Opt &o, const Nuage<PD::dim> &nu ) {
    constexpr int D = PD::dim;
    using TK = typename PD::TKernel;

    PD pd;
    double t0 = now();
    pd.build( nu.P, nu.W, nu.n, a.leaf );
    const double t_arbre = now() - t0;

    // ---- le temoin
    std::vector<TF> cpu;
    SI deb_cpu = pd.measures( cpu, a.par );
    double t_cpu = 1e300;
    for ( int r = 0; r < a.reps; ++r ) {
        t0 = now();
        deb_cpu = pd.measures( cpu, a.par );
        t_cpu = std::min( t_cpu, now() - t0 );
    }
    double somme_cpu = 0;
    for ( TF v : cpu ) somme_cpu += v;
    std::printf( "  %-24s n=%-7d %-8s  arbre %5.0f ms\n", nu.nom.c_str(), int( nu.n ), nu.W ? "Laguerre" : "Voronoi", t_arbre * 1e3 );
    std::printf( "      %-8s %8.4f s  %7.0f ns/germe            somme %.9f%s\n", "CPU", t_cpu, t_cpu / nu.n * 1e9, somme_cpu,
                 deb_cpu ? "   <-- DEBORDE" : "" );

    // ---- le GPU
    gpu::DiagrammeGpu<D,TK> g( pd.arbre );
    std::vector<SI> rang_de( nu.n );                     // identifiant -> rang dans l'arbre
    for ( SI k = 0; k < nu.n; ++k ) rang_de[ pd.ids[ k ] ] = k;
    std::vector<double> res;
    int bad = 0;
    const double moyenne = somme_cpu / nu.n;
    for ( int iv = 0; iv < int( gpu::Variante::NB ); ++iv ) {
        const gpu::Variante v = gpu::Variante( iv );
        if ( o.variante != "toutes" && o.variante != gpu::nom( v ) ) continue;
        if ( D == 3 && v != gpu::Variante::FIL && v != gpu::Variante::VOIES ) continue;   // en 3D « voies » est le warp
        const gpu::Chrono ch = g.mesures( v, a.nv( D ), o.reps_gpu, res );
        // l'ecart d'une cellule est rapporte a ELLE ( ou a la moyenne si elle est plus petite ). En
        // `float` c'est du bruit : une cellule de cote 1e-3 avec des sommets a 6e-8 pres a son aire
        // a 4e-4 pres, et les deux cotes n'arrondissent pas pareil ( `fma` contractes ou non ) ;
        // la somme, elle, est tenue a 1e-6. En `double` l'ecart est celui de l'ordre des operations
        double somme = 0, ecart = 0;
        int non_ecrites = 0;
        for ( SI i = 0; i < nu.n; ++i ) {
            if ( std::isnan( res[ i ] ) ) { if ( non_ecrites++ < 5 ) std::printf( "        cellule %d ( rang %d ) non ecrite\n", int( i ), int( rang_de[ i ] ) ); continue; }
            somme += res[ i ];
            ecart = std::max( ecart, std::fabs( res[ i ] - cpu[ i ] ) / std::max( double( cpu[ i ] ), moyenne ) );
        }
        if ( non_ecrites ) std::printf( "        %d cellules NON ECRITES\n", non_ecrites );
        if ( std::getenv( "MESURES_DEBUG" ) ) {
            int nz = 0, faux = 0;
            for ( SI i = 0; i < nu.n; ++i ) {
                if ( std::isnan( res[ i ] ) || res[ i ] == 0 ) { ++nz; continue; }
                if ( std::fabs( res[ i ] - cpu[ i ] ) > 1e-6 * std::max( double( cpu[ i ] ), moyenne ) ) { if ( faux++ < 5 ) std::printf( "        cellule %d rang %d : gpu %.6e cpu %.6e\n", int( i ), int( rang_de[ i ] ), res[ i ], double( cpu[ i ] ) ); }
            }
            std::printf( "        debug : %d nulles ou NaN, %d fausses parmi les autres\n", nz, faux );
        }
        const bool ok = non_ecrites == 0 && ch.deborde == 0 && ecart < ( sizeof( TK ) == 4 ? 1e-2 : 1e-9 ) && std::fabs( somme - 1 ) < 1e-6 + 1e-4 * std::fabs( somme_cpu - 1 );
        std::printf( "      %-8s %8.4f s  %7.1f ns/germe  x%-5.1f  somme %.9f  ecart max %.1e  retour %.0f ms%s%s\n",
                     gpu::nom( v ), ch.noyau, ch.noyau / nu.n * 1e9, t_cpu / ch.noyau, somme, ecart, ch.retour * 1e3,
                     ch.deborde ? "   <-- DEBORDE" : "", ok ? "" : "   <-- FAUX" );
        if ( ch.deborde )
            std::printf( "        %d cellules ont deborde ( --maxnv %d )\n", ch.deborde, 2 * a.nv( D ) );
        if ( ch.stats[ 0 ] )
            std::printf( "        par cellule : %.1f plans / coupes tentees, %.1f coupes effectives, %.3f excursions, %.1f boites testees\n",
                         double( ch.stats[ 0 ] ) / nu.n, double( ch.stats[ 1 ] ) / nu.n, double( ch.stats[ 2 ] ) / nu.n, double( ch.stats[ 3 ] ) / nu.n );
        bad += ! ok;
    }
    std::printf( "      televersement %.0f ms\n", g.televersement() * 1e3 );
    return bad;
}

template<int D>
int deroule( const Args &a, const Opt &o ) {
    int bad = 0;
    std::printf( "=== %dD  threads=%d kernel=%s maxnv=%d leaf=%d\n", D, a.par.threads, a.kernel.c_str(), a.nv( D ), int( a.leaf ) );
    for ( const Nuage<D> &nu : a.nuages<D>() ) {
        if ( nu.absent ) {
            std::printf( "  %-28s : ABSENT ( --cases DIR )\n", nu.nom.c_str() );
            continue;
        }
        bad += dispatch<D>( a, [ & ]( auto tag ) { return mesure<typename decltype( tag )::type>( a, o, nu ); } );
    }
    return bad;
}

} // namespace

int main( int argc, char **argv ) {
    Args a;
    Opt o;
    for ( int i = 1; i < argc; ++i ) {
        const std::string s = argv[ i ];
        if ( a.parse( s, i, argc, argv ) ) continue;
        if ( s == "--variante" && i + 1 < argc ) { o.variante = argv[ ++i ]; continue; }
        if ( s == "--reps-gpu" && i + 1 < argc ) { o.reps_gpu = std::atoi( argv[ ++i ] ); continue; }
        std::printf( "usage: mesures [options]\n" );
        Args::usage();
        std::printf( "  --variante V    fil | filreg | filregc | filmix{4,6,8,12,16} | filbrk{6,8,10,12,16} | filbrk8nu | filrot{6,8} | filnrm8 | voies | voies16 | voies32 | paquet{8,32}x{1,2,4}[S] | toutes (toutes)\n"
                     "  --reps-gpu R    repetitions du noyau GPU, minimum       (10)\n" );
        return s == "--help" || s == "-h" ? 0 : 1;
    }
    a.finalise();
    std::printf( "GPU : %s\n", gpu::carte().c_str() );

    int bad = 0;
    if ( a.dims != 3 ) bad += deroule<2>( a, o );
    if ( a.dims != 2 ) bad += deroule<3>( a, o );
    return bad ? 1 : 0;
}

// =====================================================================================
// LA CHAINE 2D SUR GPU -- le squelette. Pour l'instant : l'arbre ( CPU ), le televersement, puis
// LES MESURES ET LES FACETTES en un seul noyau, comparees au moteur CPU poste par poste.
//
// Les facettes sont le maillon qui manquait entre « mesurer des cellules » et « faire un Newton » :
// la hessienne du transport est le laplacien du graphe de Laguerre, `c_ij = |facette ij| /
// ( 2 |p_i - p_j| )` ( `solver/Laplacien.h` ), et ses coefficients sont exactement les aretes de
// chaque cellule -- que le noyau porte deja, chacune avec le `cid` de son voisin.
//
// Ce qui reste a porter, dans l'ordre : l'assemblage CSR ( comptage + somme prefixe, deux passes
// GPU ), le gradient conjugue preconditionne, la boucle de Newton avec sa recherche lineaire,
// puis les densites ( l'image : intersecter la cellule avec la grille de pixels ).
//
//   ./build/linux/x86_64/release/chaine --threads 8 --kernel float --load uniforme -n 1000000
// =====================================================================================

#include "bench/Dispatch.h"
#include "solver/Laplacien.h"
#include <cuda_runtime.h>
#include "gpu/Mesures.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace sf;

namespace {

/// une facette telle que le banc la compare : ( i, j ) tries, et le coefficient
struct Fa { int i, j; double c; };
bool avant( const Fa &a, const Fa &b ) { return a.i != b.i ? a.i < b.i : a.j < b.j; }

template<class PD>
int chaine( const Args &a, const Nuage<PD::dim> &nu, int reps_gpu, bool arbre_gpu, int iterations ) {
    constexpr int D = PD::dim;
    using TK = typename PD::TKernel;
    static_assert( D == 2, "la chaine est 2D" );

    PD pd;
    double t0 = now();
    pd.build( nu.P, nu.W, nu.n, a.leaf );
    const double t_arbre = now() - t0;

    // ---- LE TEMOIN : mesures et facettes par le moteur CPU. Le callback est appele DEPUIS LES
    //      FILS : une liste par fil, comme `solver/Newton.h`.
    std::vector<TF> cpu;
    std::vector<std::vector<Fa>> par_fil( std::max( a.par.threads, 1 ) );
    t0 = now();
    pd.measures_and_facets( cpu, a.par, [ & ]( int t, SI i, SI j, TF mes ) {
        double d2 = 0;
        for ( int d = 0; d < D; ++d ) { const double e = nu.P[ d ][ j ] - nu.P[ d ][ i ]; d2 += e * e; }
        if ( d2 > 0 ) par_fil[ t ].push_back( Fa{ int( i ), int( j ), mes / ( 2 * std::sqrt( d2 ) ) } );
    } );
    const double t_cpu = now() - t0;
    std::vector<Fa> fcpu;
    for ( const auto &v : par_fil ) fcpu.insert( fcpu.end(), v.begin(), v.end() );

    std::printf( "  %-24s n=%-8d %-8s  arbre %6.0f ms ( CPU, %d fils )\n",
                 nu.nom.c_str(), int( nu.n ), nu.W ? "Laguerre" : "Voronoi", t_arbre * 1e3, a.par.threads );
    std::printf( "      CPU       mesures + facettes %7.3f s   %6.0f ns/germe   %zu facettes\n",
                 t_cpu, t_cpu / nu.n * 1e9, fcpu.size() );

    // ---- LE GPU : le meme, en un noyau
    // ---- L'ARBRE : celui du CPU, ou bien construit SUR LE GPU et qui n'en redescend pas
    double ms_gpu = 0, t_arbre_gpu = 0;
    const double tg0 = now();
    gpu::DiagrammeGpu<D,TK> g = arbre_gpu
        ? gpu::DiagrammeGpu<D,TK>( nu.P, nu.W, int( nu.n ), int( a.leaf ), &ms_gpu, true )
        : gpu::DiagrammeGpu<D,TK>( pd.arbre );
    t_arbre_gpu = now() - tg0;
    if ( arbre_gpu )
        std::printf( "      arbre GPU %6.0f ms ( noyaux %6.1f ms ) contre %6.0f ms au CPU sur %d fils   x%.1f   %d noeuds contre %d\n",
                     t_arbre_gpu * 1e3, ms_gpu, t_arbre * 1e3, a.par.threads, t_arbre / t_arbre_gpu,
                     g.nb_noeuds(), int( pd.arbre.nodes.size() ) );
    std::vector<double> res, fl;
    std::vector<int> fj;
    const gpu::Chrono ch = g.facettes( reps_gpu, res, fj, fl );

    std::vector<Fa> fgpu;
    long long vides = 0, cotes = 0;
    fgpu.reserve( fcpu.size() );
    for ( SI i = 0; i < nu.n; ++i )
        for ( int s = 0; s < gpu::DiagrammeGpu<D,TK>::NF; ++s ) {
            const int j = fj[ size_t( s ) * nu.n + i ];
            if ( j == -1000000 ) { ++vides; continue; }
            if ( j < 0 ) { ++cotes; continue; }
            double d2 = 0;
            for ( int d = 0; d < D; ++d ) { const double e = nu.P[ d ][ j ] - nu.P[ d ][ i ]; d2 += e * e; }
            if ( d2 > 0 ) fgpu.push_back( Fa{ int( i ), j, fl[ size_t( s ) * nu.n + i ] / ( 2 * std::sqrt( d2 ) ) } );
        }

    std::printf( "      GPU       mesures + facettes %7.3f s   %6.1f ns/germe   %zu facettes   x%.1f\n",
                 ch.noyau, ch.noyau / nu.n * 1e9, fgpu.size(), t_cpu / ch.noyau );
    std::printf( "        %d registres, %d blocs par SM, occupation %.0f %%   ( descente %.0f ms, televersement %.0f ms )\n",
                 ch.regs, ch.blocs, ch.occup * 100, ch.retour * 1e3, g.televersement() * 1e3 );

    // ---- LA COMPARAISON, facette par facette
    std::sort( fcpu.begin(), fcpu.end(), avant );
    std::sort( fgpu.begin(), fgpu.end(), avant );
    double somme = 0, ecart_m = 0, moyenne = 0;
    for ( TF v : cpu ) moyenne += v;
    moyenne /= nu.n;
    for ( SI i = 0; i < nu.n; ++i ) {
        somme += res[ i ];
        ecart_m = std::max( ecart_m, std::fabs( res[ i ] - cpu[ i ] ) / std::max( double( cpu[ i ] ), moyenne ) );
    }
    if ( std::getenv( "CHAINE_DEBUG" ) ) {
        const size_t uc = size_t( std::unique( fcpu.begin(), fcpu.end(), []( const Fa &x, const Fa &y ){ return x.i == y.i && x.j == y.j; } ) - fcpu.begin() );
        std::vector<Fa> c2( fcpu );                       // `unique` a deplace : on recharge
        std::printf( "        debug : %zu facettes CPU dont %zu paires ( i, j ) distinctes ; GPU %lld cotes de boite, %lld cases vides ( soit %.2f aretes par cellule )\n", fcpu.size(), uc, cotes, vides, double( gpu::DiagrammeGpu<D,TK>::NF * nu.n - vides ) / nu.n );
        fcpu = c2;
    }
    int manque = 0, en_trop = 0;
    double ecart_c = 0, cmax = 0, csom = 0, ctot = 0, csom2 = 0;
    std::vector<double> ecarts;
    ecarts.reserve( fcpu.size() );
    for ( const Fa &f : fcpu ) ctot += f.c;
    size_t p = 0, q = 0;
    while ( p < fcpu.size() && q < fgpu.size() ) {
        if ( avant( fcpu[ p ], fgpu[ q ] ) ) { ++manque; cmax = std::max( cmax, fcpu[ p ].c ); csom += fcpu[ p ].c; ++p; continue; }
        if ( avant( fgpu[ q ], fcpu[ p ] ) ) { ++en_trop; csom2 += fgpu[ q ].c; ++q; continue; }
        // l'ecart d'un `c_ij` est rapporte a LA MOYENNE, pas a lui-meme : une arete quasi nulle a
        // une erreur relative enorme et un poids nul dans la hessienne, la juger sur elle-meme ne
        // dit rien. On garde quand meme la liste pour les quantiles.
        ecarts.push_back( std::fabs( fcpu[ p ].c - fgpu[ q ].c ) );
        ecart_c = std::max( ecart_c, ecarts.back() );
        ++p; ++q;
    }
    manque += int( fcpu.size() - p );
    en_trop += int( fgpu.size() - q );
    if ( manque && std::getenv( "CHAINE_DEBUG" ) ) {
        // la premiere cellule qui manque une facette, vidangee des deux cotes
        int cible = -1;
        { size_t pp = 0, qq = 0;
          while ( pp < fcpu.size() && qq < fgpu.size() && cible < 0 ) {
              if ( avant( fcpu[ pp ], fgpu[ qq ] ) ) cible = fcpu[ pp ].i;
              else if ( avant( fgpu[ qq ], fcpu[ pp ] ) ) ++qq;
              else { ++pp; ++qq; }
          } }
        std::printf( "        debug : cellule %d, mesure cpu %.9e gpu %.9e\n", cible, double( cpu[ cible ] ), res[ cible ] );
        std::printf( "          cpu :" );
        for ( const Fa &f : fcpu ) if ( f.i == cible ) std::printf( " (%d,%.4e)", f.j, f.c );
        std::printf( "\n          gpu :" );
        for ( int s = 0; s < gpu::DiagrammeGpu<D,TK>::NF; ++s )
            std::printf( " (%d,%.4e)", fj[ size_t( s ) * nu.n + cible ], fl[ size_t( s ) * nu.n + cible ] );
        std::printf( "\n" );
    }
    if ( manque && std::getenv( "CHAINE_DEBUG" ) )
        std::printf( "        debug : les %d facettes manquantes pesent %.3e sur %.3e ( %.4f %% ), la plus grosse %.3e contre %.3e de moyenne\n",
                     manque, csom, ctot, 100 * csom / ctot, cmax, ctot / fcpu.size() );
    // ce qui compte pour la hessienne : le POIDS des facettes non appariees, et l'ecart des autres
    // rapporte au `c` moyen -- pas a elles-memes
    const double cmoy = ctot / std::max<size_t>( fcpu.size(), 1 );
    std::sort( ecarts.begin(), ecarts.end() );
    const auto qu = [ & ]( double f ) { return ecarts.empty() ? 0.0 : ecarts[ std::min( ecarts.size() - 1, size_t( f * ecarts.size() ) ) ] / cmoy; };
    const double perdu = ( csom + csom2 ) / ctot;
    const double tol = sizeof( TK ) == 4 ? 1e-3 : 1e-9;
    // en `float` le critere porte sur le p99 : la queue des `c_ij` est faite d'aretes quasi nulles
    // que le `float` fait apparaitre ou disparaitre, de poids negligeable ( voir `perdu` )
    bool ok = perdu < tol && ( sizeof( TK ) == 4 ? qu( 0.99 ) < 1e-2 : qu( 0.9999 ) < 1e-9 );
    std::printf( "      controle  somme %.9f  ecart mesure %.1e  |  c_ij : median %.1e, p99 %.1e, p99.99 %.1e, max %.1e ( rapportes au c moyen )\n",
                 somme, ecart_m, qu( 0.5 ), qu( 0.99 ), qu( 0.9999 ), ecart_c / cmoy );
    std::printf( "                facettes manquantes %d, en trop %d -- %.2e du poids total%s\n",
                 manque, en_trop, perdu, ok ? "" : "   <-- FAUX" );
    // ---- LA HESSIENNE : assemblee sur la carte, verifiee contre `solver/Laplacien.h` par un
    //      produit `y = L x` sur un vecteur quelconque ( un seul nombre exerce toute la matrice ),
    //      plus la propriete de noyau `L . 1 = 0`, ligne par ligne.
    if ( arbre_gpu ) {
        gpu::Hessienne H;
        const double t_h = g.assemble( H );

        std::vector<Facette> fa;
        fa.reserve( fcpu.size() );
        for ( const Fa &f : fcpu ) fa.push_back( Facette{ f.i, f.j, f.c } );
        Laplacien lap;
        const double tl0 = now();
        lap.assemble( nu.n, fa );
        const double t_lc = now() - tl0;

        std::vector<double> x( nu.n ), yg( nu.n ), yc( nu.n ), un( nu.n, 1.0 );
        for ( SI i = 0; i < nu.n; ++i ) x[ i ] = std::sin( 0.7 * double( i ) + 1.0 );
        for ( SI i = 0; i < nu.n; ++i ) {
            double sc = lap.dia[ i ] * x[ i ];
            for ( SI p = lap.row[ i ]; p < lap.row[ i + 1 ]; ++p ) sc -= lap.c[ p ] * x[ lap.col[ p ] ];
            yc[ i ] = sc;
        }
        double *dx, *dy;
        cudaMalloc( &dx, nu.n * sizeof( double ) ); cudaMalloc( &dy, nu.n * sizeof( double ) );
        cudaMemcpy( dx, x.data(), nu.n * sizeof( double ), cudaMemcpyHostToDevice );
        g.applique( H, dx, dy );
        cudaMemcpy( yg.data(), dy, nu.n * sizeof( double ), cudaMemcpyDeviceToHost );
        cudaMemcpy( dx, un.data(), nu.n * sizeof( double ), cudaMemcpyHostToDevice );
        g.applique( H, dx, dy );
        std::vector<double> y1( nu.n );
        cudaMemcpy( y1.data(), dy, nu.n * sizeof( double ), cudaMemcpyDeviceToHost );
        cudaFree( dx ); cudaFree( dy );

        double ech = 0, ecart = 0, noyau = 0;
        for ( SI i = 0; i < nu.n; ++i ) ech = std::max( ech, std::fabs( yc[ i ] ) );
        for ( SI i = 0; i < nu.n; ++i ) {
            ecart = std::max( ecart, std::fabs( yg[ i ] - yc[ i ] ) );
            noyau = std::max( noyau, std::fabs( y1[ i ] ) / std::max( lap.dia[ i ], 1e-300 ) );
        }
        const bool hok = H.nnz == int( lap.row[ nu.n ] ) && ecart < 1e-9 * ech && noyau < 1e-12;
        std::printf( "      HESSIENNE %6.2f ms sur GPU contre %6.0f ms au CPU ( assemblage seul )   x%.0f   %d coefficients contre %d\n",
                     t_h, t_lc * 1e3, t_lc * 1e3 / t_h, H.nnz, int( lap.row[ nu.n ] ) );
        std::printf( "                | L x |_max %.3e, ecart au CPU %.1e ( soit %.1e relatif ), | L . 1 | %.1e%s\n",
                     ech, ecart, ech > 0 ? ecart / ech : 0.0, noyau, hok ? "" : "   <-- FAUX" );
        ok = ok && hok;
    }

    // ---- LE REGIME AMORTI : `iterations` tours de Newton ( poids neufs, majorants, mesures et
    //      facettes ), l'arbre construit UNE FOIS. C'est le cout qui compte pour un solveur : les
    //      frais fixes ( contexte CUDA, allocations ) sont derriere, et rien ne redescend.
    if ( iterations > 1 && arbre_gpu ) {
        std::vector<TF> W( nu.n );
        for ( SI i = 0; i < nu.n; ++i ) W[ i ] = nu.W ? nu.W[ i ] : TF( 0 );
        g.tour_newton( nu.W ? W.data() : nullptr );      // la chauffe
        double t_r = 0;
        if ( nu.W ) {                                    // les majorants SEULS : ni tri, ni boites
            g.refresh_poids( W.data() );
            for ( int it = 0; it < iterations; ++it ) t_r += g.refresh_poids( W.data() );
            t_r /= iterations;
            std::printf( "      REGIME    majorants refaits seuls : %6.2f ms contre %6.0f ms de construction complete au CPU   x%.0f\n",
                         t_r, t_arbre * 1e3, t_arbre * 1e3 / t_r );
        }
        double t_g = 0;
        gpu::Hessienne H2;
        for ( int it = 0; it < iterations; ++it ) {
            t_g += g.tour_newton( nu.W ? W.data() : nullptr );
            t_g += g.assemble( H2 );                     // le tour COMPLET : jusqu'a la hessienne
        }

        std::vector<TF> c2;
        const double tc0 = now();
        Laplacien lap2;
        for ( int it = 0; it < iterations; ++it ) {
            if ( nu.W ) pd.set_weights( W.data(), a.par );
            std::vector<std::vector<Facette>> pf( std::max( a.par.threads, 1 ) );
            pd.measures_and_facets( c2, a.par, [ & ]( int t, SI i, SI j, TF mes ) {
                double d2 = 0;
                for ( int d = 0; d < D; ++d ) { const double e = nu.P[ d ][ j ] - nu.P[ d ][ i ]; d2 += e * e; }
                if ( d2 > 0 ) pf[ t ].push_back( Facette{ i, j, mes / ( 2 * std::sqrt( d2 ) ) } );
            } );
            std::vector<Facette> tout;
            for ( const auto &v : pf ) tout.insert( tout.end(), v.begin(), v.end() );
            lap2.assemble( nu.n, tout );                 // le tour COMPLET, des deux cotes
        }
        const double t_c = now() - tc0;
        std::printf( "      REGIME    %d tours complets ( poids, majorants, mesures, facettes, hessienne ) : GPU %7.1f ms au tour ( %.3f s ), CPU %7.1f ms au tour ( %.3f s )   x%.1f\n",
                     iterations, t_g / iterations, t_g * 1e-3, t_c / iterations * 1e3, t_c, t_c * 1e3 / t_g );
    }
    return ! ok;
}

} // namespace

int main( int argc, char **argv ) {
    Args a;
    int reps_gpu = 10;
    bool arbre_gpu = false;
    int iterations = 1;
    for ( int i = 1; i < argc; ++i ) {
        const std::string s = argv[ i ];
        if ( a.parse( s, i, argc, argv ) ) continue;
        if ( s == "--reps-gpu" && i + 1 < argc ) { reps_gpu = std::atoi( argv[ ++i ] ); continue; }
        if ( s == "--arbre-gpu" ) { arbre_gpu = true; continue; }
        if ( s == "--iterations" && i + 1 < argc ) { iterations = std::atoi( argv[ ++i ] ); continue; }
        std::printf( "usage: chaine [options]\n" );
        Args::usage();
        std::printf( "  --reps-gpu R    repetitions du noyau GPU, minimum       (10)\n"
                     "  --arbre-gpu     construire l'arbre sur le GPU, et l'y garder\n"
                     "  --iterations K  le REGIME AMORTI : K tours de Newton apres une seule construction\n" );
        return s == "--help" || s == "-h" ? 0 : 1;
    }
    a.dims = 2;
    a.finalise();
    std::printf( "GPU : %s\n", gpu::carte().c_str() );
    int bad = 0;
    for ( const Nuage<2> &nu : a.nuages<2>() ) {
        if ( nu.absent ) { std::printf( "  %-28s : ABSENT ( --cases DIR )\n", nu.nom.c_str() ); continue; }
        bad += dispatch<2>( a, [ & ]( auto tag ) { return chaine<typename decltype( tag )::type>( a, nu, reps_gpu, arbre_gpu, iterations ); } );
    }
    return bad ? 1 : 0;
}

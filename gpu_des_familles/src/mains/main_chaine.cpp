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
int chaine( const Args &a, const Nuage<PD::dim> &nu, int reps_gpu, bool arbre_gpu ) {
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
        ? gpu::DiagrammeGpu<D,TK>( nu.P, nu.W, int( nu.n ), int( a.leaf ), &ms_gpu )
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
    const bool ok = perdu < tol && ( sizeof( TK ) == 4 ? qu( 0.99 ) < 1e-2 : qu( 0.9999 ) < 1e-9 );
    std::printf( "      controle  somme %.9f  ecart mesure %.1e  |  c_ij : median %.1e, p99 %.1e, p99.99 %.1e, max %.1e ( rapportes au c moyen )\n",
                 somme, ecart_m, qu( 0.5 ), qu( 0.99 ), qu( 0.9999 ), ecart_c / cmoy );
    std::printf( "                facettes manquantes %d, en trop %d -- %.2e du poids total%s\n",
                 manque, en_trop, perdu, ok ? "" : "   <-- FAUX" );
    return ! ok;
}

} // namespace

int main( int argc, char **argv ) {
    Args a;
    int reps_gpu = 10;
    bool arbre_gpu = false;
    for ( int i = 1; i < argc; ++i ) {
        const std::string s = argv[ i ];
        if ( a.parse( s, i, argc, argv ) ) continue;
        if ( s == "--reps-gpu" && i + 1 < argc ) { reps_gpu = std::atoi( argv[ ++i ] ); continue; }
        if ( s == "--arbre-gpu" ) { arbre_gpu = true; continue; }
        std::printf( "usage: chaine [options]\n" );
        Args::usage();
        std::printf( "  --reps-gpu R    repetitions du noyau GPU, minimum       (10)\n"
                     "  --arbre-gpu     construire l'arbre sur le GPU ( 2D, Voronoi )\n" );
        return s == "--help" || s == "-h" ? 0 : 1;
    }
    a.dims = 2;
    a.finalise();
    std::printf( "GPU : %s\n", gpu::carte().c_str() );
    int bad = 0;
    for ( const Nuage<2> &nu : a.nuages<2>() ) {
        if ( nu.absent ) { std::printf( "  %-28s : ABSENT ( --cases DIR )\n", nu.nom.c_str() ); continue; }
        bad += dispatch<2>( a, [ & ]( auto tag ) { return chaine<typename decltype( tag )::type>( a, nu, reps_gpu, arbre_gpu ); } );
    }
    return bad ? 1 : 0;
}

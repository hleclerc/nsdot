// =====================================================================================
// LA CHAINE 2D SUR GPU, DE BOUT EN BOUT, avec le moteur CPU en TEMOIN a chaque poste : l'arbre
// ( CPU ou GPU ), les mesures ET les facettes en un seul noyau, l'assemblage de la hessienne, le
// gradient conjugue preconditionne par notre multigrille, la boucle de Newton avec sa recherche
// lineaire, et LA DENSITE IMAGE.
//
// Les facettes sont le maillon entre « mesurer des cellules » et « faire un Newton » : la
// hessienne du transport est le laplacien du graphe de Laguerre, `c_ij = integrale_facette rho ds
// / ( 2 |p_i - p_j| )` ( `solver/Laplacien.h` ) -- la LONGUEUR de la facette seulement quand la
// source est uniforme. Ce sont exactement les aretes de chaque cellule, que le noyau porte deja.
//
// LES TEMOINS. Chaque poste est compare a un calcul qui ne lui doit rien : les mesures et les
// facettes au moteur CPU ; la hessienne a `solver/Laplacien.h` PUIS a une difference finie centree
// sur les mesures ( le seul controle qui ne suppose ni le signe ni le facteur ) ; le CG a un CG
// Jacobi ecrit ici et a AMGCL ; la masse sous une image a un DECOUPAGE EN PIXELS ( Sutherland-
// Hodgman ), c'est-a-dire l'algorithme que le noyau refuse de faire.
//
//   xmake run chaine --threads 8 --kernel float --load uniforme -n 1000000
//   xmake run chaine --threads 8 --arbre-gpu --image 2048 --densite toutes
//   xmake run chaine --threads 8 --arbre-gpu --image 512 --newton 2000 --image-etapes 8
// =====================================================================================

#include "bench/Dispatch.h"
#include "solver/Laplacien.h"
#include "solver/Lineaire.h"
#include <cuda_runtime.h>
#include "gpu/Mesures.h"
#include "gpu/RefAmgcl.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cctype>
#include <cstdlib>
#include <string>
#include <vector>

using namespace sf;

namespace {

/// une facette telle que le banc la compare : ( i, j ) tries, et le coefficient
struct Fa { int i, j; double c; };
bool avant( const Fa &a, const Fa &b ) { return a.i != b.i ? a.i < b.i : a.j < b.j; }

// =====================================================================================
// LA DENSITE IMAGE, COTE HOTE : la grille, et LE TEMOIN.
//
// Le temoin ne doit rien partager avec le GPU, sinon il ne temoigne de rien. Le GPU integre SUR
// LE BORD ( `gpu/Image2D.cuh` ) ; ici on fait exactement ce qu'il refuse de faire : on DECOUPE la
// cellule par les bords de pixels ( Sutherland-Hodgman ) et on somme `rho x aire`. Deux codes
// sans une ligne commune pour le meme nombre.
// =====================================================================================

struct Img {
    int W = 0, H = 0;
    std::vector<double> v;                               ///< `v[ j * W + i ]`, `j = 0` en bas
    std::vector<double> s;                               ///< la somme prefixe de la ligne, `hx` compris
    double hx() const { return 1.0 / W; }
    double hy() const { return 1.0 / H; }
    double val( int i, int j ) const { return v[ size_t( j ) * W + i ]; }
    void prepare() {                                     // pour la variante par le bord ( chronos )
        s.assign( v.size(), 0.0 );
        for ( int j = 0; j < H; ++j ) {
            double a = 0;
            for ( int i = 0; i < W; ++i ) { s[ size_t( j ) * W + i ] = a; a += v[ size_t( j ) * W + i ] * hx(); }
        }
    }
};

int borne( int i, int n ) { return i < 0 ? 0 : ( i >= n ? n - 1 : i ); }

/// Sutherland-Hodgman contre `a x + b y <= c`. Le polygone est convexe : au plus un sommet de plus.
int coupe_demi( const double *ix, const double *iy, int n, double *ox, double *oy, double a, double b, double c ) {
    int m = 0;
    for ( int i = 0, j = n - 1; i < n; j = i++ ) {
        const double sj = a * ix[ j ] + b * iy[ j ] - c, si = a * ix[ i ] + b * iy[ i ] - c;
        if ( ( sj <= 0 ) != ( si <= 0 ) ) {
            const double t = sj / ( sj - si );
            ox[ m ] = ix[ j ] + t * ( ix[ i ] - ix[ j ] );
            oy[ m ] = iy[ j ] + t * ( iy[ i ] - iy[ j ] );
            ++m;
        }
        if ( si <= 0 ) { ox[ m ] = ix[ i ]; oy[ m ] = iy[ i ]; ++m; }
    }
    return m;
}

double aire_pol( const double *x, const double *y, int n ) {
    double a = 0;
    for ( int i = 0, j = n - 1; i < n; j = i++ ) a += x[ j ] * y[ i ] - x[ i ] * y[ j ];
    return 0.5 * std::fabs( a );
}

constexpr int NMX = 600;                                 ///< `MaxNv` plus la marge des coupes

/// LA MASSE, par decoupage en pixels : la boite englobante, puis quatre demi-plans par pixel.
double masse_pixels( const Img &im, const double *px, const double *py, int nb ) {
    if ( nb < 3 ) return 0;
    double x0 = px[ 0 ], x1 = px[ 0 ], y0 = py[ 0 ], y1 = py[ 0 ];
    for ( int i = 1; i < nb; ++i ) {
        x0 = std::min( x0, px[ i ] ); x1 = std::max( x1, px[ i ] );
        y0 = std::min( y0, py[ i ] ); y1 = std::max( y1, py[ i ] );
    }
    const int i0 = borne( int( x0 * im.W ), im.W ), i1 = borne( int( x1 * im.W ), im.W );
    const int j0 = borne( int( y0 * im.H ), im.H ), j1 = borne( int( y1 * im.H ), im.H );
    double m = 0;
    double ax[ NMX ], ay[ NMX ], bx[ NMX ], by[ NMX ];
    for ( int j = j0; j <= j1; ++j )
        for ( int i = i0; i <= i1; ++i ) {
            const double r = im.val( i, j );
            if ( r == 0 ) continue;
            int n = nb;
            for ( int q = 0; q < nb; ++q ) { ax[ q ] = px[ q ]; ay[ q ] = py[ q ]; }
            n = coupe_demi( ax, ay, n, bx, by,  1,  0,  ( i + 1 ) * im.hx() );
            n = coupe_demi( bx, by, n, ax, ay, -1,  0, -i * im.hx() );
            n = coupe_demi( ax, ay, n, bx, by,  0,  1,  ( j + 1 ) * im.hy() );
            n = coupe_demi( bx, by, n, ax, ay,  0, -1, -j * im.hy() );
            if ( n >= 3 ) m += r * aire_pol( ax, ay, n );
        }
    return m;
}

/// `integrale_arete rho ds` -- le coefficient de hessienne quand la source n'est pas uniforme.
/// Temoin : on LISTE les traversees de lignes de grille, on les TRIE, et on evalue `rho` au
/// milieu de chaque morceau. Pas un increment, pas une marche : rien du parcours du GPU.
double long_ponderee( const Img &im, double x0, double y0, double x1, double y1 ) {
    const double dx = x1 - x0, dy = y1 - y0;
    const double lg = std::sqrt( dx * dx + dy * dy );
    if ( lg == 0 ) return 0;
    std::vector<double> t;
    t.reserve( 32 );
    t.push_back( 0 ); t.push_back( 1 );
    if ( dx != 0 ) {
        const int a = borne( int( std::min( x0, x1 ) * im.W ), im.W ), b = borne( int( std::max( x0, x1 ) * im.W ), im.W );
        for ( int i = a; i <= b + 1; ++i ) { const double u = ( i * im.hx() - x0 ) / dx; if ( u > 0 && u < 1 ) t.push_back( u ); }
    }
    if ( dy != 0 ) {
        const int a = borne( int( std::min( y0, y1 ) * im.H ), im.H ), b = borne( int( std::max( y0, y1 ) * im.H ), im.H );
        for ( int j = a; j <= b + 1; ++j ) { const double u = ( j * im.hy() - y0 ) / dy; if ( u > 0 && u < 1 ) t.push_back( u ); }
    }
    std::sort( t.begin(), t.end() );
    double s = 0;
    for ( size_t q = 1; q < t.size(); ++q ) {
        const double u = 0.5 * ( t[ q - 1 ] + t[ q ] );
        const int i = borne( int( ( x0 + dx * u ) * im.W ), im.W ), j = borne( int( ( y0 + dy * u ) * im.H ), im.H );
        s += im.val( i, j ) * ( t[ q ] - t[ q - 1 ] ) * lg;
    }
    return s;
}

/// LA MEME MASSE, PAR LE BORD -- la methode du GPU, mais sur le CPU. Elle ne sert PAS de temoin
/// ( elle partage l'idee du noyau, donc elle ne prouverait rien ) : elle sert a separer ce qui
/// revient a L ALGORITHME de ce qui revient a LA MACHINE, en chronometrant les deux cote a cote.
double masse_bord( const Img &im, const double *px, const double *py, int nb ) {
    if ( nb < 3 ) return 0;
    const double sref = im.s[ size_t( borne( int( py[ 0 ] * im.H ), im.H ) ) * im.W + borne( int( px[ 0 ] * im.W ), im.W ) ];
    double acc = 0;
    for ( int e = 0, f = nb - 1; e < nb; f = e++ ) {
        const double x0 = px[ f ], y0 = py[ f ], x1 = px[ e ], y1 = py[ e ];
        const double dx = x1 - x0, dy = y1 - y0;
        if ( dy == 0 ) continue;
        constexpr double INF = 1e300;
        int i = borne( int( x0 * im.W ), im.W ), j = borne( int( y0 * im.H ), im.H );
        const int si = dx > 0 ? 1 : -1, sj = dy > 0 ? 1 : -1;
        double tx = dx == 0 ? INF : ( ( dx > 0 ? ( i + 1 ) * im.hx() : i * im.hx() ) - x0 ) / dx;
        double ty = ( ( dy > 0 ? ( j + 1 ) * im.hy() : j * im.hy() ) - y0 ) / dy;
        const double ax = dx == 0 ? INF : std::fabs( im.hx() / dx ), ay = std::fabs( im.hy() / dy );
        tx = tx < 0 ? 0 : tx; ty = ty < 0 ? 0 : ty;
        double tp = 0, xp = x0;
        for ( int g = 0; g < im.W + im.H + 4; ++g ) {
            const bool par_x = tx < ty;
            double tn = par_x ? tx : ty;
            const bool der = ! ( tn < 1.0 );
            if ( der ) tn = 1.0;
            const double xc = x0 + dx * tn;
            const size_t o = size_t( j ) * im.W + i;
            acc += ( tn - tp ) * dy * ( im.s[ o ] - sref + im.v[ o ] * ( 0.5 * ( xp + xc ) - i * im.hx() ) );
            if ( der ) break;
            if ( par_x ) { i = borne( i + si, im.W ); tx += ax; } else { j = borne( j + sj, im.H ); ty += ay; }
            tp = tn; xp = xc;
        }
    }
    return std::fabs( acc );
}

/// UNE IMAGE DE SYNTHESE, faite pour etre dure : un fond lisse, un disque net, une bande fine.
/// Les DISCONTINUITES sont ce qui distingue une image d'une densite reguliere -- une methode qui
/// les rate ne se voit pas sur un fond lisse.
Img synthese( int N ) {
    Img im; im.W = N; im.H = N; im.v.assign( size_t( N ) * N, 0.0 );
    for ( int j = 0; j < N; ++j )
        for ( int i = 0; i < N; ++i ) {
            const double x = ( i + 0.5 ) / N, y = ( j + 0.5 ) / N;
            double r = 0.10 + 0.9 * std::pow( std::sin( 3 * M_PI * x ) * std::cos( 2 * M_PI * y ), 2 );
            const double dx = x - 0.30, dy = y - 0.70;
            if ( dx * dx + dy * dy < 0.15 * 0.15 ) r += 3.0;            // un disque net
            if ( std::fabs( x - y ) < 0.02 ) r += 5.0;                  // une bande fine
            im.v[ size_t( j ) * N + i ] = r;
        }
    return im;
}

/// un PGM ( P2 ascii ou P5 binaire ), la ligne du haut devenant `j = H - 1`
bool lit_pgm( const char *chemin, Img &im ) {
    std::FILE *f = std::fopen( chemin, "rb" );
    if ( ! f ) return false;
    auto jeton = [ & ]( long &val ) {
        int c;
        do {
            while ( ( c = std::fgetc( f ) ) != EOF && std::isspace( c ) ) {}
            if ( c == '#' ) { while ( ( c = std::fgetc( f ) ) != EOF && c != '\n' ) {} continue; }
            break;
        } while ( true );
        if ( c == EOF ) return false;
        long v = 0;
        while ( c != EOF && std::isdigit( c ) ) { v = v * 10 + ( c - '0' ); c = std::fgetc( f ); }
        val = v;
        return true;
    };
    char magique[ 3 ] = {};
    if ( std::fscanf( f, "%2s", magique ) != 1 || magique[ 0 ] != 'P' || ( magique[ 1 ] != '2' && magique[ 1 ] != '5' ) ) { std::fclose( f ); return false; }
    long W = 0, H = 0, mx = 0;
    if ( ! jeton( W ) || ! jeton( H ) || ! jeton( mx ) ) { std::fclose( f ); return false; }
    im.W = int( W ); im.H = int( H );
    im.v.assign( size_t( W ) * H, 0.0 );
    if ( magique[ 1 ] == '5' ) {
        std::vector<unsigned char> buf( size_t( W ) * H * ( mx > 255 ? 2 : 1 ) );
        if ( std::fread( buf.data(), 1, buf.size(), f ) != buf.size() ) { std::fclose( f ); return false; }
        for ( long j = 0; j < H; ++j )
            for ( long i = 0; i < W; ++i ) {
                const size_t o = size_t( j ) * W + i;
                const double u = mx > 255 ? ( buf[ 2 * o ] * 256.0 + buf[ 2 * o + 1 ] ) : double( buf[ o ] );
                im.v[ size_t( H - 1 - j ) * W + i ] = u / double( mx );
            }
    } else {
        for ( long j = 0; j < H; ++j )
            for ( long i = 0; i < W; ++i ) { long u = 0; if ( ! jeton( u ) ) { std::fclose( f ); return false; } im.v[ size_t( H - 1 - j ) * W + i ] = double( u ) / double( mx ); }
    }
    std::fclose( f );
    return true;
}

/// `rho <- ( 1 - f ) + f rho`, apres normalisation : `f = 0` rend Lebesgue, `f = 1` l'image nue.
/// C'est le bouton de CONTRASTE -- il dit si une difficulte vient de la methode ou de l'image.
void melange( Img &im, double f ) {
    for ( double &u : im.v ) u = ( 1 - f ) + f * u;
}

/// la masse totale ramenee a un : le Newton vise `1 / n` par cellule des deux cotes
double normalise( Img &im ) {
    double t = 0;
    for ( double u : im.v ) t += u;
    t *= im.hx() * im.hy();
    if ( t <= 0 ) return 0;
    for ( double &u : im.v ) u /= t;
    return t;
}

template<class PD>
int chaine( const Args &a, const Nuage<PD::dim> &nu, int reps_gpu, bool arbre_gpu, int iterations,
            int newton, int raff, double marge, double tol, const Img &img, int dmode, int chunk, int etapes, double tolcg, int echelle ) {
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
    auto garde = [ & ]( int t, SI i, SI j, TF mes ) {
        double d2 = 0;
        for ( int d = 0; d < D; ++d ) { const double e = nu.P[ d ][ j ] - nu.P[ d ][ i ]; d2 += e * e; }
        if ( d2 > 0 ) par_fil[ t ].push_back( Fa{ int( i ), int( j ), mes / ( 2 * std::sqrt( d2 ) ) } );
    };
    t0 = now();
    if ( img.W ) {
        // LE TEMOIN SOUS LA DENSITE : la masse par decoupage en pixels, le coefficient de
        // hessienne par `integrale_facette rho ds` -- deux calculs qui ne doivent rien au GPU.
        pd.measures_and_facets_avec( cpu, a.par, garde, [ & ]( const auto &cel, auto &&fac, SI ) -> TF {
            if ( cel.nb <= 0 ) return TF( 0 );
            double px[ NMX ], py[ NMX ];
            const int nb = std::min( int( cel.nb ), NMX );
            for ( int q = 0; q < nb; ++q ) { px[ q ] = double( cel.vx[ q ] ); py[ q ] = double( cel.vy[ q ] ); }
            for ( int i = 0, j = nb - 1; i < nb; j = i++ )
                if ( cel.cid[ j ] >= 0 ) fac( cel.cid[ j ], TF( long_ponderee( img, px[ j ], py[ j ], px[ i ], py[ i ] ) ) );
            return TF( masse_pixels( img, px, py, nb ) );
        } );
    } else {
        pd.measures_and_facets( cpu, a.par, garde );
    }
    const double t_cpu = now() - t0;
    std::vector<Fa> fcpu;
    for ( const auto &v : par_fil ) fcpu.insert( fcpu.end(), v.begin(), v.end() );

    std::printf( "  %-24s n=%-8d %-8s  arbre %6.0f ms ( CPU, %d fils )\n",
                 nu.nom.c_str(), int( nu.n ), nu.W ? "Laguerre" : "Voronoi", t_arbre * 1e3, a.par.threads );
    std::printf( "      CPU       mesures + facettes %7.3f s   %6.0f ns/germe   %zu facettes\n",
                 t_cpu, t_cpu / nu.n * 1e9, fcpu.size() );
    // CE QUI REVIENT A L ALGORITHME : le meme moteur CPU, la meme cellule, la masse par le bord
    // au lieu du decoupage en pixels. La difference est celle des deux METHODES, a machine egale.
    double t_bord = 0;
    if ( img.W ) {
        std::vector<TF> c3;
        const double tb0 = now();
        pd.measures_and_facets_avec( c3, a.par, []( int, SI, SI, TF ) {}, [ & ]( const auto &cel, auto &&, SI ) -> TF {
            if ( cel.nb <= 0 ) return TF( 0 );
            double px[ NMX ], py[ NMX ];
            const int nb = std::min( int( cel.nb ), NMX );
            for ( int q = 0; q < nb; ++q ) { px[ q ] = double( cel.vx[ q ] ); py[ q ] = double( cel.vy[ q ] ); }
            return TF( masse_bord( img, px, py, nb ) );
        } );
        t_bord = now() - tb0;
        double e = 0;
        for ( SI i = 0; i < nu.n; ++i ) e = std::max( e, std::fabs( double( c3[ i ] ) - double( cpu[ i ] ) ) );
        std::printf( "      CPU       la MEME cellule, la masse par le bord : %7.3f s contre %7.3f s par decoupage en pixels   x%.1f ( l ALGORITHME ), ecart %.1e\n",
                     t_bord, t_cpu, t_cpu / t_bord, e * nu.n );
    }

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
    // ---- LA DENSITE IMAGE : la grille monte une fois, avec la somme prefixe de ses lignes
    if ( img.W ) {
        const double mtot = g.charge_image( img.v.data(), img.W, img.H );
        g.regle_densite( dmode == 3 ? gpu::Densite::DIRECTE : gpu::Densite( dmode + 1 ), chunk );
        std::printf( "      DENSITE   image %d x %d, masse totale %.9f, mode %s ( lots de %d cellules )\n",
                     img.W, img.H, mtot, gpu::nom( g.densite() ), std::min( chunk, int( nu.n ) ) );
    }
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
    const double tolf = sizeof( TK ) == 4 ? 1e-3 : 1e-9;   // le seuil du controle des facettes
    // en `float` le critere porte sur le p99 : la queue des `c_ij` est faite d'aretes quasi nulles
    // que le `float` fait apparaitre ou disparaitre, de poids negligeable ( voir `perdu` )
    bool ok = perdu < tolf && ( sizeof( TK ) == 4 ? qu( 0.99 ) < 1e-2 : qu( 0.9999 ) < 1e-9 );
    std::printf( "      controle  somme %.9f  ecart mesure %.1e  |  c_ij : median %.1e, p99 %.1e, p99.99 %.1e, max %.1e ( rapportes au c moyen )\n",
                 somme, ecart_m, qu( 0.5 ), qu( 0.99 ), qu( 0.9999 ), ecart_c / cmoy );
    std::printf( "                facettes manquantes %d, en trop %d -- %.2e du poids total%s\n",
                 manque, en_trop, perdu, ok ? "" : "   <-- FAUX" );
    // ---- LES TROIS MODES, mesures cote a cote sur le MEME travail. Ils doivent rendre le meme
    //      nombre : ce qui change est ou le parcours a lieu, pas ce qu'il calcule.
    if ( img.W && dmode == 3 ) {
        for ( int md = 2; md <= 3; ++md ) {
            g.regle_densite( gpu::Densite( md ), chunk );
            std::vector<double> r2, l2;
            std::vector<int> j2;
            const gpu::Chrono c2 = g.facettes( reps_gpu, r2, j2, l2 );
            double e = 0;
            for ( SI i = 0; i < nu.n; ++i ) e = std::max( e, std::fabs( r2[ i ] - res[ i ] ) );
            std::printf( "      DENSITE   %-12s %7.3f s   %6.1f ns/germe   x%.2f sur directe, ecart %.1e ( noyau des cellules : %d registres, %.0f %% )\n",
                         gpu::nom( gpu::Densite( md ) ), c2.noyau, c2.noyau / nu.n * 1e9, ch.noyau / c2.noyau,
                         e / ( 1.0 / nu.n ), c2.regs, c2.occup * 100 );
        }
        g.regle_densite( gpu::Densite::DIRECTE, chunk );
    }

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

        // ---- LE SYSTEME DE NEWTON : `L d = mesures - cible`, resolu par le CG precondionne.
        //      Compare a un CG Jacobi ecrit ici, meme algorithme, pour la solution et le temps.
        std::vector<double> bb( nu.n );
        const double cible = 1.0 / double( nu.n );
        for ( SI i = 0; i < nu.n; ++i ) bb[ i ] = res[ i ] - cible;
        // `b` doit etre ORTHOGONAL AUX CONSTANTES : c'est la condition de compatibilite du
        // laplacien, et la somme des mesures ne vaut un qu'a l'arrondi pres. Les deux jauges
        // ( moyenne nulle cote GPU, `d_0 = 0` cote CPU ) ne donnent la meme direction que la.
        double mb = 0;
        for ( SI i = 0; i < nu.n; ++i ) mb += bb[ i ];
        mb /= nu.n;
        for ( SI i = 0; i < nu.n; ++i ) bb[ i ] -= mb;
        double *db, *dd;
        cudaMalloc( &db, nu.n * sizeof( double ) ); cudaMalloc( &dd, nu.n * sizeof( double ) );
        cudaMemcpy( db, bb.data(), nu.n * sizeof( double ), cudaMemcpyHostToDevice );
        const double ta0 = now();
        g.monte_amg( H );                                // la hierarchie du multigrille
        const double t_amg = now() - ta0;
        double ms_cg = 0, r_cg = 0;
        const int its = g.resout( H, db, dd, 1e-10, 20000, &ms_cg, &r_cg );
        std::vector<double> dg( nu.n );
        cudaMemcpy( dg.data(), dd, nu.n * sizeof( double ), cudaMemcpyDeviceToHost );
        cudaFree( db ); cudaFree( dd );

        // le meme CG, sur le CPU : jauge `d_0 = 0`, preconditionneur Jacobi. `CHAINE_RAPIDE`
        // le saute ( et AMGCL avec ), pour balayer des reglages sans le payer a chaque point.
        const bool rapide = std::getenv( "CHAINE_RAPIDE" ) != nullptr;
        std::vector<double> dc( nu.n, 0.0 ), rr( bb ), zz( nu.n ), pp( nu.n ), qq( nu.n );
        rr[ 0 ] = 0;                                     // sa jauge : la ligne zero est rayee
        auto applique_cpu = [ & ]( const std::vector<double> &v, std::vector<double> &y ) {
            y[ 0 ] = 0;
            for ( SI i = 1; i < nu.n; ++i ) {
                double sm = lap.dia[ i ] * v[ i ];
                for ( SI e = lap.row[ i ]; e < lap.row[ i + 1 ]; ++e )
                    if ( lap.col[ e ] ) sm -= lap.c[ e ] * v[ lap.col[ e ] ];
                y[ i ] = sm;
            }
        };
        const double tc0 = now();
        int itc = 0;
        if ( ! rapide ) {
        double nb2 = 0;
        for ( SI i = 0; i < nu.n; ++i ) nb2 += rr[ i ] * rr[ i ];
        for ( SI i = 0; i < nu.n; ++i ) zz[ i ] = i ? rr[ i ] / lap.dia[ i ] : 0.0;
        pp = zz;
        double rz = 0;
        for ( SI i = 0; i < nu.n; ++i ) rz += rr[ i ] * zz[ i ];
        for ( double r2 = nb2; itc < 20000 && r2 > 1e-20 * nb2; ++itc ) {
            applique_cpu( pp, qq );
            double pq = 0;
            for ( SI i = 0; i < nu.n; ++i ) pq += pp[ i ] * qq[ i ];
            const double al = pq != 0 ? rz / pq : 0;
            r2 = 0;
            for ( SI i = 0; i < nu.n; ++i ) { dc[ i ] += al * pp[ i ]; rr[ i ] -= al * qq[ i ]; r2 += rr[ i ] * rr[ i ]; }
            if ( r2 <= 1e-20 * nb2 ) { ++itc; break; }
            for ( SI i = 0; i < nu.n; ++i ) zz[ i ] = i ? rr[ i ] / lap.dia[ i ] : 0.0;
            double rz2 = 0;
            for ( SI i = 0; i < nu.n; ++i ) rz2 += rr[ i ] * zz[ i ];
            const double be = rz != 0 ? rz2 / rz : 0;
            rz = rz2;
            for ( SI i = 0; i < nu.n; ++i ) pp[ i ] = zz[ i ] + be * pp[ i ];
        }
        }
        const double t_cgc = now() - tc0;

        // les deux jauges ( moyenne nulle cote GPU, `d_0 = 0` cote CPU ) donnent la meme direction
        // a une constante pres : on centre les deux avant de comparer
        double mg = 0, mc = 0;
        for ( SI i = 0; i < nu.n; ++i ) { mg += dg[ i ]; mc += dc[ i ]; }
        mg /= nu.n; mc /= nu.n;
        for ( SI i = 0; i < nu.n; ++i ) { dg[ i ] -= mg; dc[ i ] -= mc; }
        double ech_d = 0, ec_d = 0;
        for ( SI i = 0; i < nu.n; ++i ) ech_d = std::max( ech_d, std::fabs( dc[ i ] ) );
        for ( SI i = 0; i < nu.n; ++i ) ec_d = std::max( ec_d, std::fabs( dg[ i ] - dc[ i ] ) );
        const bool cok = its > 0 && r_cg < 1e-9 && ( rapide || ec_d < 1e-6 * ech_d );
        std::printf( "      AMG       hierarchie montee en %6.0f ms\n", t_amg * 1e3 );
#ifdef SF_AMGCL
        if ( ! rapide )
        // LE TEMOIN DE REFERENCE : AMGCL, agregation LISSEE + SPAI0, sur le systeme reduit
        {
            Amg ref;
            ref.tol = 1e-10;
            std::vector<TF> dref;
            const double tr0 = now();
            ref.resout( lap, bb, dref );
            const double t_ref = now() - tr0;
            double mr = 0;
            for ( SI i = 0; i < nu.n; ++i ) mr += dref[ i ];
            mr /= nu.n;
            double ecr = 0, echr = 0;
            for ( SI i = 0; i < nu.n; ++i ) { echr = std::max( echr, std::fabs( dref[ i ] - mr ) ); }
            for ( SI i = 0; i < nu.n; ++i ) ecr = std::max( ecr, std::fabs( dg[ i ] - ( dref[ i ] - mr ) ) );
            std::printf( "      %-9s %4d iterations, %7.0f ms au CPU ( hierarchie %4.0f + resolution %4.0f ), ecart a nous %.1e relatif\n",
                         ref.nom(), ref.st.nb_iter, t_ref * 1e3, ref.st.t_hier * 1e3, ref.st.t_res * 1e3,
                         echr > 0 ? ecr / echr : 0.0 );
        }
#endif
        // ---- LE MEME AMGCL, MAIS SUR LA CARTE : la seule comparaison a plateforme egale.
        //      `AMGCL_VAR` choisit la configuration ( par defaut on les essaie toutes ).
        {
            std::vector<int> ptr, cl;
            std::vector<double> vl;
            const double tf0 = now();
            lap.crs_reduit( ptr, cl, vl );
            const double t_forme = now() - tf0;
            const int mred = int( nu.n ) - 1;
            std::vector<double> rb( mred ), sol( mred );
            for ( SI i = 1; i < nu.n; ++i ) rb[ i - 1 ] = bb[ i ];
            const char *choix = std::getenv( "AMGCL_VAR" );
            // `CHAINE_RAPIDE` : aucune variante, on balaie nos reglages sans payer le temoin
            for ( int v = rapide ? 5 : ( choix ? std::atoi( choix ) : 0 ); v < 5; ++v ) {
                int it = 0; double er = 0, th = 0, tr = 0;
                gpu::amgcl_cuda( mred, ptr, cl, vl, rb, sol, 1e-10, 20000, v, &it, &er, &th, &tr );
                std::vector<double> dr( nu.n, 0.0 );
                for ( SI i = 1; i < nu.n; ++i ) dr[ i ] = sol[ i - 1 ];
                double mr = 0;
                for ( SI i = 0; i < nu.n; ++i ) mr += dr[ i ];
                mr /= nu.n;
                double ec = 0, ech = 0;
                for ( SI i = 0; i < nu.n; ++i ) { ech = std::max( ech, std::fabs( dr[ i ] - mr ) ); ec = std::max( ec, std::fabs( dg[ i ] - ( dr[ i ] - mr ) ) ); }
                std::printf( "      %-24s %5d it., %7.0f ms ( mise en forme %4.0f + hierarchie %4.0f + resolution %5.0f ), residu %.1e, ecart a nous %.1e\n",
                             gpu::amgcl_cuda_nom( v ), it, ( t_forme + th + tr ) * 1e3, t_forme * 1e3, th * 1e3, tr * 1e3,
                             er, ech > 0 ? ec / ech : 0.0 );
                if ( choix ) break;
            }
        }
        std::printf( "      CG        %4d iterations, residu relatif %.1e, %7.1f ms sur GPU contre %7.0f ms au CPU ( %d it. )   x%.1f\n",
                     its, r_cg, ms_cg, t_cgc * 1e3, itc, t_cgc * 1e3 / ms_cg );
        std::printf( "                | d |_max %.3e, ecart au CG du CPU %.1e ( soit %.1e relatif )%s\n",
                     ech_d, ec_d, ech_d > 0 ? ec_d / ech_d : 0.0, cok ? "" : "   <-- FAUX" );
        ok = ok && cok;
    }

    // ---- LE NEWTON COMPLET. La hessienne du dual est `d m / d w = L` ( augmenter `w_i` pousse
    //      les plans qui bordent la cellule `i`, donc l'agrandit ), donc le pas resout `L d = m - v`
    //      et va dans le sens `w <- w - t d`.
    //
    //      LA RECHERCHE DU PAS ( Kitagawa-Merigot-Thibert ) : un pas est accepte si AUCUNE CELLULE
    //      NE DISPARAIT -- c'est la condition qui mord, la fonctionnelle duale n'etant definie que
    //      la ou toutes les cellules ont une masse -- ET si le residu decroit d'au moins `t / 2`.
    //      On divise par deux jusqu'a passer, puis on RAFFINE PAR DICHOTOMIE entre le dernier pas
    //      refuse et le premier accepte : c'est le « meilleur coefficient de relaxation », et il
    //      compte, parce qu'un pas deux fois trop petit coute une iteration de Newton entiere.
    if ( newton > 0 && arbre_gpu ) {
        const double cible = 1.0 / double( nu.n );
        const double norme = cible * std::sqrt( double( nu.n ) );
        std::vector<TF> w( nu.n, 0.0 ), w2( nu.n ), dh( nu.n );
        double *db = nullptr, *dd = nullptr;
        cudaMalloc( &db, nu.n * sizeof( double ) );
        cudaMalloc( &dd, nu.n * sizeof( double ) );

        // ---- LE CONTROLE QUI NE SUPPOSE RIEN : `d m / d w` PAR DIFFERENCE FINIE, compare a `-L`.
        //      Comparer la hessienne du GPU a celle du CPU ne dit rien si les DEUX ont la meme
        //      formule fausse -- sous une densite, `c_ij` n'est plus `| facette |` mais
        //      `integrale_facette rho ds`, et rien d'autre ne le verifie.
        {
            double *b0, *b1, *du, *dy;
            cudaMalloc( &b0, nu.n * sizeof( double ) ); cudaMalloc( &b1, nu.n * sizeof( double ) );
            cudaMalloc( &du, nu.n * sizeof( double ) ); cudaMalloc( &dy, nu.n * sizeof( double ) );
            std::vector<double> u( nu.n );
            double mu = 0;
            for ( SI i = 0; i < nu.n; ++i ) { u[ i ] = std::sin( 0.37 * double( i ) + 0.5 ); mu += u[ i ]; }
            mu /= nu.n;
            for ( SI i = 0; i < nu.n; ++i ) u[ i ] -= mu;
            cudaMemcpy( du, u.data(), nu.n * sizeof( double ), cudaMemcpyHostToDevice );
            // DIFFERENCE CENTREE : la troncature tombe de `eps` a `eps^2`, sinon le verdict se
            // confond avec elle ( une difference a droite donnait 1.6e-3 sur une hessienne juste )
            const double eps = 1e-9;
            std::vector<TF> wp( nu.n ), wm( nu.n );
            for ( SI i = 0; i < nu.n; ++i ) { wp[ i ] = TF( double( w[ i ] ) + eps * u[ i ] ); wm[ i ] = TF( double( w[ i ] ) - eps * u[ i ] ); }
            gpu::Hessienne H0;
            double mn = 0;
            g.tour_newton( w.data() ); g.assemble( H0 );
            g.tour_newton( wm.data() ); g.residu( cible, &mn, b0 );
            g.tour_newton( wp.data() ); g.residu( cible, &mn, b1 );
            g.applique( H0, du, dy );
            std::vector<double> v0( nu.n ), v1( nu.n ), lv( nu.n );
            cudaMemcpy( v0.data(), b0, nu.n * sizeof( double ), cudaMemcpyDeviceToHost );
            cudaMemcpy( v1.data(), b1, nu.n * sizeof( double ), cudaMemcpyDeviceToHost );
            cudaMemcpy( lv.data(), dy, nu.n * sizeof( double ), cudaMemcpyDeviceToHost );
            // `b` est `m` centre : la constante s'en va dans la difference, et `L u` est deja centre
            // JUGE EN QUANTILES, pas en max : sur 10^5 cellules il y en a toujours quelques-unes
            // au bord d'un changement de VOISINAGE, ou `m` n'est plus deux fois derivable et ou
            // une difference finie sur un `eps` fini ne peut pas rendre la derivee. C'est la meme
            // lecon que pour les `c_ij` -- juger une queue sur son maximum ne dit rien.
            double ech = 0, som = 0, somp = 0;
            std::vector<double> ecs( nu.n );
            for ( SI i = 0; i < nu.n; ++i ) {
                const double fd = ( v1[ i ] - v0[ i ] ) / ( 2 * eps );
                ech = std::max( ech, std::fabs( fd ) );
                som += std::fabs( fd + lv[ i ] ); somp += std::fabs( fd - lv[ i ] );
                ecs[ i ] = std::fabs( fd - lv[ i ] );
            }
            const bool plus = somp <= som;               // le signe de `d m / d w`
            if ( ! plus ) for ( SI i = 0; i < nu.n; ++i ) ecs[ i ] = std::fabs( ( v1[ i ] - v0[ i ] ) / ( 2 * eps ) + lv[ i ] );
            std::sort( ecs.begin(), ecs.end() );
            auto qq = [ & ]( double f ) { return ech > 0 ? ecs[ std::min( ecs.size() - 1, size_t( f * ecs.size() ) ) ] / ech : 0.0; };
            std::printf( "      NEWTON    d m / d w = %cL, verifie par difference finie centree : ecart median %.1e, p99 %.1e, p99.99 %.1e, max %.1e ( rapportes a | fd |_max %.2e )%s\n",
                         plus ? '+' : '-', qq( 0.5 ), qq( 0.99 ), qq( 0.9999 ), qq( 1.0 ), ech,
                         qq( 0.99 ) < 1e-5 ? "" : "   <-- LA HESSIENNE NE DERIVE PAS LES MESURES" );
            cudaFree( b0 ); cudaFree( b1 ); cudaFree( du ); cudaFree( dy );
        }

        const double tn0 = now();
        double t_diag = 0, t_lin = 0;
        int nb_diag = 0, nb_cg = 0;

        // ---- LA CONTINUATION EN CONTRASTE. Sous une image contrastee, le pas de Newton n'est pas
        //      borne par la decroissance du residu mais par LA MORT D'UNE CELLULE : a `t = 0.03`
        //      le residu baisse deja, et pourtant une cellule tombe a zero -- on perd un facteur
        //      trente sur le pas pour UNE cellule. Le remede est de ne pas attaquer l'image d'un
        //      coup : on resout sur `rho_s = ( 1 - s ) + s rho`, `s` montant de `1/K` a `1`, chaque
        //      etape partant des poids de la precedente. A `s = 0` c'est Lebesgue, ou tout va bien.
        auto charge_etape = [ & ]( double sx ) {
            if ( sx >= 1.0 ) { g.charge_image( img.v.data(), img.W, img.H ); return; }
            std::vector<double> t( img.v.size() );
            for ( size_t q = 0; q < t.size(); ++q ) t[ q ] = ( 1 - sx ) + sx * img.v[ q ];   // masse deja un
            g.charge_image( t.data(), img.W, img.H );
        };
        const int netapes = img.W ? std::max( 1, etapes ) : 1;

        double mini = 0, err = 0, t_prec = 1;
        const bool trace = std::getenv( "CHAINE_DEBUG" ) != nullptr;
        int it = 0;
        for ( int et = 1; et <= netapes; ++et ) {
        if ( img.W && netapes > 1 ) {
            charge_etape( double( et ) / netapes );
            std::printf( "      NEWTON    etape %d / %d : rho <- ( 1 - s ) + s rho, s = %.3f\n", et, netapes, double( et ) / netapes );
        }
        g.tour_newton( w.data() ); ++nb_diag;
        err = g.residu( cible, &mini, db );
        if ( et == 1 )
            std::printf( "      NEWTON    depart : residu %.3e, plus petite cellule %.2e de la cible\n",
                         err / norme, mini / cible );

        for ( int it_etape = 0; it < newton && err > tol * norme; ++it, ++it_etape ) {
            gpu::Hessienne H;
            const double tl0 = now();
            g.assemble( H );
            g.monte_amg( H );
            double ms_cg = 0, r_cg = 0;
            // NEWTON INEXACT : la tolerance du CG suit la convergence ( suite de forcage ). A
            // tolerance fixe, la direction devient du bruit des que le residu de Newton descend au
            // meme niveau -- et la recherche lineaire ne trouve plus de pas ( mesure : arret a
            // 1e-8 sur les nuages de lignes ).
            const double tol_cg = std::max( 1e-14, std::min( tolcg, 0.05 * err / norme ) );
            const int its = g.resout( H, db, dd, tol_cg, 20000, &ms_cg, &r_cg );
            t_lin += now() - tl0;
            nb_cg += its > 0 ? its : 0;
            cudaMemcpy( dh.data(), dd, nu.n * sizeof( double ), cudaMemcpyDeviceToHost );

            // l'essai d'un pas : le diagramme aux poids `w - t d`, puis les deux criteres.
            // On en tire AUSSI le nombre de cellules condamnees -- le diagramme est deja calcule,
            // le compte ne coute rien, et c'est lui qui dit si l'obstruction est locale.
            double e2 = 0, m2 = 0;
            int nc = 0, nv = 0;
            auto essai = [ & ]( double t ) {
                for ( SI i = 0; i < nu.n; ++i ) w2[ i ] = w[ i ] - t * dh[ i ];
                const double td0 = now();
                g.tour_newton( w2.data() );
                t_diag += now() - td0;
                ++nb_diag;
                e2 = g.residu( cible, &m2, nullptr, marge * mini, &nc, &nv );
                // `marge` : la plus petite cellule ne doit pas perdre plus qu'une fraction de ce
                // qu'elle vaut DEJA. Le critere KMT nu ( `marge = 0` ) autorise un pas qui la laisse
                // au bord du vide -- admissible, et desastreux pour l'iteration suivante ; une marge
                // ABSOLUE, elle, serait increvable au depart, ou les cellules sont deja minuscules.
                const bool vivant = m2 > 0 && m2 >= marge * mini, baisse = e2 <= ( 1 - t / 2 ) * err;
                if ( trace )
                    std::printf( "          essai t=%.3e : condamnees %7d dont %7d vides / %d, plus petite %.2e ( seuil %.2e ) %s, residu %.6e contre %.6e %s\n",
                                 t, nc, nv, int( nu.n ), m2 / cible, marge * mini / cible, vivant ? "ok" : "REFUS",
                                 e2 / norme, ( 1 - t / 2 ) * err / norme, baisse ? "ok" : "REFUS" );
                return vivant && baisse;
            };

            // L'ECHELLE LOGARITHMIQUE, une fois, pour voir la COURBE des condamnees : elle dit si
            // un pas est refuse pour trois cellules ou pour dix mille. ( `--echelle K` : les K
            // premieres iterations. Chaque barreau coute un diagramme, donc c'est un diagnostic. )
            if ( it_etape < echelle ) {
                // LES TROIS ECHELLES ( voir `BilanCel` ). Les comptes sont CUMULES : la colonne
                // « < f » donne le nombre de cellules SOUS la fraction `f`, ce qui se lit
                // directement comme « combien de cellules tomberaient sous f si je prenais ce pas ».
                g.garde_mesures();                       // l'etat de reference : avant le pas
                const gpu::BilanCel b0 = g.bilan( cible );
                std::printf( "      ECHELLE   etape %d, iteration %d : residu %.4e, mesures de %.2e a %.2e fois la cible\n",
                             et, it_etape + 1, err / norme, b0.mini / cible, b0.maxi / cible );
                std::printf( "                                                 = 0    < 1e-6    < 1e-5    < 1e-4    < 1e-3    < 1e-2    < 1e-1      < 1\n" );
                auto cumul = [ & ]( const char *nom, const long long *h ) {
                    std::printf( "                              %-12s", nom );
                    long long c = 0;
                    for ( int k = 0; k < 8; ++k ) { c += h[ k ]; std::printf( " %9lld", c ); }
                    std::printf( "\n" );
                };
                std::printf( "        t =          0   ( l'etat de depart )\n" );
                cumul( "m / cible", b0.par_cible );
                cumul( "m / max", b0.par_max );
                for ( double t = 1; t > 1e-7; t *= 0.5 ) {
                    essai( t );
                    const gpu::BilanCel b = g.bilan( cible );
                    std::printf( "        t = %10.3e   residu %.6e   plus petite %.2e%s\n",
                                 t, e2 / norme, m2 / cible,
                                 e2 <= ( 1 - t / 2 ) * err ? "   <- le residu, lui, baisse assez" : "" );
                    cumul( "m / cible", b.par_cible );
                    cumul( "m / max", b.par_max );
                    cumul( "m / m avant", b.par_ref );   // octaves : 1/64, 1/32, ... 1/2, 1
                    if ( nc == 0 ) break;
                }
                std::printf( "                ( les deux premieres lignes sont en DECADES, la troisieme en OCTAVES : 1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 1 )\n" );
            }

            // LE DEPART DE LA RECHERCHE : le pas precedent DOUBLE, pas `1`. Sous une image
            // contrastee le pas admissible est de l'ordre de `1e-3` et ne remonte que lentement ;
            // repartir de `1` a chaque fois coutait huit a treize diagrammes d'essai par iteration
            // pour retrouver ce qu'on savait deja.
            double t = std::min( 1.0, 2 * t_prec ), t_ok = 0;
            int essais = 0;
            for ( ; t > 1e-13; t *= 0.5 ) { ++essais; if ( essai( t ) ) { t_ok = t; break; } }
            if ( t_ok == 0 ) { std::printf( "      NEWTON    pas trouve, arret\n" ); et = netapes; break; }
            // LE MEILLEUR COEFFICIENT : entre le dernier refuse et le premier accepte
            double lo = t_ok, hi = std::min( 1.0, 2 * t_ok );
            for ( int k = 0; k < raff && t_ok < 1.0; ++k ) {
                const double mid = 0.5 * ( lo + hi );
                ++essais;
                if ( essai( mid ) ) { lo = mid; t_ok = mid; } else hi = mid;
            }
            t_prec = t_ok;
            for ( SI i = 0; i < nu.n; ++i ) w[ i ] -= t_ok * dh[ i ];
            const double td0 = now();
            g.tour_newton( w.data() );                   // l'etat doit finir sur le pas RETENU
            t_diag += now() - td0;
            ++nb_diag;
            err = g.residu( cible, &mini, db );
            std::printf( "      NEWTON %2d  pas %.4f ( %d essais ), CG %4d it., residu %.3e, plus petite %.2e\n",
                         it + 1, t_ok, essais, its, err / norme, mini / cible );
        }
        }
        const double t_tot = now() - tn0;
        std::printf( "      NEWTON    %d iterations, %.2f s ( %d diagrammes %.2f s, %d it. de CG %.2f s ), residu final %.3e%s\n",
                     it, t_tot, nb_diag, t_diag, nb_cg, t_lin, err / norme,
                     err <= tol * norme ? "" : "   <-- PAS CONVERGE" );
        cudaFree( db ); cudaFree( dd );
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
    int iterations = 1, newton = 0, raff = 0;
    double marge = 0.5, tol = 1e-7;
    Img img;
    int dmode = 3, chunk = 1 << 20;
    const char *pgm = nullptr;
    int taille_img = 0;
    double fmel = 1.0;
    int etapes = 1;
    double tolcg = 1e-2;
    int echelle = 0;
    for ( int i = 1; i < argc; ++i ) {
        const std::string s = argv[ i ];
        if ( a.parse( s, i, argc, argv ) ) continue;
        if ( s == "--reps-gpu" && i + 1 < argc ) { reps_gpu = std::atoi( argv[ ++i ] ); continue; }
        if ( s == "--arbre-gpu" ) { arbre_gpu = true; continue; }
        if ( s == "--iterations" && i + 1 < argc ) { iterations = std::atoi( argv[ ++i ] ); continue; }
        if ( s == "--newton" && i + 1 < argc ) { newton = std::atoi( argv[ ++i ] ); continue; }
        if ( s == "--raffine" && i + 1 < argc ) { raff = std::atoi( argv[ ++i ] ); continue; }
        if ( s == "--marge" && i + 1 < argc ) { marge = std::atof( argv[ ++i ] ); continue; }
        if ( s == "--tol" && i + 1 < argc ) { tol = std::atof( argv[ ++i ] ); continue; }
        if ( s == "--image" && i + 1 < argc ) { taille_img = std::atoi( argv[ ++i ] ); continue; }
        if ( s == "--image-pgm" && i + 1 < argc ) { pgm = argv[ ++i ]; continue; }
        if ( s == "--chunk" && i + 1 < argc ) { chunk = std::atoi( argv[ ++i ] ); continue; }
        if ( s == "--image-melange" && i + 1 < argc ) { fmel = std::atof( argv[ ++i ] ); continue; }
        if ( s == "--image-etapes" && i + 1 < argc ) { etapes = std::atoi( argv[ ++i ] ); continue; }
        if ( s == "--tol-cg" && i + 1 < argc ) { tolcg = std::atof( argv[ ++i ] ); continue; }
        if ( s == "--echelle" && i + 1 < argc ) { echelle = std::atoi( argv[ ++i ] ); continue; }
        if ( s == "--densite" && i + 1 < argc ) {
            const std::string m = argv[ ++i ];
            dmode = m == "directe" ? 0 : m == "depot" ? 1 : m == "depot-arete" ? 2 : 3;
            continue;
        }
        std::printf( "usage: chaine [options]\n" );
        Args::usage();
        std::printf( "  --reps-gpu R    repetitions du noyau GPU, minimum       (10)\n"
                     "  --arbre-gpu     construire l'arbre sur le GPU, et l'y garder\n"
                     "  --iterations K  le REGIME AMORTI : K tours de Newton apres une seule construction\n"
                     "  --newton K      le NEWTON COMPLET, au plus K iterations\n"
                     "  --raffine R     dichotomies pour le meilleur pas apres la premiere acceptation (0 : nuisible)\n"
                     "  --marge F       toute cellule doit garder F fois ce qu elle vaut deja (0.5)\n"
                     "  --tol T         residu relatif vise sur les mesures (1e-7)\n"
                     "  --image N       LA DENSITE IMAGE : une grille N x N de synthese ( fond lisse, disque net, bande fine )\n"
                     "  --image-pgm F   la meme, lue dans un PGM ( P2 ou P5 )\n"
                     "  --densite M     directe | depot | depot-arete | toutes  (toutes)\n"
                     "  --chunk C       cellules par lot dans les modes a depot (1048576)\n"
                     "  --image-melange F  rho <- ( 1 - F ) + F rho : le CONTRASTE, F = 0 rend Lebesgue (1)\n"
                     "  --image-etapes K   LA CONTINUATION : K etapes de s = 1/K a 1, chacune partant de la precedente (1)\n"
                     "  --tol-cg T      plafond de la suite de forcage du CG (1e-2)\n"
                     "  --echelle K     le balayage LOGARITHMIQUE du pas aux K premieres iterations : la courbe\n"
                     "                  des cellules condamnees, pour voir si l obstruction est locale\n" );
        return s == "--help" || s == "-h" ? 0 : 1;
    }
    a.dims = 2;
    a.finalise();
    if ( pgm ) {
        if ( ! lit_pgm( pgm, img ) ) { std::printf( "image illisible : %s\n", pgm ); return 1; }
    } else if ( taille_img > 0 ) {
        img = synthese( taille_img );
    }
    if ( img.W ) { normalise( img ); melange( img, fmel ); normalise( img ); img.prepare(); }
    std::printf( "GPU : %s\n", gpu::carte().c_str() );
    int bad = 0;
    for ( const Nuage<2> &nu : a.nuages<2>() ) {
        if ( nu.absent ) { std::printf( "  %-28s : ABSENT ( --cases DIR )\n", nu.nom.c_str() ); continue; }
        bad += dispatch<2>( a, [ & ]( auto tag ) { return chaine<typename decltype( tag )::type>( a, nu, reps_gpu, arbre_gpu, iterations, newton, raff, marge, tol, img, dmode, chunk, etapes, tolcg, echelle ); } );
    }
    return bad ? 1 : 0;
}

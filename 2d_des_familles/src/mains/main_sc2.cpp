// =====================================================================================
// LES SUR-CELLULES EN DEUX PHASES, DE BOUT EN BOUT.
//
// Phase 1 : chaque cellule contre son agregat et l'anneau 1. Les cellules obtenues enveloppent les
//           vraies -- somme des aires ~3 au lieu de 1.
// Enceinte : par agregat, la boite de ces cellules et le plus grand `|v - p_i|`.
// Table    : par agregat, les agregats de l'ANNEAU 2 ET AU-DELA qui peuvent encore le manger.
// Phase 2  : on repart de la cellule de phase 1 et on ajoute ce que la table designe.
//
// LE TEMOIN EST LA SOMME DES AIRES. Elle doit valoir 1 : si une seule coupe est ratee, une cellule
// deborde sur sa voisine et la somme monte. C'est le seul controle qui teste la COMPLETUDE, et
// c'est tout l'objet de la manoeuvre.
// =====================================================================================

#include "bench/Bench.h"
#include "spatial_accel/AaBsp.h"
#include "supercell/Agregats.h"
#include "supercell/FournisseurSC.h"
#include "supercell/FournisseurSC2.h"
#include "supercell/Noyau2DEtats.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

using namespace pd;
using namespace pd::bench;
using namespace pd::supercell;

static double maintenant() {
    using namespace std::chrono;
    return duration<double>( steady_clock::now().time_since_epoch() ).count();
}

static constexpr int MAXNB = 64;

static inline unsigned etale( unsigned x ) {
    x &= 0xffffu; x = ( x | ( x << 8 ) ) & 0x00ff00ffu; x = ( x | ( x << 4 ) ) & 0x0f0f0f0fu;
    x = ( x | ( x << 2 ) ) & 0x33333333u; x = ( x | ( x << 1 ) ) & 0x55555555u; return x;
}
static inline unsigned morton2( double x, double y ) {
    return etale( (unsigned) std::min( 65535.0, std::max( 0.0, x * 65535.0 ) ) )
       | ( etale( (unsigned) std::min( 65535.0, std::max( 0.0, y * 65535.0 ) ) ) << 1 );
}

/// MORTON LOCAL : les membres de chaque agregat, tries entre eux. `etape1` les laisse dans l'ordre
/// de generation ( coherence spatiale mesuree : 0.973, c'est a dire aucune ) ; apres tri, 0.652.
static void trie_interne_morton( const Cloud<2> &cl, Gros<2> &G ) {
    std::vector<SI> inv( G.mem.begin(), G.mem.end() );
    std::vector<SI> med( G.ns, -1 );
    for ( SI a = 0; a < G.ns; ++a ) if ( G.median[ a ] >= 0 ) med[ a ] = inv[ G.median[ a ] ];
    for ( SI a = 0; a < G.ns; ++a )
        std::sort( G.mem.begin() + G.mdeb[ a ], G.mem.begin() + G.mdeb[ a + 1 ],
                   [ & ]( SI x, SI y ) {
                       return morton2( cl.P[0][x], cl.P[1][x] ) < morton2( cl.P[0][y], cl.P[1][y] ); } );
    for ( int d = 0; d < 2; ++d ) {
        for ( SI k = 0; k < cl.n; ++k ) G.Pp[ d ][ k ] = cl.P[ d ][ G.mem[ k ] ];
        G.Ppp[ d ] = G.Pp[ d ].data();
    }
    if ( G.pese ) for ( SI k = 0; k < cl.n; ++k ) G.Wp[ k ] = cl.W[ G.mem[ k ] ];
    std::vector<SI> pos( cl.n );
    for ( SI k = 0; k < cl.n; ++k ) pos[ G.mem[ k ] ] = k;
    for ( SI a = 0; a < G.ns; ++a ) if ( med[ a ] >= 0 ) G.median[ a ] = pos[ med[ a ] ];
}

template<bool POIDS>
static int deroule( const Cloud<2> &cl, Gros<2> &G, int n, bool morton_local ) {
    if ( morton_local ) trie_interne_morton( cl, G );

    // ---------------- PHASE 1
    std::vector<noyau2d::Atelier<MAXNB>> cel( n );
    double t = maintenant();
    for ( SI ag = 0; ag < G.ns; ++ag )
    for ( SI i = G.mdeb[ ag ]; i < G.mdeb[ ag + 1 ]; ++i ) {
        noyau2d::FournisseurSC1<2,POIDS> f( &G, (int) ag, (int) i );
        noyau2d::etats::moteur( &f, &cel[ i ] );
    }
    const double t1 = maintenant() - t;

    double s1 = 0;
    for ( int i = 0; i < n; ++i ) {
        const auto &c = cel[ i ];
        for ( int v = 0; v < c.nb; ++v ) { const int w = v + 1 < c.nb ? v + 1 : 0;
            s1 += (double) c.vx[v] * c.vy[w] - (double) c.vx[w] * c.vy[v]; }
    }

    // ---------------- LES ENCEINTES
    t = maintenant();
    std::vector<noyau2d::Enceinte> enc( G.ns );
    for ( SI a = 0; a < G.ns; ++a ) {
        auto &e = enc[ a ];
        e.lo[0] = e.lo[1] = 1e30f; e.hi[0] = e.hi[1] = -1e30f; e.r2 = 0;
        for ( SI i = G.mdeb[ a ]; i < G.mdeb[ a + 1 ]; ++i ) {
            const auto &c = cel[ i ];
            const float px = (float) G.Ppp[0][i], py = (float) G.Ppp[1][i];
            for ( int v = 0; v < c.nb; ++v ) {
                e.lo[0] = std::min( e.lo[0], c.vx[v] ); e.hi[0] = std::max( e.hi[0], c.vx[v] );
                e.lo[1] = std::min( e.lo[1], c.vy[v] ); e.hi[1] = std::max( e.hi[1], c.vy[v] );
                const float ex = c.vx[v] - px, ey = c.vy[v] - py;
                e.r2 = std::max( e.r2, ex * ex + ey * ey );
            }
        }
    }
    const double t_enc = maintenant() - t;

    // ---------------- LA TABLE : anneau 2 et au-dela
    t = maintenant();
    std::vector<double> px( n ), py( n ), pw( n );
    for ( int i = 0; i < n; ++i ) { px[i] = G.Ppp[0][i]; py[i] = G.Ppp[1][i];
                                    pw[i] = POIDS ? G.Wp[i] : 0.0; }
    AaBspT<2> arbre;
    arbre.build( px.data(), py.data(), nullptr, n, 10 );
    std::vector<int> agr_de( n );                        // indice permute -> agregat
    for ( SI a = 0; a < G.ns; ++a )
        for ( SI i = G.mdeb[ a ]; i < G.mdeb[ a + 1 ]; ++i ) agr_de[ i ] = (int) a;

    noyau2d::TableSC T;
    T.deb.assign( G.ns + 1, 0 );
    T.lo0.assign( G.ns, 1e30f ); T.lo1.assign( G.ns, 1e30f );
    T.hi0.assign( G.ns, -1e30f ); T.hi1.assign( G.ns, -1e30f );
    T.wmax.assign( G.ns, 0.f );
    for ( SI a = 0; a < G.ns; ++a )
        for ( SI i = G.mdeb[ a ]; i < G.mdeb[ a + 1 ]; ++i ) {
            T.lo0[a] = std::min( T.lo0[a], (float) G.Ppp[0][i] );
            T.hi0[a] = std::max( T.hi0[a], (float) G.Ppp[0][i] );
            T.lo1[a] = std::min( T.lo1[a], (float) G.Ppp[1][i] );
            T.hi1[a] = std::max( T.hi1[a], (float) G.Ppp[1][i] );
            if constexpr ( POIDS ) T.wmax[a] = std::max( T.wmax[a], (float) G.Wp[i] );
        }
    std::vector<int> vu( G.ns, -1 ), proche( G.ns, -1 );
    std::vector<int> pile( 64 );
    for ( SI a = 0; a < G.ns; ++a ) {
        // l'anneau 0 et 1 sont deja dans la cellule : on les marque pour les sauter
        proche[ a ] = (int) a;
        for ( SI q = G.adeb[ a ]; q < G.adeb[ a + 1 ]; ++q ) proche[ G.adj[ q ] ] = (int) a;
        const auto &e = enc[ a ];
        int haut = 0; pile[ haut++ ] = 0;
        while ( haut > 0 ) {
            const int h = pile[ --haut ];
            const auto &nd = arbre.nodes[ h ];
            float d2 = 0;                                // dist^2( enceinte, boite du noeud )
            for ( int d = 0; d < 2; ++d ) {
                const float lo = (float) nd.lo[d], hi = (float) nd.hi[d];
                const float t2 = e.hi[d] < lo ? lo - e.hi[d] : ( e.lo[d] > hi ? e.lo[d] - hi : 0.f );
                d2 += t2 * t2;
            }
            if ( d2 >= e.r2 ) continue;                  // aucun germe de ce sous-arbre ne peut rien
            if ( nd.right < 0 ) {
                for ( SI k = nd.beg; k < nd.end; ++k ) {
                    const int b = agr_de[ (int) arbre.order[ k ] ];
                    if ( proche[ b ] == (int) a || vu[ b ] == (int) a ) continue;
                    vu[ b ] = (int) a;
                    T.ag.push_back( b );
                }
                continue;
            }
            if ( haut + 2 > (int) pile.size() ) pile.resize( pile.size() * 2 );
            pile[ haut++ ] = (int) nd.right; pile[ haut++ ] = h + 1;
        }
        T.deb[ a + 1 ] = (int) T.ag.size();
    }
    const double t_tab = maintenant() - t;

    // ---------------- PHASE 2
    t = maintenant();
    for ( SI a = 0; a < G.ns; ++a )
    for ( SI i = G.mdeb[ a ]; i < G.mdeb[ a + 1 ]; ++i ) {
        noyau2d::FournisseurSC2<2,POIDS> f( &G, &T, (int) a, (int) i );
        noyau2d::etats::moteur_depuis( &f, &cel[ i ] );
    }
    const double t2 = maintenant() - t;

    double s2 = 0; long long cotes = 0;
    for ( int i = 0; i < n; ++i ) {
        const auto &c = cel[ i ];
        if ( c.nb <= 0 ) continue;
        cotes += c.nb;
        for ( int v = 0; v < c.nb; ++v ) { const int w = v + 1 < c.nb ? v + 1 : 0;
            s2 += (double) c.vx[v] * c.vy[w] - (double) c.vx[w] * c.vy[v]; }
    }

    printf( "  phase 1  %7.4f s | somme %.4f  ( enveloppe, donc > 1 )\n", t1, 0.5 * s1 );
    printf( "  enceintes%7.4f s | table %7.4f s | %.2f agregats par ligne ( anneau >= 2 )\n",
            t_enc, t_tab, double( T.ag.size() ) / G.ns );
    printf( "  phase 2  %7.4f s | somme %.9f  <- doit valoir 1 | %.2f cotes\n",
            t2, 0.5 * s2, double( cotes ) / n );
    printf( "  TOTAL    %7.4f s  ( %.1f ns/germe )\n",
            t1 + t_enc + t_tab + t2, ( t1 + t_enc + t_tab + t2 ) * 1e9 / n );
    return 0;
}

int main( int argc, char **argv ) {
    int n = 200000; SI rho = 8; double frac = 0; bool morton_local = true;
    for ( int i = 1; i + 1 < argc; i += 2 ) {
        const std::string o = argv[ i ];
        if      ( o == "-n"        ) n = atoi( argv[ i + 1 ] );
        else if ( o == "--rho"     ) rho = atoi( argv[ i + 1 ] );
        else if ( o == "--weights" ) frac = atof( argv[ i + 1 ] );
        else if ( o == "--morton-local" ) morton_local = atoi( argv[ i + 1 ] ) != 0;
        else { fprintf( stderr, "option inconnue %s\n", argv[ i ] ); return 1; }
    }
    Cloud<2> cl;
    cl.nom = "uniforme"; cl.n = n;
    cl.c[0].resize( n ); cl.c[1].resize( n ); cl.w.assign( n, 0 );
    std::mt19937 gen( 12345 );
    std::uniform_real_distribution<double> u( 0, 1 );
    const double h2 = 1.0 / n;
    for ( int i = 0; i < n; ++i ) {
        cl.c[0][i] = u( gen ); cl.c[1][i] = u( gen ); cl.w[i] = frac * h2 * u( gen );
    }
    cl.P[0] = cl.c[0].data(); cl.P[1] = cl.c[1].data();
    cl.W = frac > 0 ? cl.w.data() : nullptr;

    Gros<2> G;
    const BilanGros b = cl.W ? etape1<2, true >( cl, rho, true, 10, G )
                             : etape1<2, false>( cl, rho, true, 10, G );
    printf( "n = %d, rho = %d, %s, morton local %s | %lld agregats, degre %.2f\n",
            n, (int) rho, frac > 0 ? "Laguerre" : "Voronoi", morton_local ? "oui" : "non",
            (long long) b.ns, b.deg );
    return frac > 0 ? deroule<true>( cl, G, n, morton_local )
                    : deroule<false>( cl, G, n, morton_local );
}

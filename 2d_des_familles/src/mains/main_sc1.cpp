// =====================================================================================
// LA PHASE 1 DES SUR-CELLULES, MESUREE -- et ce qu'on mesure d'abord n'est pas le temps.
//
// Le noyau tient huit sommets dans des registres. L'etape 4 d'origine coupe PAR AGREGAT, donc les
// plans arrivent groupes par voisin et non par distance, et la cellule traverse des etats
// transitoires bien plus larges que sa forme finale -- c'est ce qui force `CellSC` a 256 sommets
// en 3D. Si ce transitoire depasse huit la moitie du temps, le SIMD ne sert plus a rien.
//
// LE CHIFFRE QUI DECIDE est donc le PIRE ETAT INTERMEDIAIRE, pas la seconde. On le mesure avec un
// espion qui regarde passer l'`Etat`, et on compare deux ordres :
//
//   `--ordre-sortant 0`  l'agregat dans l'ordre memoire, puis l'anneau 1 dans l'ordre de la
//                        connectivite grossiere -- ce que fait l'etape 4 aujourd'hui
//   `--ordre-sortant 1`  l'agregat en s'ECARTANT du germe, puis l'anneau 1 par distance
//
// La somme des aires est ATTENDUE SUPERIEURE A 1 : la phase 1 ne coupe qu'avec l'agregat et
// l'anneau 1, donc les cellules obtenues enveloppent les vraies. C'est la sur-cellule, et l'ecart
// a 1 dit ce qu'il reste a faire a la phase 2.
// =====================================================================================

#include "bench/Bench.h"
#include "supercell/Agregats.h"
#include "supercell/FournisseurSC.h"
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

/// ---- LE CODE DE MORTON, seize bits par axe, entrelaces.
static inline unsigned etale( unsigned x ) {
    x &= 0xffffu; x = ( x | ( x << 8 ) ) & 0x00ff00ffu; x = ( x | ( x << 4 ) ) & 0x0f0f0f0fu;
    x = ( x | ( x << 2 ) ) & 0x33333333u; x = ( x | ( x << 1 ) ) & 0x55555555u; return x;
}
static inline unsigned morton2( double x, double y ) {
    return etale( (unsigned) std::min( 65535.0, std::max( 0.0, x * 65535.0 ) ) )
       | ( etale( (unsigned) std::min( 65535.0, std::max( 0.0, y * 65535.0 ) ) ) << 1 );
}

/// LA COHERENCE SPATIALE DE L'ORDRE INTERNE : distance moyenne entre membres CONSECUTIFS en
/// indice, rapportee a la distance moyenne entre deux membres pris au hasard dans le meme
/// agregat. Un rapport de 1 dit que l'ordre des indices ne porte AUCUNE information spatiale --
/// et alors un parcours « en s'ecartant » ne vaut pas mieux qu'un parcours sequentiel.
static double coherence( const Gros<2> &G ) {
    double consec = 0, quelconque = 0; long long nc = 0, nq = 0;
    for ( SI a = 0; a < G.ns; ++a ) {
        const SI d0 = G.mdeb[ a ], d1 = G.mdeb[ a + 1 ];
        for ( SI u = d0; u + 1 < d1; ++u ) {
            const double ex = G.Ppp[0][u+1] - G.Ppp[0][u], ey = G.Ppp[1][u+1] - G.Ppp[1][u];
            consec += std::sqrt( ex * ex + ey * ey ); ++nc;
        }
        for ( SI u = d0; u < d1; ++u ) for ( SI v = u + 1; v < d1; ++v ) {
            const double ex = G.Ppp[0][v] - G.Ppp[0][u], ey = G.Ppp[1][v] - G.Ppp[1][u];
            quelconque += std::sqrt( ex * ex + ey * ey ); ++nq;
        }
    }
    return ( consec / std::max( 1LL, nc ) ) / ( quelconque / std::max( 1LL, nq ) );
}

/// TRIE LES MEMBRES DE CHAQUE AGREGAT EN MORTON, et refait tout ce qui en depend.
///
/// `etape1` remplit `mem` en balayant le nuage dans son ordre D'ORIGINE : les membres d'un
/// agregat s'y retrouvent donc dans l'ordre de generation, sans rapport avec l'espace. Le tri
/// interne est ce qui donne un sens a « i+1, i-1, i+2... ».
static void trie_interne_morton( const Cloud<2> &cl, Gros<2> &G ) {
    std::vector<SI> inv( cl.n );                         // permute -> original, avant remaniement
    for ( SI k = 0; k < cl.n; ++k ) inv[ k ] = G.mem[ k ];
    std::vector<SI> med_orig( G.ns, -1 );
    for ( SI a = 0; a < G.ns; ++a ) if ( G.median[ a ] >= 0 ) med_orig[ a ] = inv[ G.median[ a ] ];

    for ( SI a = 0; a < G.ns; ++a ) {
        const SI d0 = G.mdeb[ a ], d1 = G.mdeb[ a + 1 ];
        std::sort( G.mem.begin() + d0, G.mem.begin() + d1, [ & ]( SI x, SI y ) {
            return morton2( cl.P[0][x], cl.P[1][x] ) < morton2( cl.P[0][y], cl.P[1][y] );
        } );
    }
    for ( int d = 0; d < 2; ++d ) {
        for ( SI k = 0; k < cl.n; ++k ) G.Pp[ d ][ k ] = cl.P[ d ][ G.mem[ k ] ];
        G.Ppp[ d ] = G.Pp[ d ].data();
    }
    if ( G.pese ) for ( SI k = 0; k < cl.n; ++k ) G.Wp[ k ] = cl.w.empty() ? 0 : cl.W[ G.mem[ k ] ];
    std::vector<SI> pos( cl.n );
    for ( SI k = 0; k < cl.n; ++k ) pos[ G.mem[ k ] ] = k;
    for ( SI a = 0; a < G.ns; ++a ) if ( med_orig[ a ] >= 0 ) G.median[ a ] = pos[ med_orig[ a ] ];
}

/// il ne change rien a ce qui est propose : il regarde passer l'etat.
template<class F>
struct Espion {
    using Local = noyau2d::Local<F>;
    F f;
    int *pire;
    long long *cand;
    long long *hors;                                     ///< coupes tentees pendant une EXCURSION
    double *somme_nb;                                    ///< taille moyenne de la cellule en cours
    template<class Etat> bool suivant( const Etat &e, Local &l, noyau2d::Plan &p ) {
        if ( e.nb > *pire ) *pire = e.nb;
        if ( e.nb > 8 ) ++*hors;                         // c'est CA que coute un mauvais ordre :
        *somme_nb += e.nb;                               // le noyau est sorti de ses registres
        const bool r = f.suivant( e, l, p );
        *cand += r;
        return r;
    }
};

/// la variante « ordre memoire » : l'agregat puis l'anneau 1, sans rien trier.
template<int D, bool POIDS>
struct FournisseurSC1Brut {
    struct Local { bool amorce = false; int k = 0, fin = 0, ia = 0, na = 0; int ann[ 24 ]; };
    const Gros<D> *G; int i0, a0; float x0, y0, w0 = 0;
    FournisseurSC1Brut( const Gros<D> *G, int a0, int i0 )
        : G( G ), i0( i0 ), a0( a0 ),
          x0( (float) G->Ppp[0][i0] ), y0( (float) G->Ppp[1][i0] ),
          w0( POIDS ? (float) G->Wp[i0] : 0.f ) {}
    template<class Etat> bool suivant( const Etat &, Local &l, noyau2d::Plan &p ) {
        if ( ! l.amorce ) {
            l.amorce = true;
            l.k = (int) G->mdeb[ a0 ]; l.fin = (int) G->mdeb[ a0 + 1 ];
            l.na = 0;
            for ( SI q = G->adeb[ a0 ]; q < G->adeb[ a0 + 1 ] && l.na < 24; ++q )
                if ( (int) G->adj[ q ] != a0 ) l.ann[ l.na++ ] = (int) G->adj[ q ];
        }
        for ( ;; ) {
            if ( l.k < l.fin ) {
                const int j = l.k++;
                if ( j == i0 ) continue;
                const float xj = (float) G->Ppp[0][j], yj = (float) G->Ppp[1][j];
                p.dx = xj - x0; p.dy = yj - y0;
                p.off = 0.5f * ( p.dx * ( xj + x0 ) + p.dy * ( yj + y0 ) );
                if constexpr ( POIDS ) p.off += 0.5f * ( w0 - (float) G->Wp[j] );
                p.id = j;
                return true;
            }
            if ( l.ia >= l.na ) return false;
            const int b = l.ann[ l.ia++ ];
            l.k = (int) G->mdeb[ b ]; l.fin = (int) G->mdeb[ b + 1 ];
        }
    }
};

template<class Fourn>
static void passe( const Gros<2> &G, int n, const char *nom, int reps ) {
    double best = 1e30, aire = 0;
    int pire = 0; long long cand = 0, cotes = 0, excur = 0; double moy = 0;
    for ( int r = 0; r < reps; ++r ) {
        double som = 0, snb = 0; long long c = 0, e = 0, nc = 0, nh = 0; int pi = 0;
        const double t0 = maintenant();
        // `ag` et non `a` : l'accumulateur d'aire s'appelait `a`, et l'indice d'agregat l'avait
        // MASQUE -- chaque terme du lacet ecrivait dans le compteur de boucle, qui repartait en
        // arriere. La boucle ne finissait jamais, et aucun garde-fou pose dans le noyau ne pouvait
        // le voir puisque le noyau, lui, allait bien.
        for ( SI ag = 0; ag < G.ns; ++ag )
        for ( SI i = G.mdeb[ ag ]; i < G.mdeb[ ag + 1 ]; ++i ) {
            noyau2d::Atelier<MAXNB> at;
            Espion<Fourn> f{ Fourn( &G, (int) ag, (int) i ), &pi, &nc, &nh, &snb };
            noyau2d::etats::moteur( &f, &at );
            if ( at.nb <= 0 ) continue;
            if ( at.nb > 8 ) ++e;
            c += at.nb;
            for ( int v = 0; v < at.nb; ++v ) { const int w = v + 1 < at.nb ? v + 1 : 0;
                som += (double) at.vx[v] * at.vy[w] - (double) at.vx[w] * at.vy[v]; }
        }
        const double dt = maintenant() - t0;
        if ( dt < best ) { best = dt; aire = 0.5 * som; cotes = c; excur = nh; cand = nc; pire = pi;
                           moy = snb / std::max( 1LL, nc ); }
    }
    printf( "  %-22s %7.4f s  %6.1f ns/germe | pire %3d | nb moyen %5.2f"
            " | %6.2f %% des coupes hors registres | somme %.4f\n",
            nom, best, best * 1e9 / n, pire, moy, 100.0 * excur / std::max( 1LL, cand ), aire );
}

/// LE STOCKAGE CSR DES CELLULES DE PHASE 1, et sa verification.
///
/// On ne garde que la CONNECTIVITE : `nb` entiers par cellule, pas un point. La geometrie se
/// refabrique -- le sommet `i` est l'intersection des coupes `i-1` et `i`. Ce qu'on verifie ici
/// est que la cellule reconstruite est bien celle qu'on avait, parce que si ce n'est pas vrai
/// toute la phase 2 repart d'un mensonge.
template<bool POIDS>
static void verifie_csr( const Gros<2> &G, int n ) {
    std::vector<int> deb( n + 1, 0 ), plat;
    std::vector<float> vx0( (size_t) n * 16 ), vy0( (size_t) n * 16 );
    std::vector<int> nb0( n, 0 );
    plat.reserve( (size_t) n * 8 );

    // --- phase 1, en gardant les `cid`
    for ( SI ag = 0; ag < G.ns; ++ag )
    for ( SI i = G.mdeb[ ag ]; i < G.mdeb[ ag + 1 ]; ++i ) {
        noyau2d::Atelier<MAXNB> at;
        noyau2d::FournisseurSC1<2,POIDS> f( &G, (int) ag, (int) i );
        noyau2d::etats::moteur( &f, &at );
        nb0[ i ] = at.nb;
        if ( at.nb > 0 && at.nb <= 16 )
            for ( int v = 0; v < at.nb; ++v ) {
                vx0[ (size_t) i * 16 + v ] = at.vx[ v ];
                vy0[ (size_t) i * 16 + v ] = at.vy[ v ];
                plat.push_back( at.cid[ v ] );
            }
        deb[ i + 1 ] = (int) plat.size();
    }

    // --- reconstruction depuis les seuls `cid`
    double pire = 0; int mauvais = 0, faits = 0;
    for ( SI ag = 0; ag < G.ns; ++ag )
    for ( SI i = G.mdeb[ ag ]; i < G.mdeb[ ag + 1 ]; ++i ) {
        const int nb = deb[ i + 1 ] - deb[ i ];
        if ( nb < 3 ) continue;
        const float x0 = (float) G.Ppp[0][i], y0 = (float) G.Ppp[1][i];
        const float w0 = POIDS ? (float) G.Wp[i] : 0.f;
        alignas( 32 ) float rx[ 16 ], ry[ 16 ];
        noyau2d::reconstruit<16>( rx, ry, &plat[ deb[ i ] ], nb,
            [ & ]( int id, float &dx, float &dy, float &off ) {
                if ( id < 0 ) { noyau2d::plan_domaine( id, dx, dy, off ); return; }
                const float xj = (float) G.Ppp[0][id], yj = (float) G.Ppp[1][id];
                dx = xj - x0; dy = yj - y0;
                off = 0.5f * ( dx * ( xj + x0 ) + dy * ( yj + y0 ) );
                if constexpr ( POIDS ) off += 0.5f * ( w0 - (float) G.Wp[id] );
            } );
        ++faits;
        double m = 0;
        for ( int v = 0; v < nb; ++v ) {
            const double ex = rx[ v ] - vx0[ (size_t) i * 16 + v ];
            const double ey = ry[ v ] - vy0[ (size_t) i * 16 + v ];
            m = std::max( m, std::sqrt( ex * ex + ey * ey ) );
        }
        if ( m > 1e-5 ) ++mauvais;
        pire = std::max( pire, m );
    }

    const double o_csr = ( double( plat.size() ) * 4 + double( n + 1 ) * 4 ) / n;
    printf( "\n  CSR : %.1f octets/dirac ( %d entiers au total ) | %d cellules refaites"
            " | ecart max %.2e | %d au-dela de 1e-5\n",
            o_csr, (int) plat.size(), faits, pire, mauvais );
    printf( "  ( pour memoire : le masque de bits de `Memoire.h` coute 8 o/dirac,"
            " mais oblige a refaire la meme liste )\n" );
}

int main( int argc, char **argv ) {
    int n = 200000, reps = 3; SI rho = 8; double frac = 0; bool morton = true;
    bool tri_interne = false;
    for ( int i = 1; i + 1 < argc; i += 2 ) {
        const std::string o = argv[ i ];
        if      ( o == "-n"        ) n      = atoi( argv[ i + 1 ] );
        else if ( o == "--reps"    ) reps   = atoi( argv[ i + 1 ] );
        else if ( o == "--rho"     ) rho    = atoi( argv[ i + 1 ] );
        else if ( o == "--weights" ) frac   = atof( argv[ i + 1 ] );
        else if ( o == "--ordre"   ) morton = std::string( argv[ i + 1 ] ) == "morton";
        else if ( o == "--tri-interne" ) tri_interne = atoi( argv[ i + 1 ] ) != 0;
        else { fprintf( stderr, "option inconnue %s\n", argv[ i ] ); return 1; }
    }

    Cloud<2> cl;
    cl.nom = "uniforme"; cl.n = n;
    cl.c[ 0 ].resize( n ); cl.c[ 1 ].resize( n ); cl.w.assign( n, 0 );
    std::mt19937 gen( 12345 );
    std::uniform_real_distribution<double> u( 0, 1 );
    const double h2 = 1.0 / n;
    for ( int i = 0; i < n; ++i ) {
        cl.c[ 0 ][ i ] = u( gen ); cl.c[ 1 ][ i ] = u( gen );
        cl.w[ i ] = frac * h2 * u( gen );
    }
    cl.P[ 0 ] = cl.c[ 0 ].data(); cl.P[ 1 ] = cl.c[ 1 ].data();
    cl.W = frac > 0 ? cl.w.data() : nullptr;

    Gros<2> G;
    const BilanGros b = cl.W ? etape1<2, true >( cl, rho, morton, 10, G )
                             : etape1<2, false>( cl, rho, morton, 10, G );
    printf( "n = %d, rho = %d, ordre %s, %s | %lld agregats, |A| median %lld, degre %.2f\n",
            n, (int) rho, morton ? "morton" : "bsp", frac > 0 ? "Laguerre" : "Voronoi",
            (long long) b.ns, (long long) b.amed, b.deg );
    printf( "  coherence de l'ordre interne : %.3f  ( 1.0 = aucune information spatiale )\n",
            coherence( G ) );
    if ( tri_interne ) {
        trie_interne_morton( cl, G );
        printf( "  apres tri Morton interne     : %.3f\n", coherence( G ) );
    }
    printf( "  ( somme attendue > 1 : la phase 1 enveloppe les vraies cellules )\n\n" );

    if ( frac > 0 ) {
        passe<FournisseurSC1Brut<2,true>>( G, n, "ordre memoire", reps );
        passe<noyau2d::FournisseurSC1<2,true>>( G, n, "ordre sortant + trie", reps );
    } else {
        passe<FournisseurSC1Brut<2,false>>( G, n, "ordre memoire", reps );
        passe<noyau2d::FournisseurSC1<2,false>>( G, n, "ordre sortant + trie", reps );
    }
    if ( frac > 0 ) verifie_csr<true >( G, n );
    else            verifie_csr<false>( G, n );
    return 0;
}

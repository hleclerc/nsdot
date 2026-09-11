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
#include "supercell/FournisseurBsp.h"
#include "supercell/EnceinteDop.h"
#include "supercell/FournisseurSC2.h"
#include "supercell/Noyau2DEtats.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cmath>
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

template<bool POIDS, class ENC>
static int deroule( const Cloud<2> &cl, Gros<2> &G, int n, bool morton_local, bool verif, int cap,
                    bool filtre1 ) {
    if ( morton_local ) trie_interne_morton( cl, G );

    // ---------------- LES BOITES DE GERMES, PAR AGREGAT
    //
    // Elles ne dependent que des germes, donc elles sont pretes AVANT la phase 1 et les deux phases
    // s'en servent : la phase 1 pour filtrer l'anneau 1, la phase 2 pour filtrer la ligne de table.
    double t = maintenant();
    noyau2d::BoitesAgregats bo( G.ns );
    for ( SI a = 0; a < G.ns; ++a ) {
        auto &B = bo[ a ];
        B.vide();
        for ( SI i = G.mdeb[ a ]; i < G.mdeb[ a + 1 ]; ++i )
            B.ajoute( (float) G.Ppp[0][i], (float) G.Ppp[1][i] );
        if constexpr ( POIDS ) {
            // le MEME majorant que l'arbre BSP, sur la tranche de l'agregat : `weight_majorant`
            // choisit lui-meme entre l'affine ajuste et le constant.
            const WMaj wm = weight_majorant<2>( G.mdeb[a], G.mdeb[a+1],
                [ & ]( SI k, Vec<2> &y, TF &w ) { y[0] = G.Ppp[0][k]; y[1] = G.Ppp[1][k];
                                                  w = G.Wp[k]; } );
            B.a[0] = wm.a[0]; B.a[1] = wm.a[1]; B.b = (float) wm.b;
        }
    }
    const double t_bo = maintenant() - t;

    // ---------------- PHASE 1
    std::vector<noyau2d::Atelier<MAXNB>> cel( n );
    t = maintenant();
    for ( SI ag = 0; ag < G.ns; ++ag )
    for ( SI i = G.mdeb[ ag ]; i < G.mdeb[ ag + 1 ]; ++i ) {
        noyau2d::FournisseurSC1<2,POIDS> f( &G, filtre1 ? &bo : nullptr, (int) ag, (int) i );
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
    std::vector<ENC> enc( G.ns );
    for ( SI a = 0; a < G.ns; ++a ) {
        auto &e = enc[ a ];
        float th = 0;
        if constexpr ( ENC::tourne ) {
            // L'AXE PRINCIPAL DES SOMMETS, en deux passes : l'orientation doit etre connue avant
            // de projeter. Les moments sont centres pour ne pas etre domines par la position.
            double sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0; long m = 0;
            for ( SI i = G.mdeb[a]; i < G.mdeb[a+1]; ++i ) { const auto &c = cel[i];
                for ( int v = 0; v < c.nb; ++v ) { sx += c.vx[v]; sy += c.vy[v]; ++m; } }
            if ( m > 0 ) {
                const double mx = sx / m, my = sy / m;
                for ( SI i = G.mdeb[a]; i < G.mdeb[a+1]; ++i ) { const auto &c = cel[i];
                    for ( int v = 0; v < c.nb; ++v ) {
                        const double ex = c.vx[v] - mx, ey = c.vy[v] - my;
                        sxx += ex * ex; syy += ey * ey; sxy += ex * ey; } }
                th = ENC::axe_principal( (float) sxx, (float) syy, (float) sxy );
            }
        }
        e.oriente( th );
        e.vide();
        for ( SI i = G.mdeb[ a ]; i < G.mdeb[ a + 1 ]; ++i ) {
            const auto &c = cel[ i ];
            const float px = (float) G.Ppp[0][i], py = (float) G.Ppp[1][i];
            const float wi = POIDS ? (float) G.Wp[i] : 0.f;
            for ( int v = 0; v < c.nb; ++v ) {
                e.ajoute( c.vx[v], c.vy[v] );
                const float ex = c.vx[v] - px, ey = c.vy[v] - py;
                e.r2 = std::max( e.r2, ex * ex + ey * ey - wi );
            }
        }
    }
    const double t_enc = maintenant() - t;

    // OU EST LE MOU ? Trois nombres par agregat, moyennes ensuite :
    //   `aire( E_A ) / somme des aires de phase 1`  ce que la FORME de l'enceinte coute
    //   `sqrt( r2 ) / h`                            ce que la PORTEE coute, en pas de germe
    // Si le premier est proche de 1, resserrer la forme ne rendra rien.
    {
        const double h = 1.0 / std::sqrt( (double) n );
        double q_forme = 0, q_portee = 0; int nq = 0;
        for ( SI a = 0; a < G.ns; ++a ) {
            const auto &e = enc[ a ];
            double u = 0;
            for ( SI i = G.mdeb[a]; i < G.mdeb[a+1]; ++i ) { const auto &c = cel[i];
                double s = 0;
                for ( int v = 0; v < c.nb; ++v ) { const int w = v+1<c.nb?v+1:0;
                    s += (double) c.vx[v]*c.vy[w] - (double) c.vx[w]*c.vy[v]; }
                u += 0.5 * ( s < 0 ? -s : s ); }
            if ( u <= 0 ) continue;
            q_forme += ( (double) e.hi[0] - e.lo[0] ) * ( (double) e.hi[ ENC::nb_dirs/2 ] - e.lo[ ENC::nb_dirs/2 ] ) / u;
            q_portee += std::sqrt( std::max( 0.0f, e.r2 ) ) / h;
            ++nq;
        }
        printf( "  enceintes: aire de la 1re base / aire des cellules %.2f | portee moyenne %.2f h\n",
                q_forme / nq, q_portee / nq );
        std::vector<float> r( G.ns );
        for ( SI a = 0; a < G.ns; ++a ) r[a] = std::sqrt( std::max( 0.f, enc[a].r2 ) ) / (float) h;
        (void) q_forme;
        std::sort( r.begin(), r.end() );
        printf( "  portee : mediane %.2f h | 90%% %.2f | 99%% %.2f | 99.9%% %.2f | max %.2f\n",
                r[ G.ns/2 ], r[ G.ns*9/10 ], r[ G.ns*99/100 ], r[ G.ns*999/1000 ], r[ G.ns-1 ] );
    }

    // ---------------- LA TABLE : anneau 2 et au-dela
    t = maintenant();
    std::vector<double> px( n ), py( n ), pw( n );
    for ( int i = 0; i < n; ++i ) { px[i] = G.Ppp[0][i]; py[i] = G.Ppp[1][i];
                                    pw[i] = POIDS ? G.Wp[i] : 0.0; }
    AaBspT<2> arbre;
    arbre.build( px.data(), py.data(), POIDS ? pw.data() : nullptr, n, 10 );
    std::vector<int> agr_de( n );                        // indice permute -> agregat
    for ( SI a = 0; a < G.ns; ++a )
        for ( SI i = G.mdeb[ a ]; i < G.mdeb[ a + 1 ]; ++i ) agr_de[ i ] = (int) a;

    noyau2d::TableSC T;
    T.deb.assign( G.ns + 1, 0 );
    T.bo = &bo;
    // LE GARDE-FOU. Mesure : en Voronoi 0.1 % des agregats -- 25 sur 25000 -- ont une cellule de
    // phase 1 qui couvre presque tout le carre ( portee 186 a 528 h ), et leurs lignes font a elles
    // seules LA MOITIE de la table. En Laguerre fort c'est 1 % des agregats dont la ligne contient
    // TOUS les autres. Pour ces agregats-la l'enceinte n'enveloppe plus rien : aucune forme, si
    // serree soit-elle, ne les rattrapera.
    //
    // On ne les rattrape donc pas -- on les SORT. Des que la ligne depasse `cap`, on l'abandonne et
    // l'agregat est marque `direct` : ses membres seront calcules par le fournisseur BSP, depuis le
    // carre, comme s'il n'y avait pas de sur-cellules. C'est exact par construction, ca coute le
    // prix du BSP sur une poignee de cellules, et ca borne le pire cas par une RESSOURCE ( la
    // longueur de ligne ) et non par un seuil geometrique.
    std::vector<char> direct( G.ns, 0 );
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
            const float nlo[2] = { (float) nd.lo[0], (float) nd.lo[1] };
            const float nhi[2] = { (float) nd.hi[0], (float) nd.hi[1] };
            const bool hors = POIDS
                ? e.ecarte_p( nlo, nhi, (float) nd.wm.a[0], (float) nd.wm.a[1], (float) nd.wm.b )
                : e.ecarte( nlo, nhi );
            if ( hors ) continue;                        // aucun germe de ce sous-arbre ne peut rien
            if ( nd.right < 0 ) {
                for ( SI k = nd.beg; k < nd.end; ++k ) {
                    const int b = agr_de[ (int) arbre.order[ k ] ];
                    if ( proche[ b ] == (int) a || vu[ b ] == (int) a ) continue;
                    vu[ b ] = (int) a;
                    T.ag.push_back( b );
                }
                if ( (int) T.ag.size() - T.deb[ a ] > cap ) { direct[ a ] = 1; break; }
                continue;
            }
            if ( haut + 2 > (int) pile.size() ) pile.resize( pile.size() * 2 );
            pile[ haut++ ] = (int) nd.right; pile[ haut++ ] = h + 1;
        }
        if ( direct[ a ] ) T.ag.resize( T.deb[ a ] );
        T.deb[ a + 1 ] = (int) T.ag.size();
    }
    const double t_tab = maintenant() - t;
    {
        std::vector<int> l( G.ns );
        for ( SI a = 0; a < G.ns; ++a ) l[a] = T.deb[a+1] - T.deb[a];
        std::sort( l.begin(), l.end() );
        long long tot = 0; for ( int v : l ) tot += v;
        long long q = 0; SI k = G.ns; while ( k > 0 && q < tot / 2 ) q += l[ --k ];
        printf( "  lignes : mediane %d | 90%% %d | 99%% %d | max %d | la moitie du total vient des %.1f%% plus longues\n",
                l[G.ns/2], l[G.ns*9/10], l[G.ns*99/100], l[G.ns-1], 100.0 * ( G.ns - k ) / G.ns );
    }

    // ---------------- PHASE 2
    t = maintenant();
    long long nd = 0;
    for ( SI a = 0; a < G.ns; ++a ) {
        if ( direct[ a ] ) {                             // l'agregat sorti : le BSP, depuis le carre
            for ( SI i = G.mdeb[ a ]; i < G.mdeb[ a + 1 ]; ++i ) {
                noyau2d::FournisseurBsp<AaBspT<2>,POIDS> f( &arbre, (float) G.Ppp[0][i],
                        (float) G.Ppp[1][i], POIDS ? (float) G.Wp[i] : 0.f, (int) i );
                cel[ i ] = noyau2d::Atelier<MAXNB>();
                noyau2d::etats::moteur( &f, &cel[ i ] );
                ++nd;
            }
            continue;
        }
        for ( SI i = G.mdeb[ a ]; i < G.mdeb[ a + 1 ]; ++i ) {
            noyau2d::FournisseurSC2<2,POIDS> f( &G, &T, (int) a, (int) i );
            noyau2d::etats::moteur_depuis( &f, &cel[ i ] );
        }
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
    printf( "  phase 2  %7.4f s | somme %.9f  <- doit valoir 1 | %.2f cotes | %lld cellules au BSP ( %.3f %% )\n",
            t2, 0.5 * s2, double( cotes ) / n, nd, 100.0 * nd / n );
    const double tt = t_bo + t1 + t_enc + t_tab + t2;
    printf( "  TOTAL    %7.4f s  ( %.1f ns/germe ) | boites %.4f\n", tt, tt * 1e9 / n, t_bo );

    if ( ! verif )
        return 0;

    // ---------------- LA REFERENCE : le fournisseur BSP, meme noyau, meme jeu de germes.
    //
    // C'est le SEUL controle qui teste la completude cellule par cellule -- la somme des aires ne
    // dit que « il manque quelque part ». Et c'est en meme temps le temoin de vitesse demande :
    // meme machine, meme case, meme noyau, seule la strategie d'elagage change.
    std::vector<float> qx( n ), qy( n ), qw( n ); std::vector<int> qid( n );
    for ( int k = 0; k < n; ++k ) {
        qx[ k ] = (float) arbre.seed_x( k ); qy[ k ] = (float) arbre.seed_y( k );
        qw[ k ] = (float) arbre.seed_w( k ); qid[ k ] = (int) arbre.order[ k ];
    }
    std::vector<noyau2d::Atelier<MAXNB>> ref( n );
    double tb = 1e30;
    for ( int r = 0; r < 3; ++r ) {
        const double t0 = maintenant();
        for ( int k = 0; k < n; ++k ) {
            noyau2d::Atelier<MAXNB> at;
            noyau2d::FournisseurBsp<AaBspT<2>,POIDS> f( &arbre, qx[ k ], qy[ k ], qw[ k ], qid[ k ] );
            noyau2d::etats::moteur( &f, &at );
            ref[ qid[ k ] ] = at;
        }
        tb = std::min( tb, maintenant() - t0 );
    }
    double sr = 0; long long cr = 0;
    for ( int i = 0; i < n; ++i ) { const auto &c = ref[ i ];
        cr += c.nb;
        for ( int v = 0; v < c.nb; ++v ) { const int w = v + 1 < c.nb ? v + 1 : 0;
            sr += (double) c.vx[v] * c.vy[w] - (double) c.vx[w] * c.vy[v]; } }

    auto rot_eq = []( const int *a, const int *b, int m ) {
        for ( int r = 0; r < m; ++r ) { bool ok = true;
            for ( int v = 0; v < m; ++v ) if ( a[ ( v + r ) % m ] != b[ v ] ) { ok = false; break; }
            if ( ok ) return true; } return m == 0; };
    auto aire_de = []( const noyau2d::Atelier<MAXNB> &c ) {
        double a = 0;
        for ( int v = 0; v < c.nb; ++v ) { const int w = v + 1 < c.nb ? v + 1 : 0;
            a += (double) c.vx[v] * c.vy[w] - (double) c.vx[w] * c.vy[v]; }
        return 0.5 * a; };
    // LA COMBINATOIRE ET L'AIRE NE DISENT PAS LA MEME CHOSE. Un plan qui passe a l'epaisseur du
    // `float` d'un sommet ajoute un sommet SANS ajouter d'aire : c'est le plancher du `float`, pas
    // une coupe ratee. Seul l'ECART D'AIRE distingue les deux.
    // L'ECART EST RAPPORTE A L'AIRE MOYENNE `1 / n` : c'est la seule echelle qui ait un sens, et
    // elle rend le seuil independant de `n`. Le plancher du `float` vit vers 1e-4 relatif ; une
    // coupe ratee se voit a 1e-2 et au-dela.
    const double am = 1.0 / n;
    int faux = 0; double ecart_max = 0, ecart_tot = 0; int h[ 3 ] = { 0, 0, 0 };
    for ( int i = 0; i < n; ++i ) {
        const auto &c = cel[ i ]; const auto &d = ref[ i ];
        if ( ! ( c.nb == d.nb && rot_eq( c.cid, d.cid, c.nb ) ) ) ++faux;
        const double e = ( aire_de( c ) - aire_de( d ) ) / am;
        ecart_tot += e > 0 ? e : -e;
        if ( e > ecart_max ) ecart_max = e;
        if ( e > 1e-3 ) ++h[ 0 ];
        if ( e > 1e-2 ) ++h[ 1 ];
        if ( e > 1e-1 ) ++h[ 2 ];
    }
    printf( "  ---- reference : fournisseur BSP ( notre dernier AaBsp ), meme noyau\n" );
    printf( "  AaBsp    %7.4f s  ( %.1f ns/germe ) | somme %.9f | %.2f cotes\n",
            tb, tb * 1e9 / n, 0.5 * sr, double( cr ) / n );
    printf( "  sur-cellules / AaBsp : x%.2f | %d / %d combinatoires differentes\n",
            tb / tt, faux, n );
    printf( "  ecart d'aire ( en aires moyennes ) : max %.3e | > 1e-3 : %d | > 1e-2 : %d | > 1e-1 : %d\n",
            ecart_max, h[ 0 ], h[ 1 ], h[ 2 ] );
    return 0;
}

int main( int argc, char **argv ) {
    int n = 200000; SI rho = 8; double frac = 0; bool morton_local = true; bool verif = false; int cap = 48; bool filtre1 = false; int dop = 2; bool rot = false;
    for ( int i = 1; i + 1 < argc; i += 2 ) {
        const std::string o = argv[ i ];
        if      ( o == "-n"        ) n = atoi( argv[ i + 1 ] );
        else if ( o == "--rho"     ) rho = atoi( argv[ i + 1 ] );
        else if ( o == "--weights" ) frac = atof( argv[ i + 1 ] );
        else if ( o == "--morton-local" ) morton_local = atoi( argv[ i + 1 ] ) != 0;
        else if ( o == "--verif"   ) verif = atoi( argv[ i + 1 ] ) != 0;
        else if ( o == "--cap"     ) cap = atoi( argv[ i + 1 ] );
        else if ( o == "--filtre1" ) filtre1 = atoi( argv[ i + 1 ] ) != 0;
        else if ( o == "--dop"     ) dop = atoi( argv[ i + 1 ] );
        else if ( o == "--rot"     ) rot = atoi( argv[ i + 1 ] ) != 0;
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
    printf( "n = %d, rho = %d, %s, morton local %s | %lld agregats, degre %.2f | enceinte %d-DOP%s, cap %d\n",
            n, (int) rho, frac > 0 ? "Laguerre" : "Voronoi", morton_local ? "oui" : "non",
            (long long) b.ns, b.deg, dop, rot ? " tourne" : "", cap );
    // les variantes d'enceinte, choisies a la ligne de commande et instanciees ici
    #define LANCE( K, R ) ( frac > 0                                                              \
        ? deroule<true , noyau2d::Dop<K,R>>( cl, G, n, morton_local, verif, cap, filtre1 )        \
        : deroule<false, noyau2d::Dop<K,R>>( cl, G, n, morton_local, verif, cap, filtre1 ) )
    if ( ! rot ) { if ( dop == 2 ) return LANCE( 2, false );
                   if ( dop == 4 ) return LANCE( 4, false );
                   if ( dop == 8 ) return LANCE( 8, false ); }
    else         { if ( dop == 2 ) return LANCE( 2, true  );
                   if ( dop == 4 ) return LANCE( 4, true  );
                   if ( dop == 8 ) return LANCE( 8, true  ); }
    fprintf( stderr, "--dop doit valoir 2, 4 ou 8\n" );
    return 1;
    #undef LANCE
}

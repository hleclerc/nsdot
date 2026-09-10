// =====================================================================================
// LE FOURNISSEUR BSP, EN PARALLELE.
//
// Le parallelisme est ici architecturalement gratuit, et il faut dire pourquoi plutot que de le
// constater : chaque cellule est independante, l'arbre est en LECTURE SEULE, et l'etat du parcours
// -- la pile, la position dans la feuille -- vit dans le `Local` que le moteur loge dans SA frame,
// donc un par thread par cellule. Il n'y a rien a proteger, rien a atomiser, rien a mettre en pool.
//
// CE QUI N'EST PAS GRATUIT, ET QUE LE BANC MESURE : le DECOUPAGE. `blocks` donne a chaque thread
// une tranche contigue de l'ordre de l'arbre, donc une region de l'espace ; `strided` les fait tous
// balayer la meme region au meme instant. Meme travail, memes resultats, mais pas la meme localite
// -- et c'est ce que `sdot` fait aujourd'hui, donc ce qu'il faut chiffrer.
//
// L'aire est accumulee PAR THREAD, sur des lignes de cache distinctes. Sans le remplissage, huit
// threads qui incrementent huit `double` voisins passent leur temps a s'invalider mutuellement.
// =====================================================================================

#include "supercell/FournisseurBsp.h"
#include "supercell/Noyau2DEtats.h"
#include "util/parallel.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace pd;

static double now() {
    using namespace std::chrono;
    return duration<double>( steady_clock::now().time_since_epoch() ).count();
}

static constexpr int MAXNB = 64;

/// une case par thread, sur sa propre ligne de cache.
struct alignas( 64 ) Accu { double aire = 0; long long cotes = 0; };

template<bool POIDS>
static double une_passe( const AaBspT<2> &arbre, const std::vector<float> &px,
                         const std::vector<float> &py, const std::vector<float> &pw,
                         const std::vector<int> &ids, int nb_threads, Split split, bool pin,
                         double &aire, long long &cotes ) {
    const int n = (int) px.size();
    std::vector<Accu> acc( nb_threads <= 1 ? 1 : nb_threads );

    const double t0 = now();
    parallel_for( n, nb_threads, split, pin, [ & ]( SI k, int t ) {
        noyau2d::Atelier<MAXNB> at;
        noyau2d::FournisseurBsp<AaBspT<2>,POIDS> f( &arbre, px[ k ], py[ k ], pw[ k ], ids[ k ] );
        noyau2d::etats::moteur( &f, &at );
        if ( at.nb <= 0 )
            return;
        double a = 0;
        for ( int v = 0; v < at.nb; ++v ) {
            const int w = v + 1 < at.nb ? v + 1 : 0;
            a += (double) at.vx[ v ] * at.vy[ w ] - (double) at.vx[ w ] * at.vy[ v ];
        }
        acc[ t ].aire += 0.5 * a;
        acc[ t ].cotes += at.nb;
    } );
    const double dt = now() - t0;

    aire = 0; cotes = 0;
    for ( const Accu &a : acc ) { aire += a.aire; cotes += a.cotes; }
    return dt;
}

int main( int argc, char **argv ) {
    int n = 1000000, reps = 3, tmax = 0;
    double frac = 0;                                     // poids en fraction de h^2 ; 0 = Voronoi
    Split split = Split::blocks;
    bool pin = true;
    for ( int i = 1; i + 1 < argc; i += 2 ) {
        const std::string o = argv[ i ];
        if      ( o == "-n"        ) n     = atoi( argv[ i + 1 ] );
        else if ( o == "--reps"    ) reps  = atoi( argv[ i + 1 ] );
        else if ( o == "--threads" ) tmax  = atoi( argv[ i + 1 ] );
        else if ( o == "--weights" ) frac  = atof( argv[ i + 1 ] );
        else if ( o == "--split"   ) split = std::string( argv[ i + 1 ] ) == "strided"
                                             ? Split::strided : Split::blocks;
        else if ( o == "--no-pin"  ) pin   = atoi( argv[ i + 1 ] ) == 0;
        else { fprintf( stderr, "option inconnue %s\n", argv[ i ] ); return 1; }
    }
    if ( tmax <= 0 ) tmax = (int) std::thread::hardware_concurrency();

    std::mt19937 gen( 12345 );
    std::uniform_real_distribution<double> u( 0, 1 );
    std::vector<double> ax( n ), ay( n ), aw( n );
    const double h2 = 1.0 / n;
    for ( int i = 0; i < n; ++i ) { ax[ i ] = u( gen ); ay[ i ] = u( gen ); aw[ i ] = frac * h2 * u( gen ); }

    AaBspT<2> arbre;
    const double t_arbre = now();
    arbre.build( ax.data(), ay.data(), frac > 0 ? aw.data() : nullptr, n, 10 );
    const double d_arbre = now() - t_arbre;

    std::vector<float> px( n ), py( n ), pw( n );
    std::vector<int> ids( n );
    for ( int k = 0; k < n; ++k ) {
        px[ k ] = (float) arbre.seed_x( k ); py[ k ] = (float) arbre.seed_y( k );
        pw[ k ] = (float) arbre.seed_w( k ); ids[ k ] = (int) arbre.order[ k ];
    }

    printf( "n = %d, %s, arbre construit en %.3f s ( hors chrono )\n",
            n, frac > 0 ? "Laguerre" : "Voronoi", d_arbre );
    printf( "decoupage %s, threads %s\n\n",
            split == Split::blocks ? "blocks" : "strided", pin ? "epingles" : "libres" );
    printf( "  threads   temps      ns/germe   acceleration   efficacite   somme des aires\n" );

    double t1 = 0;
    for ( int t = 1; t <= tmax; t *= 2 ) {
        double best = 1e30, aire = 0; long long cotes = 0;
        for ( int r = 0; r < reps; ++r ) {
            double a; long long c;
            const double dt = frac > 0
                ? une_passe<true >( arbre, px, py, pw, ids, t, split, pin, a, c )
                : une_passe<false>( arbre, px, py, pw, ids, t, split, pin, a, c );
            if ( dt < best ) { best = dt; aire = a; cotes = c; }
        }
        if ( t == 1 ) t1 = best;
        printf( "  %5d   %8.4f s  %7.1f    x%6.2f      %5.1f %%     %.9f\n",
                t, best, best * 1e9 / n, t1 / best, 100.0 * t1 / best / t, aire );
        (void) cotes;
    }
    return 0;
}

// LA VERIFICATION. C'est le seul binaire qui connait TOUS les accelerateurs, et c'est sa raison
// d'etre : les faire tomber d'accord germe par germe.
//
// Deux echelles, et il faut les deux :
//
//   * `--petit` (defaut) : contre le BALAYAGE COMPLET, qui n'a aucune ligne de code geometrique en
//     commun avec un arbre. Si les deux s'accordent germe par germe, ce n'est pas deux fois le meme
//     bug. Borne a quelques milliers de germes -- l'oracle est en `O( n^2 )`.
//   * `--grand` : les accelerateurs les uns contre les autres, au `n` demande, sans oracle. Un
//     desaccord peut n'apparaitre qu'en grand -- profondeur d'arbre, voisinages plus fournis -- et
//     la somme des mesures, seul controle du banc, ne le montre que s'il ne se compense pas.
//
//   xmake run pd_check --help

#include "spatial_accel/AaBsp.h"
#include "spatial_accel/AaBsp4.h"
#include "spatial_accel/AaBspHull.h"
#include "spatial_accel/AaBspMemo.h"
#include "spatial_accel/AaBspPack.h"
#include "spatial_accel/AaBspPacked.h"
#include "spatial_accel/AaBspPre.h"
#include "spatial_accel/FrontPd.h"
#include "spatial_accel/Grid.h"
#include "spatial_accel/ObBsp.h"
#include "bench/Bench.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace pd;
using namespace pd::bench;

namespace {

/// Un accelerateur, construit et mesure sur le meme nuage. `CellBox` reste vrai partout : ce qu'on
/// verifie est la GEOMETRIE, pas une variante de reglage.
template<class Accel, class Cell, int D>
std::vector<TF> mesure( const Cloud<D> &cl, SI leaf ) {
    Accel t;
    t.build( cl.P, cl.W, cl.n, leaf );
    std::vector<TF> r;
    PowerDiagram<Cell, Accel, true, false, false, true> pd{ t };
    pd.measures( r, 1, Split::blocks, false );
    return r;
}

/// LE MEME VOLUME PAR L'AUTRE CHEMIN. `Cell3::measure()` accumule les faces SANS les ordonner ;
/// `measure_by_cycles()` reconstruit les cycles et eventaille chaque face depuis un sommet. Les
/// deux ne partagent ni la facon de retrouver les faces, ni la decomposition en triangles, ni le
/// point d'ou partent les tetraedres -- ils ne partagent que les sommets et les aretes. En 3D il
/// n'existe aucune reference exterieure pour le volume d'un polyedre : leur accord EST le temoin.
template<class Accel, class Cell>
std::vector<TF> mesure_par_cycles( const Cloud<3> &cl, SI leaf ) {
    Accel t;
    t.build( cl.P, cl.W, cl.n, leaf );
    PowerDiagram<Cell, Accel, true, false, false, true> pd{ t };
    std::vector<TF> r( cl.n, TF( 0 ) );
    Cell c;
    for ( SI k = 0; k < cl.n; ++k ) {
        pd.make_cell( c, k );
        r[ t.seed_id( k ) ] = c.measure_by_cycles();
    }
    return r;
}

/// L'ecart maximum germe par germe. Ecrire PAR INDICE D'ORIGINE est ce qui rend la comparaison
/// possible : une cellule fausse d'un cote et fausse a l'envers de l'autre passerait une
/// comparaison de sommes.
double ecart( const std::vector<TF> &a, const std::vector<TF> &b ) {
    double m = 0;
    for ( size_t i = 0; i < a.size(); ++i )
        m = std::max( m, std::fabs( double( a[ i ] - b[ i ] ) ) );
    return m;
}

bool dit( const char *nom, double e, double tol ) {
    std::printf( "  %-10s vs balayage complet : ecart max %.3e%s\n", nom, e,
                 e < tol ? "" : "   <-- ECHEC" );
    return e < tol;
}

int check_2d( const Args &a ) {
    using Cell = CellSoAT<64>;
    Cloud<2> cl;
    if ( ! a.load.empty() ) {
        // le balayage complet est en `O( n^2 )` : on ne prend qu'un PREFIXE du fichier. Les germes
        // y sont tires independamment, donc un prefixe suit exactement la meme loi -- et ses poids,
        // qui resolvaient le probleme ENTIER, ne resolvent plus rien pour lui : beaucoup de
        // cellules sont vides, ce qui rend la verification PLUS severe, pas moins.
        if ( ! load_cloud( a.load, cl ) )
            return 1;
        const SI n = std::min( cl.n, SI( 3000 ) );
        for ( int d = 0; d < 2; ++d ) cl.c[ d ].resize( n );
        cl.w.resize( n );
        cl.finish();
        std::printf( "=== 2D  nuage '%s', les %d premiers germes\n", a.load.c_str(), int( cl.n ) );
    } else {
        uniform_cloud( cl, 2000, a.seed, a.wscale );
        std::printf( "=== 2D  uniforme, %d germes, poids x%.3g\n", int( cl.n ), a.wscale );
    }

    front_rate = 2;
    const std::vector<TF> ref = mesure<EverySeed, Cell, 2>( cl, a.leaf );
    TF s = 0;
    for ( TF v : ref ) s += v;
    std::printf( "  balayage complet : somme %.12f\n", double( s ) );

    bool ok = std::fabs( double( s ) - 1 ) < 1e-12;
    ok &= dit( "bsp",    ecart( ref, mesure<AaBsp,       Cell, 2>( cl, a.leaf ) ), 1e-12 );
    ok &= dit( "bsp4",   ecart( ref, mesure<AaBsp4,      Cell, 2>( cl, a.leaf ) ), 1e-12 );
    ok &= dit( "bsp4l",  ecart( ref, mesure<AaBsp4L,     Cell, 2>( cl, a.leaf ) ), 1e-12 );
    ok &= dit( "obsp",   ecart( ref, mesure<ObBsp,       Cell, 2>( cl, a.leaf ) ), 1e-12 );
    ok &= dit( "packed", ecart( ref, mesure<AaBspPacked, Cell, 2>( cl, a.leaf ) ), 1e-12 );
    ok &= dit( "grille", ecart( ref, mesure<Grid,        Cell, 2>( cl, a.leaf ) ), 1e-12 );
    // la pre-passe ne doit RIEN changer : la cellule est l'intersection de tous les demi-plans,
    // donc couper par un sous-ensemble avant le reste doit rendre le meme polygone, au bit pres si
    // l'ordre des coupes effectives est le meme et a l'arrondi sinon.
    ok &= dit( "pre",    ecart( ref, mesure<AaBspPre,    Cell, 2>( cl, a.leaf ) ), 1e-12 );
    ok &= dit( "pack",   ecart( ref, mesure<AaBspPack,   Cell, 2>( cl, a.leaf ) ), 1e-12 );
    ok &= dit( "hull",   ecart( ref, mesure<AaBspHull,   Cell, 2>( cl, a.leaf ) ), 1e-12 );
    ok &= dit( "front",  ecart( ref, mesure<FrontPd,     Cell, 2>( cl, a.leaf ) ), 1e-12 );
    std::printf( "  => %s\n", ok ? "OK" : "ECHEC" );
    return ok ? 0 : 1;
}

/// LE MEME EN 3D, et c'est la que le balayage complet vaut le plus cher : il n'y a pas de reference
/// exterieure pour un polyedre, donc l'accord entre deux chemins de code independants EST la
/// preuve. La somme des volumes doit valoir 1 -- les cellules pavent le cube -- ce qui attrape
/// separement une erreur de `cut` et une erreur de `measure`.
int check_3d( const Args &a ) {
    using Cell = Cell3T<128>;
    Cloud<3> cl;
    uniform_cloud( cl, 1500, a.seed, a.wscale );
    std::printf( "=== 3D  uniforme, %d germes, poids x%.3g\n", int( cl.n ), a.wscale );

    const std::vector<TF> ref = mesure<EverySeed3, Cell, 3>( cl, a.leaf );
    TF s = 0;
    for ( TF v : ref ) s += v;
    std::printf( "  balayage complet : somme %.12f\n", double( s ) );

    bool ok = std::fabs( double( s ) - 1 ) < 1e-11;
    const std::vector<TF> rb = mesure<AaBsp3, Cell, 3>( cl, a.leaf );
    ok &= dit( "bsp",    ecart( ref, rb ), 1e-12 );
    ok &= dit( "grille", ecart( ref, mesure<Grid3,  Cell, 3>( cl, a.leaf ) ), 1e-12 );

    const double ec = ecart( rb, mesure_par_cycles<AaBsp3, Cell>( cl, a.leaf ) );
    std::printf( "  volume : accumulation vs cycles : ecart max %.3e%s\n", ec,
                 ec < 1e-15 ? "" : "   <-- ECHEC" );
    ok &= ec < 1e-15;
    std::printf( "  => %s\n", ok ? "OK" : "ECHEC" );
    return ok ? 0 : 1;
}

/// LES ACCELERATEURS ENTRE EUX, au `n` demande et sans oracle : on nomme les pires desaccords.
int grand( const Args &a ) {
    using Cell = CellSoAT<64>;
    int bad = 0;
    for ( const Cloud<2> &cl : suite_2d( a ) ) {
        if ( cl.absent ) continue;
        std::printf( "=== 2D  %s\n", cl.nom.c_str() );
        front_rate = 4;
        const std::vector<TF> rb = mesure<AaBsp, Cell, 2>( cl, a.leaf );
        TF s = 0;
        for ( TF v : rb ) s += v;
        std::printf( "  bsp : somme %.12f\n", double( s ) );
        auto contre = [ & ]( const char *nom, const std::vector<TF> &r ) {
            std::vector<std::pair<double, SI>> mauvais;
            for ( SI i = 0; i < cl.n; ++i ) {
                const double d = std::fabs( double( rb[ i ] - r[ i ] ) );
                if ( d > 1e-15 )
                    mauvais.emplace_back( d, i );
            }
            std::sort( mauvais.rbegin(), mauvais.rend() );
            std::printf( "  %-8s vs bsp : %d germes > 1e-15%s\n", nom, int( mauvais.size() ),
                         mauvais.empty() ? "" : ", les pires :" );
            for ( size_t u = 0; u < std::min<size_t>( mauvais.size(), 3 ); ++u )
                std::printf( "      germe %7d  ecart %.3e   bsp %.6e   %s %.6e\n",
                             int( mauvais[ u ].second ), mauvais[ u ].first,
                             double( rb[ mauvais[ u ].second ] ), nom,
                             double( r[ mauvais[ u ].second ] ) );
            bad += ! mauvais.empty();
        };
        contre( "front", mesure<FrontPd,   Cell, 2>( cl, a.leaf ) );
        contre( "hull",  mesure<AaBspHull, Cell, 2>( cl, a.leaf ) );
        contre( "pack",  mesure<AaBspPack, Cell, 2>( cl, a.leaf ) );
        contre( "grille", mesure<Grid,     Cell, 2>( cl, a.leaf ) );
    }
    for ( const Cloud<3> &cl : suite_3d( a ) ) {
        if ( cl.absent ) continue;
        std::printf( "=== 3D  %s\n", cl.nom.c_str() );
        const std::vector<TF> rb = mesure<AaBsp3, Cell3T<128>, 3>( cl, a.leaf );
        TF s = 0;
        for ( TF v : rb ) s += v;
        std::printf( "  bsp : somme %.12f\n", double( s ) );
        const std::vector<TF> rg = mesure<Grid3, Cell3T<128>, 3>( cl, a.leaf );
        std::printf( "  grille vs bsp : ecart max %.3e\n", ecart( rb, rg ) );
        bad += ecart( rb, rg ) > 1e-12;
    }
    return bad ? 1 : 0;
}

} // namespace

int main( int argc, char **argv ) {
    Args a;
    bool gros = false;
    for ( int i = 1; i < argc; ++i ) {
        const std::string s = argv[ i ];
        if ( parse_commun( a, s, i, argc, argv ) ) continue;
        if ( s == "--grand" ) gros = true;
        else {
            std::printf( "usage: pd_check [options]\n" );
            usage_commun();
            std::printf( "  --grand         les accelerateurs entre eux au n demande, sans oracle\n"
                         "  --weights W     poids aleatoires : c'est le reglage qui rend la\n"
                         "                  verification severe (des cellules VIDES apparaissent)\n" );
            return s == "--help" || s == "-h" ? 0 : 1;
        }
    }
    finalise( a );
    front_threads = 1;
    hull_threads = 1;

    if ( gros )
        return grand( a );
    int bad = 0;
    if ( a.dims != 3 ) bad += check_2d( a );
    if ( a.dims != 2 ) bad += check_3d( a );
    return bad;
}

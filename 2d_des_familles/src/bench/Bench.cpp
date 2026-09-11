#include "bench/Bench.h"
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <random>
#include <thread>

namespace pd::bench {

double now() {
    using namespace std::chrono;
    return duration<double>( steady_clock::now().time_since_epoch() ).count();
}

// --------------------------------------------------------------------------------- les options

bool parse_commun( Args &a, const std::string &s, int &i, int argc, char **argv ) {
    auto val = [ & ]() { return i + 1 < argc ? argv[ ++i ] : ""; };
    if      ( s == "-n" )            a.n = std::atoi( val() );
    else if ( s == "--reps" )        a.reps = std::atoi( val() );
    else if ( s == "--threads" )     a.threads = std::atoi( val() );
    else if ( s == "--leaf" )        a.leaf = std::atoi( val() );
    else if ( s == "--maxnv" )       a.maxnv = std::atoi( val() );
    else if ( s == "--no-cellbox" )  a.cellbox = false;
    else if ( s == "--skip-inside" ) a.skipin = true;
    else if ( s == "--split" )       a.split = std::string( val() ) == "strided" ? Split::strided : Split::blocks;
    else if ( s == "--no-pin" )      a.pin = false;
    else if ( s == "--seed" )        a.seed = unsigned( std::atoi( val() ) );
    else if ( s == "--weights" )     a.wscale = std::atof( val() );
    else if ( s == "--load" )        a.load = val();
    else if ( s == "--2d" )          a.dims = 2;
    else if ( s == "--3d" )          a.dims = 3;
    else return false;
    return true;
}

void usage_commun() {
    std::printf(
        "  -n N            germes du cas « uniforme »        (1000000)\n"
        "  --reps R        repetitions chronometrees         (3)\n"
        "  --threads T     0 = autant que de coeurs          (0)\n"
        "  --leaf L        germes par feuille                (10)\n"
        "  --maxnv M       sommets max par cellule           (32 en 2D, 128 en 3D)\n"
        "  --2d / --3d     ne derouler QUE cette dimension   (les deux)\n"
        "  --load FILE     UN nuage, au lieu de la suite\n"
        "  --no-cellbox    pas de boite de cellule (sommets directement)\n"
        "  --skip-inside   pas de test pour une boite contenant le germe\n"
        "  --split S       blocks | strided                  (blocks)\n"
        "  --no-pin        ne pas epingler les threads\n"
        "  --seed S        graine du tirage                  (0)\n"
        "  --weights W     poids aleatoires du cas uniforme, en fraction de h^2\n" );
}

void finalise( Args &a ) {
    if ( a.threads <= 0 )
        a.threads = int( std::thread::hardware_concurrency() );
}

// --------------------------------------------------------------------------------- les nuages

namespace {

/// le lecteur commun : `n`, puis `n * ( D + 1 )` nombres, les `#` sautes.
template<int D>
bool load_impl( const std::string &path, Cloud<D> &cl, bool bavard ) {
    std::FILE *f = std::fopen( path.c_str(), "rb" );
    if ( ! f ) {
        if ( bavard )
            std::printf( "impossible d'ouvrir '%s'\n", path.c_str() );
        return false;
    }
    std::fseek( f, 0, SEEK_END );
    const long sz = std::ftell( f );
    std::fseek( f, 0, SEEK_SET );
    std::string buf( size_t( sz ) + 1, '\0' );
    const size_t rd = std::fread( buf.data(), 1, size_t( sz ), f );
    std::fclose( f );
    buf.resize( rd + 1 );
    buf[ rd ] = 0;

    const char *p = buf.data(), *e = p + rd;
    auto num = [ & ]( TF &out ) {
        for ( ;; ) {
            while ( p < e && ( *p == ' ' || *p == '\t' || *p == '\n' || *p == '\r' ) ) ++p;
            if ( p < e && *p == '#' ) { while ( p < e && *p != '\n' ) ++p; continue; }
            break;
        }
        if ( p >= e ) return false;
        char *q = nullptr;
        out = std::strtod( p, &q );
        if ( q == p ) return false;
        p = q;
        return true;
    };

    TF v;
    if ( ! num( v ) ) {
        if ( bavard ) std::printf( "'%s' : pas de nombre de germes\n", path.c_str() );
        return false;
    }
    const SI n = SI( v );
    for ( int d = 0; d < D; ++d ) cl.c[ d ].assign( n, TF( 0 ) );
    cl.w.assign( n, TF( 0 ) );
    for ( SI i = 0; i < n; ++i ) {
        for ( int d = 0; d < D; ++d )
            if ( ! num( cl.c[ d ][ i ] ) ) {
                if ( bavard ) std::printf( "'%s' : tronque au germe %d\n", path.c_str(), int( i ) );
                return false;
            }
        if ( ! num( cl.w[ i ] ) ) {
            if ( bavard ) std::printf( "'%s' : tronque au germe %d\n", path.c_str(), int( i ) );
            return false;
        }
    }
    cl.finish();
    return true;
}

template<int D>
void uniform_impl( Cloud<D> &cl, SI n, unsigned seed, double wscale ) {
    std::mt19937_64 rng( seed );
    std::uniform_real_distribution<TF> uni( 0.001, 0.999 );
    for ( int d = 0; d < D; ++d ) {
        cl.c[ d ].resize( n );
        for ( SI i = 0; i < n; ++i ) cl.c[ d ][ i ] = uni( rng );
    }
    cl.w.assign( n, TF( 0 ) );
    if ( wscale != 0 ) {
        const TF h = std::pow( TF( 1 ) / n, TF( 1 ) / D );
        std::uniform_real_distribution<TF> uw( -wscale * h * h, wscale * h * h );
        for ( SI i = 0; i < n; ++i ) cl.w[ i ] = uw( rng );
    }
    cl.finish();
}

template<int D>
Cloud<D> depuis_fichier( const char *path, const char *nom, int nv ) {
    Cloud<D> cl;
    cl.nom = nom;
    cl.nv = nv;
    if ( ! load_impl<D>( path, cl, false ) )
        cl.absent = true;
    return cl;
}

} // namespace

bool load_cloud( const std::string &path, Cloud<2> &cl, bool bavard ) { return load_impl<2>( path, cl, bavard ); }
bool load_cloud( const std::string &path, Cloud<3> &cl, bool bavard ) { return load_impl<3>( path, cl, bavard ); }
void uniform_cloud( Cloud<2> &cl, SI n, unsigned s, double w ) { uniform_impl<2>( cl, n, s, w ); }
void uniform_cloud( Cloud<3> &cl, SI n, unsigned s, double w ) { uniform_impl<3>( cl, n, s, w ); }

/// LES CAS IMPORTANTS EN 2D.
///
///   * uniforme -- le regime facile, celui ou la cellule est toujours autour de son germe ;
///   * lignes / Voronoi -- densite etalee sur plusieurs ordres de grandeur, poids nuls ;
///   * lignes / aires egales -- le vrai cas dur : les poids EMMENENT la cellule loin de son germe.
std::vector<Cloud<2>> suite_2d( const Args &a ) {
    std::vector<Cloud<2>> r;
    if ( a.dims == 3 )
        return r;                               // `--load` designe alors un nuage 3D
    if ( ! a.load.empty() ) {
        Cloud<2> cl;
        cl.nom = a.load;
        if ( ! load_cloud( a.load, cl ) ) cl.absent = true;
        r.push_back( std::move( cl ) );
        return r;
    }
    Cloud<2> u;
    u.nom = "uniforme n=" + std::to_string( a.n );
    uniform_cloud( u, a.n, a.seed, a.wscale );
    r.push_back( std::move( u ) );
    r.push_back( depuis_fichier<2>( "cases/lines5_n100000_s0.005_voronoi.txt", "lignes / Voronoi", 64 ) );
    r.push_back( depuis_fichier<2>( "cases/lines5_n100000_s0.005_equal.txt", "lignes / aires egales", 64 ) );
    return r;
}

/// LES CAS IMPORTANTS EN 3D, et ce sont les MEMES QUESTIONS.
///
/// Le nuage dur y est fait de germes serres autour de quelques PLANS traversant le cube -- l'analogue
/// exact des lignes en 2D : une variete de codimension 1, donc une densite qui s'effondre entre deux
/// nappes. Et le jeu de poids « volumes egaux » y joue le meme role, pour la meme raison.
std::vector<Cloud<3>> suite_3d( const Args &a ) {
    std::vector<Cloud<3>> r;
    if ( ! a.load.empty() ) {
        // un fichier n'annonce pas sa dimension : c'est `--3d` qui la donne. Sans lui, `--load`
        // designe un nuage 2D et la suite 3D est vide.
        if ( a.dims != 3 )
            return r;
        Cloud<3> cl;
        cl.nom = a.load;
        cl.nv = 128;
        if ( ! load_cloud( a.load, cl ) ) cl.absent = true;
        r.push_back( std::move( cl ) );
        return r;
    }
    Cloud<3> u;
    u.nom = "uniforme n=" + std::to_string( a.n );
    uniform_cloud( u, a.n, a.seed, a.wscale );
    r.push_back( std::move( u ) );
    r.push_back( depuis_fichier<3>( "cases/planes4_n100000_s0.02_voronoi.txt", "plans / Voronoi", 128 ) );
    r.push_back( depuis_fichier<3>( "cases/planes4_n100000_s0.02_equal.txt", "plans / volumes egaux", 128 ) );
    return r;
}

// --------------------------------------------------------------------------------- l'affichage

namespace {
void ligne_impl( const std::string &nom, SI n, const Mesure &m, int maxnv ) {
    std::printf( "  %-24s n=%-7d nv=%-3d : %8.3f s (%7.0f ns/germe)  arbre %6.0f ms  somme %.9f%s\n",
                 nom.c_str(), int( n ), maxnv, m.t, m.t / n * 1e9, m.t_build * 1e3, m.somme,
                 m.ok ? "" : "   <-- FAUX" );
    if ( m.novf )
        std::printf( "      ATTENTION : %d coupes ont DEBORDE %d sommets -- leur mesure est fausse."
                     " Relancer avec --maxnv %d.\n", int( m.novf ), maxnv, 2 * maxnv );
}
} // namespace

void ligne( const Cloud<2> &cl, const Mesure &m, int maxnv ) { ligne_impl( cl.nom, cl.n, m, maxnv ); }
void ligne( const Cloud<3> &cl, const Mesure &m, int maxnv ) { ligne_impl( cl.nom, cl.n, m, maxnv ); }

} // namespace pd::bench

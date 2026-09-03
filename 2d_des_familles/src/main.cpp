// Le banc : des diracs AU HASARD dans le carre unite, les aires de leurs cellules de Laguerre, et
// un chrono. La somme des aires doit valoir 1 -- c'est le seul controle, et il suffit : une
// cellule fausse d'un cote et fausse a l'envers de l'autre est ce qu'il attrape.
//
//   xmake run pd2d --help

#include "AaBsp.h"
#include "AaBsp4.h"
#include "ObBsp.h"
#include "Newton.h"
#ifdef _OPENMP
#include <omp.h>
#endif
#include "AaBspPacked.h"
#include "AaBspHull.h"
#include "AaBspPack.h"
#include "AaBspPre.h"
#include "Grid.h"
#include "Cell.h"
#include "PowerDiagram.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

using namespace pd2d;

namespace {

struct Args {
    SI   n           = 1000000;
    int  reps        = 3;
    int  threads     = 0;          ///< 0 -> autant que de coeurs
    SI   leaf        = 10;
    std::string tree = "bsp";
    int  maxnv   = 32;
    bool cellbox = true;
    bool skipin  = false;
    Split split      = Split::blocks;
    bool pin         = true;
    bool check       = false;
    unsigned seed    = 0;
    double   wscale  = 0;          ///< 0 = pas de poids
    std::string load;              ///< un fichier de `cases/`, au lieu du tirage uniforme
    bool stats       = false;      ///< compter ce que le parcours fait, au lieu de le chronometrer
    bool majorant    = false;      ///< comparer les majorants de poids, au lieu de mesurer
    bool cross       = false;      ///< comparer les accelerateurs entre eux au `n` demande
    bool baisse      = false;      ///< la parabole ABAISSEE : un enclos par paquet
    bool psigrid     = false;      ///< le critere `min_B h_i > M( B )` sur une grille reguliere
    bool front       = false;      ///< l ETALEMENT sur grille : amorce, front, et ce qu il manque
    bool enclos      = false;      ///< mesurer l'ENCLOS par sous-echantillon, au lieu de mesurer
    SI   prerate     = 16;         ///< la pre-passe coupe d'abord contre un germe sur `prerate`
    bool newton      = false;      ///< resoudre le probleme d'aires egales, au lieu de mesurer
    double ntol      = 1e-6;       ///< arret : `max_i |a_i - nu| <= ntol * nu`
    int  nmax        = 100;        ///< iterations de Newton au maximum
    double cgtol     = 1e-10;      ///< arret du gradient conjugue, en residu RELATIF
    int  cgmax       = 20000;
    std::string solver = "amg";    ///< amg (AMGCL) | chol (Eigen) | cg (maison)
    int  amgvar      = 0;          ///< 0 = SA+spai0, 1 = SA+Gauss-Seidel, 2 = Ruge-Stuben+GS
    SI   msratio     = 1;          ///< multi-echelle : germes par paquet d un niveau au suivant
                                   ///< 1 = AUCUN (defaut : la piste n aboutit pas encore)
    SI   msmin       = 500;        ///< ... taille du niveau le plus grossier
    double mstol     = 1e-2;       ///< ... tolerance des niveaux grossiers
    int  mspasses    = 4;          ///< ... passes de rattrapage des cellules vides
    double msmarge   = 0;          ///< ... de combien on releve, en fraction de la cible
    bool memo        = false;      ///< garder d une iteration a l autre les coupes de la feuille
    bool memobits    = true;       ///< ... les coupes de la boite d origine, un bit chacune
};

/// Lire un nuage produit par `cases/gen_cases.py` : des lignes `#` de commentaire, `n`, puis `n`
/// fois « x y w ».
///
/// Pourquoi ce chargeur existe : le tirage uniforme est le seul regime ou une cellule est toujours
/// autour de son germe et ou une grille est bien remplie. Un accelerateur peut n'etre juste que
/// sous ces deux hypotheses sans qu'on le voie -- il faut donc pouvoir lui donner un nuage qui les
/// viole, et le meme d'une execution a l'autre.
///
/// `strtod` sur un tampon lu d'un coup, et non `operator>>` : a 1e6 germes, soit 3e6 nombres, le
/// second coute plusieurs secondes -- plus que la mesure qu'on veut chronometrer.
bool load_cloud( const std::string &path, std::vector<TF> &X, std::vector<TF> &Y, std::vector<TF> &W ) {
    std::FILE *f = std::fopen( path.c_str(), "rb" );
    if ( ! f ) {
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
    if ( ! num( v ) ) { std::printf( "'%s' : pas de nombre de germes\n", path.c_str() ); return false; }
    const SI n = SI( v );
    X.resize( n ); Y.resize( n ); W.resize( n );
    for ( SI i = 0; i < n; ++i ) {
        if ( ! num( X[ i ] ) || ! num( Y[ i ] ) || ! num( W[ i ] ) ) {
            std::printf( "'%s' : coupe apres %d germes sur %d\n", path.c_str(), int( i ), int( n ) );
            return false;
        }
    }
    return true;
}

/// Les poids, A L'ECHELLE : un plan est decale de `dw / ( 2 |d1 - d0| )`, donc pour que le decalage
/// soit une fraction de l'espacement `h ~ n^(-1/2)`, il faut `dw ~ h^2`. Des poids « au hasard entre
/// -1 et 1 » videraient presque toutes les cellules et le banc ne mesurerait plus rien.
const TF *make_weights( const Args &a, SI n, std::mt19937_64 &rng, std::vector<TF> &W ) {
    if ( a.wscale == 0 )
        return nullptr;
    const TF h2 = TF( 1 ) / n;                          // `h^2` avec `h = n^(-1/2)` en 2D
    std::uniform_real_distribution<TF> uw( -1, 1 );
    W.resize( n );
    for ( SI i = 0; i < n; ++i )
        W[ i ] = TF( a.wscale ) * h2 * uw( rng );
    return W.data();
}

/// Le tableau de poids, ou `nullptr` s'ils sont TOUS NULS. Un fichier `voronoi` declare des poids
/// parce que le format en a une colonne, mais des poids nuls sont le cas euclidien : les garder
/// ferait payer au diagramme un majorant identiquement nul (mesure : 12 %), et surtout ferait
/// mesurer autre chose que ce qu'on croit comparer.
const TF *weights_or_null( const std::vector<TF> &W ) {
    for ( TF v : W )
        if ( v != 0 )
            return W.data();
    return nullptr;
}

double now() {
    using namespace std::chrono;
    return duration<double>( steady_clock::now().time_since_epoch() ).count();
}

/// Le corps du banc, une fois l'accelerateur choisi. Le tour de chauffe est HORS chrono, et on
/// garde le MINIMUM : c'est la mesure la moins bruitee d'un temps qu'on veut comparer.
template<class Accel, class Cell, bool CellBox, bool SkipIn, bool Weighted>
int run( const Args &a, const std::vector<TF> &X, const std::vector<TF> &Y, const TF *W ) {
    Accel tree;
    const double t_build_0 = now();
    tree.build( X.data(), Y.data(), W, a.n, a.leaf );
    const double t_build = now() - t_build_0;

    PowerDiagram<Cell, Accel, CellBox, SkipIn, false, Weighted> pd{ tree };
    std::vector<TF> res;

    pd.measures( res, a.threads, a.split, a.pin );      // chauffe

    double best = 1e300;
    for ( int r = 0; r < a.reps; ++r ) {
        const double t0 = now();
        pd.measures( res, a.threads, a.split, a.pin );
        best = std::min( best, now() - t0 );
    }

    TF sum = 0;
    for ( TF v : res )
        sum += v;

    const SI novf = pd.nb_overflow.load();
    std::printf( "  %-6s %-7s nv=%-2d box=%d in=%d n=%d threads=%d leaf=%d : %.3f s (%.0f ns/germe)"
                 "  arbre %.0f ms  somme %.9f\n",
                 Accel::name, a.split == Split::blocks ? "blocks" : "strided",
                 Cell::max_nb_vertices, int( CellBox ), int( SkipIn ),
                 int( a.n ), a.threads, int( a.leaf ), best, best / a.n * 1e9, t_build * 1e3, double( sum ) );
    if ( novf )
        std::printf( "  ATTENTION : %d coupes ont DEBORDE %d sommets -- leur aire est fausse."
                     "  Relancer avec --maxnv %d.\n",
                     int( novf ), Cell::max_nb_vertices, 2 * Cell::max_nb_vertices );

    return std::fabs( double( sum ) - 1.0 ) < 1e-9 && novf == 0 ? 0 : 1;
}

/// La verification : le meme nuage par le balayage COMPLET, qui n'a aucune ligne de code
/// geometrique en commun avec l'arbre. Si les deux s'accordent germe par germe, ce n'est pas deux
/// fois le meme bug.
int check( const Args &a ) {
    std::mt19937_64 rng( a.seed );
    std::uniform_real_distribution<TF> uni( 0.001, 0.999 );
    SI n = 2000;
    std::vector<TF> X, Y, W;
    const TF *Wp = nullptr;

    if ( ! a.load.empty() ) {
        // le balayage complet est en `O( n^2 )` : on ne prend qu'un PREFIXE du fichier. Les germes
        // y sont tires independamment, donc un prefixe suit exactement la meme loi -- et ses poids,
        // qui resolvaient le probleme ENTIER, ne resolvent plus rien pour lui : beaucoup de
        // cellules sont vides, ce qui rend la verification plus severe, pas moins.
        if ( ! load_cloud( a.load, X, Y, W ) )
            return 1;
        n = std::min( SI( X.size() ), SI( 3000 ) );
        X.resize( n ); Y.resize( n ); W.resize( n );
        Wp = weights_or_null( W );
        std::printf( "  nuage '%s', les %d premiers germes\n", a.load.c_str(), int( n ) );
    } else {
        X.resize( n ); Y.resize( n );
        for ( SI i = 0; i < n; ++i ) { X[ i ] = uni( rng ); Y[ i ] = uni( rng ); }
        Wp = make_weights( a, n, rng, W );
    }

    EverySeed   es;  es.build( X.data(), Y.data(), Wp, n, 0 );
    AaBsp       bs;  bs.build( X.data(), Y.data(), Wp, n, a.leaf );
    AaBspPacked pk;  pk.build( X.data(), Y.data(), Wp, n, a.leaf );
    Grid        gr;  gr.build( X.data(), Y.data(), Wp, n, a.leaf );
    AaBspPre    pr;  pr.build( X.data(), Y.data(), Wp, n, a.leaf );
    AaBsp4      b4;  b4.build( X.data(), Y.data(), Wp, n, a.leaf );
    ObBsp       ob;  ob.build( X.data(), Y.data(), Wp, n, a.leaf );
    AaBsp4L     bl;  bl.build( X.data(), Y.data(), Wp, n, a.leaf );

    std::vector<TF> ref, dfs, pkd, grd, prd, q4d, qld, obd;
    PowerDiagram<CellSoA, EverySeed> p_ref{ es };
    PowerDiagram<CellSoA, AaBsp> p_dfs{ bs };
    PowerDiagram<CellSoA, AaBspPacked, false> p_pk{ pk };
    PowerDiagram<CellSoA, Grid, false> p_gr{ gr };
    PowerDiagram<CellSoA, AaBspPre> p_pr{ pr };
    PowerDiagram<CellSoA, AaBsp4> p_q4{ b4 };
    PowerDiagram<CellSoA, AaBsp4L> p_ql{ bl };
    PowerDiagram<CellSoA, ObBsp>  p_ob{ ob };
    p_q4 .measures( q4d, 1, Split::blocks, false );
    p_ob .measures( obd, 1, Split::blocks, false );
    p_ql .measures( qld, 1, Split::blocks, false );
    p_pr .measures( prd, 1, Split::blocks, false );
    p_ref.measures( ref, 1, Split::blocks, false );
    p_dfs.measures( dfs, 1, Split::blocks, false );
    p_pk .measures( pkd, 1, Split::blocks, false );
    p_gr .measures( grd, 1, Split::blocks, false );
    const SI novf = p_ref.nb_overflow.load();
    if ( novf )
        std::printf( "  ATTENTION : %d coupes ont DEBORDE %d sommets (balayage complet)\n",
                     int( novf ), int( CellSoA::max_nb_vertices ) );

    // le balayage complet range ses resultats PAR INDICE D'ORIGINE, comme tous les autres, donc la
    // comparaison est germe par germe et non pas seulement sur la somme -- une cellule fausse d'un
    // cote et fausse a l'envers de l'autre passerait une comparaison de sommes.
    TF sr = 0, md = 0, mp = 0, mg = 0, mr = 0, m4 = 0, ml = 0, mo = 0;
    for ( SI i = 0; i < n; ++i ) {
        sr += ref[ i ];
        md = std::max( md, std::fabs( ref[ i ] - dfs[ i ] ) );
        mp = std::max( mp, std::fabs( ref[ i ] - pkd[ i ] ) );
        mg = std::max( mg, std::fabs( ref[ i ] - grd[ i ] ) );
        mr = std::max( mr, std::fabs( ref[ i ] - prd[ i ] ) );
        m4 = std::max( m4, std::fabs( ref[ i ] - q4d[ i ] ) );
        ml = std::max( ml, std::fabs( ref[ i ] - qld[ i ] ) );
        mo = std::max( mo, std::fabs( ref[ i ] - obd[ i ] ) );
    }
    std::printf( "  balayage complet : somme %.12f\n", double( sr ) );
    std::printf( "  bsp       vs lui : ecart max %.3e\n", double( md ) );
    std::printf( "  bsp4      vs lui : ecart max %.3e\n", double( m4 ) );
    std::printf( "  bsp4l     vs lui : ecart max %.3e\n", double( ml ) );
    std::printf( "  obsp      vs lui : ecart max %.3e\n", double( mo ) );
    std::printf( "  packed    vs lui : ecart max %.3e\n", double( mp ) );
    std::printf( "  grille    vs lui : ecart max %.3e\n", double( mg ) );
    // la pre-passe ne doit RIEN changer : la cellule est l'intersection de tous les demi-plans,
    // donc couper par un sous-ensemble avant le reste doit rendre le meme polygone, au bit pres si
    // l'ordre des coupes effectives est le meme et a l'arrondi sinon.
    std::printf( "  pre-passe vs lui : ecart max %.3e\n", double( mr ) );

    const bool ok = std::fabs( double( sr ) - 1 ) < 1e-12 && md < 1e-12
                 && mp < 1e-12 && mg < 1e-12 && mr < 1e-12 && m4 < 1e-12 && ml < 1e-12 && mo < 1e-12
                 && novf == 0;
    std::printf( "  => %s\n", ok ? "OK" : "ECHEC" );
    return ok ? 0 : 1;
}

/// Ce que le parcours a REELLEMENT fait, par cellule. Un seul thread, compteurs non atomiques.
///
/// Ce qu'on cherche a distinguer : un nuage difficile peut couter cher pour deux raisons opposees,
/// et elles n'appellent pas le meme remede.
///   * la cellule a BEAUCOUP DE COTES (anisotropie, voisinage etendu) -> `coupes effectives` monte,
///     et c'est un maillage adapte a la forme qui aiderait ;
///   * la cellule est LOIN de son germe -> `boites` monte sans que `coupes effectives` bouge : le
///     parcours cherche longtemps avant de trouver de quoi couper, parce qu'il part du mauvais
///     endroit. Aucun maillage ne corrige ca ; il faut partir d'ailleurs.
template<class Accel, class Cell, bool CellBox>
void stats_one( const Accel &tree ) {
    PowerDiagram<Cell, Accel, CellBox, false, true> pd{ tree };
    const SI n = tree.nb_seeds();
    double sv = 0, sr = 0;
    for ( SI k = 0; k < n; ++k ) {
        Cell c;
        pd.make_cell( c, k );
        sv += c.nb;
        const TF p0x = tree.seed_x( k ), p0y = tree.seed_y( k );
        TF m = 0;
        for ( SI v = 0; v < c.nb; ++v ) {
            const TF ex = c.vx[ v ] - p0x, ey = c.vy[ v ] - p0y;
            m = std::max( m, ex * ex + ey * ey );
        }
        sr += std::sqrt( double( m ) );
    }
    const double d = double( n );
    std::printf( "  %-8s boites %8.1f  balayees %7.1f  gardees %6.1f  coupes tentees %6.1f"
                 "  effectives %5.2f  sommets %5.2f  rayon %.4f\n",
                 Accel::name, pd.st_boxes / d, pd.st_sweep / d, pd.st_kept / d, pd.st_tried / d,
                 pd.st_done / d, sv / d, sr / d );
}

int stats( const Args &a, const std::vector<TF> &X, const std::vector<TF> &Y, const TF *W ) {
    AaBsp bs;  bs.build( X.data(), Y.data(), W, a.n, a.leaf );
    Grid  gr;  gr.build( X.data(), Y.data(), W, a.n, a.leaf );
    AaBspPre pr;  pr.build( X.data(), Y.data(), W, a.n, a.leaf );
    AaBspPack pa;  pa.build( X.data(), Y.data(), W, a.n, a.leaf );
    AaBsp4 b4;  b4.build( X.data(), Y.data(), W, a.n, a.leaf );
    AaBsp4L bl;  bl.build( X.data(), Y.data(), W, a.n, a.leaf );
    ObBsp  ob;  ob.build( X.data(), Y.data(), W, a.n, a.leaf );
    stats_one<AaBsp, CellSoAT<64>, true>( bs );
    // « bsp4 » teste QUATRE boites par visite au lieu d une : « boites » doit monter et
    // « gardees » aussi -- ce qui compte est le temps, pas ce compteur.
    stats_one<AaBsp4, CellSoAT<64>, true>( b4 );
    stats_one<AaBsp4L, CellSoAT<64>, true>( bl );
    stats_one<ObBsp, CellSoAT<64>, true>( ob );
    // « pack » : la liste plate. Ce qu'il faut y lire est le nombre de BOITES -- s'il est plus
    // petit que celui du Bsp, c'est que la descente testait surtout des noeuds internes.
    stats_one<AaBspPack, CellSoAT<64>, true>( pa );
    // « pre » compte les DEUX passes : c'est bien le total qu'il faut comparer a « bsp », puisque
    // la pre-passe est un cout qu'on paie pour en economiser un autre.
    stats_one<AaBspPre, CellSoAT<64>, true>( pr );
    stats_one<Grid, CellSoAT<64>, false>( gr );
    return 0;
}

/// LA VERIFICATION A GRANDE ECHELLE : les accelerateurs les uns contre les autres, au `n` demande.
///
/// `--check` est borne a quelques milliers de germes par son oracle en `O( n^2 )`. Or un desaccord
/// peut n'apparaitre qu'en grand -- profondeur d'arbre, voisinages plus fournis, cellules plus
/// fines -- et la somme des aires, qui est le seul controle du banc, ne le montre que s'il ne se
/// compense pas. On compare donc germe par germe, sans oracle, et on nomme les pires.
int cross( const Args &a, const std::vector<TF> &X, const std::vector<TF> &Y, const TF *W ) {
    using Cell = CellSoAT<64>;
    AaBsp    bs;  bs.build( X.data(), Y.data(), W, a.n, a.leaf );
    AaBspHull pk;  pk.build( X.data(), Y.data(), W, a.n, a.leaf );
    Grid     gr;  gr.build( X.data(), Y.data(), W, a.n, a.leaf );

    std::vector<TF> rb, rp, rg;
    PowerDiagram<Cell, AaBsp>          pb{ bs };  pb.measures( rb, 1, Split::blocks, false );
    PowerDiagram<Cell, AaBspHull>      pp{ pk };  pp.measures( rp, 1, Split::blocks, false );
    PowerDiagram<Cell, Grid, false>    pg{ gr };  pg.measures( rg, 1, Split::blocks, false );

    double sb = 0, mp = 0, mg = 0;
    std::vector<std::pair<double,SI>> bad;
    for ( SI i = 0; i < a.n; ++i ) {
        sb += rb[ i ];
        const double dp = std::fabs( double( rb[ i ] - rp[ i ] ) );
        mp = std::max( mp, dp );
        mg = std::max( mg, std::fabs( double( rb[ i ] - rg[ i ] ) ) );
        if ( dp > 1e-15 )
            bad.emplace_back( dp, i );
    }
    std::sort( bad.rbegin(), bad.rend() );
    std::printf( "  bsp : somme %.12f   hull vs bsp : ecart max %.3e (%d germes > 1e-15)"
                 "   grille vs bsp : %.3e\n",
                 sb, mp, int( bad.size() ), mg );
    for ( size_t u = 0; u < std::min<size_t>( bad.size(), 5 ); ++u ) {
        const SI i = bad[ u ].second;
        std::printf( "    germe %7d  ( %.6f, %.6f )  w %.3e   bsp %.6e   hull %.6e   paquet %d de %d\n",
                     int( i ), double( X[ i ] ), double( Y[ i ] ), W ? double( W[ i ] ) : 0.0,
                     double( rb[ i ] ), double( rp[ i ] ),
                     int( pk.leaf_of[ 0 ] ), int( pk.loff.size() - 1 ) );
    }
    return bad.empty() && mg < 1e-12 ? 0 : 1;
}

/// L'ENCLOS PAR SOUS-ECHANTILLON, mesure : de combien la cellule calculee contre `S` seul est trop
/// grande, ce qu'elle coute, et CE QU'ELLE PERMETTRAIT si on s'en servait comme index.
///
/// = Premiere table : l'enclos
///
/// Le pendant 2D exact du tableau de `cases/sub1d.py`. La 1D predisait que le rapport des MESURES
/// vaut `n / |S|` (un enclos contient environ `n / |S|` cellules), donc que celui des RAYONS vaut
/// `sqrt( n / |S| )` en 2D.
///
/// = Seconde table : les PAQUETS, c'est-a-dire l'index qu'on n'a pas encore
///
/// L'idee testee ici : se passer du BSP et n'utiliser que le pavage GROSSIER. On range chaque germe
/// fin `j` dans les cellules grossieres que son enclos rencontre ; les candidats de `k` sont
/// l'union des paquets des cellules grossieres que SON enclos rencontre. Plus une boite a tester,
/// plus de descente -- deux listes contigues.
///
/// Pourquoi c'est un sur-ensemble : si `C( j )` et `C( k )` se touchent en `x`, alors `x` est dans
/// les deux enclos (chacun contient sa cellule), et la cellule grossiere `D_r` qui gagne en `x`
/// rencontre donc les deux enclos. `r` est dans les deux listes, et `j` sort du paquet de `r`.
///
/// Les cellules grossieres qu'un enclos rencontre se lisent GRATUITEMENT : ce sont les `cid` de ses
/// aretes. En effet `D_r inter C_S( j ) = D_r inter { h_j <= h_r }` -- sur `D_r`, `h_r` EST le
/// minimum sur `S` -- donc l'intersection est d'aire non nulle exactement quand la bissectrice
/// `( j, r )` coupe `D_r`, c'est-a-dire quand `r` borde l'enclos. Le seul cas qui echappe est celui
/// d'une cellule grossiere AVALEE tout entiere : elle ne borde rien. C'est pourquoi la colonne
/// « manques » est la : elle compte, sur les vrais voisins de Laguerre de `k`, ceux que la regle ne
/// proposerait pas. Elle doit valoir zero, sans quoi la regle demande un correctif.
int enclos_stats( const Args &a, const std::vector<TF> &X, const std::vector<TF> &Y, const TF *W ) {
    using Cell = CellSoAT<64>;

    auto quant = []( std::vector<double> &v, double q ) {
        return v.empty() ? 0.0 : v[ std::min( v.size() - 1, size_t( q * v.size() ) ) ];
    };
    auto reach = []( const Cell &c, TF p0x, TF p0y ) {
        TF m = 0;
        for ( SI v = 0; v < c.nb; ++v ) {
            const TF ex = c.vx[ v ] - p0x, ey = c.vy[ v ] - p0y;
            m = std::max( m, ex * ex + ey * ey );
        }
        return std::sqrt( double( m ) );
    };

    const SI rates[] = { 2, 4, 8, 16, 32, 64 };
    struct Row { int r; double qr[3], qm[3], nn, qc[3]; long long miss; double bpre, bfull, tried; };
    std::vector<Row> rows;

    for ( SI r : rates ) {
        pre_rate = r;
        AaBspSub t;
        t.build( X.data(), Y.data(), W, a.n, a.leaf );

        const SI n = t.nb_seeds(), m = t.sub.nb_seeds();
        std::vector<SI> loc( a.n, -1 ), pos( a.n, -1 );
        for ( SI u = 0; u < m; ++u ) loc[ t.sub.order[ u ] ] = u;
        for ( SI k = 0; k < n; ++k ) pos[ t.full.order[ k ] ] = k;

        PowerDiagram<Cell, AaBspSub, true, false, true> pe{ t };        // contre `S` seul
        PowerDiagram<Cell, AaBsp,    true, false, true> pf{ t.full };   // la vraie cellule

        // ---- passe 1 : les enclos, et pour chacun la liste des cellules grossieres qu'il borde.
        std::vector<SI> noff( n + 1, 0 ), nval;
        std::vector<double> re( n ), me( n );
        nval.reserve( size_t( 7 ) * n );
        for ( SI k = 0; k < n; ++k ) {
            Cell ce;
            pe.make_cell( ce, k );
            re[ k ] = reach( ce, t.seed_x( k ), t.seed_y( k ) );
            me[ k ] = double( ce.measure() );
            const size_t b0 = nval.size();
            for ( SI v = 0; v < ce.nb; ++v ) {
                const SI id = ce.cid[ v ];
                if ( id < 0 )
                    continue;                       // une arete du DOMAINE, pas un voisin
                bool seen = false;
                for ( size_t u = b0; u < nval.size(); ++u )
                    seen |= nval[ u ] == id;
                if ( ! seen )
                    nval.push_back( id );
            }
            noff[ k + 1 ] = SI( nval.size() );
        }

        // ---- les PAQUETS : la relation inverse, en CSR.
        std::vector<SI> boff( m + 2, 0 ), bval( nval.size() );
        for ( SI v : nval )
            ++boff[ loc[ v ] + 2 ];
        for ( SI u = 1; u < m + 2; ++u )
            boff[ u ] += boff[ u - 1 ];
        for ( SI k = 0; k < n; ++k )
            for ( SI u = noff[ k ]; u < noff[ k + 1 ]; ++u )
                bval[ boff[ loc[ nval[ u ] ] + 1 ]++ ] = k;

        // ---- passe 2 : la vraie cellule, la liste de candidats, et les MANQUES.
        std::vector<SI> stamp( n, -1 );
        std::vector<double> qr, qm, qc;
        qr.reserve( n ); qm.reserve( n ); qc.reserve( n );
        long long miss = 0, snn = 0;
        for ( SI k = 0; k < n; ++k ) {
            Cell cf;
            pf.make_cell( cf, k );

            SI cnt = 0;
            stamp[ k ] = k;                         // le germe lui-meme n'est pas son candidat
            for ( SI u = noff[ k ]; u < noff[ k + 1 ]; ++u ) {
                const SI rr = loc[ nval[ u ] ];
                for ( SI w = boff[ rr ]; w < boff[ rr + 1 ]; ++w ) {
                    const SI j = bval[ w ];
                    if ( stamp[ j ] != k ) { stamp[ j ] = k; ++cnt; }
                }
            }
            snn += noff[ k + 1 ] - noff[ k ];
            qc.push_back( double( cnt ) );

            for ( SI v = 0; v < cf.nb; ++v ) {      // les VRAIS voisins sont-ils tous proposes ?
                const SI id = cf.cid[ v ];
                if ( id >= 0 && stamp[ pos[ id ] ] != k )
                    ++miss;
            }

            const double rf = reach( cf, t.seed_x( k ), t.seed_y( k ) );
            if ( rf > 0 ) qr.push_back( re[ k ] / rf );
            const double mf = double( cf.measure() );
            if ( mf > 0 ) qm.push_back( me[ k ] / mf );
        }
        std::sort( qr.begin(), qr.end() );
        std::sort( qm.begin(), qm.end() );
        std::sort( qc.begin(), qc.end() );

        const double d = double( n );
        rows.push_back( Row{ int( r ),
            { quant( qr, .5 ), quant( qr, .9 ), qr.empty() ? 0 : qr.back() },
            { quant( qm, .5 ), quant( qm, .9 ), qm.empty() ? 0 : qm.back() },
            double( snn ) / d,
            { quant( qc, .5 ), quant( qc, .9 ), qc.empty() ? 0 : qc.back() },
            miss, pe.st_boxes / d, pf.st_boxes / d, pf.st_tried / d } );
    }

    std::printf( "  %-6s %28s %28s %17s\n", "", "RAYON enclos / vrai",
                 "MESURE enclos / vraie", "boites/cellule" );
    std::printf( "  %-6s %8s %8s %8s   %8s %8s %8s   %8s %8s\n", "n/|S|",
                 "median", "dec. 9", "max", "median", "dec. 9", "max", "pre", "pleine" );
    for ( const Row &w : rows )
        std::printf( "  %-6d %8.2f %8.2f %8.1f   %8.1f %8.1f %8.0f   %8.1f %8.1f\n", w.r,
                     w.qr[ 0 ], w.qr[ 1 ], w.qr[ 2 ], w.qm[ 0 ], w.qm[ 1 ], w.qm[ 2 ],
                     w.bpre, w.bfull );

    std::printf( "\n  LES PAQUETS : ce que couterait un index fait du seul pavage grossier\n" );
    std::printf( "  %-6s %9s %27s %8s %19s\n", "n/|S|", "cellules", "CANDIDATS par cellule",
                 "", "le BSP d'aujourd'hui" );
    std::printf( "  %-6s %9s %8s %8s %8s %8s   %9s %9s\n", "", "grossieres",
                 "median", "dec. 9", "max", "manques", "tentees", "boites" );
    for ( const Row &w : rows )
        std::printf( "  %-6d %9.2f %8.0f %8.0f %8.0f %8lld   %9.1f %9.1f\n", w.r, w.nn,
                     w.qc[ 0 ], w.qc[ 1 ], w.qc[ 2 ], w.miss, w.tried, w.bfull );
    return 0;
}

/// LE CRITERE DE L'ENVELOPPE SUR UNE GRILLE : le troisieme index candidat, mesure avant d'etre ecrit.
///
/// = Le critere, et il est exact
///
/// Sur une boite `B` de l'espace, `min_B psi = min_i min_B h_i` se calcule EXACTEMENT, et
///
///     M( B ) := min_i max_B h_i     majore     max_B psi
///
/// puisque `psi <= h_i` partout, pour tout `i`. D'ou, pour un dirac `i` :
///
///     min_B h_i > M( B )   =>   C( i ) inter B = vide
///
/// car alors `h_i > M( B ) >= max_B psi >= psi` en tout point de `B`. C'est un critere que la
/// grille actuelle N'A PAS : elle s'arrete sur un rayon, d'ou ses 4546 boites par cellule sur le
/// cas dur. Et il est STATIQUE -- `M` se calcule une fois pour tous les diracs, alors que le test
/// du BSP depend de la cellule en cours.
///
/// = Ce qui est mesure ici
///
/// `M( B )` est pris comme `max_B h_r` ou `r` est le dirac qui GAGNE au centre de `B` (obtenu en
/// rasterisant les vraies cellules). C'est un `min` sur un singleton, donc un majorant valide, et
/// le plus serre qu'un seul dirac puisse donner.
///
/// Puis, boite par boite, on compte les diracs que le critere retient ; on inverse ; et pour chaque
/// dirac on prend l'union des listes des boites ou il a ete retenu -- ses candidats. La colonne
/// « manques » verifie que ses VRAIS voisins de Laguerre y sont tous : ici elle doit valoir zero
/// par construction, le critere etant une implication et non une heuristique.
///
/// = L'ecueil qu'on ne mesure pas ici
///
/// L'enonce parle d'INONDATION : partir de la boite du dirac et s'etendre aux voisines. Mais rien
/// ne garantit que la boite de `p_i` soit retenue -- sur le cas dur, un autre dirac gagne en `p_i`,
/// c'est toute la difficulte du nuage. On construit donc par BOITE et non par dirac, ce qui evite
/// la question du point de depart et celle de la connexite du domaine retenu.
/// QUI GAGNE EN `x` : `argmin_j ( |x - p_j|^2 - w_j )`, cherche dans l'arbre. Rend l'indice
/// D'ORIGINE. C'est la seule chose qu'un pavage ait besoin de stocker pour porter un majorant
/// AFFINE de `psi` : toutes les paraboles partagent `|x|^2`, donc `psi - |x|^2` est un min de
/// fonctions affines, donc CONCAVE, et un majorant affine minimal en est un hyperplan d'appui --
/// c'est-a-dire la partie affine de la parabole d'un dirac qui gagne quelque part dans la tuile.
/// Stocker le proprietaire EST stocker un majorant affine.
SI gagnant( const AaBsp &tr, TF x, TF y ) {
    TF best = 1e300;
    SI bi = -1;
    tr.for_each_candidate_at( x, y, -1,
        [ & ]( TF lox, TF loy, TF hix, TF hiy, const WMaj &wm ) {
            const TF ex = x < lox ? lox - x : ( x > hix ? x - hix : TF( 0 ) );
            const TF ey = y < loy ? loy - y : ( y > hiy ? y - hiy : TF( 0 ) );
            const TF sx = TF( wm.ax ) * lox, tx = TF( wm.ax ) * hix;
            const TF sy = TF( wm.ay ) * loy, ty = TF( wm.ay ) * hiy;
            const TF wmax = ( sx > tx ? sx : tx ) + ( sy > ty ? sy : ty ) + wm.b;
            return ex * ex + ey * ey - wmax < best;
        },
        [ & ]( TF px, TF py, TF w, SI id ) {
            const TF dx = px - x, dy = py - y;
            const TF h = dx * dx + dy * dy - w;
            if ( h < best ) { best = h; bi = id; }
            return true;
        },
        [] { return TF( 0 ); } );
    return bi;
}

/// LE FRONT SUR UNE GRILLE REGULIERE.
///
/// L'objectif : remplacer la marche dans le BSP par un ETALEMENT de proche en proche, tant que la
/// parabole du dirac peut passer sous le majorant. Trois choses a etablir, et c'est ce que cette
/// mesure fait :
///
///   1. L'AMORCE. Le front est valide depuis n'importe quelle tuile rencontrant `Lag_i` -- celle-ci
///      etant convexe, les tuiles qu'elle rencontre forment un ensemble CONNEXE, et elles passent
///      toutes le critere. Mais la tuile de `p_i` n'est PAS garantie retenue : sur le cas dur c'est
///      un autre dirac qui gagne en `p_i`. On amorce donc par une DESCENTE sur
///      `phi_i( x ) = h_i( x ) - psi( x )`, qui est un MAX de fonctions affines donc CONVEXE, et
///      qui vaut zero exactement sur `Lag_i`. Au centre d'une tuile, `psi` vaut exactement
///      `h_proprietaire`, donc `phi_i` s'y evalue sans rien chercher.
///
///   2. LA TAILLE DU FRONT. C'est elle, et non le nombre de candidats, qui remplacera les 42 a 135
///      tests de boite du BSP.
///
///   3. QU'IL NE MANQUE RIEN. On rasterise la VRAIE cellule et on verifie que chacune de ses tuiles
///      est dans le front.
int front_stats( const Args &a, const std::vector<TF> &X, const std::vector<TF> &Y, const TF *W ) {
    using Cell = CellSoAT<64>;
    AaBsp bs;
    bs.build( X.data(), Y.data(), W, a.n, a.leaf );
    PowerDiagram<Cell, AaBsp, true, false, true> pf{ bs };
    const SI n = a.n;
    auto poids = [ & ]( SI i ) { return W ? W[ i ] : TF( 0 ); };

    std::printf( "  %-5s %9s %8s   %8s %8s %8s   %8s %8s %9s\n", "g", "tuiles", "orphel.",
                 "marche", "echecs", "hors", "front", "front max", "manques" );

    for ( SI g : { SI( 64 ), SI( 128 ), SI( 256 ), SI( 512 ) } ) {
        const SI nb = g * g;
        const TF hh = TF( 1 ) / g;

        // ---- A. LE PROPRIETAIRE de chaque tuile, par requete. Parallele sans partage, une requete
        //         par tuile -- et c'est la seule chose pour laquelle le BSP reste necessaire.
        std::vector<SI> owner( nb, -1 );
        const double t0 = now();
        parallel_for( nb, a.threads, Split::blocks, false, [ & ]( SI b, int ) {
            owner[ b ] = gagnant( bs, ( b % g + TF( 0.5 ) ) * hh, ( b / g + TF( 0.5 ) ) * hh );
        } );
        const double t_own = now() - t0;

        SI orphelines = 0;
        for ( SI b = 0; b < nb; ++b )
            orphelines += owner[ b ] < 0;

        // `min_B ( h_i - h_r )`, exact : les deux paraboles ont le meme `|x|^2`, donc la difference
        // est AFFINE et son minimum sur la boite est a un coin. `|p_i|^2 - |p_r|^2` est ecrit
        // `dx ( px + rx )` et non litteralement : developpe, il ne rend pas zero quand `i == r`
        // (cf. le meme piege dans `--psigrid`), et le proprietaire s'excluait de sa propre liste.
        auto ecart = [ & ]( SI i, SI b, bool au_centre ) {
            const SI r = owner[ b ];
            const TF dx = X[ i ] - X[ r ], dy = Y[ i ] - Y[ r ];
            const TF e = dx * ( X[ i ] + X[ r ] ) + dy * ( Y[ i ] + Y[ r ] ) - poids( i ) + poids( r );
            const TF x0 = ( b % g ) * hh, y0 = ( b / g ) * hh;
            if ( au_centre )
                return e - 2 * ( dx * ( x0 + hh / 2 ) + dy * ( y0 + hh / 2 ) );
            return e - 2 * ( dx > 0 ? dx * ( x0 + hh ) : dx * x0 )
                     - 2 * ( dy > 0 ? dy * ( y0 + hh ) : dy * y0 );
        };

        // ---- B. LE FRONT, par dirac
        std::vector<double> lg_desc( n, 0 ), lg_front( n, 0 );
        std::atomic<SI> echecs{ 0 }, hors{ 0 }, manques{ 0 }, front_max{ 0 };
        const int nth = std::max( a.threads, 1 );
        std::vector<std::vector<SI>> vu( nth, std::vector<SI>( nb, -1 ) );
        std::vector<std::vector<SI>> pile( nth );

        const double t1 = now();
        parallel_for( n, a.threads, Split::blocks, false, [ & ]( SI k, int th ) {
            const SI i = bs.seed_id( k );

            // 1. L'AMORCE. Le front n'est valide que depuis une tuile qui RENCONTRE `Lag_i` -- pas
            //    seulement une tuile retenue : l'ensemble retenu peut avoir plusieurs composantes,
            //    et partir de la mauvaise fait manquer la cellule.
            //
            //    On descend `phi_i( x ) = h_i( x ) - psi( x )`, qui est un MAX de fonctions affines
            //    donc CONVEXE, nul exactement sur `Lag_i`. Au centre d'une tuile, `psi` vaut
            //    exactement `h_proprietaire`, donc `phi_i` s'y evalue sans rien chercher. Le
            //    CERTIFICAT est `proprietaire == i` : le centre est alors dans `Lag_i`.
            //
            //    ESSAYE ET REJETE : suivre la direction de descente `p_i - p_r` (« s'eloigner du
            //    concurrent »), qui est le vrai gradient de la piece affine courante. C'est une
            //    marche de visibilite ordinaire, mais elle n'a pas de critere d'arret utilisable :
            //    quand le dirac ne possede AUCUNE tuile -- 96 % d'entre eux a g=64 -- elle court
            //    jusqu'a sa borne. Mesure : 480 a 2810 pas contre 7 a 58 pour le glouton.
            SI b = std::min<SI>( g - 1, SI( Y[ i ] / hh ) ) * g + std::min<SI>( g - 1, SI( X[ i ] / hh ) );
            SI pas = 0;
            for ( ; owner[ b ] != i && pas < 4 * g; ++pas ) {
                const TF cur = ecart( i, b, true );
                SI best = b;
                TF bv = cur;
                const SI bx = b % g, by = b / g;
                for ( int dy = -1; dy <= 1; ++dy )
                    for ( int dx = -1; dx <= 1; ++dx ) {
                        const SI nx = bx + dx, ny = by + dy;
                        if ( ( dx == 0 && dy == 0 ) || nx < 0 || ny < 0 || nx >= g || ny >= g )
                            continue;
                        const TF v = ecart( i, ny * g + nx, true );
                        if ( v < bv ) { bv = v; best = ny * g + nx; }
                    }
                if ( best == b )
                    break;
                b = best;
            }
            lg_desc[ i ] = pas;

            // `hors` : on n'a PAS le certificat -- soit la cellule est plus petite qu'une tuile et
            // aucun centre ne lui appartient, soit le glouton a cale. On part quand meme de la
            // tuile atteinte si elle est retenue, et on compte separement.
            if ( owner[ b ] != i ) {
                ++hors;
                if ( ! ( ecart( i, b, false ) <= 0 ) ) {
                    ++echecs;
                    ++manques;                          // sans amorce du tout, la cellule est manquee
                    return;
                }
            }

            // 2. L'ETALEMENT : tant que la parabole peut passer sous le majorant, au sens large.
            std::vector<SI> &pi = pile[ th ];
            std::vector<SI> &vi = vu[ th ];
            pi.clear();
            pi.push_back( b );
            vi[ b ] = i;
            SI nf = 0;
            for ( SI t = 0; t < SI( pi.size() ); ++t ) {
                const SI c = pi[ t ];
                ++nf;
                const SI cx = c % g, cy = c / g;
                for ( int dy = -1; dy <= 1; ++dy )
                    for ( int dx = -1; dx <= 1; ++dx ) {
                        const SI nx = cx + dx, ny = cy + dy;
                        if ( nx < 0 || ny < 0 || nx >= g || ny >= g )
                            continue;
                        const SI d = ny * g + nx;
                        if ( vi[ d ] == i )
                            continue;
                        if ( ecart( i, d, false ) <= 0 ) { vi[ d ] = i; pi.push_back( d ); }
                    }
            }
            lg_front[ i ] = nf;
            SI fm = front_max.load( std::memory_order_relaxed );
            while ( nf > fm && ! front_max.compare_exchange_weak( fm, nf ) )
                ;

            // 3. LE CONTROLE : la vraie cellule, rasterisee, doit etre entierement dans le front.
            Cell c;
            pf.make_cell( c, k );
            if ( ! c.nb )
                return;
            TF lox, loy, hix, hiy;
            c.bounds( lox, loy, hix, hiy );
            const SI i0 = std::max<SI>( 0, SI( lox / hh ) ), i1 = std::min<SI>( g - 1, SI( hix / hh ) );
            const SI j0 = std::max<SI>( 0, SI( loy / hh ) ), j1 = std::min<SI>( g - 1, SI( hiy / hh ) );
            for ( SI jj = j0; jj <= j1; ++jj )
                for ( SI ii = i0; ii <= i1; ++ii ) {
                    const TF qx = ( ii + TF( 0.5 ) ) * hh, qy = ( jj + TF( 0.5 ) ) * hh;
                    bool in = true;
                    for ( SI v = 0; v < c.nb && in; ++v )
                        in = c.cdx[ v ] * qx + c.cdy[ v ] * qy <= c.co[ v ];
                    if ( in && vu[ th ][ jj * g + ii ] != i )
                        ++manques;
                }
        } );
        const double t_front = now() - t1;

        double sd = 0, sf = 0;
        for ( SI i = 0; i < n; ++i ) { sd += lg_desc[ i ]; sf += lg_front[ i ]; }
        std::printf( "  %-5d %9d %8d   %8.2f %8d %8d   %8.1f %8d %9d   [%.0f ms + %.0f ms]\n",
                     int( g ), int( nb ), int( orphelines ), sd / n, int( echecs.load() ),
                     int( hors.load() ), sf / n, int( front_max.load() ), int( manques.load() ),
                     1e3 * t_own, 1e3 * t_front );
    }
    return 0;
}

int psigrid_stats( const Args &a, const std::vector<TF> &X, const std::vector<TF> &Y, const TF *W ) {
    using Cell = CellSoAT<64>;
    auto quant = []( std::vector<double> &v, double q ) {
        return v.empty() ? 0.0 : v[ std::min( v.size() - 1, size_t( q * v.size() ) ) ];
    };

    AaBsp bs;
    bs.build( X.data(), Y.data(), W, a.n, a.leaf );
    PowerDiagram<Cell, AaBsp, true, false, true> pf{ bs };
    const SI n = bs.nb_seeds();

    std::vector<SI> pos( a.n, -1 );
    for ( SI k = 0; k < n; ++k )
        pos[ bs.order[ k ] ] = k;

    struct Row { int g; double ql[ 3 ], qc[ 3 ]; long long miss, vrais; double maxlen; };

    std::printf( "  %-7s %7s %31s   %31s\n", "", "", "M SCALAIRE", "BISSECTRICE (proprietaire)" );
    std::printf( "  %-7s %7s %7s %7s %6s   %7s %7s %6s %6s %10s\n", "g x g", "diracs",
                 "retenus", "candid.", "manq.", "retenus", "candid.", "manq.", "vrais", "arete max" );

    for ( SI g : { SI( 32 ), SI( 64 ), SI( 128 ), SI( 256 ) } ) {
        const SI nb = g * g;
        const TF hh = TF( 1 ) / g;

        // ---- A. le proprietaire du centre de chaque boite, par RASTERISATION des vraies cellules.
        //         Les cellules pavent le domaine, donc chaque centre en a exactement un.
        std::vector<SI> owner( nb, -1 );
        for ( SI k = 0; k < n; ++k ) {
            Cell c;
            pf.make_cell( c, k );
            if ( ! c.nb )
                continue;
            TF lox, loy, hix, hiy;
            c.bounds( lox, loy, hix, hiy );
            const SI i0 = std::max<SI>( 0, SI( lox / hh ) ), i1 = std::min<SI>( g - 1, SI( hix / hh ) );
            const SI j0 = std::max<SI>( 0, SI( loy / hh ) ), j1 = std::min<SI>( g - 1, SI( hiy / hh ) );
            for ( SI j = j0; j <= j1; ++j )
                for ( SI i = i0; i <= i1; ++i ) {
                    const TF cx = ( i + TF( 0.5 ) ) * hh, cy = ( j + TF( 0.5 ) ) * hh;
                    bool in = true;
                    for ( SI v = 0; v < c.nb && in; ++v )
                        in = c.cdx[ v ] * cx + c.cdy[ v ] * cy <= c.co[ v ];
                    if ( in )
                        owner[ j * g + i ] = k;
                }
        }

        // ---- B. `M[ b ] = max_B h_owner`, donc le maximum sur les QUATRE COINS (`|x-p|^2` est
        //         convexe, son maximum sur une boite est a un coin).
        std::vector<TF> M( nb, 0 );
        SI orphelines = 0;
        for ( SI b = 0; b < nb; ++b ) {
            const SI k = owner[ b ];
            if ( k < 0 ) { M[ b ] = 1e300; ++orphelines; continue; }
            const TF px = bs.seed_x( k ), py = bs.seed_y( k ), pw = bs.seed_w( k );
            const TF x0 = ( b % g ) * hh, y0 = ( b / g ) * hh;
            TF m = -1e300;
            for ( int u = 0; u < 4; ++u ) {
                const TF ex = x0 + ( u & 1 ) * hh - px, ey = y0 + ( u >> 1 ) * hh - py;
                m = std::max( m, ex * ex + ey * ey - pw );
            }
            M[ b ] = m;
        }

        // ---- C/D. par boite, les diracs que le critere retient ; puis la relation inverse et,
        //           pour chaque dirac, l'union des listes des boites ou il a ete retenu.
        //
        // DEUX criteres, et la difference est tout l'interet de la mesure :
        //
        //   `M` SCALAIRE  -- `min_B h_i > M( B )`, ce que la grille stocke litteralement. Son mou
        //      vaut au moins l'OSCILLATION de `psi` sur la boite, puisque `M( B )` doit majorer
        //      `max_B psi` alors qu'on le compare a un minimum.
        //
        //   BISSECTRICE   -- `min_B ( h_i - h_r ) > 0` avec `r` le proprietaire de la boite. C'est
        //      strictement plus fort, au meme cout : `psi <= h_r`, donc `h_i > h_r` sur `B` suffit.
        //      Et `h_i - h_r` est AFFINE -- le `|x|^2` s'en va -- donc son minimum sur la boite est
        //      exact, a un coin. Le mou en `osc( psi )` disparait : ce qui reste est « le dirac `i`
        //      bat-il le proprietaire quelque part dans la boite ».
        auto mesure = [ & ]( bool bissectrice, Row &w ) {
            std::vector<std::vector<SI>> lst( nb );
            parallel_for( nb, a.threads, Split::blocks, false, [ & ]( SI b, int ) {
                const TF x0 = ( b % g ) * hh, y0 = ( b / g ) * hh;
                const TF x1 = x0 + hh, y1 = y0 + hh;
                const TF mb = M[ b ];
                const SI r = owner[ b ];
                const TF rx = bs.seed_x( r ), ry = bs.seed_y( r ), rw = bs.seed_w( r );
                std::vector<SI> &o = lst[ b ];
                for ( SI k = 0; k < n; ++k ) {
                    const TF px = bs.seed_x( k ), py = bs.seed_y( k ), pw = bs.seed_w( k );
                    bool keep;
                    if ( bissectrice ) {
                        // `|p_i|^2 - |p_r|^2` NON pas litteralement mais en `dx ( px + rx )` :
                        // ecrite telle quelle, la difference des carres ne rend pas zero quand
                        // `i == r` (`fl( px^2 + py^2 ) - px^2 - py^2` vaut environ 1e-19, parfois
                        // POSITIF), et le proprietaire de la boite s'excluait alors de sa propre
                        // liste -- 8700 voisins manques sur l'uniforme a g=256, dont des aretes de
                        // 7.5e-3, plus longues qu'une cellule entiere. Meme lecon que pour
                        // l'interpolation de `Cell::cut` : la forme factorisee est exacte la ou
                        // l'autre ne l'est pas.
                        const TF dx = px - rx, dy = py - ry;
                        const TF e = dx * ( px + rx ) + dy * ( py + ry ) - pw + rw;
                        keep = e - 2 * ( dx > 0 ? dx * x1 : dx * x0 )
                                 - 2 * ( dy > 0 ? dy * y1 : dy * y0 ) <= 0;
                    } else {
                        const TF ex = px < x0 ? x0 - px : ( px > x1 ? px - x1 : TF( 0 ) );
                        const TF ey = py < y0 ? y0 - py : ( py > y1 ? py - y1 : TF( 0 ) );
                        keep = ex * ex + ey * ey - pw <= mb;
                    }
                    if ( keep )
                        o.push_back( k );
                }
            } );

            std::vector<SI> boff( n + 2, 0 );
            for ( const std::vector<SI> &o : lst )
                for ( SI k : o )
                    ++boff[ k + 2 ];
            for ( SI u = 1; u < n + 2; ++u )
                boff[ u ] += boff[ u - 1 ];
            std::vector<SI> bval( boff[ n + 1 ] );
            for ( SI b = 0; b < nb; ++b )
                for ( SI k : lst[ b ] )
                    bval[ boff[ k + 1 ]++ ] = b;

            std::vector<double> ql, qc;
            for ( const std::vector<SI> &o : lst )
                ql.push_back( double( o.size() ) );
            std::vector<SI> stamp( n, -1 );
            long long miss = 0, vrais = 0;
            double maxlen = 0;
            for ( SI k = 0; k < n; ++k ) {
                SI cnt = 0;
                stamp[ k ] = k;
                for ( SI u = boff[ k ]; u < boff[ k + 1 ]; ++u )
                    for ( SI j : lst[ bval[ u ] ] )
                        if ( stamp[ j ] != k ) { stamp[ j ] = k; ++cnt; }
                qc.push_back( double( cnt ) );

                // Un voisin « manque » -- mais `cid` marque une coupe meme quand l'arete qu'elle
                // porte a ete reduite a RIEN par les coupes suivantes. Deux cellules qui ne se
                // touchent qu'en un point ne sont pas voisines, et aucun index n'a a les proposer.
                // On mesure donc AUSSI la longueur de l'arete manquee : si elle est nulle, c'est la
                // reference qui est sale, pas le critere.
                Cell cf;
                pf.make_cell( cf, k );
                for ( SI v = 0; v < cf.nb; ++v ) {
                    const SI id = cf.cid[ v ];
                    if ( id < 0 || stamp[ pos[ id ] ] == k )
                        continue;
                    ++miss;
                    const SI vn = ( v + 1 ) % cf.nb;
                    const TF ex = cf.vx[ vn ] - cf.vx[ v ], ey = cf.vy[ vn ] - cf.vy[ v ];
                    const double len = std::sqrt( double( ex * ex + ey * ey ) );
                    maxlen = std::max( maxlen, len );
                    if ( len > 1e-9 )
                        ++vrais;
                    if ( bissectrice && len > 1e-6 && vrais <= 3 ) {
                        // le MILIEU de l'arete manquee est, par construction, dans les deux
                        // cellules : la boite qui le contient doit retenir les deux germes.
                        const TF mx = ( cf.vx[ v ] + cf.vx[ vn ] ) / 2;
                        const TF my = ( cf.vy[ v ] + cf.vy[ vn ] ) / 2;
                        const SI bi = std::min<SI>( g - 1, SI( mx / hh ) );
                        const SI bj = std::min<SI>( g - 1, SI( my / hh ) );
                        const SI b  = bj * g + bi;
                        const SI kj = pos[ id ], r = owner[ b ];
                        bool ini = false, inj = false;
                        for ( SI q : lst[ b ] ) { ini |= q == k; inj |= q == kj; }
                        std::printf( "    MANQUE g=%d : k=%d (%.6f,%.6f) j=%d (%.6f,%.6f) "
                                     "arete %.3e milieu (%.6f,%.6f) boite %d "
                                     "[%d germes, proprio %d] k dedans=%d j dedans=%d\n",
                                     int( g ), int( k ), double( bs.seed_x( k ) ), double( bs.seed_y( k ) ),
                                     int( kj ), double( bs.seed_x( kj ) ), double( bs.seed_y( kj ) ),
                                     len, double( mx ), double( my ), int( b ),
                                     int( lst[ b ].size() ), int( r ), int( ini ), int( inj ) );
                    }
                }
            }
            std::sort( ql.begin(), ql.end() );
            std::sort( qc.begin(), qc.end() );
            w = Row{ int( g ), { quant( ql, .5 ), quant( ql, .9 ), ql.back() },
                                { quant( qc, .5 ), quant( qc, .9 ), qc.back() }, miss, vrais, maxlen };
        };

        if ( orphelines )
            std::printf( "  (g=%d : %d boites sur %d SANS proprietaire)\n",
                         int( g ), int( orphelines ), int( nb ) );
        Row wa, wb;
        mesure( false, wa );
        mesure( true,  wb );
        std::printf( "  %-7d %7.1f %7.0f %7.0f %6lld   %7.0f %7.0f %6lld %6lld %10.2e\n",
                     int( g ), double( n ) / nb,
                     wa.ql[ 0 ], wa.qc[ 0 ], wa.miss,
                     wb.ql[ 0 ], wb.qc[ 0 ], wb.miss, wb.vrais, wb.maxlen );
    }
    // la reference, sur des compteurs NEUFS : `pf` en a accumule autant de passes qu'il y a de
    // resolutions, et un diagnostic qui se compare a un chiffre faux ne vaut rien.
    PowerDiagram<Cell, AaBsp, true, false, true> pr{ bs };
    for ( SI k = 0; k < n; ++k ) { Cell c; pr.make_cell( c, k ); }
    std::printf( "  (le BSP d'aujourd'hui : %.1f coupes tentees et %.1f boites par cellule)\n",
                 pr.st_tried / double( n ), pr.st_boxes / double( n ) );
    return 0;
}

/// CE QUE VAUDRAIT UN MAJORANT DE DEGRE 2, mesure AVANT de l'ecrire.
///
/// Un majorant vaut ce que vaut l'ETALEMENT de ses residus, `max( w - q ) - min( w - q )` : c'est
/// exactement le mou de la borne, puisque `b` est releve jusqu'au pire germe et que tous les autres
/// paient l'ecart. On compare donc, noeud par noeud, l'etalement de quatre ajustements :
///
///   constant  [ 1 ]                        -- la borne classique
///   affine    [ 1, x, y ]                  -- celle qu'on vient de mettre, en O( 1 )
///   diagonal  [ 1, x, y, x^2, y^2 ]        -- degre 2 SANS terme croise, encore en O( 1 )
///   complet   [ 1, x, y, x^2, y^2, x y ]   -- degre 2 entier, qui lui n'est PAS separable
///
/// Le complet n'est pas utilisable ; il est la pour dire si c'est le degre qui manque ou la
/// restriction a la diagonale.
///
/// La precaution qui decide de tout : un ajustement a `k` parametres sur `m` points resserre
/// l'etalement MEME QUAND IL N'Y A RIEN A AJUSTER, d'un facteur `sqrt( 1 - ( k - 1 ) / ( m - 1 ) )`.
/// Une feuille a dix germes et le diagonal cinq parametres : il y gagnerait 0.75 sur du bruit pur.
/// On affiche donc ce plancher a cote, et seul ce qui passe NETTEMENT dessous compte.
double fit_spread( const std::vector<double> &B, const std::vector<double> &w, int m, int nb ) {
    // equations normales `B^T B c = B^T w`, resolues par Gauss avec pivot partiel. `nb <= 6`, donc
    // le cout ne compte pas : c'est un diagnostic, pas un chemin chaud.
    std::vector<double> A( nb * nb, 0 ), r( nb, 0 );
    for ( int i = 0; i < m; ++i ) {
        for ( int u = 0; u < nb; ++u ) {
            r[ u ] += B[ i * nb + u ] * w[ i ];
            for ( int v = 0; v < nb; ++v )
                A[ u * nb + v ] += B[ i * nb + u ] * B[ i * nb + v ];
        }
    }
    for ( int u = 0; u < nb; ++u ) {
        int piv = u;
        for ( int v = u + 1; v < nb; ++v )
            if ( std::fabs( A[ v * nb + u ] ) > std::fabs( A[ piv * nb + u ] ) )
                piv = v;
        if ( std::fabs( A[ piv * nb + u ] ) < 1e-300 )
            return -1;                                  // systeme degenere : on ne compte pas
        if ( piv != u ) {
            for ( int v = 0; v < nb; ++v )
                std::swap( A[ u * nb + v ], A[ piv * nb + v ] );
            std::swap( r[ u ], r[ piv ] );
        }
        for ( int v = u + 1; v < nb; ++v ) {
            const double f = A[ v * nb + u ] / A[ u * nb + u ];
            for ( int k = u; k < nb; ++k )
                A[ v * nb + k ] -= f * A[ u * nb + k ];
            r[ v ] -= f * r[ u ];
        }
    }
    for ( int u = nb - 1; u >= 0; --u ) {
        for ( int v = u + 1; v < nb; ++v )
            r[ u ] -= A[ u * nb + v ] * r[ v ];
        r[ u ] /= A[ u * nb + u ];
    }

    double lo = 1e300, hi = -1e300;
    for ( int i = 0; i < m; ++i ) {
        double f = 0;
        for ( int u = 0; u < nb; ++u )
            f += B[ i * nb + u ] * r[ u ];
        const double e = w[ i ] - f;
        lo = std::min( lo, e ); hi = std::max( hi, e );
    }
    return hi - lo;
}

int majorant_stats( const Args &a, const std::vector<TF> &X, const std::vector<TF> &Y, const TF *W ) {
    if ( ! W ) {
        std::printf( "  pas de poids : rien a majorer\n" );
        return 0;
    }
    AaBsp bs;
    bs.build( X.data(), Y.data(), W, a.n, a.leaf );

    // par TAILLE de noeud : c'est elle qui dit si le majorant sert a rejeter un sous-arbre entier
    // (gros noeud, haut dans l'arbre) ou juste une feuille.
    struct Acc { double s1 = 0, s2 = 0, s3 = 0, chance2 = 0, chance3 = 0; long long m = 0, n = 0; };
    std::vector<Acc> by( 32 );

    for ( const AaBsp::Node &nd : bs.nodes ) {
        const int m = int( nd.end - nd.beg );
        if ( m < 12 )
            continue;                                   // trop peu de points pour que 6 parametres aient un sens
        int bucket = 0;
        while ( ( 1 << ( bucket + 1 ) ) <= m )
            ++bucket;

        double mx = 0, my = 0;
        for ( SI k = nd.beg; k < nd.end; ++k ) { mx += bs.px[ k ]; my += bs.py[ k ]; }
        mx /= m; my /= m;

        std::vector<double> B( size_t( m ) * 6 ), w( m );
        for ( int i = 0; i < m; ++i ) {
            const SI k = nd.beg + i;
            const double qx = bs.px[ k ] - mx, qy = bs.py[ k ] - my;
            B[ i * 6 + 0 ] = 1; B[ i * 6 + 1 ] = qx; B[ i * 6 + 2 ] = qy;
            B[ i * 6 + 3 ] = qx * qx; B[ i * 6 + 4 ] = qy * qy; B[ i * 6 + 5 ] = qx * qy;
            w[ i ] = bs.pw[ k ];
        }
        // les colonnes ne sont pas contigues pour `nb < 6` : on recopie le prefixe voulu.
        auto sub = [ & ]( int nb ) {
            std::vector<double> C( size_t( m ) * nb );
            for ( int i = 0; i < m; ++i )
                for ( int u = 0; u < nb; ++u )
                    C[ i * nb + u ] = B[ i * 6 + u ];
            return fit_spread( C, w, m, nb );
        };

        const double s0 = sub( 1 ), s1 = sub( 3 ), s2 = sub( 5 ), s3 = fit_spread( B, w, m, 6 );
        if ( s0 <= 0 || s1 < 0 || s2 < 0 || s3 < 0 )
            continue;
        Acc &A = by[ bucket ];
        A.s1 += s1 / s0; A.s2 += s2 / s0; A.s3 += s3 / s0;
        A.chance2 += std::sqrt( std::max( 0.0, 1.0 - 4.0 / ( m - 1 ) ) );
        A.chance3 += std::sqrt( std::max( 0.0, 1.0 - 5.0 / ( m - 1 ) ) );
        A.m += m; ++A.n;
    }

    std::printf( "  etalement des residus, RAPPORTE au majorant constant (plus petit = plus serre)\n" );
    std::printf( "  %-14s %7s   %8s   %8s %8s   %8s %8s\n",
                 "germes/noeud", "noeuds", "affine", "diagonal", "(hasard)", "complet", "(hasard)" );
    for ( size_t b = 0; b < by.size(); ++b ) {
        const Acc &A = by[ b ];
        if ( ! A.n )
            continue;
        std::printf( "  %6d - %-6d %7lld   %8.3f   %8.3f %8.3f   %8.3f %8.3f\n",
                     1 << b, ( 2 << b ) - 1, A.n, A.s1 / A.n,
                     A.s2 / A.n, A.chance2 / A.n, A.s3 / A.n, A.chance3 / A.n );
    }
    return 0;
}

/// LA PARABOLE ABAISSEE en 2D : combien de germes reste-t-il a tester ?
///
/// Le portage direct de `cases/sub1d_baisse.py`. On prend `|S| = n / rho` germes (stratifies : un
/// par sous-arbre du Bsp), on rattache chaque germe `m` au `k` de `S` le PLUS PROCHE -- son paquet
/// -- et on ABAISSE la parabole de `k` juste assez pour qu'elle minore tout son paquet :
///
///     h_k( x ) - h_m( x ) = -2 ( p_k - p_m ) . x + |p_k|^2 - |p_m|^2 - w_k + w_m
///
/// est AFFINE (le `|x|^2` est commun), donc son maximum sur `E_m` -- l'enclos de `m` contre `S`
/// seul, deja un convexe -- est atteint a un SOMMET, et se lit sans approximation.
///
/// `I_k = { x : h_k - delta_k <= h_j , j de S }` contient alors toutes les cellules du paquet :
///
///     x dans Lag( m ) ==> h_m <= h_j sur S ==> h_k - delta_k <= h_j sur S ==> x dans I_k
///
/// C'est un diagramme de puissance ou CHAQUE cellule est calculee avec son propre poids augmente
/// -- donc `|S|` convexes qui se CHEVAUCHENT au lieu d'un pavage. Deux germes ne peuvent se couper
/// que si leurs paquets se chevauchent : la colonne « manq. » verifie que le certificat tient, en
/// reprenant tous les couples reellement adjacents du vrai diagramme.
///
/// Ce qui est simule et ce qui ne l'est pas : on suppose, comme demande, qu'on sait trouver les
/// chevauchements vite. La colonne « boite » dit ce que couterait la version bon marche -- ne
/// tester que les boites englobantes des `I_k` au lieu des convexes eux-memes.
int baisse_stats( const Args &a, const std::vector<TF> &X, const std::vector<TF> &Y, const TF *W ) {
    using Cell = CellSoAT<64>;
    const SI rates[] = { 2, 4, 8, 16, 32, 64 };

    auto quant = []( std::vector<double> v, double q ) {
        if ( v.empty() ) return 0.0;
        std::sort( v.begin(), v.end() );
        return v[ std::min( v.size() - 1, size_t( q * v.size() ) ) ];
    };

    // OU EST LE DEBORDEMENT ? `delta_k` est le max de `h_k - h_m` sur `E_m`, qui CONTIENT
    // `Lag( m )` mais lui est bien plus gros. La demonstration, elle, n'a besoin de
    // `h_k - delta_k <= h_m` que sur `Lag( m )` : un `delta` calcule sur la vraie cellule reste
    // donc VALIDE (la colonne « manq. » le confirme), simplement pas calculable sans la reponse.
    // C'est un ORACLE, et il partage la question en deux -- si les chiffres s'effondrent, c'est
    // `E_m` qu'il faut resserrer ; s'ils ne bougent pas, c'est la GEOMETRIE du paquet qui coute, et
    // aucun raffinement de `delta` n'y changera rien.
    //
    // ESSAYE ET MESURE par ailleurs : rattacher `m` au proprietaire de `p_m` dans le diagramme de
    // puissance grossier au lieu du germe le plus proche. Sans poids les deux regles COINCIDENT
    // (`argmin_j h_j( p_m )` est alors le plus proche) ; avec poids la seconde s'effondre -- 22436
    // candidats contre 42 sur le cas dur. La pente de `h_k - h_m` vaut `2 | p_k - p_m |`, une
    // quantite de l'espace des GERMES : le proprietaire de `p_m` peut y etre tres loin.
    // TROIS VARIANTES de la parabole abaissee. `h_q( x ) - omega = |x|^2 + a . x + b` avec
    // `q = -a/2` : abaisser seulement `w` fixe `a = -2 p_k` et n'optimise que `b`, donc ne corrige
    // pas la PENTE de `h_k - h_m`, qui vaut `2 | p_k - p_m |`. Deplacer aussi le germe, c'est
    // choisir `a` -- et le barycentre du paquet est le choix bon marche.
    static const char *noms[] = { "delta seul, germe = le plus proche de S",
                                  "delta + germe VIRTUEL au barycentre du paquet",
                                  "delta seul, pris sur Lag( m ) -- ORACLE, non calculable" };
    for ( int mode = 0; mode < 3; ++mode ) {
    std::printf( "  -- %s\n", noms[ mode ] );
    std::printf( "  %5s %8s %8s %8s %8s %8s %8s %9s %8s %7s\n", "n/|S|", "|S|", "paq/paq",
                 "vrais", "aire I/p", "cand.", "cd. d9", "cd. max", "cd.boite", "manq." );

    for ( SI r : rates ) {
        pre_rate = r;
        AaBspSub t;
        t.build( X.data(), Y.data(), W, a.n, a.leaf );
        const SI n = t.nb_seeds(), m = t.sub.nb_seeds();

        std::vector<SI> loc( a.n, -1 ), pos( a.n, -1 );
        for ( SI u = 0; u < m; ++u ) loc[ t.sub.order[ u ] ] = u;
        for ( SI k = 0; k < n; ++k ) pos[ t.full.order[ k ] ] = k;

        // ---- 1. LES PAQUETS : le germe de `S` le plus proche. Le parcours du Bsp sert TEL QUEL --
        //         `may_cut` devient « cette boite peut-elle contenir plus proche que le meilleur ».
        std::vector<SI> pack( n, -1 );
        parallel_for( n, a.threads, Split::blocks, a.pin, [ & ]( SI k, int ) {
            const TF px = t.full.px[ k ], py = t.full.py[ k ];
            TF best = 1e30;
            SI bu = -1;
            t.sub.for_each_candidate_at( px, py, -1,
                [ & ]( TF lox, TF loy, TF hix, TF hiy, const WMaj & ) {
                    const TF ex = px < lox ? lox - px : ( px > hix ? px - hix : TF( 0 ) );
                    const TF ey = py < loy ? loy - py : ( py > hiy ? py - hiy : TF( 0 ) );
                    return ex * ex + ey * ey <= best;
                },
                [ & ]( TF x, TF y, TF, SI id ) {
                    const TF ex = x - px, ey = y - py;
                    const TF d = ex * ex + ey * ey;
                    if ( d < best ) { best = d; bu = loc[ id ]; }
                    return true;
                },
                [ & ]() { return best; } );
            pack[ k ] = bu;
        } );
        std::vector<SI> psz( m, 0 );
        for ( SI k = 0; k < n; ++k )
            ++psz[ pack[ k ] ];

        // le germe VIRTUEL du paquet : son barycentre. Il n'appartient plus au nuage, ce que la
        // demonstration autorise -- elle ne demande `h_q - delta <= h_m` que sur `B_m`.
        std::vector<TF> qx( m, 0 ), qy( m, 0 ), qw( m, 0 );
        for ( SI u = 0; u < m; ++u ) { qx[ u ] = t.sub.px[ u ]; qy[ u ] = t.sub.py[ u ];
                                       qw[ u ] = t.sub.seed_w( u ); }
        if ( mode == 1 ) {
            std::fill( qx.begin(), qx.end(), TF( 0 ) );
            std::fill( qy.begin(), qy.end(), TF( 0 ) );
            std::fill( qw.begin(), qw.end(), TF( 0 ) );
            for ( SI k = 0; k < n; ++k ) { qx[ pack[ k ] ] += t.full.px[ k ];
                                           qy[ pack[ k ] ] += t.full.py[ k ]; }
            for ( SI u = 0; u < m; ++u ) { qx[ u ] /= psz[ u ]; qy[ u ] /= psz[ u ]; }
        }

        // ---- 2. L'ABAISSEMENT. `E_m` est la cellule de `m` contre `S` seul : `AaBspSub` la donne
        //         directement. Le maximum de la difference affine y est a un SOMMET.
        PowerDiagram<Cell, AaBspSub, true, false, false> pe{ t };
        PowerDiagram<Cell, AaBsp,    true, false, false> pf{ t.full };
        std::vector<TF> dv( n, TF( 0 ) );
        parallel_for( n, a.threads, Split::blocks, a.pin, [ & ]( SI k, int ) {
            Cell ce;
            if ( mode == 2 ) pf.make_cell( ce, k ); else pe.make_cell( ce, k );
            const SI u = pack[ k ];
            const TF mx = t.full.px[ k ], my = t.full.py[ k ], mw = t.full.seed_w( k );
            const TF rx = qx[ u ], ry = qy[ u ], rw = qw[ u ];
            const TF dx = rx - mx, dy = ry - my;
            // `|p_r|^2 - |p_m|^2` FACTORISE, et pas litteralement : ecrit `rx*rx - mx*mx` il ne rend
            // pas exactement zero quand `r == m`, et le paquet s'abaisserait d'un cran pour rien --
            // c'est le defaut qui avait fait rater 8700 voisins au critere de la grille.
            const TF e = dx * ( rx + mx ) + dy * ( ry + my ) - rw + mw;
            TF best = mode == 1 ? TF( -1e30 ) : TF( 0 );
            for ( SI v = 0; v < ce.nb; ++v ) {
                const TF s = e - 2 * ( dx * ce.vx[ v ] + dy * ce.vy[ v ] );
                best = s > best ? s : best;
            }
            dv[ k ] = best;
        } );
        std::vector<TF> delta( m, mode == 1 ? TF( -1e30 ) : TF( 0 ) );
        for ( SI k = 0; k < n; ++k )
            delta[ pack[ k ] ] = std::max( delta[ pack[ k ] ], dv[ k ] );
        // la MARGE de tangence, cf. `AaBspPack` : deux paquets adjacents se touchent en mesure
        // nulle, et l'arrondi les separe. Grossir `delta` reste un certificat.
        for ( SI u = 0; u < m; ++u )
            delta[ u ] += TF( 1e-9 );

        // ---- 3. LES CONVEXES `I_k`, ranges en CSR. C'est le diagramme de `S` ou chaque cellule est
        //         calculee avec SON poids augmente -- donc `|S|` convexes qui se recouvrent.
        std::vector<SI> voff( m + 1, 0 );
        std::vector<TF> vxs, vys, bb( 4 * size_t( m ), 0 );
        std::vector<double> ai( m, 0 );
        vxs.reserve( size_t( 10 ) * m ); vys.reserve( size_t( 10 ) * m );
        for ( SI u = 0; u < m; ++u ) {
            const TF ux = qx[ u ], uy = qy[ u ];
            const TF uw = qw[ u ] + delta[ u ];
            // le germe virtuel n'est PAS dans `S` : aucun plan a sauter. Le representant, lui,
            // peut se sauter parce que `delta >= 0` rend `h_k - delta <= h_k` automatique.
            const SI uid = mode == 1 ? SI( -1 ) : t.sub.order[ u ];
            Cell c;
            const OneSeed os{ t.sub, ux, uy, uw, uid };
            PowerDiagram<Cell, OneSeed, true, false, false>{ os }.make_cell( c, 0 );
            TF x0 = 1e30, y0 = 1e30, x1 = -1e30, y1 = -1e30;
            for ( SI v = 0; v < c.nb; ++v ) {
                vxs.push_back( c.vx[ v ] ); vys.push_back( c.vy[ v ] );
                x0 = std::min( x0, c.vx[ v ] ); x1 = std::max( x1, c.vx[ v ] );
                y0 = std::min( y0, c.vy[ v ] ); y1 = std::max( y1, c.vy[ v ] );
            }
            voff[ u + 1 ] = SI( vxs.size() );
            bb[ 4 * u + 0 ] = x0; bb[ 4 * u + 1 ] = y0;
            bb[ 4 * u + 2 ] = x1; bb[ 4 * u + 3 ] = y1;
            ai[ u ] = double( c.measure() );
        }

        // deux convexes sont disjoints SSI une normale d'arete de l'un les separe (axe separateur).
        // `<=` partout : deux convexes qui se touchent par une arete comptent comme se chevauchant,
        // et c'est bien ce qu'on veut -- deux cellules adjacentes se touchent exactement ainsi.
        auto demi = [ & ]( SI p, SI q ) {
            const SI b0 = voff[ p ], e0 = voff[ p + 1 ], b1 = voff[ q ], e1 = voff[ q + 1 ];
            for ( SI i = b0; i < e0; ++i ) {
                const SI j = i + 1 < e0 ? i + 1 : b0;
                const TF nx = vys[ j ] - vys[ i ], ny = vxs[ i ] - vxs[ j ];
                TF a0 = 1e30, a1 = -1e30, c0 = 1e30, c1 = -1e30;
                for ( SI k = b0; k < e0; ++k ) {
                    const TF s = nx * vxs[ k ] + ny * vys[ k ];
                    a0 = std::min( a0, s ); a1 = std::max( a1, s );
                }
                for ( SI k = b1; k < e1; ++k ) {
                    const TF s = nx * vxs[ k ] + ny * vys[ k ];
                    c0 = std::min( c0, s ); c1 = std::max( c1, s );
                }
                // TOLERANCE, et elle n'est pas cosmetique : deux paquets ADJACENTS se touchent
                // exactement le long d'une arete, donc s'intersectent en mesure nulle. Un test
                // strict les declare separes des que l'arrondi ecarte les deux convexes de 1e-17,
                // et le certificat se met a « manquer » 3754 couples a rho = 2 -- la ou delta est
                // petit et ou les `I_k` sont donc presque le pavage grossier lui-meme. La normale
                // n'est pas unitaire : sa longueur est celle de l'arete, d'ou le facteur.
                const TF eps = TF( 1e-12 ) * std::sqrt( nx * nx + ny * ny );
                if ( c0 > a1 + eps || a0 > c1 + eps )
                    return true;
            }
            return false;
        };
        // l'ECART des deux convexes, normalise : > 0 = separes. Sert au diagnostic des manques --
        // de la tangence a 1e-16 et un vrai trou ne se corrigent pas de la meme facon.
        auto ecart = [ & ]( SI p, SI q ) {
            TF g = -1e30;
            for ( int s2 = 0; s2 < 2; ++s2 ) {
                const SI P0 = s2 ? q : p, Q0 = s2 ? p : q;
                const SI b0 = voff[ P0 ], e0 = voff[ P0 + 1 ], b1 = voff[ Q0 ], e1 = voff[ Q0 + 1 ];
                for ( SI i = b0; i < e0; ++i ) {
                    const SI j = i + 1 < e0 ? i + 1 : b0;
                    const TF nx = vys[ j ] - vys[ i ], ny = vxs[ i ] - vxs[ j ];
                    const TF ln = std::sqrt( nx * nx + ny * ny );
                    if ( ln == 0 )
                        continue;
                    TF a0 = 1e30, a1 = -1e30, c0 = 1e30, c1 = -1e30;
                    for ( SI k = b0; k < e0; ++k ) {
                        const TF s3 = ( nx * vxs[ k ] + ny * vys[ k ] ) / ln;
                        a0 = std::min( a0, s3 ); a1 = std::max( a1, s3 );
                    }
                    for ( SI k = b1; k < e1; ++k ) {
                        const TF s3 = ( nx * vxs[ k ] + ny * vys[ k ] ) / ln;
                        c0 = std::min( c0, s3 ); c1 = std::max( c1, s3 );
                    }
                    g = std::max( g, std::max( c0 - a1, a0 - c1 ) );
                }
            }
            return double( g );
        };
        auto chevauche = [ & ]( SI p, SI q ) {
            if ( voff[ p + 1 ] == voff[ p ] || voff[ q + 1 ] == voff[ q ] )
                return false;
            const TF eb = TF( 1e-12 );
            if ( bb[ 4 * q + 0 ] > bb[ 4 * p + 2 ] + eb || bb[ 4 * p + 0 ] > bb[ 4 * q + 2 ] + eb ||
                 bb[ 4 * q + 1 ] > bb[ 4 * p + 3 ] + eb || bb[ 4 * p + 1 ] > bb[ 4 * q + 3 ] + eb )
                return false;
            return ! demi( p, q ) && ! demi( q, p );
        };

        // ---- 4. LES CHEVAUCHEMENTS. On suppose qu'on sait les trouver vite ; ici une grille sur
        //         les boites suffit, sa maille etant reduite tant que les insertions debordent.
        SI g = std::max<SI>( 4, std::min<SI>( 512, SI( std::sqrt( double( m ) ) ) ) );
        auto plage = [ & ]( SI u, SI gg, SI &i0, SI &j0, SI &i1, SI &j1 ) {
            auto cl = [ & ]( TF v ) { return std::max<SI>( 0, std::min<SI>( gg - 1, SI( v * gg ) ) ); };
            i0 = cl( bb[ 4 * u + 0 ] ); j0 = cl( bb[ 4 * u + 1 ] );
            i1 = cl( bb[ 4 * u + 2 ] ); j1 = cl( bb[ 4 * u + 3 ] );
        };
        for ( ;; ) {
            long long tot = 0;
            for ( SI u = 0; u < m; ++u )
                if ( voff[ u + 1 ] > voff[ u ] ) {
                    SI i0, j0, i1, j1; plage( u, g, i0, j0, i1, j1 );
                    tot += (long long)( i1 - i0 + 1 ) * ( j1 - j0 + 1 );
                }
            if ( tot <= 64ll * m || g <= 4 )
                break;
            g /= 2;
        }
        std::vector<SI> goff( size_t( g ) * g + 2, 0 ), gval;
        for ( SI u = 0; u < m; ++u )
            if ( voff[ u + 1 ] > voff[ u ] ) {
                SI i0, j0, i1, j1; plage( u, g, i0, j0, i1, j1 );
                for ( SI j = j0; j <= j1; ++j )
                    for ( SI i = i0; i <= i1; ++i )
                        ++goff[ j * g + i + 2 ];
            }
        for ( size_t q = 1; q < goff.size(); ++q )
            goff[ q ] += goff[ q - 1 ];
        gval.resize( goff.back() );
        for ( SI u = 0; u < m; ++u )
            if ( voff[ u + 1 ] > voff[ u ] ) {
                SI i0, j0, i1, j1; plage( u, g, i0, j0, i1, j1 );
                for ( SI j = j0; j <= j1; ++j )
                    for ( SI i = i0; i <= i1; ++i )
                        gval[ goff[ j * g + i + 1  ]++ ] = u;
            }

        std::vector<double> cand( m, 0 ), cbox( m, 0 ), nov( m, 0 );
        std::vector<std::vector<SI>> mark( a.threads, std::vector<SI>( m, -1 ) );
        parallel_for( m, a.threads, Split::blocks, a.pin, [ & ]( SI u, int th ) {
            if ( voff[ u + 1 ] == voff[ u ] )
                return;
            auto &mk = mark[ th ];
            SI i0, j0, i1, j1; plage( u, g, i0, j0, i1, j1 );
            double cb = 0, ce = 0, no = 0;
            for ( SI j = j0; j <= j1; ++j )
                for ( SI i = i0; i <= i1; ++i )
                    for ( SI q = goff[ j * g + i ]; q < goff[ j * g + i + 1 ]; ++q ) {
                        const SI v = gval[ q ];
                        if ( mk[ v ] == u )
                            continue;
                        mk[ v ] = u;
                        if ( bb[ 4 * v + 0 ] > bb[ 4 * u + 2 ] + TF( 1e-12 ) ||
                             bb[ 4 * u + 0 ] > bb[ 4 * v + 2 ] + TF( 1e-12 ) ||
                             bb[ 4 * v + 1 ] > bb[ 4 * u + 3 ] + TF( 1e-12 ) ||
                             bb[ 4 * u + 1 ] > bb[ 4 * v + 3 ] + TF( 1e-12 ) )
                            continue;
                        cb += psz[ v ];
                        if ( demi( u, v ) || demi( v, u ) )
                            continue;
                        ce += psz[ v ]; no += 1;
                    }
            cand[ u ] = ce; cbox[ u ] = cb; nov[ u ] = no;
        } );

        // ---- 5. LE CERTIFICAT, et l'aire vraie de chaque paquet. On reprend TOUS les couples
        //         reellement adjacents du vrai diagramme : aucun ne doit manquer.
        std::vector<double> ar( n, 0 );
        std::vector<long long> mv( n, 0 ), rv( n, 0 );
        std::vector<double> gv( n, 0 );
        std::vector<std::vector<std::pair<SI,SI>>> adj( a.threads );
        parallel_for( n, a.threads, Split::blocks, a.pin, [ & ]( SI k, int th ) {
            Cell c;
            pf.make_cell( c, k );
            ar[ k ] = double( c.measure() );
            for ( SI v = 0; v < c.nb; ++v ) {
                const SI id = c.cid[ v ];
                if ( id < 0 )
                    continue;
                ++rv[ k ];
                if ( ! chevauche( pack[ k ], pack[ pos[ id ] ] ) ) {
                    ++mv[ k ];
                    gv[ k ] = std::max( gv[ k ], ecart( pack[ k ], pack[ pos[ id ] ] ) );
                }
                adj[ th ].emplace_back( pack[ k ], pack[ pos[ id ] ] );
            }
        } );
        // LA REFERENCE : combien de paquets sont REELLEMENT adjacents. Sans elle « 13.3 paquets
        // retenus » ne dit pas si c'est le critere qui est large ou le regroupement qui est cher.
        std::vector<std::pair<SI,SI>> ap;
        for ( auto &v : adj ) ap.insert( ap.end(), v.begin(), v.end() );
        std::sort( ap.begin(), ap.end() );
        const double vrai = double( std::unique( ap.begin(), ap.end() ) - ap.begin() ) / m;

        long long miss = 0, rel = 0;
        double gmax = 0;
        std::vector<double> par( m, 0 );
        for ( SI k = 0; k < n; ++k ) {
            miss += mv[ k ]; rel += rv[ k ]; gmax = std::max( gmax, gv[ k ] );
            par[ pack[ k ] ] += ar[ k ];
        }

        // par GERME et non par paquet : un paquet gros pese plus lourd, et c'est le germe qui paie.
        std::vector<double> cg( n ), deb;
        for ( SI k = 0; k < n; ++k )
            cg[ k ] = cand[ pack[ k ] ];
        double sov = 0, sbo = 0;
        for ( SI u = 0; u < m; ++u ) {
            sov += nov[ u ];
            sbo += cbox[ u ] * psz[ u ];
            if ( par[ u ] > 0 )
                deb.push_back( ai[ u ] / par[ u ] );
        }
        std::printf( "  %5d %8d %8.1f %8.1f %8.2f %8.0f %8.0f %9.0f %8.0f %7lld\n",
                     int( r ), int( m ), sov / m, vrai, quant( deb, 0.5 ), quant( cg, 0.5 ),
                     quant( cg, 0.9 ), quant( cg, 1.0 ), sbo / n, miss );
        if ( miss )
            std::printf( "        ^ ecart max des couples manques : %.2e\n", gmax );
        (void)rel;
    }
    }

    AaBsp bs;
    bs.build( X.data(), Y.data(), W, a.n, a.leaf );
    PowerDiagram<Cell, AaBsp, true, false, true> pr{ bs };
    for ( SI k = 0; k < a.n; ++k ) { Cell c; pr.make_cell( c, k ); }
    std::printf( "  (le BSP d'aujourd'hui : %.1f coupes tentees et %.1f boites par cellule)\n",
                 double( pr.st_tried ) / a.n, double( pr.st_boxes ) / a.n );
    return 0;
}

/// Le nombre d iterations de CG, ou rien du tout quand on a factorise.
inline const char *cg_txt( int nb ) {
    static char buf[ 64 ];
    if ( ! nb ) return "";
    std::snprintf( buf, sizeof( buf ), " (%d iterations)", nb );
    return buf;
}

/// UN NIVEAU de la hierarchie multi-echelle : un nuage grossier, et la masse cible de chaque germe
/// grossier -- qui est celle de TOUT SON PAQUET.
struct Niveau {
    SI              R = 1, m = 0;
    std::vector<SI> cl;                 ///< place dans `order` -> paquet
    std::vector<SI> beg, end;           ///< la tranche de chaque paquet
    std::vector<TF> X, Y, nu;           ///< le nuage grossier
};

/// LES PAQUETS SONT DES NOEUDS DU BSP, pas des tranches a pas fixe de sa permutation.
///
/// ESSAYE ET REJETE : decouper `order` en blocs de `R` positions consecutives. C'etait tentant --
/// le BSP coupe a la MEDIANE, donc un sous-arbre EST une tranche contigue -- mais la reciproque est
/// fausse : les frontieres de sous-arbres tombent aux medianes (10000, 5000, 2500, 1250, 625,
/// 313...), qui ne sont pas des multiples de `R`. Un bloc sur deux enjambait donc deux sous-arbres,
/// parfois cousins eloignes, et la prolongation donnait LE MEME POIDS a deux germes a l'autre bout
/// du carre. Mesure : le niveau grossier convergeait en 3 iterations a pas plein, et le niveau
/// suivant demarrait avec des cellules VIDES et un residu de 67 fois la cible -- pire que depuis
/// `w = 0`.
///
/// Ici on descend l'arbre et on coupe des qu'un sous-arbre ne depasse plus `R` germes : les paquets
/// sont alors compacts par construction, et restent des tranches contigues de `order`.
Niveau faire_niveau( const AaBsp &tr, const std::vector<TF> &X, const std::vector<TF> &Y, SI R ) {
    const SI n = SI( tr.order.size() );
    Niveau L;
    L.R = R;
    L.cl.assign( n, -1 );

    std::vector<SI> pile{ 0 };
    while ( ! pile.empty() ) {
        const SI h = pile.back();
        pile.pop_back();
        const AaBsp::Node &nd = tr.nodes[ h ];
        if ( nd.end - nd.beg <= R || nd.right < 0 ) {
            for ( SI k = nd.beg; k < nd.end; ++k )
                L.cl[ k ] = L.m;
            L.beg.push_back( nd.beg );
            L.end.push_back( nd.end );
            ++L.m;
        } else {
            pile.push_back( h + 1 );                    // PREORDRE : le fils gauche est juste a cote
            pile.push_back( nd.right );
        }
    }

    // le representant est le germe le plus proche du barycentre du paquet -- pas le premier venu :
    // la prolongation donne son poids a tout le paquet, donc il doit en etre au centre.
    L.X.resize( L.m ); L.Y.resize( L.m ); L.nu.resize( L.m );
    for ( SI c = 0; c < L.m; ++c ) {
        const SI b = L.beg[ c ], e = L.end[ c ];
        TF cx = 0, cy = 0;
        for ( SI k = b; k < e; ++k ) { cx += X[ tr.order[ k ] ]; cy += Y[ tr.order[ k ] ]; }
        cx /= ( e - b ); cy /= ( e - b );
        SI best = b;
        TF bd = -1;
        for ( SI k = b; k < e; ++k ) {
            const TF dx = X[ tr.order[ k ] ] - cx, dy = Y[ tr.order[ k ] ] - cy;
            const TF d = dx * dx + dy * dy;
            if ( bd < 0 || d < bd ) { bd = d; best = k; }
        }
        L.X[ c ] = X[ tr.order[ best ] ];
        L.Y[ c ] = Y[ tr.order[ best ] ];
        L.nu[ c ] = TF( e - b ) / n;                    // la masse AGREGEE du paquet
    }
    return L;
}

/// LA BASELINE : un probleme de transport resolu de bout en bout, chronometre par poste.
///
/// Ce qu'on veut en lire n'est pas le temps total mais sa REPARTITION. Un accelerateur qui doit
/// survivre d'une iteration a l'autre ne se juge que la : s'il faut douze diagrammes pour
/// converger, un index qui coute une passe et en fait gagner un quart se rembourse -- alors qu'il
/// est une perte seche sur une passe isolee.
///
/// = LE MULTI-ECHELLE, et ce qu'il repare
///
/// Depuis `w = 0` sur un nuage tres non uniforme, l'amortissement rampe : les cellules de Voronoi
/// s'etalent sur six ordres de grandeur, la direction de Newton est enorme devant la region ou
/// toutes les cellules restent non vides, et le pas tombe a 1/256. Mesure sur le nuage de lignes :
/// SEIZE iterations a ramper avant six iterations quadratiques.
///
/// Le remede n'est pas de mieux amortir, c'est de PARTIR D'AILLEURS. On resout d'abord le meme
/// probleme sur `n / R` representants portant la masse agregee de leur paquet -- moins cher, et
/// surtout bien mieux conditionne, puisque les masses cibles y sont comparables par construction.
/// Puis on PROLONGE : chaque germe fin recoit le poids de son representant.
///
/// Pourquoi cette prolongation est la bonne : tous les germes d'un paquet portant alors LE MEME
/// poids, la bissectrice de puissance entre deux d'entre eux est la mediatrice ordinaire. Le
/// diagramme fin est donc, a l'interieur d'un paquet, un VORONOI -- et globalement la solution
/// grossiere. L'erreur restante est purement LOCALE, et c'est exactement le regime ou Newton prend
/// des pas pleins.
template<class Cell, bool BOX, bool IN, class Tree>
int newton_go( const Args &a, const std::vector<TF> &X, const std::vector<TF> &Y, const TF *Wref ) {
    const SI n = a.n;

    // l'arbre du niveau FIN est bati une fois pour toutes, sur des poids nuls, et ne sera plus que
    // rafraichi : les positions ne bougent pas d'une iteration de Newton a l'autre. Sa permutation
    // sert AUSSI a decouper les paquets de tous les niveaux grossiers.
    Tree tr;
    std::vector<TF> zero( n, TF( 0 ) );
    const double tb = now();
    tr.build( X.data(), Y.data(), zero.data(), n, a.leaf );
    if constexpr ( requires ( Tree &t ) { t.bits; } ) { tr.bits = a.memobits; }
    const double t_avant = now() - tb;              // hors de `tot`, donc a rajouter a la fin
    double t_arbre = t_avant;

#ifdef _OPENMP
    // AMGCL est parallelise en OpenMP, le diagramme en `std::thread` : sans ca les deux moities du
    // chronometre ne tourneraient pas sur le meme nombre de coeurs.
    omp_set_num_threads( a.threads );
#endif
    Newton nw;
    nw.quel = a.solver == "chol" ? 1 : ( a.solver == "cg" ? 2 : 0 );
    nw.variante = a.amgvar;

    const double t0 = now();

    // ---- les niveaux, du plus fin au plus grossier.
    //
    // ESSAYE ET REJETE : decouper les paquets dans l'arbre du DIAGRAMME. Ses feuilles portent
    // `--leaf` germes (dix), donc aucun paquet ne peut etre plus petit que dix : quel que soit
    // `--ms-ratio`, la DERNIERE prolongation etait toujours un saut d'un facteur dix. Mesure : les
    // niveaux grossiers convergeaient tous en quatre a cinq iterations a pas plein, et le niveau
    // fin stagnait des sa premiere iteration. On batit donc un second arbre, a feuilles de
    // `--ms-ratio` germes, qui ne sert qu'a decouper -- et qu'on relache aussitot les niveaux
    // construits, parce qu'il pese deux fois plus de noeuds que celui du diagramme.
    std::vector<Niveau> niv;
    if ( a.msratio > 1 ) {
        AaBsp hier;
        const double th0 = now();
        hier.build( X.data(), Y.data(), zero.data(), n, a.msratio );
        t_arbre += now() - th0;
        for ( SI R = a.msratio; ; R *= a.msratio ) {
            Niveau L = faire_niveau( hier, X, Y, R );
            if ( L.m < a.msmin )
                break;
            if ( ! niv.empty() && L.m == niv.back().m )
                continue;
            niv.push_back( std::move( L ) );
        }
    }

    // LA PROLONGATION, par c-transformee : `w_i = -psi_grossier( p_i )`. Voir `psi_min`.
    std::vector<TF> wp;                                 // la solution du niveau precedent
    AaBsp tsup;                                         // et son arbre, pour l'interroger
    auto prolonge = [ & ]( AaBsp &tc, const std::vector<TF> &wc, const TF *Xf, const TF *Yf,
                           SI mf, std::vector<TF> &wf ) {
        refresh_weights( tc, wc.data(), a.threads, a.split, a.pin );
        wf.assign( mf, TF( 0 ) );
        parallel_for( mf, a.threads, a.split, a.pin, [ & ]( SI i, int ) {
            wf[ i ] = -psi_min( tc, Xf[ i ], Yf[ i ] );
        } );
    };

    for ( int li = int( niv.size() ) - 1; li >= 0; --li ) {
        const Niveau &L = niv[ li ];
        std::vector<TF> w0( L.m, TF( 0 ) );
        if ( li + 1 < int( niv.size() ) )
            prolonge( tsup, wp, L.X.data(), L.Y.data(), L.m, w0 );

        AaBsp tl;
        const double tl0 = now();
        tl.build( L.X.data(), L.Y.data(), w0.data(), L.m, a.leaf );
        t_arbre += now() - tl0;
        PowerDiagram<Cell, AaBsp, BOX, IN, false, true> pl{ tl };
        if ( li + 1 < int( niv.size() ) )
            rattrape_vides( pl, tl, L.X.data(), L.Y.data(), L.m, w0, TF( a.msmarge ) / L.m, a.mspasses,
                            a.threads, a.split, a.pin, true );
        nw.n = L.m;
        nw.nu = L.nu;
        std::printf( "    -- niveau R=%d, %d germes\n", int( L.R ), int( L.m ) );
        nw.resout<Cell>( pl, tl, L.X.data(), L.Y.data(), w0, TF( a.mstol ), a.nmax,
                         TF( a.cgtol ), a.cgmax, a.threads, a.split, a.pin, true );
        wp = nw.w;
        tsup = std::move( tl );
    }

    // ---- le niveau FIN : chaque germe recoit le poids du paquet auquel il appartient.
    std::vector<TF> w0( n, TF( 0 ) );
    if ( ! niv.empty() ) {
        prolonge( tsup, wp, X.data(), Y.data(), n, w0 );
    }

    PowerDiagram<Cell, Tree, BOX, IN, false, true> pd{ tr };
    if ( ! niv.empty() )
        rattrape_vides( pd, tr, X.data(), Y.data(), n, w0, TF( a.msmarge ) / n, a.mspasses, a.threads,
                        a.split, a.pin, true );
    nw.n = n;
    nw.nu.assign( n, TF( 1 ) / n );
    if ( ! niv.empty() )
        std::printf( "    -- niveau R=1, %d germes\n", int( n ) );
    const bool ok = nw.resout<Cell>( pd, tr, X.data(), Y.data(), w0, TF( a.ntol ), a.nmax,
                                     TF( a.cgtol ), a.cgmax, a.threads, a.split, a.pin, true );
    const double tot = now() - t0;

    const double total = tot + t_avant;
    const double autre = total - t_arbre - nw.t_diag - nw.t_maj - nw.t_syst - nw.t_cg;
    std::printf( "  newton %s (max|a-nu|/nu = %.2e) : n=%d threads=%d nv=%d box=%d leaf=%d %s %d iterations,"
                 " %d diagrammes (%d reculs), solveur %s%s\n",
                 nw.fin, double( nw.reste ), int( n ), a.threads, Cell::max_nb_vertices, int( BOX ),
                 int( a.leaf ), Tree::name, nw.nb_iter, nw.nb_diag, nw.nb_recul,
                 nw.quel == 0 ? "AMGCL" : ( nw.quel == 1 ? "Cholesky creux" : "gradient conjugue" ),
                 cg_txt( nw.nb_cg ) );
    std::printf( "         arbres %.3f | diagrammes %.3f | majorants %.3f | assemblage %.3f"
                 " | resolution %.3f | reste %.3f | TOTAL %.3f s\n",
                 t_arbre, nw.t_diag, nw.t_maj, nw.t_syst, nw.t_cg, autre, total );
    std::printf( "         soit %.0f %% de diagramme, %.3f s par diagramme, %.1f us/germe en tout\n",
                 100 * nw.t_diag / total, nw.t_diag / std::max( nw.nb_diag, 1 ), 1e6 * total / n );
    std::printf( "         resolution en detail : mise en forme %.3f | hierarchie/analyse %.3f"
                 " | resolution %.3f | descente %.3f\n",
                 nw.t_tri, nw.t_ana, nw.t_fac, nw.t_sol );
    if ( nw.nb_ana )
        std::printf( "         (%d montages pour %d iterations, pire residu lineaire %.2e)\n",
                     nw.nb_ana, nw.nb_iter, double( nw.pire_lin ) );

    if constexpr ( requires ( const Tree &t ) { t.nb_rejoue; } )
        std::printf( "         memo : %.2f coupes rejouees dans %.2f feuilles, et %.1f boites"
                     " testees par cellule (compteurs NON atomiques : a lire a --threads 1)\n",
                     double( tr.nb_rejoue ) / std::max( tr.nb_cell, 1LL ),
                     double( tr.nb_feuilles ) / std::max( tr.nb_cell, 1LL ),
                     double( tr.nb_boites ) / std::max( tr.nb_cell, 1LL ) );

    const SI novf = pd.nb_overflow.load();
    if ( novf )
        std::printf( "  ATTENTION : %d coupes ont DEBORDE %d sommets pendant la resolution.\n",
                     int( novf ), Cell::max_nb_vertices );

    // La verification qui ne coute rien : le nuage `_equal` PORTE deja la solution, obtenue par
    // L-BFGS dans `gen_cases.py`. Elle n'est definie qu'a une constante pres, donc on recale sur
    // le germe 0 -- exactement la jauge que Newton impose.
    if ( Wref ) {
        TF m = 0, ampl = 0;
        for ( SI i = 0; i < n; ++i ) {
            m = std::max( m, std::fabs( ( nw.w[ i ] - nw.w[ 0 ] ) - ( Wref[ i ] - Wref[ 0 ] ) ) );
            ampl = std::max( ampl, std::fabs( Wref[ i ] - Wref[ 0 ] ) );
        }
        std::printf( "  contre les poids du fichier : ecart max %.3e sur une amplitude %.3e\n",
                     double( m ), double( ampl ) );
    }
    return ok && novf == 0 ? 0 : 1;
}

} // namespace

int main( int argc, char **argv ) {
    Args a;
    for ( int i = 1; i < argc; ++i ) {
        const std::string s = argv[ i ];
        auto val = [ & ]() { return i + 1 < argc ? argv[ ++i ] : ""; };
        if      ( s == "-n" )        a.n = std::atoi( val() );
        else if ( s == "--reps" )    a.reps = std::atoi( val() );
        else if ( s == "--threads" ) a.threads = std::atoi( val() );
        else if ( s == "--leaf" )    a.leaf = std::atoi( val() );
        else if ( s == "--tree" )    a.tree = val();
        else if ( s == "--maxnv" )   a.maxnv = std::atoi( val() );
        else if ( s == "--no-cellbox" ) a.cellbox = false;
        else if ( s == "--skip-inside" ) a.skipin = true;
        else if ( s == "--split" )   a.split = std::string( val() ) == "strided" ? Split::strided : Split::blocks;
        else if ( s == "--no-pin" )  a.pin = false;
        else if ( s == "--check" )   a.check = true;
        else if ( s == "--seed" )    a.seed = unsigned( std::atoi( val() ) );
        else if ( s == "--weights" ) a.wscale = std::atof( val() );
        else if ( s == "--load" )    a.load = val();
        else if ( s == "--stats" )   a.stats = true;
        else if ( s == "--majorant" ) a.majorant = true;
        else if ( s == "--enclos" )  a.enclos = true;
        else if ( s == "--psigrid" ) a.psigrid = true;
        else if ( s == "--front" )   a.front = true;
        else if ( s == "--baisse" )  a.baisse = true;
        else if ( s == "--cross" )   a.cross = true;
        else if ( s == "--pre-overlap" ) pre_overlap = true;
        else if ( s == "--pre-rate" ) a.prerate = std::atoi( val() );
        else if ( s == "--newton" )  a.newton = true;
        else if ( s == "--newton-tol" ) a.ntol = std::atof( val() );
        else if ( s == "--newton-max" ) a.nmax = std::atoi( val() );
        else if ( s == "--cg-tol" )  a.cgtol = std::atof( val() );
        else if ( s == "--cg-max" )  a.cgmax = std::atoi( val() );
        else if ( s == "--solver" )  a.solver = val();
        else if ( s == "--amg-var" ) a.amgvar = std::atoi( val() );
        else if ( s == "--ms-ratio" ) a.msratio = std::atoi( val() );
        else if ( s == "--ms-min" ) a.msmin = std::atoi( val() );
        else if ( s == "--ms-tol" ) a.mstol = std::atof( val() );
        else if ( s == "--ms-passes" ) a.mspasses = std::atoi( val() );
        else if ( s == "--ms-marge" ) a.msmarge = std::atof( val() );
        else if ( s == "--memo" )    a.memo = true;
        else if ( s == "--no-memo-bits" ) a.memobits = false;
        else if ( s == "--pack-rate" ) pack_rate = std::atoi( val() );
        else if ( s == "--hull-rate" ) hull_rate = std::atoi( val() );
        else if ( s == "--no-hull-init" ) hull_init = false;
        else {
            std::printf(
                "usage: pd2d [options]\n"
                "  -n N            nombre de germes            (%d)\n"
                "  --reps R        repetitions chronometrees   (%d)\n"
                "  --threads T     0 = autant que de coeurs    (%d)\n"
                "  --leaf L        germes par feuille du BSP   (%d)\n"
                "  --tree T        bsp | bsp4 | obsp | hull | pack | pre | packed | grid | all   (bsp)\n"
                "  --tree bsp4     le meme BSP a QUATRE fils par noeud (test des 4 chez le pere)\n"
                "  --tree bsp4l    ... mais chaque fils teste a SA sortie de pile\n"
                "  --tree obsp     ... et coupes NON ALIGNEES sur les axes (boites alignees)\n"
                "  --hull-rate R   « hull » : sur-cellules anisotropes, R germes par paquet\n"
                "  --no-hull-init  ... mais partir du domaine et non de la sur-cellule\n"
                "                  « pack » : l index par paquets ; --pack-rate = rho,\n"
                "                  --leaf reste l unite d elagage (sous-arbre du paquet)\n"
                "  --pre-rate R    « pre » : coupe d'abord contre un germe sur R  (%d)\n"
                "  --maxnv M       sommets max par cellule    (%d)\n"
                "  --no-cellbox    pas de boite de cellule (sommets directement)\n"
                "  --skip-inside   pas de test pour une boite contenant le germe\n"
                "  --split S       blocks | strided            (blocks)\n"
                "  --no-pin        ne pas epingler les threads\n"
                "  --check         verifie contre le balayage complet, puis sort\n"
                "  --seed S        graine du tirage            (%u)\n"
                "  --weights W     poids aleatoires, en fraction de h^2 (0 = aucun)\n"
                "  --load FILE     un nuage de cases/ (« n » puis « x y w »), au lieu du tirage\n"
                "  --stats         compte boites/coupes par cellule au lieu de chronometrer\n"
                "  --majorant      compare les majorants de poids degre 0 / 1 / 2 par noeud\n"
                "  --cross         compare les accelerateurs entre eux au n demande\n"
                "  --pre-overlap   « pre » : recouvre les deux arbres, donc chaque germe de S est\n"
                "                  coupe DEUX FOIS par le meme plan -- le cas degenere, expres\n"
                "  --psigrid       mesure le critere min_B h_i > M(B) sur une grille reguliere\n"
                "  --front         l ETALEMENT sur grille : amorce par descente, front, manques\n"
                "  --baisse        la parabole ABAISSEE : un enclos par paquet, et ce qu il reste\n"
                "  --enclos        mesure l'enclos par sous-echantillon (rayon, mesure, cout)\n"
                "  --newton        RESOUT le probleme d aires egales (Newton amorti, w_0 = 0)\n"
                "  --newton-tol T  ... arret sur max|a_i - nu| / nu           (%.0e)\n"
                "  --newton-max K  ... iterations au maximum                  (%d)\n"
                "  --cg-tol T      ... arret du gradient conjugue, relatif    (%.0e)\n"
                "  --solver S      ... amg (AMGCL, defaut) | chol (Eigen) | cg (maison)\n"
                "  --amg-var V     ... 0 = agregation+spai0 | 1 = agregation+GS | 2 = Ruge-Stuben+GS\n"
                "  --ms-ratio R    ... MULTI-ECHELLE : rapport entre deux niveaux, 1 = aucun (%d)\n"
                "  --ms-min M      ... taille du niveau le plus grossier          (%d)\n"
                "  --ms-tol T      ... tolerance des niveaux grossiers            (%.0e)\n"
                "  --memo          garde d une iteration a l autre les coupes de la feuille\n"
                "                  (un bit par germe) et le germe qui a vide la cellule\n"
                "  --no-memo-bits  ... la memoire eteinte, pour mesurer ce que le reste coute\n",
                int( a.n ), a.reps, a.threads, int( a.leaf ), int( a.prerate ), a.maxnv, a.seed,
                a.ntol, a.nmax, a.cgtol, int( a.msratio ), int( a.msmin ), a.mstol );
            return s == "--help" || s == "-h" ? 0 : 1;
        }
    }
    if ( a.threads <= 0 )
        a.threads = int( std::thread::hardware_concurrency() );

    if ( a.check )
        return check( a );

    // le nuage : uniforme dans le carre unite, borne loin des cotes pour qu'aucune cellule ne soit
    // vide -- un germe sur le bord donnerait une cellule degeneree qui ne mesure rien d'interessant.
    std::mt19937_64 rng( a.seed );
    std::uniform_real_distribution<TF> uni( 0.001, 0.999 );
    std::vector<TF> X, Y, W;
    const TF *Wp = nullptr;
    if ( ! a.load.empty() ) {
        if ( ! load_cloud( a.load, X, Y, W ) )
            return 1;
        a.n = SI( X.size() );
        Wp = weights_or_null( W );
        std::printf( "  nuage '%s' : %d germes, %s\n", a.load.c_str(), int( a.n ),
                     Wp ? "avec poids" : "poids tous nuls (cas euclidien)" );
    } else {
        X.resize( a.n ); Y.resize( a.n );
        for ( SI i = 0; i < a.n; ++i ) { X[ i ] = uni( rng ); Y[ i ] = uni( rng ); }
        Wp = make_weights( a, a.n, rng, W );
    }

    pre_rate = a.prerate;
    hull_threads = a.threads;

    if ( a.newton ) {
        // les poids du fichier ne servent PAS de depart -- on part de zero, comme demande -- mais
        // de temoin quand ils resolvent deja le probleme d'aires egales.
        const TF *ref = a.load.find( "equal" ) != std::string::npos ? Wp : nullptr;
        auto go_n = [ & ]( auto cell_tag ) {
            using Cell = decltype( cell_tag );
            auto avec = [ & ]( auto tree_tag ) {
                using Tree = decltype( tree_tag );
                if ( a.cellbox && ! a.skipin ) return newton_go<Cell, true, false, Tree>( a, X, Y, ref );
                if ( a.cellbox &&   a.skipin ) return newton_go<Cell, true, true, Tree >( a, X, Y, ref );
                if ( ! a.cellbox && ! a.skipin ) return newton_go<Cell, false, false, Tree>( a, X, Y, ref );
                return newton_go<Cell, false, true, Tree>( a, X, Y, ref );
            };
            return a.memo ? avec( AaBspMemo{} ) : avec( AaBsp{} );
        };
        switch ( a.maxnv ) {
            case 16: return go_n( CellSoAT<16>{} );
            case 24: return go_n( CellSoAT<24>{} );
            case 48: return go_n( CellSoAT<48>{} );
            case 64: return go_n( CellSoAT<64>{} );
            default: return go_n( CellSoAT<32>{} );
        }
    }

    if ( a.cross )
        return cross( a, X, Y, Wp );

    if ( a.baisse )
        return baisse_stats( a, X, Y, Wp );

    if ( a.front )
        return front_stats( a, X, Y, Wp );

    if ( a.psigrid )
        return psigrid_stats( a, X, Y, Wp );

    if ( a.enclos )
        return enclos_stats( a, X, Y, Wp );

    if ( a.majorant )
        return majorant_stats( a, X, Y, Wp );

    if ( a.stats )
        return stats( a, X, Y, Wp );

    // le dispatch : les options sont des parametres de TEMPLATE (elles doivent disparaitre a la
    // compilation, sans quoi on mesurerait le branchement), donc on instancie les combinaisons
    // qu'on veut pouvoir choisir a l'execution.
    auto go = [ & ]( auto cell_tag, auto box_tag, auto in_tag ) {
        using Cell = decltype( cell_tag );
        constexpr bool BOX = decltype( box_tag )::value;
        constexpr bool IN  = decltype( in_tag )::value;
        // les poids sont eux aussi un parametre de TEMPLATE : sans eux, tout le majorant doit
        // disparaitre a la compilation, sinon le cas euclidien paie une borne qui vaut zero.
        auto weighted = [ & ]( auto w_tag ) {
            constexpr bool W = decltype( w_tag )::value;
            if ( a.tree == "all" )
                return run<EverySeed, Cell, BOX, IN, W>( a, X, Y, Wp );
            if ( a.tree == "grid" )
                return run<Grid, Cell, BOX, IN, W>( a, X, Y, Wp );
            if ( a.tree == "packed" )
                return run<AaBspPacked, Cell, BOX, IN, W>( a, X, Y, Wp );
            if ( a.tree == "hull" )
                return run<AaBspHull, Cell, BOX, IN, W>( a, X, Y, Wp );
            if ( a.tree == "pack" )
                return run<AaBspPack, Cell, BOX, IN, W>( a, X, Y, Wp );
            if ( a.tree == "bsp4" )
                return run<AaBsp4, Cell, BOX, IN, W>( a, X, Y, Wp );
            if ( a.tree == "bsp4l" )
                return run<AaBsp4L, Cell, BOX, IN, W>( a, X, Y, Wp );
            if ( a.tree == "obsp" )
                return run<ObBsp, Cell, BOX, IN, W>( a, X, Y, Wp );
            if ( a.tree == "pre" )
                return run<AaBspPre, Cell, BOX, IN, W>( a, X, Y, Wp );
            return run<AaBsp, Cell, BOX, IN, W>( a, X, Y, Wp );
        };
        return Wp ? weighted( std::true_type{} ) : weighted( std::false_type{} );
    };
    auto with_opts = [ & ]( auto cell_tag ) {
        if ( a.cellbox && ! a.skipin ) return go( cell_tag, std::true_type{}, std::false_type{} );
        if ( a.cellbox &&   a.skipin ) return go( cell_tag, std::true_type{}, std::true_type{} );
        if ( ! a.cellbox && ! a.skipin ) return go( cell_tag, std::false_type{}, std::false_type{} );
        return go( cell_tag, std::false_type{}, std::true_type{} );
    };
    switch ( a.maxnv ) {
        case 12: return with_opts( CellSoAT<12>{} );
        case 16: return with_opts( CellSoAT<16>{} );
        case 24: return with_opts( CellSoAT<24>{} );
        case 48: return with_opts( CellSoAT<48>{} );
        case 64: return with_opts( CellSoAT<64>{} );
        default: return with_opts( CellSoAT<32>{} );
    }
}

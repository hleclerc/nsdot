// Le MEME calcul que `test_PowerDiagram::pd accelerated`, mais par CGAL : un diagramme de
// puissance 2D sur `n` germes du carre unite, les cellules clippees sur ce carre, et leurs AIRES.
//
// = Pourquoi un programme a part, et pas un test
//
// C'est un ETALON, pas une verification : ce qu'on lui demande est un temps, dans les memes
// conditions que le notre (meme machine, meme n, meme FP64), et la somme des aires comme seul
// controle -- elle doit valoir 1 a l'arrondi pres, sans quoi le chiffre ne mesure rien.
//
// = Ce que CGAL fait, et ce qu'il ne fait pas
//
// `Regular_triangulation_2` est la triangulation reguliere (le dual du diagramme de puissance) :
// elle rend l'ADJACENCE, pas les cellules. La cellule d'un germe se reconstruit en tournant autour
// de lui (`incident_faces`) et en reliant les CENTRES DE PUISSANCE des faces incidentes -- c'est le
// polygone dual. Deux choses en sortent qui n'existent pas de notre cote :
//
//   * un germe « caché » (`hidden`) n'a pas de cellule du tout : c'est le cas qu'un poids assez bas
//     produit, et CGAL le retire de la triangulation au lieu de lui laisser une cellule vide.
//   * une cellule de BORD est infinie. CGAL ne connait pas notre domaine, donc le clip sur le carre
//     unite est fait ici, a la main (Sutherland-Hodgman), apres coup.
//
// Le clip est donc DE NOTRE COTE dans cette comparaison, alors qu'il est dans le kernel du notre.
// C'est le seul endroit ou les deux ne font pas exactement le meme travail, et il joue en faveur
// de CGAL (notre boite est coupee AVANT les bissectrices, la sienne apres) -- a garder en tete en
// lisant l'ecart.
//
// = Le noyau
//
// `Exact_predicates_inexact_constructions_kernel` : predicats exacts (c'est ce qui rend la
// triangulation robuste), constructions en `double`. C'est le choix normal pour une mesure, et
// c'est celui qui se compare a notre FP64. Un noyau a constructions exactes serait un autre banc.

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Regular_triangulation_2.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

// Le NOYAU sert directement de traits : `CGAL::Regular_triangulation_traits_2`, l'adaptateur
// d'autrefois, n'existe plus en CGAL 6 -- un noyau porte desormais son propre `Weighted_point_2`.
using K    = CGAL::Exact_predicates_inexact_constructions_kernel;
using Rt   = CGAL::Regular_triangulation_2<K>;
using Wp   = Rt::Weighted_point;
using Bare = Rt::Bare_point;

namespace {

struct P2 { double x, y; };

// Sutherland-Hodgman contre UN demi-plan `a.x + b.y <= c`, en place dans `poly`.
void clip_half( std::vector<P2> &poly, double a, double b, double c, std::vector<P2> &tmp ) {
    tmp.clear();
    const std::size_t n = poly.size();
    for ( std::size_t i = 0; i < n; ++i ) {
        const P2 &p = poly[ i ];
        const P2 &q = poly[ ( i + 1 ) % n ];
        const double sp = a * p.x + b * p.y - c;
        const double sq = a * q.x + b * q.y - c;
        if ( sp <= 0 )
            tmp.push_back( p );
        if ( ( sp < 0 && sq > 0 ) || ( sp > 0 && sq < 0 ) ) {
            const double t = sp / ( sp - sq );
            tmp.push_back( { p.x + t * ( q.x - p.x ), p.y + t * ( q.y - p.y ) } );
        }
    }
    poly.swap( tmp );
}

double area( const std::vector<P2> &poly ) {
    double a = 0;
    const std::size_t n = poly.size();
    for ( std::size_t i = 0; i < n; ++i ) {
        const P2 &p = poly[ i ];
        const P2 &q = poly[ ( i + 1 ) % n ];
        a += p.x * q.y - q.x * p.y;
    }
    return 0.5 * ( a < 0 ? -a : a );
}

/// Un nuage de `2d_des_familles/cases/` : des lignes `#`, puis `n`, puis `n` fois « x y w ».
///
/// La convention de poids est la MEME des deux cotes -- CGAL minimise `|x - p|^2 - w` comme nous --
/// donc les fichiers se lisent tels quels, sans conversion, et les deux calculent bien le meme
/// diagramme. C'est ce qui permet de comparer les temps sur les nuages DURS, ou l'uniforme cesse
/// d'etre representatif.
bool load_cloud( const std::string &path, std::vector<Wp> &pts ) {
    std::FILE *f = std::fopen( path.c_str(), "rb" );
    if ( ! f ) {
        std::printf( "impossible d'ouvrir '%s'\n", path.c_str() );
        return false;
    }
    std::fseek( f, 0, SEEK_END );
    const long sz = std::ftell( f );
    std::fseek( f, 0, SEEK_SET );
    std::string buf( std::size_t( sz ) + 1, '\0' );
    const std::size_t rd = std::fread( buf.data(), 1, std::size_t( sz ), f );
    std::fclose( f );
    buf[ rd ] = 0;

    const char *p = buf.data(), *e = p + rd;
    auto num = [ & ]( double &out ) {
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

    double v;
    if ( ! num( v ) ) return false;
    const int n = int( v );
    pts.clear();
    pts.reserve( n );
    for ( int i = 0; i < n; ++i ) {
        double x, y, w;
        if ( ! num( x ) || ! num( y ) || ! num( w ) )
            return false;
        pts.emplace_back( Bare( x, y ), w );
    }
    return true;
}

} // namespace

int main( int argc, char **argv ) {
    int n = 1000000, reps = 3;
    std::string load;
    for ( int i = 1; i < argc; ++i ) {
        const std::string s = argv[ i ];
        if ( s == "--load" && i + 1 < argc )       load = argv[ ++i ];
        else if ( s == "--reps" && i + 1 < argc )  reps = std::atoi( argv[ ++i ] );
        else                                       n = std::atoi( s.c_str() );
    }

    std::vector<Wp> pts;
    if ( ! load.empty() ) {
        if ( ! load_cloud( load, pts ) )
            return 1;
        n = int( pts.size() );
        std::printf( "  nuage '%s' : %d germes\n", load.c_str(), n );
    } else {
        // le MEME nuage que le banc python, dans l'esprit : uniforme dans [ 0.01, 0.99 ]^2. Pas les
        // memes tirages (deux generateurs differents), ce qui n'a pas d'importance pour un temps.
        std::mt19937_64 rng( 0 );
        std::uniform_real_distribution<double> uni( 0.01, 0.99 );
        pts.reserve( n );
        for ( int i = 0; i < n; ++i )
            pts.emplace_back( Bare( uni( rng ), uni( rng ) ), 0.0 );
    }

    double best = 1e300, last_sum = 0;
    for ( int r = 0; r < reps; ++r ) {
        const auto t0 = std::chrono::steady_clock::now();

        // l'insertion en VRAC : CGAL trie alors les points lui-meme (Hilbert) et insere avec
        // localisation spatiale, ce qui est de loin le chemin le plus rapide -- l'insertion une par
        // une mesurerait surtout le cout de la localisation.
        Rt rt;
        rt.insert( pts.begin(), pts.end() );

        // les cellules, une par sommet fini : le polygone dual, puis le clip sur le carre unite.
        double sum = 0;
        std::vector<P2> poly, tmp;
        for ( auto v = rt.finite_vertices_begin(); v != rt.finite_vertices_end(); ++v ) {
            poly.clear();
            poly.push_back( { 0, 0 } ); poly.push_back( { 1, 0 } );
            poly.push_back( { 1, 1 } ); poly.push_back( { 0, 1 } );

            // couper par la bissectrice de puissance avec chaque voisin. Passer par les VOISINS
            // plutot que par les centres de puissance des faces evite d'avoir a traiter a part les
            // faces infinies : un demi-plan est un demi-plan, borne ou non.
            const auto &p0 = v->point();
            Rt::Vertex_circulator c = rt.incident_vertices( v ), done( c );
            if ( c != nullptr ) {
                do {
                    if ( rt.is_infinite( c ) )
                        continue;
                    const auto &p1 = c->point();
                    const double dx = p1.x() - p0.x(), dy = p1.y() - p0.y();
                    const double off = dx * ( p0.x() + p1.x() ) / 2 + dy * ( p0.y() + p1.y() ) / 2
                                     + ( p0.weight() - p1.weight() ) / 2;
                    clip_half( poly, dx, dy, off, tmp );
                    if ( poly.size() < 3 )
                        break;
                } while ( ++c != done );
            }
            sum += poly.size() >= 3 ? area( poly ) : 0.0;
        }

        const double dt = std::chrono::duration<double>( std::chrono::steady_clock::now() - t0 ).count();
        best = std::min( best, dt );
        last_sum = sum;
    }

    std::printf( "  %d germes en 2D : %.3f s (%.0f ns/germe), somme des mesures %.6f\n",
                 n, best, best / n * 1e9, last_sum );
    return 0;
}

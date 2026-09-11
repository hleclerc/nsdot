// L'ETALON 3D, pendant de `power_2d.cpp` : ce que coute a CGAL la TRIANGULATION REGULIERE d'un
// nuage pondere 3D, et l'extraction de l'adjacence.
//
// = Ce qu'il mesure, et surtout ce qu'il NE mesure PAS
//
// Il chronometre deux choses :
//
//   1. `Regular_triangulation_3` sur `n` points ponderes -- l'insertion par plage, donc avec le
//      tri spatial que CGAL fait tout seul ;
//   2. le tour de chaque sommet (`finite_adjacent_vertices`), qui rend la LISTE DES VOISINS.
//
// Il ne construit AUCUNE cellule : ni les sommets duals, ni les faces, ni le clip sur le cube, ni
// les volumes. Le chiffre qui en sort est donc une BORNE INFERIEURE du prix qu'aurait CGAL pour
// faire le meme travail que nous. C'est ce qu'on veut savoir en premier : si cette borne depasse
// deja notre temps complet, la question est reglee sans ecrire l'extraction.
//
// Le controle de sanite est le nombre moyen de voisins : une cellule de Laguerre 3D poissonienne
// en a environ 15.5, et c'est ce que notre banc compte de son cote (15.2 sur ce nuage).
//
// = Le noyau
//
// `Exact_predicates_inexact_constructions_kernel`, comme en 2D : predicats exacts -- ce qui rend la
// triangulation robuste, et c'est precisement ce que notre coupe n'a pas besoin d'etre -- et
// constructions en `double`, pour se comparer a notre FP64.

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Regular_triangulation_3.h>

// NOS en-tetes, pris tels quels dans le banc. Ils sont sans dependance -- c'est tout l'objet de
// `2d_des_familles` -- donc les inclure ici n'y fait entrer ni CGAL ni gmp : la dependance ne va
// que dans ce sens. `--cells` mesure alors la chaine COMPLETE : triangulation CGAL, adjacence,
// puis NOTRE coupe et NOTRE mesure, avec les 15.5 vrais voisins au lieu des 87.8 proposes.
#include "geometry/Cell3.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

using K    = CGAL::Exact_predicates_inexact_constructions_kernel;
using Rt   = CGAL::Regular_triangulation_3<K>;
using Wp   = Rt::Weighted_point;
using Bare = Rt::Bare_point;

namespace {

double now() {
    using namespace std::chrono;
    return duration<double>( steady_clock::now().time_since_epoch() ).count();
}

/// Un nuage de `2d_des_familles/cases/` en 3D : des lignes `#`, puis `n`, puis `n` fois « x y z w ».
/// Meme convention de poids que nous -- CGAL minimise `|x - p|^2 - w` -- donc rien a convertir.
bool load_cloud( const std::string &path, std::vector<Wp> &pts ) {
    std::FILE *f = std::fopen( path.c_str(), "rb" );
    if ( ! f ) { std::printf( "impossible d'ouvrir '%s'\n", path.c_str() ); return false; }
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
        double x, y, z, w;
        if ( ! num( x ) || ! num( y ) || ! num( z ) || ! num( w ) ) return false;
        pts.emplace_back( Bare( x, y, z ), w );
    }
    return true;
}

} // namespace

int main( int argc, char **argv ) {
    int n = 200000, reps = 3;
    bool cells = false, tri = false;
    std::string load;
    for ( int i = 1; i < argc; ++i ) {
        const std::string s = argv[ i ];
        if ( s == "--load" && i + 1 < argc ) load = argv[ ++i ];
        else if ( s == "--reps" && i + 1 < argc ) reps = std::atoi( argv[ ++i ] );
        else if ( s == "--cells" ) cells = true;
        else if ( s == "--sort" ) { cells = true; tri = true; }
        else if ( s == "--help" ) {
            std::printf( "usage: power_3d [n] [--load FICHIER] [--reps R] [--cells]\n"
                         "  --cells  construit aussi les cellules avec NOTRE noyau, et somme\n" );
            return 0;
        } else n = std::atoi( argv[ i ] );
    }

    std::vector<Wp> pts;
    if ( ! load.empty() ) {
        if ( ! load_cloud( load, pts ) ) return 1;
        n = int( pts.size() );
    } else {
        std::mt19937_64 gen( 0 );
        std::uniform_real_distribution<double> u( 0.0, 1.0 );
        pts.reserve( n );
        for ( int i = 0; i < n; ++i ) pts.emplace_back( Bare( u( gen ), u( gen ), u( gen ) ), 0.0 );
    }

    double tt = 1e30, ta = 1e30, tc = 1e30;
    double voisins = 0, somme = 0;
    std::size_t caches = 0, sommets = 0;
    for ( int r = 0; r < reps; ++r ) {
        const double t0 = now();
        Rt rt( pts.begin(), pts.end() );
        const double t1 = now();

        double vs = 0;
        std::vector<Rt::Vertex_handle> adj;
        for ( auto v = rt.finite_vertices_begin(); v != rt.finite_vertices_end(); ++v ) {
            adj.clear();
            rt.finite_adjacent_vertices( v, std::back_inserter( adj ) );
            vs += double( adj.size() );
        }
        const double t2 = now();

        double sv = 0;
        if ( cells ) {
            std::vector<Rt::Vertex_handle> ad;
            for ( auto v = rt.finite_vertices_begin(); v != rt.finite_vertices_end(); ++v ) {
                ad.clear();
                rt.finite_adjacent_vertices( v, std::back_inserter( ad ) );
                const auto &wp0 = v->point();
                const double x0 = wp0.x(), y0 = wp0.y(), z0 = wp0.z(), w0 = wp0.weight();
                pd::Cell3T<128> c;
                c.init_as_unit_cube();
                pd::SI id = 0;
                if ( tri ) {
                    // DU PLUS PROCHE AU PLUS LOIN, au sens de la puissance. Ici les 15.5 plans sont
                    // TOUS des faces de la cellule finale : l'ordre ne change donc pas le resultat,
                    // seulement la taille des polyedres INTERMEDIAIRES -- et la coupe balaie tous
                    // les sommets a chaque fois.
                    std::sort( ad.begin(), ad.end(), [ & ]( Rt::Vertex_handle a, Rt::Vertex_handle b ) {
                        const auto &pa = a->point(); const auto &pb = b->point();
                        const double da = ( pa.x() - x0 ) * ( pa.x() - x0 ) + ( pa.y() - y0 ) * ( pa.y() - y0 )
                                        + ( pa.z() - z0 ) * ( pa.z() - z0 ) - pa.weight();
                        const double db = ( pb.x() - x0 ) * ( pb.x() - x0 ) + ( pb.y() - y0 ) * ( pb.y() - y0 )
                                        + ( pb.z() - z0 ) * ( pb.z() - z0 ) - pb.weight();
                        return da < db;
                    } );
                }
                for ( auto u : ad ) {
                    const auto &wp1 = u->point();
                    const double dx = wp1.x() - x0, dy = wp1.y() - y0, dz = wp1.z() - z0;
                    const double off = ( w0 - wp1.weight() ) / 2
                                     + dx * ( x0 + wp1.x() ) / 2
                                     + dy * ( y0 + wp1.y() ) / 2
                                     + dz * ( z0 + wp1.z() ) / 2;
                    c.cut( pd::Vec<3>{ dx, dy, dz }, off, id++ );
                }
                sv += c.measure();
            }
        }
        const double t3 = now();

        tt = std::min( tt, t1 - t0 );
        ta = std::min( ta, t2 - t1 );
        if ( cells ) { tc = std::min( tc, t3 - t2 ); somme = sv; }
        sommets = rt.number_of_vertices();
        caches = std::size_t( n ) - sommets;
        voisins = vs / double( sommets );
    }

    std::printf( "%-46s n=%d\n", load.empty() ? "uniforme" : load.c_str(), n );
    std::printf( "   triangulation  %7.3f s (%7.0f ns/germe)\n", tt, 1e9 * tt / n );
    std::printf( "   adjacence      %7.3f s (%7.0f ns/germe)\n", ta, 1e9 * ta / n );
    if ( cells ) {
        std::printf( "   coupe + mesure %7.3f s (%7.0f ns/germe)\n", tc, 1e9 * tc / n );
        std::printf( "   CHAINE ENTIERE %7.3f s (%7.0f ns/germe)   voisins %.2f   caches %zu"
                     "   somme %.9f\n",
                     tt + ta + tc, 1e9 * ( tt + ta + tc ) / n, voisins, caches, somme );
    } else
        std::printf( "   BORNE INF      %7.3f s (%7.0f ns/germe)   voisins %.2f   caches %zu\n",
                     tt + ta, 1e9 * ( tt + ta ) / n, voisins, caches );
    return 0;
}

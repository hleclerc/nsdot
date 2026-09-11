// LE PENDANT 2D DE `power_3d_light.cpp` : ce que couterait une version allegee de CGAL en 2D.
//
// La question posee est celle du NOYAU. `Exact_predicates_inexact_constructions_kernel` (Epick)
// evalue chaque predicat d'orientation / in_power_circle d'abord en ARITHMETIQUE D'INTERVALLES ; si
// l'intervalle contient zero -- c'est-a-dire si le signe n'est pas certain -- il rejoue le predicat
// en exact. `-DLIGHT_KERNEL` remplace le tout par `Simple_cartesian<double>` : le predicat devient
// un determinant FP64 nu, sans intervalle et sans repli. C'est EXACTEMENT ce que fait notre coupe.
//
// La difference des deux binaires est donc le prix de la robustesse, et rien d'autre : meme
// algorithme, meme structure, meme ordre d'insertion (Hilbert, insertion en vrac).
//
//   `--cells` enchaine NOTRE cellule 2D (`CellSoAT`) et NOTRE mesure sur les voisins rendus par la
//   triangulation. C'est la « rejoue de la connectivite » : on ne tente que les coupes FINALES.
//
// Attention : sans filtre, une degenerescence peut casser la triangulation. Les controles sont le
// nombre de sommets et la somme des aires (qui doit valoir 1).

#ifdef LIGHT_KERNEL
#   include <CGAL/Simple_cartesian.h>
using K = CGAL::Simple_cartesian<double>;
static constexpr const char *nom_noyau = "Simple_cartesian<double> (predicats FP64 nus)";
#else
#   include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
using K = CGAL::Exact_predicates_inexact_constructions_kernel;
static constexpr const char *nom_noyau = "Epick (predicats filtres exacts)";
#endif

#include <CGAL/Regular_triangulation_2.h>

#include "geometry/Cell.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>
#include "dump_csr.h"

using Rt   = CGAL::Regular_triangulation_2<K>;
using Wp   = Rt::Weighted_point;
using Bare = Rt::Bare_point;

namespace {

std::size_t sommets_att( const Rt &rt ) { return rt.number_of_vertices(); }

double now() {
    using namespace std::chrono;
    return duration<double>( steady_clock::now().time_since_epoch() ).count();
}

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
        double x, y, w;
        if ( ! num( x ) || ! num( y ) || ! num( w ) ) return false;
        pts.emplace_back( Bare( x, y ), w );
    }
    return true;
}

} // namespace

int main( int argc, char **argv ) {
    int n = 1000000, reps = 3;
    int decoy = 0;
    bool cells = false;
    std::string load, dump;
    for ( int i = 1; i < argc; ++i ) {
        const std::string s = argv[ i ];
        if ( s == "--load" && i + 1 < argc ) load = argv[ ++i ];
        else if ( s == "--dump" && i + 1 < argc ) dump = argv[ ++i ];
        else if ( s == "--reps" && i + 1 < argc ) reps = std::atoi( argv[ ++i ] );
        else if ( s == "--cells" ) cells = true;
        else if ( s == "--decoy" && i + 1 < argc ) decoy = std::atoi( argv[ ++i ] );
        else if ( s == "--help" ) {
            std::printf( "usage: power_2d_light [n] [--load F] [--reps R] [--cells]\n" );
            return 0;
        } else n = std::atoi( argv[ i ] );
    }

    std::vector<Wp> pts;
    if ( ! load.empty() ) {
        if ( ! load_cloud( load, pts ) ) return 1;
        n = int( pts.size() );
    } else {
        std::mt19937_64 gen( 0 );
        std::uniform_real_distribution<double> u( 0.01, 0.99 );
        pts.reserve( n );
        for ( int i = 0; i < n; ++i ) pts.emplace_back( Bare( u( gen ), u( gen ) ), 0.0 );
    }

    double tt = 1e30, ta = 1e30, tc = 1e30;
    double voisins = 0, somme = 0;
    std::size_t sommets = 0, caches = 0;

    for ( int r = 0; r < reps; ++r ) {
        const double t0 = now();
        Rt rt;
        rt.insert( pts.begin(), pts.end() );
        const double t1 = now();

        // L'ADJACENCE. En 2D elle est DEJA legere : `incident_vertices` est un circulateur qui suit
        // les faces autour du sommet, sans conteneur associatif temporaire -- contrairement a la
        // version 3D. Il n'y a donc pas de variante « maison » a opposer ici.
        std::vector<Rt::Vertex_handle> adj;
        double vs = 0;
        for ( auto v = rt.finite_vertices_begin(); v != rt.finite_vertices_end(); ++v ) {
            adj.clear();
            Rt::Vertex_circulator c = rt.incident_vertices( v ), done( c );
            if ( c != nullptr ) do {
                if ( ! rt.is_infinite( c ) ) adj.push_back( c );
            } while ( ++c != done );
            vs += double( adj.size() );
        }
        const double t2 = now();

        // LA REJOUE DE LA CONNECTIVITE. En deux temps, et c'est le point : la LISTE finale des
        // voisins est batie d'abord, HORS chronometre (c'est le travail que notre accelerateur
        // remplace, pas celui qu'on veut mesurer) ; puis on ne chronometre que la coupe et la
        // mesure, sur cette liste et sur elle seule. Ce que ca donne est donc le plancher
        // geometrique : le temps d'UNE cellule si l'on savait d'avance, sans jamais se tromper,
        // quels germes la coupent.
        double sv = 0;
        if ( cells ) {
            std::vector<double> moi, vois;                  // `moi` : x,y,w du germe. `vois` : idem
            std::vector<std::size_t> off( 1, 0 );           // decoupage de `vois` par germe
            moi.reserve( 3 * sommets_att( rt ) );
            for ( auto v = rt.finite_vertices_begin(); v != rt.finite_vertices_end(); ++v ) {
                const auto &p0 = v->point();
                moi.push_back( p0.x() ); moi.push_back( p0.y() ); moi.push_back( p0.weight() );
                Rt::Vertex_circulator c = rt.incident_vertices( v ), done( c );
                if ( c != nullptr ) do {
                    if ( rt.is_infinite( c ) ) continue;
                    const auto &p1 = c->point();
                    vois.push_back( p1.x() ); vois.push_back( p1.y() ); vois.push_back( p1.weight() );
                } while ( ++c != done );
                off.push_back( vois.size() / 3 );
            }

            const double u0 = now();
            for ( std::size_t i = 0; i + 1 < off.size(); ++i ) {
                const double x0 = moi[ 3 * i ], y0 = moi[ 3 * i + 1 ], w0 = moi[ 3 * i + 2 ];
                pd::CellSoAT<64> cl;
                cl.init_as_unit_square();
                pd::SI id = 0;
                for ( std::size_t j = off[ i ]; j < off[ i + 1 ]; ++j ) {
                    const double x1 = vois[ 3 * j ], y1 = vois[ 3 * j + 1 ], w1 = vois[ 3 * j + 2 ];
                    const double dx = x1 - x0, dy = y1 - y0;
                    const double o = dx * ( x0 + x1 ) / 2 + dy * ( y0 + y1 ) / 2 + ( w0 - w1 ) / 2;
                    cl.cut( dx, dy, o, id++ );
                }
                // LES LEURRES : `--decoy K` ajoute K plans PRIS SUR DE VRAIS GERMES, mais assez
                // loin pour qu'aucun ne coupe. La pente en K est le prix d'UNE COUPE INUTILE --
                // celle que notre accelerateur tente et qui ne rapporte rien.
                for ( int d = 0; d < decoy; ++d ) {
                    const std::size_t j = ( i + 50 + std::size_t( d ) * 7 ) % ( off.size() - 1 );
                    const double x1 = moi[ 3 * j ], y1 = moi[ 3 * j + 1 ], w1 = moi[ 3 * j + 2 ];
                    const double dx = x1 - x0, dy = y1 - y0;
                    const double o = dx * ( x0 + x1 ) / 2 + dy * ( y0 + y1 ) / 2 + ( w0 - w1 ) / 2;
                    cl.cut( dx, dy, o, id++ );
                }
                sv += cl.measure();
            }
            tc = std::min( tc, now() - u0 );
            somme = sv;

            // LE VIDAGE, une seule fois et HORS chronometre (voir la version 3D).
            if ( ! dump.empty() && r == 0 ) {
                std::vector<double> ref( off.size() - 1 );
                for ( std::size_t i = 0; i + 1 < off.size(); ++i ) {
                    const double x0 = moi[ 3 * i ], y0 = moi[ 3 * i + 1 ], w0 = moi[ 3 * i + 2 ];
                    pd::CellSoAT<64> cl;
                    cl.init_as_unit_square();
                    pd::SI id = 0;
                    for ( std::size_t j = off[ i ]; j < off[ i + 1 ]; ++j ) {
                        const double x1 = vois[ 3 * j ], y1 = vois[ 3 * j + 1 ], w1 = vois[ 3 * j + 2 ];
                        const double dx = x1 - x0, dy = y1 - y0;
                        const double o = dx * ( x0 + x1 ) / 2 + dy * ( y0 + y1 ) / 2 + ( w0 - w1 ) / 2;
                        cl.cut( dx, dy, o, id++ );
                    }
                    ref[ i ] = cl.measure();
                }
                dump_csr( dump, 2, moi, off, vois, ref );
            }
        }
        tt = std::min( tt, t1 - t0 );
        ta = std::min( ta, t2 - t1 );
        sommets = rt.number_of_vertices();
        caches = std::size_t( n ) - sommets;
        voisins = vs / double( sommets );
    }

    std::printf( "%-46s n=%d   noyau %s\n", load.empty() ? "uniforme" : load.c_str(), n, nom_noyau );
    std::printf( "   triangulation  %7.3f s (%7.0f ns/germe)\n", tt, 1e9 * tt / n );
    std::printf( "   adjacence      %7.3f s (%7.0f ns/germe)\n", ta, 1e9 * ta / n );
    if ( cells ) {
        std::printf( "   coupe + mesure %7.3f s (%7.0f ns/germe)   leurres %d\n", tc, 1e9 * tc / n, decoy );
        std::printf( "   CHAINE ENTIERE %7.3f s (%7.0f ns/germe)   voisins %.2f   caches %zu   somme %.9f\n",
                     tt + ta + tc, 1e9 * ( tt + ta + tc ) / n, voisins, caches, somme );
    } else
        std::printf( "   BORNE INF      %7.3f s (%7.0f ns/germe)   voisins %.2f   caches %zu\n",
                     tt + ta, 1e9 * ( tt + ta ) / n, voisins, caches );
    return 0;
}

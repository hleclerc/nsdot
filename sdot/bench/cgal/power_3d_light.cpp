// CE QUE COUTERAIT UNE VERSION ALLEGEE DE L'ALGORITHME DE CGAL.
//
// `power_3d.cpp` mesure CGAL tel qu'on l'emploie normalement. Celui-ci demonte le chiffre en trois
// postes qu'on peut allumer separement, pour savoir ce qui, dans les nanosecondes de CGAL, tient a
// l'ALGORITHME (l'insertion incrementale dans une triangulation reguliere, qu'il faudrait de toute
// facon ecrire) et ce qui tient a la GENERICITE de la bibliotheque (predicats filtres, requetes
// d'adjacence generiques) -- c'est-a-dire au plafond de ce qu'une reecriture maison pourrait gagner.
//
//   1. LE NOYAU.  `-DLIGHT_KERNEL` remplace `Exact_predicates_inexact_constructions_kernel` par
//      `Simple_cartesian<double>` : plus de filtre par intervalles, plus de repli exact, les
//      predicats deviennent des determinants en FP64 -- exactement ce que notre coupe fait deja.
//      La difference des deux binaires EST le prix de la robustesse.
//
//      Attention : sans filtre, une degenerescence peut faire boucler ou casser la triangulation.
//      Le nuage uniforme n'en a pas ; le controle est le nombre de sommets et la somme des volumes.
//
//   2. L'ADJACENCE.  `--adj fast` remplace `finite_adjacent_vertices` -- qui passe par un conteneur
//      associatif temporaire a chaque sommet -- par le tour des cellules incidentes avec
//      dedoublonnage O(1) sur un ESTAMPILLE range dans le sommet (`with_info`). C'est ce qu'on
//      ecrirait soi-meme, et ca ne change rien a l'algorithme : la meme structure est lue.
//
//   3. LA CHAINE.  `--cells` enchaine NOTRE coupe et NOTRE mesure sur les voisins rendus, comme
//      dans `power_3d.cpp`, pour que le total soit comparable au banc.

#ifdef LIGHT_KERNEL
#   include <CGAL/Simple_cartesian.h>
using K = CGAL::Simple_cartesian<double>;
static constexpr const char *nom_noyau = "Simple_cartesian<double> (predicats FP64 nus)";
#else
#   include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
using K = CGAL::Exact_predicates_inexact_constructions_kernel;
static constexpr const char *nom_noyau = "Epick (predicats filtres exacts)";
#endif

#include <CGAL/Regular_triangulation_3.h>
#include <CGAL/Regular_triangulation_vertex_base_3.h>
#include <CGAL/Regular_triangulation_cell_base_3.h>
#include <CGAL/Triangulation_vertex_base_with_info_3.h>

#include "geometry/Cell3.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>
#include "dump_csr.h"

using Vbb = CGAL::Regular_triangulation_vertex_base_3<K>;
using Vb  = CGAL::Triangulation_vertex_base_with_info_3<int, K, Vbb>;
using Cb  = CGAL::Regular_triangulation_cell_base_3<K>;
using Tds = CGAL::Triangulation_data_structure_3<Vb, Cb>;
using Rt   = CGAL::Regular_triangulation_3<K, Tds>;
using Wp   = Rt::Weighted_point;
using Bare = Rt::Bare_point;
using Vh   = Rt::Vertex_handle;

namespace {

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
        double x, y, z, w;
        if ( ! num( x ) || ! num( y ) || ! num( z ) || ! num( w ) ) return false;
        pts.emplace_back( Bare( x, y, z ), w );
    }
    return true;
}

} // namespace

int main( int argc, char **argv ) {
    int n = 200000, reps = 3;
    int decoy = 0;
    bool cells = false, fast = false;
    std::string load, dump;
    for ( int i = 1; i < argc; ++i ) {
        const std::string s = argv[ i ];
        if ( s == "--load" && i + 1 < argc ) load = argv[ ++i ];
        else if ( s == "--dump" && i + 1 < argc ) dump = argv[ ++i ];
        else if ( s == "--reps" && i + 1 < argc ) reps = std::atoi( argv[ ++i ] );
        else if ( s == "--cells" ) cells = true;
        else if ( s == "--decoy" && i + 1 < argc ) decoy = std::atoi( argv[ ++i ] );
        else if ( s == "--adj" && i + 1 < argc ) fast = std::string( argv[ ++i ] ) == "fast";
        else if ( s == "--help" ) {
            std::printf( "usage: power_3d_light [n] [--load F] [--reps R] [--cells] [--adj fast]\n" );
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
    std::size_t sommets = 0, caches = 0;

    for ( int r = 0; r < reps; ++r ) {
        const double t0 = now();
        Rt rt( pts.begin(), pts.end() );
        const double t1 = now();

        // L'ESTAMPILLE : `info` porte le numero du sommet dont on fait le tour. Deux cellules
        // incidentes partagent une arete, donc proposent le meme voisin -- c'est ce doublon que le
        // conteneur associatif de CGAL elimine, et qu'un entier elimine aussi.
        int stamp = 0;
        for ( auto v = rt.finite_vertices_begin(); v != rt.finite_vertices_end(); ++v )
            v->info() = -1;

        double vs = 0;
        std::vector<Vh> adj;
        std::vector<Rt::Cell_handle> inc;
        for ( auto v = rt.finite_vertices_begin(); v != rt.finite_vertices_end(); ++v ) {
            adj.clear();
            if ( fast ) {
                inc.clear();
                rt.incident_cells( v, std::back_inserter( inc ) );
                ++stamp;
                for ( auto c : inc )
                    for ( int i = 0; i < 4; ++i ) {
                        const Vh u = c->vertex( i );
                        if ( u == Vh( v ) || rt.is_infinite( u ) || u->info() == stamp ) continue;
                        u->info() = stamp;
                        adj.push_back( u );
                    }
            } else
                rt.finite_adjacent_vertices( v, std::back_inserter( adj ) );
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
            std::vector<double> moi, vois;                  // x,y,z,w du germe / de chaque voisin
            std::vector<std::size_t> off( 1, 0 );
            stamp = 0;
            for ( auto v = rt.finite_vertices_begin(); v != rt.finite_vertices_end(); ++v )
                v->info() = -1;
            for ( auto v = rt.finite_vertices_begin(); v != rt.finite_vertices_end(); ++v ) {
                adj.clear();
                if ( fast ) {
                    inc.clear();
                    rt.incident_cells( v, std::back_inserter( inc ) );
                    ++stamp;
                    for ( auto c : inc )
                        for ( int i = 0; i < 4; ++i ) {
                            const Vh u = c->vertex( i );
                            if ( u == Vh( v ) || rt.is_infinite( u ) || u->info() == stamp ) continue;
                            u->info() = stamp;
                            adj.push_back( u );
                        }
                } else
                    rt.finite_adjacent_vertices( v, std::back_inserter( adj ) );

                const auto &wp0 = v->point();
                moi.push_back( wp0.x() ); moi.push_back( wp0.y() );
                moi.push_back( wp0.z() ); moi.push_back( wp0.weight() );
                for ( auto u : adj ) {
                    const auto &wp1 = u->point();
                    vois.push_back( wp1.x() ); vois.push_back( wp1.y() );
                    vois.push_back( wp1.z() ); vois.push_back( wp1.weight() );
                }
                off.push_back( vois.size() / 4 );
            }

            const double u0 = now();
            for ( std::size_t i = 0; i + 1 < off.size(); ++i ) {
                const double x0 = moi[ 4 * i ], y0 = moi[ 4 * i + 1 ],
                             z0 = moi[ 4 * i + 2 ], w0 = moi[ 4 * i + 3 ];
                pd::Cell3T<128> c;
                c.init_as_unit_cube();
                pd::SI id = 0;
                for ( std::size_t j = off[ i ]; j < off[ i + 1 ]; ++j ) {
                    const double x1 = vois[ 4 * j ], y1 = vois[ 4 * j + 1 ],
                                 z1 = vois[ 4 * j + 2 ], w1 = vois[ 4 * j + 3 ];
                    const double dx = x1 - x0, dy = y1 - y0, dz = z1 - z0;
                    const double o = ( w0 - w1 ) / 2 + dx * ( x0 + x1 ) / 2
                                   + dy * ( y0 + y1 ) / 2 + dz * ( z0 + z1 ) / 2;
                    c.cut( pd::Vec<3>{ dx, dy, dz }, o, id++ );
                }
                // LES LEURRES : `--decoy K` ajoute K plans PRIS SUR DE VRAIS GERMES, mais assez
                // loin pour qu'aucun ne coupe. La pente en K est le prix d'UNE COUPE INUTILE --
                // celle que notre accelerateur tente et qui ne rapporte rien.
                for ( int d = 0; d < decoy; ++d ) {
                    const std::size_t j = ( i + 50 + std::size_t( d ) * 7 ) % ( off.size() - 1 );
                    const double x1 = moi[ 4 * j ], y1 = moi[ 4 * j + 1 ],
                                 z1 = moi[ 4 * j + 2 ], w1 = moi[ 4 * j + 3 ];
                    const double dx = x1 - x0, dy = y1 - y0, dz = z1 - z0;
                    const double o = ( w0 - w1 ) / 2 + dx * ( x0 + x1 ) / 2
                                   + dy * ( y0 + y1 ) / 2 + dz * ( z0 + z1 ) / 2;
                    c.cut( pd::Vec<3>{ dx, dy, dz }, o, id++ );
                }
                sv += c.measure();
            }
            tc = std::min( tc, now() - u0 );
            somme = sv;

            // LE VIDAGE, une seule fois et HORS chronometre : la boucle chronometree ci-dessus ne
            // doit pas porter un `if` de plus, et le volume de reference se recalcule ici pour
            // rien de mesure.
            if ( ! dump.empty() && r == 0 ) {
                std::vector<double> ref( off.size() - 1 );
                for ( std::size_t i = 0; i + 1 < off.size(); ++i ) {
                    const double x0 = moi[ 4 * i ], y0 = moi[ 4 * i + 1 ],
                                 z0 = moi[ 4 * i + 2 ], w0 = moi[ 4 * i + 3 ];
                    pd::Cell3T<128> c;
                    c.init_as_unit_cube();
                    pd::SI id = 0;
                    for ( std::size_t j = off[ i ]; j < off[ i + 1 ]; ++j ) {
                        const double x1 = vois[ 4 * j ], y1 = vois[ 4 * j + 1 ],
                                     z1 = vois[ 4 * j + 2 ], w1 = vois[ 4 * j + 3 ];
                        const double dx = x1 - x0, dy = y1 - y0, dz = z1 - z0;
                        const double o = ( w0 - w1 ) / 2 + dx * ( x0 + x1 ) / 2
                                       + dy * ( y0 + y1 ) / 2 + dz * ( z0 + z1 ) / 2;
                        c.cut( pd::Vec<3>{ dx, dy, dz }, o, id++ );
                    }
                    ref[ i ] = c.measure();
                }
                dump_csr( dump, 3, moi, off, vois, ref );
            }
        }
        tt = std::min( tt, t1 - t0 );
        ta = std::min( ta, t2 - t1 );
        sommets = rt.number_of_vertices();
        caches = std::size_t( n ) - sommets;
        voisins = vs / double( sommets );
    }

    std::printf( "%-46s n=%d   noyau %s   adjacence %s\n",
                 load.empty() ? "uniforme" : load.c_str(), n, nom_noyau, fast ? "MAISON" : "CGAL" );
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

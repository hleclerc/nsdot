// =====================================================================================
// LE TEMOIN EXTERIEUR EN 3D : CGAL, SEQUENTIEL, BORNE AU CUBE, VOLUMES COMPRIS.
//
// La premiere version de ce banc ne bornait pas et ne rendait que des sommets duals. Elle mesurait
// donc MOINS DE TRAVAIL QUE LE NOTRE, et le rapport qu'elle donnait etait un minorant. Ici CGAL
// fait exactement la meme chose que nous : chaque cellule est bornee au cube unite, et son VOLUME
// est calcule. La somme doit valoir 1, comme chez nous -- c'est le temoin d'exactitude.
//
// = BORNER AU CUBE : LES POINTS MIROIRS
//
// La facon classique, et elle est EXACTE. On ajoute, pour chaque germe `q` proche d'une face `F`
// du cube, son symetrique `q'` par rapport a `F`, de MEME POIDS. Le plan radical de `( q, w )` et
// `( q', w )` est alors exactement `F`.
//
// Et les miroirs ne coupent rien A L'INTERIEUR du cube : si `F` est `x_d = 0`, alors pour tout `x`
// du cube et tout germe `q`, `( x_d + q_d )^2 >= ( x_d - q_d )^2` puisque les deux sont positifs,
// donc `q'` n'est jamais plus proche que `q`. La cellule d'un germe original dans le diagramme
// augmente est donc EXACTEMENT sa cellule vraie intersectee avec le cube.
//
// ON NE MIROITE QUE LA COQUILLE d'epaisseur `--coquille` fois l'espacement `h = n^(-1/3)` : miroiter
// tous les germes couterait sept fois la triangulation pour une raison qui n'est pas celle de
// CGAL. C'est un choix heuristique, donc il est VERIFIE : toute cellule d'un germe original doit
// etre bornee et tenir dans le cube, et la somme des volumes doit valoir 1.
//
// LE DEFAUT DE 2 h EST UNE MESURE, et la verification est ce qui l'a donne :
//
//   coquille   points triangules ( n=200000 )   hors du cube   volume total
//     1 h              220 464                      1265        1.000166352
//     1,5 h            230 923                         9        1.000000452
//     2 h              241 091                         0        1.000000000
//     3 h              261 529                         0        1.000000000
//
// Un volume qui depasse 1 est exactement le symptome attendu : une cellule non bornee par le cube
// deborde sur sa voisine. La verification n'est donc pas decorative -- elle a trouve la valeur.
//
// = LES VOLUMES, ET LE PIEGE DE LAGUERRE
//
// La cellule d'un sommet `v` a une FACE par arete finie incidente a `v`, et les sommets de cette
// face sont les duals des tetraedres qui tournent autour de l'arete. On triangule chaque face en
// eventail depuis son premier sommet et on somme les tetraedres, en valeur absolue -- ce qui
// dispense d'orienter les faces, la cellule etant convexe.
//
// MAIS L'APEX DE L'EVENTAIL NE PEUT PAS ETRE LE GERME. En Voronoi le germe est dans sa cellule ;
// EN LAGUERRE IL N'Y EST PAS FORCEMENT, et l'eventail en valeur absolue compte alors deux fois ce
// qui depasse. Prendre le germe donnait un volume total de 1,0166 au lieu de 1 -- l'erreur etait
// invisible en Voronoi et franche en Laguerre. L'apex est donc le BARYCENTRE des sommets duals de
// la cellule, qui est dedans par convexite ; c'est exactement ce que fait notre `Cellule3::volume`,
// et pour la meme raison.
//
// = LES GERMES CACHES
//
// En Laguerre, un germe dont la boule est contenue dans une autre est CACHE : il n'est pas sommet
// de la triangulation et sa cellule est vide. On les compte, et leur volume nul entre bien dans la
// somme -- c'est aussi ce que nous rendons.
//
// L'insertion se fait PAR PLAGE : CGAL trie alors spatialement et prend son chemin rapide.
// =====================================================================================
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Regular_triangulation_3.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <random>
#include <string>
#include <vector>

using K  = CGAL::Exact_predicates_inexact_constructions_kernel;
using RT = CGAL::Regular_triangulation_3<K>;
using Wp = RT::Weighted_point;
using Pt = RT::Bare_point;

static double now() { using namespace std::chrono;
    return duration<double>( steady_clock::now().time_since_epoch() ).count(); }

int main( int argc, char **argv ) {
    int    n = 100000;
    double frac = 0, coquille = 2;
    for ( int i = 1; i < argc; ++i ) {
        const std::string o = argv[ i ];
        if      ( o == "--n"        && i + 1 < argc ) n        = atoi( argv[ ++i ] );
        else if ( o == "--poids"    && i + 1 < argc ) frac     = atof( argv[ ++i ] );
        else if ( o == "--coquille" && i + 1 < argc ) coquille = atof( argv[ ++i ] );
        else { fprintf( stderr, "option inconnue %s\n", argv[ i ] ); return 1; }
    }

    // LE MEME NUAGE QUE `pd_bspf3d` : `mt19937( 12345 )`, uniforme, et `dw ~ h^2` avec
    // `h ~ n^(-1/3)`, de sorte que le decalage d'un plan soit une fraction de l'espacement.
    std::mt19937 gen( 12345 );
    std::uniform_real_distribution<float> u( 0, 1 );
    const double h  = std::pow( double( n ), -1.0 / 3.0 );
    const double h2 = h * h;
    std::vector<Wp> pts; pts.reserve( n );
    for ( int i = 0; i < n; ++i ) {
        const double x = u( gen ), y = u( gen ), z = u( gen );
        pts.emplace_back( Pt( x, y, z ), frac * h2 * u( gen ) );
    }

    // ---- LES MIROIRS, sur la coquille seulement. Hors chronometre : c'est notre echafaudage, pas
    // le travail de CGAL. ( Le cout de la triangulation des miroirs, lui, EST compte. )
    const double ep = coquille * h;
    std::vector<Wp> tout = pts;
    for ( const Wp &p : pts ) {
        const double c[ 3 ] = { p.point().x(), p.point().y(), p.point().z() };
        for ( int d = 0; d < 3; ++d ) {
            double q[ 3 ] = { c[0], c[1], c[2] };
            if ( c[ d ] < ep )     { q[ d ] = -c[ d ];     tout.emplace_back( Pt( q[0], q[1], q[2] ), p.weight() ); }
            q[0] = c[0]; q[1] = c[1]; q[2] = c[2];
            if ( c[ d ] > 1 - ep ) { q[ d ] = 2 - c[ d ];  tout.emplace_back( Pt( q[0], q[1], q[2] ), p.weight() ); }
        }
    }

    // ---- 1. la triangulation. L'insertion par PLAGE trie spatialement : c'est le chemin rapide.
    const double t0 = now();
    RT rt;
    rt.insert( tout.begin(), tout.end() );
    const double t_tri = now() - t0;

    // ---- 2. les cellules BORNEES et leurs VOLUMES.
    const double t1 = now();
    double        volume = 0;
    long long     bornees = 0, ouvertes = 0, dehors = 0, faces = 0, sommets_duals = 0;
    std::vector<RT::Edge>        aretes;
    std::vector<Pt>              duals;
    std::vector<size_t>          debuts;                 ///< le debut de chaque face dans `duals`
    for ( auto v = rt.finite_vertices_begin(); v != rt.finite_vertices_end(); ++v ) {
        const Pt p = v->point().point();
        // UN GERME ORIGINAL a ses trois coordonnees dans le cube ; un miroir en a une dehors.
        if ( p.x() < 0 || p.x() > 1 || p.y() < 0 || p.y() > 1 || p.z() < 0 || p.z() > 1 )
            continue;

        aretes.clear();
        rt.incident_edges( v, std::back_inserter( aretes ) );
        duals.clear(); debuts.clear();
        bool ok = true;
        for ( const RT::Edge &e : aretes ) {
            if ( rt.is_infinite( e ) ) { ok = false; break; }
            const size_t d0 = duals.size();
            RT::Cell_circulator c = rt.incident_cells( e ), fin = c;
            do {
                if ( rt.is_infinite( c ) ) { ok = false; break; }
                duals.push_back( rt.dual( c ) );
            } while ( ++c != fin );
            if ( ! ok ) break;
            if ( duals.size() - d0 < 3 ) { duals.resize( d0 ); continue; }
            debuts.push_back( d0 );
        }
        if ( ! ok ) { ++ouvertes; continue; }
        debuts.push_back( duals.size() );
        const long long nf = (long long) debuts.size() - 1;
        if ( nf < 4 ) { ++bornees; continue; }           // degenere : volume nul

        // L'APEX EST LE BARYCENTRE des sommets duals, pas le germe -- voir l'en-tete.
        double gx = 0, gy = 0, gz = 0;
        for ( const Pt &d : duals ) { gx += d.x(); gy += d.y(); gz += d.z(); }
        gx /= duals.size(); gy /= duals.size(); gz /= duals.size();

        double vol = 0;
        for ( long long f = 0; f < nf; ++f ) {
            const size_t a0 = debuts[ f ], a1 = debuts[ f + 1 ];
            const Pt &d0 = duals[ a0 ];
            for ( size_t j = a0 + 1; j + 1 < a1; ++j ) {
                const Pt &a = duals[ j ], &b = duals[ j + 1 ];
                const double ax = a.x() - d0.x(), ay = a.y() - d0.y(), az = a.z() - d0.z();
                const double bx = b.x() - d0.x(), by = b.y() - d0.y(), bz = b.z() - d0.z();
                const double cx = gx - d0.x(), cy = gy - d0.y(), cz = gz - d0.z();
                const double det = ax * ( by * cz - bz * cy )
                                 - ay * ( bx * cz - bz * cx )
                                 + az * ( bx * cy - by * cx );
                vol += ( det < 0 ? -det : det ) / 6;
            }
        }
        ++bornees; faces += nf; sommets_duals += (long long) duals.size(); volume += vol;
    }
    const double t_cell = now() - t1;

    // ---- 3. LA VERIFICATION DE LA COQUILLE, hors chronometre : toute cellule d'un germe original
    // doit tenir dans le cube. Sinon la coquille est trop mince et le resultat est faux.
    for ( auto v = rt.finite_vertices_begin(); v != rt.finite_vertices_end(); ++v ) {
        const Pt p = v->point().point();
        if ( p.x() < 0 || p.x() > 1 || p.y() < 0 || p.y() > 1 || p.z() < 0 || p.z() > 1 ) continue;
        std::vector<RT::Cell_handle> cs;
        rt.incident_cells( v, std::back_inserter( cs ) );
        bool in = true;
        for ( auto c : cs ) {
            if ( rt.is_infinite( c ) ) { in = false; break; }
            const Pt d = rt.dual( c );
            const double e = 1e-9;
            in &= d.x() >= -e && d.x() <= 1 + e && d.y() >= -e && d.y() <= 1 + e
               && d.z() >= -e && d.z() <= 1 + e;
        }
        if ( ! in ) ++dehors;
    }

    const long long caches = n - bornees - ouvertes;
    printf( "CGAL %d.%d en 3D, n = %d, %s -- BORNE AU CUBE, volumes compris\n",
            CGAL_VERSION_MAJOR, CGAL_VERSION_MINOR, n, frac > 0 ? "Laguerre" : "Voronoi" );
    printf( "  %zu points triangules ( %d germes + %zu miroirs, coquille %.1f h )\n",
            tout.size(), n, tout.size() - pts.size(), coquille );
    printf( "  triangulation      : %8.4f s   ( %7.1f ns/germe )\n", t_tri, t_tri * 1e9 / n );
    printf( "  cellules + volumes : %8.4f s   ( %7.1f ns/germe )\n", t_cell, t_cell * 1e9 / n );
    printf( "  TOTAL              : %8.4f s   ( %7.1f ns/germe )\n",
            t_tri + t_cell, ( t_tri + t_cell ) * 1e9 / n );
    printf( "\n  volume total %.9f   <- doit valoir 1\n", volume );
    printf( "  cellules bornees %lld, caches %lld, encore OUVERTES %lld, hors du cube %lld\n",
            bornees, caches, ouvertes, dehors );
    if ( ouvertes || dehors )
        printf( "  !! la coquille est trop mince : augmenter --coquille\n" );
    printf( "  faces par cellule %.2f, sommets duals par face %.2f\n",
            double( faces ) / ( bornees ? bornees : 1 ),
            double( sommets_duals ) / ( faces ? faces : 1 ) );
    printf( "CSVCG3 n=%d poids=%.1f tri=%.1f cell=%.1f tot=%.1f vol=%.9f faces=%.2f ouv=%lld dehors=%lld pts=%zu\n",
            n, frac, t_tri * 1e9 / n, t_cell * 1e9 / n, ( t_tri + t_cell ) * 1e9 / n, volume,
            double( faces ) / ( bornees ? bornees : 1 ), ouvertes, dehors, tout.size() );
    return 0;
}

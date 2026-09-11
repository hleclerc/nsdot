// =====================================================================================
// LE TEMOIN EXTERIEUR : CGAL, SEQUENTIEL.
//
// Triangulation REGULIERE ( le dual de Laguerre ) puis extraction des cellules en tournant autour
// des faces incidentes a chaque sommet.
//
// LE CAS EST LE MEME QUE CELUI DE `pd_sc2` -- meme graine, meme loi, meme formule de poids -- pour
// que les deux nombres soient comparables sans precaution.
//
// CE QUE LA COMPARAISON NE DIT PAS. CGAL ne borne pas au domaine : les cellules du bord sont NON
// BORNEES et il les rend telles quelles, la ou nous coupons au carre unite. Son temps est donc un
// MINORANT de ce que couterait le meme travail que le notre. On compte les cellules entierement
// interieures a part, c'est la seule partie directement comparable.
//
// L'insertion se fait PAR PLAGE : CGAL trie alors spatialement et prend son chemin rapide. Inserer
// point par point serait le desavantager pour rien.
// =====================================================================================
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Regular_triangulation_2.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

using K  = CGAL::Exact_predicates_inexact_constructions_kernel;
using RT = CGAL::Regular_triangulation_2<K>;
using Wp = RT::Weighted_point;
using Pt = RT::Bare_point;

static double now() { using namespace std::chrono;
    return duration<double>( steady_clock::now().time_since_epoch() ).count(); }

int main( int argc, char **argv ) {
    const int n = argc > 1 ? atoi( argv[ 1 ] ) : 200000;
    const double frac = argc > 2 ? atof( argv[ 2 ] ) : 0.0;

    std::mt19937 gen( 12345 );
    std::uniform_real_distribution<double> u( 0, 1 );
    std::vector<Wp> pts; pts.reserve( n );
    const double h2 = 1.0 / n;
    for ( int i = 0; i < n; ++i )
        pts.emplace_back( Pt( u( gen ), u( gen ) ), frac * h2 * u( gen ) );

    // ---- 1. la triangulation. L'insertion par PLAGE trie spatialement : c'est le chemin rapide.
    const double t0 = now();
    RT rt;
    rt.insert( pts.begin(), pts.end() );
    const double t_tri = now() - t0;

    // ---- 2. les cellules. Pour chaque sommet fini, on tourne autour de ses faces incidentes et
    // on prend les duals ( centres circonscrits ponderes ). Les sommets du bord ont une cellule
    // NON BORNEE : on les compte a part, faute de domaine dans CGAL.
    const double t1 = now();
    double aire = 0;
    long long bornees = 0, non_bornees = 0, cotes = 0, dedans = 0;
    double aire_dedans = 0;
    for ( auto v = rt.finite_vertices_begin(); v != rt.finite_vertices_end(); ++v ) {
        RT::Face_circulator f = rt.incident_faces( v ), fin = f;
        bool ouverte = false;
        double a = 0;
        Pt premier, precedent;
        int c = 0;
        do {
            if ( rt.is_infinite( f ) ) { ouverte = true; break; }
            const Pt d = rt.dual( f );
            if ( c == 0 ) premier = d;
            else a += precedent.x() * d.y() - d.x() * precedent.y();
            precedent = d;
            ++c;
        } while ( ++f != fin );
        if ( ouverte || c < 3 ) { ++non_bornees; continue; }
        a += precedent.x() * premier.y() - premier.x() * precedent.y();
        const double s = 0.5 * ( a < 0 ? -a : a );
        aire += s;
        ++bornees; cotes += c;
        // la meme cellule, mais seulement si elle tient DANS le carre unite : c'est la seule
        // partie comparable a ce que nous calculons, puisque nous bornons au domaine.
        bool in = true;
        RT::Face_circulator g = rt.incident_faces( v ), gfin = g;
        do { const Pt d = rt.dual( g );
             in &= d.x() >= 0 && d.x() <= 1 && d.y() >= 0 && d.y() <= 1; } while ( ++g != gfin );
        if ( in ) { aire_dedans += s; ++dedans; }
    }
    const double t_cell = now() - t1;

    printf( "CGAL %d.%d, n = %d, %s\n", CGAL_VERSION_MAJOR, CGAL_VERSION_MINOR, n,
            frac > 0 ? "Laguerre" : "Voronoi" );
    printf( "  triangulation      : %8.4f s   ( %7.1f ns/germe )\n", t_tri, t_tri * 1e9 / n );
    printf( "  extraction cellules: %8.4f s   ( %7.1f ns/germe )\n", t_cell, t_cell * 1e9 / n );
    printf( "  TOTAL              : %8.4f s   ( %7.1f ns/germe )\n",
            t_tri + t_cell, ( t_tri + t_cell ) * 1e9 / n );
    printf( "  cellules bornees %lld, non bornees %lld, %.2f cotes\n",
            bornees, non_bornees, double( cotes ) / ( bornees ? bornees : 1 ) );
    printf( "  aire totale %.3f  ( CGAL ne borne pas au domaine )\n", aire );
    printf( "  dont %lld cellules ENTIEREMENT dans le carre : aire %.6f\n", dedans, aire_dedans );
    return 0;
}

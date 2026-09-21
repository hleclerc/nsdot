// =====================================================================================
// LA MEMOIRE EN 3D, SA BORNE SUPERIEURE. La question de `2d_des_familles` ( journal, `--memo` ) :
// que rend le fait de proposer D'ABORD les diracs qui portaient une face de la cellule a la passe
// precedente ? Ici les deux passes ont LES MEMES POIDS, donc chaque souvenir est exact -- c'est
// le mieux que l'idee puisse rendre, et ce qu'une boucle de Newton ne peut qu'eroder.
//
// Quatre mesures par nuage, minimum de `--reps` :
//   sans memoire        le chemin ordinaire ( `cellule` ) ;
//   temoin              le chemin MEMO a vide : le prix du code de plus ;
//   avec memoire        les voisins d'hier proposes d'abord, le parcours en complement ;
//   les voisins seuls   pas de parcours du tout : le plancher, ce que coutent les coupes utiles.
// Et par cellule : plans proposes, boites testees, voisins.
//
// `--perime T` ( nuages a poids ) : les souvenirs pris a `T * W`, le diagramme mesure a `W` -- ce
// que Newton fait subir a la memoire, en pire ( `T = 0` : les souvenirs de Voronoi ).
//
//   xmake run memo --threads 8
//   xmake run memo --load uniforme -n 1000000 --3d
//   xmake run memo --perime 0.9
// =====================================================================================

#include "bench/Dispatch.h"
#include <cmath>
#include <cstdio>
#include <string>

using namespace sf;

namespace {

double perime = -1;                                      ///< < 0 : souvenirs exacts

template<class PD>
int mesure( const Args &a, const Nuage<3> &nu ) {
    const SI n = nu.n;
    PD pd;
    pd.build( nu.P, nu.W, n, a.leaf );
    std::vector<TF> w_perime;
    if ( perime >= 0 && nu.W ) {                         // les souvenirs viennent d'AUTRES poids
        w_perime.resize( n );
        for ( SI i = 0; i < n; ++i ) w_perime[ i ] = TF( perime ) * nu.W[ i ];
        pd.set_weights( w_perime.data(), a.par );
    }
    std::vector<SI> rang( n );
    for ( SI k = 0; k < n; ++k ) rang[ pd.ids[ k ] ] = k;

    // ---- sans memoire ( aux vrais poids )
    std::vector<TF> res;
    if ( ! w_perime.empty() ) pd.set_weights( nu.W, a.par );
    pd.measures( res, a.par );
    double t_sans = 1e300;
    for ( int r = 0; r < a.reps; ++r ) { const double t0 = now(); pd.measures( res, a.par ); t_sans = std::min( t_sans, now() - t0 ); }
    TF somme_sans = 0;
    for ( TF v : res ) somme_sans += v;

    // ---- les souvenirs : les voisins de chaque cellule, en rangs, CSR ( aux poids perimes s'il y a lieu )
    if ( ! w_perime.empty() ) pd.set_weights( w_perime.data(), a.par );
    const int nth = std::max( a.par.threads, 1 );
    std::vector<std::vector<d2::SI32>> vois( n );
    std::atomic<SI> nb_vois{ 0 };
    parallel_for( n, a.par, [ & ]( SI k, int ) {
        typename PD::Cell cel;
        pd.cellule( k, cel );
        d2::SI32 out[ PD::max_nv ];
        const int m = cel.voisins( out, PD::max_nv );
        for ( int q = 0; q < m; ++q ) if ( out[ q ] >= 0 ) vois[ k ].push_back( d2::SI32( rang[ out[ q ] ] ) );
        nb_vois += SI( vois[ k ].size() );
    } );
    std::vector<SI> beg( n + 1, 0 );
    for ( SI k = 0; k < n; ++k ) beg[ k + 1 ] = beg[ k ] + SI( vois[ k ].size() );
    std::vector<d2::SI32> pre( beg[ n ] );
    for ( SI k = 0; k < n; ++k ) std::copy( vois[ k ].begin(), vois[ k ].end(), pre.begin() + beg[ k ] );
    if ( ! w_perime.empty() ) pd.set_weights( nu.W, a.par );

    // ---- une passe MEMO parametree : `avec` la memoire ou a vide, `parcours` ou non
    std::vector<std::vector<unsigned char>> saute( nth, std::vector<unsigned char>( n, 0 ) );
    std::vector<SI> prop_th( nth ), boites_th( nth ), coupees_th( nth );
    std::atomic<SI> deborde{ 0 };
    auto passe = [ & ]( bool avec, bool parcours ) {
        res.assign( n, TF( 0 ) );
        for ( int t = 0; t < nth; ++t ) { prop_th[ t ] = 0; boites_th[ t ] = 0; coupees_th[ t ] = 0; }
        deborde = 0;
        parallel_for( n, a.par, [ & ]( SI k, int t ) {
            typename PD::Cell cel;
            const d2::SI32 *p = pre.data() + beg[ k ];
            const int np = avec ? int( beg[ k + 1 ] - beg[ k ] ) : 0;
            unsigned char *sa = saute[ t ].data();
            for ( int q = 0; q < np; ++q ) sa[ p[ q ] ] = 1;
            int prop = 0, boites = 0, coupees = 0;
            if ( ! pd.cellule_memo( k, cel, p, np, avec ? sa : nullptr, prop, boites, coupees, parcours ) ) ++deborde;
            else res[ pd.ids[ k ] ] = PD::mesure( cel );
            for ( int q = 0; q < np; ++q ) sa[ p[ q ] ] = 0;
            prop_th[ t ] += prop; boites_th[ t ] += boites; coupees_th[ t ] += coupees;
        } );
        TF s = 0;
        for ( TF v : res ) s += v;
        SI prop = 0, boites = 0, coupees = 0;
        for ( int t = 0; t < nth; ++t ) { prop += prop_th[ t ]; boites += boites_th[ t ]; coupees += coupees_th[ t ]; }
        return std::tuple{ s, prop, boites, coupees };
    };
    auto chrono = [ & ]( bool avec, bool parcours ) {
        passe( avec, parcours );                         // chauffe
        double t = 1e300;
        for ( int r = 0; r < a.reps; ++r ) { const double t0 = now(); passe( avec, parcours ); t = std::min( t, now() - t0 ); }
        return t;
    };
    const double t_temoin = chrono( false, true );
    const auto [ s_temoin, prop_temoin, boites_temoin, coupees_temoin ] = passe( false, true );
    const double t_avec = chrono( true, true );
    const auto [ s_avec, prop_avec, boites_avec, coupees_avec ] = passe( true, true );
    const double t_seuls = chrono( true, false );
    const auto [ s_seuls, prop_seuls, boites_seuls, coupees_seuls ] = passe( true, false );

    std::printf( "  %-24s n=%-7d %-8s  %.2f voisins par cellule%s\n", nu.nom.c_str(), int( n ), nu.W ? "Laguerre" : "Voronoi", double( nb_vois.load() ) / n,
                 w_perime.empty() ? "" : ( "  ( souvenirs pris a " + std::to_string( perime ) + " x W )" ).c_str() );
    std::printf( "     %-22s %8.3f s   somme %.9f\n", "sans memoire", t_sans, double( somme_sans ) );
    std::printf( "     %-22s %8.3f s   somme %.9f   %6.2f plans proposes, %6.2f boites testees, %6.2f coupes effectives par cellule\n", "temoin ( MEMO a vide )", t_temoin, double( s_temoin ), double( prop_temoin ) / n, double( boites_temoin ) / n, double( coupees_temoin ) / n );
    std::printf( "     %-22s %8.3f s   somme %.9f   %6.2f plans proposes, %6.2f boites testees, %6.2f coupes effectives par cellule   ( %+.1f %% / sans, %+.1f %% / temoin )\n", "avec memoire", t_avec, double( s_avec ), double( prop_avec ) / n, double( boites_avec ) / n, double( coupees_avec ) / n, 100 * ( t_avec / t_sans - 1 ), 100 * ( t_avec / t_temoin - 1 ) );
    std::printf( "     %-22s %8.3f s   somme %.9f   %6.2f plans proposes,                       %6.2f coupes effectives par cellule   ( %+.1f %% / sans : le plancher )\n", "les voisins seuls", t_seuls, double( s_seuls ), double( prop_seuls ) / n, double( coupees_seuls ) / n, 100 * ( t_seuls / t_sans - 1 ) );
    const bool ok = std::fabs( s_avec - somme_sans ) < 1e-9 && ( ! w_perime.empty() || std::fabs( s_seuls - somme_sans ) < 1e-9 ) && deborde == 0;
    if ( ! ok ) std::printf( "     <-- FAUX ( ecart %.2e / %.2e, %d debordements )\n", double( s_avec - somme_sans ), double( s_seuls - somme_sans ), int( deborde.load() ) );
    return ! ok;
}

} // namespace

int main( int argc, char **argv ) {
    Args a;
    a.n = 100000;
    a.dims = 3;
    for ( int i = 1; i < argc; ++i ) {
        const std::string s = argv[ i ];
        if ( a.parse( s, i, argc, argv ) ) continue;
        if ( s == "--perime" && i + 1 < argc ) { perime = std::atof( argv[ ++i ] ); continue; }
        std::printf( "usage: memo [options]   ( 3D seulement )\n" );
        Args::usage();
        std::printf( "  --perime T      les souvenirs pris a T * W, le diagramme a W ( nuages a poids )\n" );
        return s == "--help" || s == "-h" ? 0 : 1;
    }
    a.finalise();
    std::printf( "=== 3D  threads=%d kernel=%s maxnv=%d leaf=%d reps=%d\n", a.par.threads, a.kernel.c_str(), a.nv( 3 ), int( a.leaf ), a.reps );
    int bad = 0;
    for ( const Nuage<3> &nu : a.nuages<3>() ) {
        if ( nu.absent ) { std::printf( "  %-28s : ABSENT ( --cases DIR )\n", nu.nom.c_str() ); continue; }
        bad += dispatch<3>( a, [ & ]( auto tag ) { return mesure<typename decltype( tag )::type>( a, nu ); } );
    }
    return bad ? 1 : 0;
}

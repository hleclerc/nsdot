// =====================================================================================
// LES DENSITES HETEROGENES : un plancher plus une somme de gaussiennes isotropes sur le carre
// ( `Densite.h` ), des germes uniformes, et Newton qui doit emmener les cellules dans les pics.
//
//   xmake run densite --sigma 0.05                       Newton direct, depuis Voronoi
//   xmake run densite --sigma 0.02 --conv 0.5            la continuation : la densite convolee par
//                                                        une gaussienne de largeur 0.5, resolue,
//                                                        puis 0.25, ... jusqu'a 0
//   xmake run densite --sigma 0.02 --conv 0.5 --predire s2
//                                                        et les poids de depart extrapoles par
//                                                        `dw / ds` ( en `s^2` ) a chaque etape
//   xmake run densite --check                            la masse par circulation contre une
//                                                        quadrature de surface, la derivee contre
//                                                        des differences finies
//
// 2D seulement.
// =====================================================================================

#include "bench/Dispatch.h"
#include "bench/Trames.h"
#include "solver/Densite.h"
#include "solver/Lineaire.h"
#include "solver/Newton.h"
#ifdef _OPENMP
#include <omp.h>
#endif
#include <cstdio>
#include <functional>
#include <random>
#include <sstream>
#include <string>

using namespace sf;

namespace {

struct Opts {
    NewtonOptions newton;
    std::string   solver = "chol";
    int           amgvar = Amg::RS_GS;
    TF            sigma = 0.05;       ///< l'echelle des largeurs ( multiplie celles du jeu )
    int           nb_gauss = 4;       ///< le jeu par defaut : 3 ou 4 gaussiennes
    std::string   gauss;              ///< "cx,cy,sigma,masse;..." a la place du jeu
    TF            plancher = 0;       ///< la fraction de la masse dans le plancher uniforme
    std::string   diracs = "uniforme";///< uniforme | rho
    std::vector<TF> conv;             ///< les largeurs de convolution, decroissantes, 0 en dernier
    TF            conv0 = 0, conv_ratio = 2, conv_min = 0;
    std::string   predire = "non";    ///< non | s | s2 : extrapoler w par `dw / ds`
    bool          check = false;
    std::string   ecrire, dump;
};

/// le jeu par defaut : des centres pas trop symetriques, des masses et des largeurs inegales
Densite densite_de( const Opts &o ) {
    Densite rho;
    if ( ! o.gauss.empty() ) {
        std::stringstream ss( o.gauss );
        std::string item;
        while ( std::getline( ss, item, ';' ) ) {
            Gaussienne g;
            if ( std::sscanf( item.c_str(), "%lf,%lf,%lf,%lf", &g.cx, &g.cy, &g.sigma, &g.masse ) == 4 )
                rho.g.push_back( g );
        }
    } else if ( o.nb_gauss <= 3 ) {
        rho.g = { { 0.30, 0.30, 1.0 * o.sigma, 0.4 }, { 0.72, 0.38, 0.7 * o.sigma, 0.3 }, { 0.42, 0.76, 1.3 * o.sigma, 0.3 } };
    } else {
        rho.g = { { 0.26, 0.30, 1.0 * o.sigma, 0.35 }, { 0.72, 0.26, 0.7 * o.sigma, 0.25 },
                  { 0.34, 0.74, 1.3 * o.sigma, 0.25 }, { 0.76, 0.70, 1.0 * o.sigma, 0.15 } };
    }
    TF mt = 0;
    for ( const Gaussienne &g : rho.g ) mt += g.masse;
    for ( Gaussienne &g : rho.g ) g.masse *= ( 1 - o.plancher ) / mt;
    rho.plancher = o.plancher;
    return rho;
}

/// la masse EXACTE de la densite sur le carre ( la convolution comprise )
TF masse_carre( const Densite &rho ) {
    TF m = rho.plancher;
    for ( SI k = 0; k < SI( rho.g.size() ); ++k ) {
        const TF sig = rho.sigma_eff( k ) * M_SQRT2;
        const Gaussienne &g = rho.g[ k ];
        m += g.masse * TF( 0.25 ) * ( std::erf( ( 1 - g.cx ) / sig ) + std::erf( g.cx / sig ) )
                                  * ( std::erf( ( 1 - g.cy ) / sig ) + std::erf( g.cy / sig ) );
    }
    return m;
}

/// des germes tires SELON `rho` ( rejet )
Nuage<2> nuage_selon( const Densite &rho, SI n, unsigned graine ) {
    Nuage<2> nu;
    nu.nom = "selon rho n=" + std::to_string( n );
    std::mt19937_64 rng( graine + 17 );
    std::uniform_real_distribution<TF> uni( 0.001, 0.999 ), u01( 0, 1 );
    const TF rmax = rho.max_rho();
    while ( SI( nu.c[ 0 ].size() ) < n ) {
        const TF x = uni( rng ), y = uni( rng );
        if ( u01( rng ) * rmax <= rho.rho( x, y ) ) { nu.c[ 0 ].push_back( x ); nu.c[ 1 ].push_back( y ); }
    }
    nu.w.assign( n, TF( 0 ) );
    nu.finish();
    return nu;
}

/// LA VERIFICATION : la circulation contre une quadrature de surface ( l'eventail depuis le
/// centroide, subdivise jusqu'a `h < sigma / 8`, Gauss a 7 points par triangle ), et la derivee
/// contre des differences finies. Quelques cellules, les plus proches des centres et une au hasard.
template<class PD>
void verifie( const PD &pd, const Nuage<2> &nu, Densite rho, const Parallel &par ) {
    const SI n = nu.n;
    std::vector<SI> kk;
    for ( const Gaussienne &g : rho.g ) {               // le germe le plus proche de chaque centre
        SI best = 0; TF db = 1e30;
        for ( SI k = 0; k < n; ++k ) {
            const TF dx = pd.c[ 0 ][ k ] - g.cx, dy = pd.c[ 1 ][ k ] - g.cy, d2 = dx * dx + dy * dy;
            if ( d2 < db ) { db = d2; best = k; }
        }
        kk.push_back( best );
    }
    kk.push_back( n / 3 );
    kk.push_back( 2 * n / 3 );
    const TF gx[ 7 ] = { 1. / 3, 0.0597158717, 0.4701420641, 0.4701420641, 0.7974269853, 0.1012865073, 0.1012865073 };
    const TF gy[ 7 ] = { 1. / 3, 0.4701420641, 0.0597158717, 0.4701420641, 0.1012865073, 0.7974269853, 0.1012865073 };
    const TF gw[ 7 ] = { 0.225, 0.1323941527, 0.1323941527, 0.1323941527, 0.1259391805, 0.1259391805, 0.1259391805 };
    TF smin = 1e30;
    for ( SI k = 0; k < SI( rho.g.size() ); ++k ) smin = std::min( smin, rho.sigma_eff( k ) );
    for ( SI k : kk ) {
        typename PD::Cell cel;
        pd.cellule( k, cel );
        TF dds = 0;
        const TF m = rho.mesure( cel, []( int, TF ) {}, &dds );
        // la quadrature de surface
        TF cx = 0, cy = 0;
        for ( int i = 0; i < cel.nb; ++i ) { cx += cel.vx[ i ]; cy += cel.vy[ i ]; }
        cx /= cel.nb; cy /= cel.nb;
        TF q = 0;
        SI nt = 0;
        std::function<void( TF, TF, TF, TF, TF, TF )> tri = [ & ]( TF ax, TF ay, TF bx, TF by, TF ox, TF oy ) {
            const TF h = std::max( { std::hypot( bx - ax, by - ay ), std::hypot( ox - ax, oy - ay ), std::hypot( ox - bx, oy - by ) } );
            if ( h > smin / 8 ) {
                const TF mx = ( ax + bx ) / 2, my = ( ay + by ) / 2, px = ( bx + ox ) / 2, py = ( by + oy ) / 2, qx = ( ox + ax ) / 2, qy = ( oy + ay ) / 2;
                tri( ax, ay, mx, my, qx, qy ); tri( mx, my, bx, by, px, py ); tri( qx, qy, px, py, ox, oy ); tri( mx, my, px, py, qx, qy );
                return;
            }
            ++nt;
            const TF aire = std::fabs( ( bx - ax ) * ( oy - ay ) - ( ox - ax ) * ( by - ay ) ) / 2;
            for ( int j = 0; j < 7; ++j ) {
                const TF x = ax + gx[ j ] * ( bx - ax ) + gy[ j ] * ( ox - ax ), y = ay + gx[ j ] * ( by - ay ) + gy[ j ] * ( oy - ay );
                q += aire * gw[ j ] * rho.rho( x, y );
            }
        };
        for ( int i = 0, j = cel.nb - 1; i < cel.nb; j = i++ )
            tri( cel.vx[ j ], cel.vy[ j ], cel.vx[ i ], cel.vy[ i ], cx, cy );
        // la derivee par differences finies
        const TF s0 = rho.s, ds = std::max( TF( 1e-4 ) * std::max( s0, smin ), TF( 1e-6 ) );
        Densite rp = rho, rm = rho;
        rp.s = s0 + ds; rm.s = std::max( TF( 0 ), s0 - ds );
        const TF df = ( rp.mesure( cel, []( int, TF ) {} ) - rm.mesure( cel, []( int, TF ) {} ) ) / ( rp.s - rm.s );
        std::printf( "  germe %6d ( %.3f, %.3f ) %2d sommets : masse %.10e  surface %.10e  ecart %.2e  |  d/ds %.6e  dif. finies %.6e  ecart %.2e  ( %d triangles )\n",
                     int( pd.ids[ k ] ), pd.c[ 0 ][ k ], pd.c[ 1 ][ k ], cel.nb, m, q, std::fabs( m - q ) / std::max( q, TF( 1e-300 ) ),
                     dds, df, std::fabs( dds - df ) / std::max( std::fabs( df ), TF( 1e-300 ) ), int( nt ) );
    }
    // la somme des masses contre la masse exacte du carre
    std::vector<TF> a;
    pd.measures_and_facets_avec( a, par, []( int, SI, SI, TF ) {}, [ & ]( const typename PD::Cell &cel, auto &&fac, SI ) { return rho.mesure( cel, fac ); } );
    TF sa = 0, amin = 1e30;
    for ( TF v : a ) { sa += v; amin = std::min( amin, v ); }
    std::printf( "  somme des masses %.12f  exacte %.12f  ecart %.2e  ( plus petite %.2e, moyenne %.2e )\n",
                 sa, masse_carre( rho ), std::fabs( sa - masse_carre( rho ) ), amin, sa / n );
}

template<class PD, class Lin>
int lance( const Args &a, const Opts &o, const Nuage<2> &nu, Lin &lin ) {
    const SI n = nu.n;
    Densite rho = densite_de( o );
    std::printf( "  densite : %d gaussiennes", int( rho.g.size() ) );
    for ( const Gaussienne &g : rho.g ) std::printf( "  ( %.2f, %.2f ; sigma %.4f, masse %.3f )", g.cx, g.cy, g.sigma, g.masse );
    std::printf( "  plancher %.3f  ;  germes %s, n = %d\n", rho.plancher, o.diracs.c_str(), int( n ) );

    double t0 = now();
    PD pd;
    pd.build( nu.P, nullptr, n, a.leaf );
    const double t_arbre = now() - t0;
#ifdef _OPENMP
    omp_set_num_threads( a.par.threads );
#endif
    if ( o.check ) {
        std::printf( "-- verification, s = 0\n" );
        verifie( pd, nu, rho, a.par );
        for ( TF s : { TF( 0.02 ), TF( 0.3 ) } ) {
            std::printf( "-- verification, s = %g\n", s );
            Densite r2 = rho; r2.s = s;
            verifie( pd, nu, r2, a.par );
        }
        return 0;
    }

    Newton<PD,Lin> nw( pd, lin, nu.P, a.par, o.newton );
    nw.rho = &rho;
    nw.derivee = o.predire != "non";
    Trames trames;                                       // une trame par etape, le diagramme converge
    const bool dump = ! o.dump.empty() && trames.ouvre( o.dump );

    std::vector<TF> w( n, TF( 0 ) ), dwds, dw, a_p, da_p, w_try, a_try, da_try;
    std::vector<Facette> fa_p, fa_try;
    const double debut = now();
    int tot_it = 0, tot_diag = 0, tot_recul = 0;
    bool ok = true;
    std::vector<std::string> lignes;
    auto norme_res = [ & ]( const std::vector<TF> &a ) { TF r = 0; for ( SI i = 0; i < n; ++i ) r += ( a[ i ] - nw.nu[ i ] ) * ( a[ i ] - nw.nu[ i ] ); return std::sqrt( r ); };
    for ( SI e = 0; e < SI( o.conv.size() ); ++e ) {
        const TF s = o.conv[ e ];
        rho.s = s;
        const TF M = masse_carre( rho );
        nw.nu.assign( n, M / n );
        nw.st = NewtonStats{};
        lin.st = StatsLin{};
        std::printf( "-- s = %g : masse sur le carre %.6f, rho max %.4g, %s\n", s, M, rho.max_rho(),
                     e == 0 ? "depuis Voronoi" : o.predire == "non" ? "depuis les poids precedents" : "depuis les poids extrapoles" );
        const double te = now();
        // L'EXTRAPOLATION, GARDEE : `w + theta dw`, theta = 1, 1/2, ... tant qu'une cellule passe sous
        // le plancher ( la moitie de la plus petite masse du depart sans extrapolation, comme
        // l'amortissement ) ou que le residu l2 n'est pas meilleur ; a defaut, le depart sans.
        bool mesure = false;
        int nb_essais = 0;
        TF theta = 0;
        if ( ! dw.empty() ) {
            std::vector<TF> *pda = nw.derivee ? &da_p : nullptr, *pdt = nw.derivee ? &da_try : nullptr;
            nw.mesures_et_facettes( w, a_p, fa_p, pda );
            TF plancher = nw.nu[ 0 ], r_p = norme_res( a_p );
            for ( SI i = 0; i < n; ++i ) plancher = std::min( plancher, a_p[ i ] );
            plancher *= TF( 0.5 );
            w_try.resize( n );
            for ( theta = 1; theta >= TF( 1. / 16 ); theta /= 2 ) {
                ++nb_essais;
                for ( SI i = 0; i < n; ++i ) w_try[ i ] = w[ i ] + theta * dw[ i ];
                nw.mesures_et_facettes( w_try, a_try, fa_try, pdt );
                TF amin = a_try[ 0 ];
                for ( SI i = 0; i < n; ++i ) amin = std::min( amin, a_try[ i ] );
                const TF r_t = norme_res( a_try );
                if ( amin >= plancher && r_t < r_p ) break;
                std::printf( "   theta %.4f refuse : plus petite masse %.2e ( plancher %.2e ), |r|_2 %.3e ( sans : %.3e )\n",
                             theta, amin, plancher, r_t, r_p );
            }
            if ( theta >= TF( 1. / 16 ) ) {
                w.swap( w_try ); nw.a.swap( a_try ); nw.fa.swap( fa_try ); if ( nw.derivee ) nw.da.swap( da_try );
                std::printf( "   extrapolation retenue : theta %.4f ( %d essais ), |r|_2 %.3e contre %.3e sans\n", theta, nb_essais, norme_res( nw.a ), r_p );
            } else {
                theta = 0;
                nw.a.swap( a_p ); nw.fa.swap( fa_p ); if ( nw.derivee ) nw.da.swap( da_p );
                std::printf( "   extrapolation REFUSEE ( %d essais ) : depart sans\n", nb_essais );
            }
            mesure = true;
        }
        const bool fini = nw.resout( w, mesure );
        const double dt = now() - te;
        const NewtonStats &st = nw.st;
        ok = ok && fini;
        tot_it += st.nb_iter; tot_diag += st.nb_diag; tot_recul += st.nb_recul;
        char buf[ 512 ];
        std::snprintf( buf, sizeof( buf ), "| %-8g | %.2f (%d) | %.2e | %d | %d (%d) | %.2e | %.2f s | %s |",
                       s, theta, nb_essais, double( st.reste0 ), st.nb_iter, st.nb_diag, st.nb_recul, double( st.reste ), dt, st.fin );
        lignes.push_back( buf );
        std::printf( "   newton %s : depart %.2e, %d iterations, %d diagrammes ( %d reculs ), reste %.2e, %.2f s"
                     " [ diag %.2f  asm %.2f  lin %.2f ]%s\n",
                     st.fin, double( st.reste0 ), st.nb_iter, st.nb_diag, st.nb_recul, double( st.reste ), dt,
                     st.t_diag, st.t_asm, st.t_lin, st.nb_deborde ? ( "  DEBORDEMENTS : " + std::to_string( st.nb_deborde ) + " cellules ( --maxnv )" ).c_str() : "" );
        w = nw.w;
        dw.clear();
        if ( dump ) {
            pd.set_weights( w.data(), a.par );
            trames.ecrit( pd, nw.nu, a.par, "densite", double( s ), st.nb_iter, double( st.reste ), st.nb_recul );
        }
        if ( ! fini && st.fin != std::string( "STAGNATION" ) ) break;

        // l'extrapolation vers la largeur suivante : `L dw/ds = dnu/ds - da/ds`
        if ( e + 1 < SI( o.conv.size() ) && o.predire != "non" ) {
            const TF s2 = o.conv[ e + 1 ];
            Laplacien L;
            L.assemble( n, nw.fa );
            std::vector<TF> b( n );
            TF sda = 0;
            for ( SI i = 0; i < n; ++i ) sda += nw.da[ i ];
            for ( SI i = 0; i < n; ++i ) b[ i ] = sda / n - nw.da[ i ];
            if ( ! lin.resout( L, b, dwds ) ) { std::printf( "   extrapolation : solveur lineaire en echec\n" ); continue; }
            // en `s` : w += ( s2 - s ) dw/ds ; en `s^2` : dw/d(s^2) = dw/ds / ( 2 s ), w += ( s2^2 - s^2 ) dw/d(s^2)
            const TF coef = o.predire == "s2" ? ( s2 * s2 - s * s ) / ( 2 * s ) : ( s2 - s );
            TF amp = 0, ampw = 0;
            dw.resize( n );
            for ( SI i = 0; i < n; ++i ) { dw[ i ] = coef * dwds[ i ]; amp = std::max( amp, std::fabs( dw[ i ] ) ); ampw = std::max( ampw, std::fabs( w[ i ] ) ); }
            std::printf( "   extrapolation vers s = %g ( en %s ) : |dw|max %.3e sur des poids d'amplitude %.3e\n", s2, o.predire.c_str(), amp, ampw );
        }
    }
    const double total = now() - debut + t_arbre;
    std::printf( "  TOTAL : %d iterations, %d diagrammes ( %d reculs ), %.2f s ( arbre %.3f )  --  %s\n",
                 tot_it, tot_diag, tot_recul, total, t_arbre, ok ? "converge" : "PAS CONVERGE" );
    std::printf( "  | s | theta (essais) | depart | it | diag (reculs) | reste | temps | fin |\n  |---|---|---|---|---|---|---|---|\n" );
    for ( const std::string &l : lignes ) std::printf( "  %s\n", l.c_str() );

    if ( ! o.ecrire.empty() ) {
        char entete[ 256 ];
        std::snprintf( entete, sizeof( entete ), "# densite : %d gaussiennes, sigma %g, poids obtenus par newton ( %s )\n", int( rho.g.size() ), o.sigma, nw.st.fin );
        if ( ecrit_nuage<2>( o.ecrire, nu, w.data(), entete ) ) std::printf( "  poids ecrits dans '%s'\n", o.ecrire.c_str() );
    }
    return ok ? 0 : 1;
}

} // namespace

int main( int argc, char **argv ) {
    Args a;
    a.n = 100000;
    a.dims = 2;
    Opts o;
    for ( int i = 1; i < argc; ++i ) {
        const std::string s = argv[ i ];
        auto val = [ & ]() { return i + 1 < argc ? argv[ ++i ] : ""; };
        if ( a.parse( s, i, argc, argv ) ) continue;
        else if ( s == "--sigma" )      o.sigma = std::atof( val() );
        else if ( s == "--nb-gauss" )   o.nb_gauss = std::atoi( val() );
        else if ( s == "--gauss" )      o.gauss = val();
        else if ( s == "--plancher" )   o.plancher = std::atof( val() );
        else if ( s == "--diracs" )     o.diracs = val();
        else if ( s == "--conv" )       o.conv0 = std::atof( val() );
        else if ( s == "--conv-ratio" ) o.conv_ratio = std::atof( val() );
        else if ( s == "--conv-min" )   o.conv_min = std::atof( val() );
        else if ( s == "--conv-liste" ) {
            std::stringstream ss( val() ); std::string it;
            while ( std::getline( ss, it, ',' ) ) o.conv.push_back( std::atof( it.c_str() ) );
        }
        else if ( s == "--predire" )    o.predire = val();
        else if ( s == "--check" )      o.check = true;
        else if ( s == "--solver" )     o.solver = val();
        else if ( s == "--amg-var" )    o.amgvar = std::atoi( val() );
        else if ( s == "--newton-tol" ) o.newton.tol = std::atof( val() );
        else if ( s == "--newton-max" ) o.newton.maxit = std::atoi( val() );
        else if ( s == "--t-min" )      o.newton.t_min = std::atof( val() );
        else if ( s == "--quiet" )      o.newton.trace = false;
        else if ( s == "--ecrire" )     o.ecrire = val();
        else if ( s == "--dump" )       o.dump = val();
        else {
            std::printf( "usage: densite [options]\n" );
            Args::usage();
            std::printf(
                "  --sigma S       l'echelle des largeurs du jeu de gaussiennes         (0.05)\n"
                "  --nb-gauss N    le jeu : 3 ou 4 gaussiennes                            (4)\n"
                "  --gauss SPEC    \"cx,cy,sigma,masse;...\" a la place du jeu\n"
                "  --plancher F    la fraction de la masse dans un plancher uniforme    (0)\n"
                "  --diracs D      uniforme | rho ( germes tires selon la densite )    (uniforme)\n"
                "  --conv S0       la continuation : largeurs S0, S0/R, ... >= Smin, puis 0   (0 : direct)\n"
                "  --conv-ratio R                                                           (2)\n"
                "  --conv-min S                                                             (0 : jusqu'a sigma/4)\n"
                "  --conv-liste L  les largeurs explicites, \"0.5,0.2,0.1,0\"\n"
                "  --predire P     non | s | s2 : extrapoler les poids par dw/ds vers l'etape suivante  (non)\n"
                "  --check         verifier la mesure ( circulation contre surface, derivee contre differences finies )\n"
                "  --solver S      chol ( Eigen, defaut ) | amg\n"
                "  --amg-var V     0 = agregation+spai0 | 1 = agregation+GS | 2 = Ruge-Stuben+GS  (2)\n"
                "  --newton-tol T  arret sur max|a_i - nu| / nu             (1e-6)\n"
                "  --newton-max K  iterations au maximum                    (100)\n"
                "  --t-min T       sous ce pas, STAGNATION                  (1e-10)\n"
                "  --quiet         pas de trace par iteration\n"
                "  --ecrire FILE   ecrire les poids trouves au format de cases/\n"
                "  --dump FILE     les trames ( JSONL ) : le diagramme converge de chaque etape\n" );
            return s == "--help" || s == "-h" ? 0 : 1;
        }
    }
    a.finalise();
    if ( o.conv.empty() ) {
        if ( o.conv0 > 0 ) {
            const TF smin = o.conv_min > 0 ? o.conv_min : o.sigma / 4;
            for ( TF s = o.conv0; s >= smin * ( 1 - 1e-12 ); s /= o.conv_ratio ) o.conv.push_back( s );
        }
        o.conv.push_back( 0 );
    }

    const Opts &oc = o;
    const Nuage<2> nu = o.diracs == "rho" ? nuage_selon( densite_de( o ), a.n, a.graine ) : nuage_uniforme<2>( a.n, a.graine, 0 );
    return dispatch<2>( a, [ & ]( auto tag ) {
        using PD = typename decltype( tag )::type;
#ifdef SF_EIGEN
        if ( oc.solver == "chol" ) {
            Cholesky lin;
            return lance<PD>( a, oc, nu, lin );
        }
#endif
#ifdef SF_AMGCL
        Amg lin;
        lin.variante = oc.amgvar;
        return lance<PD>( a, oc, nu, lin );
#else
        std::printf( "  AMGCL absent : --solver chol\n" );
        return 1;
#endif
    } );
}

// =====================================================================================
// LES DENSITES HETEROGENES : un plancher plus une somme de gaussiennes isotropes sur le carre
// ( `Densite.h` ), des germes uniformes, et Newton qui doit emmener les cellules dans les pics.
//
//   xmake run densite --sigma 0.05                       Newton direct, depuis Voronoi
//   xmake run densite --sigma 0.02 --conv 0.5            la continuation : la densite convolee par
//                                                        une gaussienne de largeur 0.5, resolue,
//                                                        puis 0.25, ... jusqu'a 0
//   xmake run densite --sigma 0.02 --conv 0.5 --ordre 2  et les poids de depart extrapoles a
//                                                        l'ordre 2 en `s` a chaque etape
//   xmake run densite --sigma 0.02 --melange 0.5         l'autre chemin : `( 1 - t ) + t rho`, le
//                                                        plancher 0.5, 0.25, ... 1e-3, puis 0
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
#include "solver/PremierOrdre.h"
#ifdef _OPENMP
#include <omp.h>
#endif
#include <atomic>
#include <cstdio>
#include <functional>
#include <limits>
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
    std::string   chemin = "conv";    ///< conv ( la largeur `s` decroit vers 0 ) | melange ( `t` croit vers 1 )
    std::vector<TF> liste;            ///< les valeurs du parametre, dans l'ordre ( la derniere : la densite elle-meme )
    TF            conv0 = 0, conv_ratio = 2, conv_min = 0;
    TF            mel0 = 0, mel_ratio = 2, mel_min = 1e-3;   ///< le melange : les PLANCHERS `1 - t`
    int           ordre = 0;          ///< l'extrapolation : 0 ( poids precedents ), 1 ( tangente ), 2
    std::string   variable = "s";     ///< s | s2 : la variable de l'extrapolation sur le chemin conv
    TF            fd = 0.25;          ///< ordre 2 : le pas des differences finies, en fraction du pas
    std::string   garde = "global";   ///< global ( theta ) | cellule ( les pincees relevees seules, puis theta )
    int           passes = 6;         ///< garde par cellule : passes de relevement au plus
    bool          check = false;
    std::string   ecrire, dump;
    std::string   methode = "newton"; ///< newton | lbfgs | cg, a chaque etape ( `PremierOrdre.h` )
    PremierOrdreOptions po;
    std::string   courbe;             ///< CSV : le residu apres chaque pas, contre les diagrammes et le temps
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
        Densite rp = rho, rm = rho;
        if ( rho.chemin == Densite::MELANGE ) { rp.t = rho.t + 1e-4; rm.t = rho.t - 1e-4; }
        else { const TF ds = std::max( TF( 1e-4 ) * std::max( rho.s, smin ), TF( 1e-6 ) ); rp.s = rho.s + ds; rm.s = std::max( TF( 0 ), rho.s - ds ); }
        const TF df = ( rp.mesure( cel, []( int, TF ) {} ) - rm.mesure( cel, []( int, TF ) {} ) ) / ( rho.chemin == Densite::MELANGE ? rp.t - rm.t : rp.s - rm.s );
        std::printf( "  germe %6d ( %.3f, %.3f ) %2d sommets : masse %.10e  surface %.10e  ecart %.2e  |  d/dlambda %.6e  dif. finies %.6e  ecart %.2e  ( %d triangles )\n",
                     int( pd.ids[ k ] ), pd.c[ 0 ][ k ], pd.c[ 1 ][ k ], cel.nb, m, q, std::fabs( m - q ) / std::max( q, TF( 1e-300 ) ),
                     dds, df, std::fabs( dds - df ) / std::max( std::fabs( df ), TF( 1e-300 ) ), int( nt ) );
    }
    // la somme des masses contre la masse exacte du carre
    std::vector<TF> a;
    pd.measures_and_facets_avec( a, par, []( int, SI, SI, TF ) {}, [ & ]( const typename PD::Cell &cel, auto &&fac, SI ) { return rho.mesure( cel, fac ); } );
    TF sa = 0, amin = 1e30;
    for ( TF v : a ) { sa += v; amin = std::min( amin, v ); }
    std::printf( "  somme des masses %.12f  exacte %.12f  ecart %.2e  ( plus petite %.2e, moyenne %.2e )\n",
                 sa, rho.masse_carre(), std::fabs( sa - rho.masse_carre() ), amin, sa / n );
}

/// la valeur du parametre du chemin dans la VARIABLE d'extrapolation, et retour
TF var_de( const Opts &o, TF lam ) { return o.variable == "s2" ? lam * lam : lam; }
TF lam_de( const Opts &o, TF v )   { return o.variable == "s2" ? std::sqrt( std::max( v, TF( 0 ) ) ) : v; }

template<class PD, class Lin>
int lance( const Args &a, const Opts &o, const Nuage<2> &nu, Lin &lin ) {
    const SI n = nu.n;
    Densite rho = densite_de( o );
    const bool melange = o.chemin == "melange";
    rho.chemin = melange ? Densite::MELANGE : Densite::CONV;
    std::printf( "  densite : %d gaussiennes", int( rho.g.size() ) );
    for ( const Gaussienne &g : rho.g ) std::printf( "  ( %.2f, %.2f ; sigma %.4f, masse %.3f )", g.cx, g.cy, g.sigma, g.masse );
    std::printf( "  plancher %.3f  ;  germes %s, n = %d  ;  chemin %s, %d etapes, extrapolation d'ordre %d en %s\n",
                 rho.plancher, o.diracs.c_str(), int( n ), o.chemin.c_str(), int( o.liste.size() ), o.ordre, melange ? "t" : o.variable.c_str() );

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
        std::printf( "-- verification, melange t = 0.3\n" );
        Densite r3 = rho; r3.t = 0.3; r3.chemin = Densite::MELANGE;
        verifie( pd, nu, r3, a.par );
        return 0;
    }

    // le parametre du chemin : `s` ( la convolution ) ou `t` ( le melange )
    auto regle = [ & ]( TF lam ) { if ( melange ) rho.t = lam; else rho.s = lam; };

    Newton<PD,Lin> nw( pd, lin, nu.P, a.par, o.newton );
    nw.rho = &rho;
    nw.derivee = o.ordre > 0;
    PremierOrdre<PD,Lin> po( nw, o.po );
    Trames trames;                                       // une trame par etape, le diagramme converge
    const bool dump = ! o.dump.empty() && trames.ouvre( o.dump );
    FILE *courbe = o.courbe.empty() ? nullptr : std::fopen( o.courbe.c_str(), "a" );
    const std::string etiquette = o.methode == "newton" ? ( o.newton.pas == NewtonOptions::ESSAI_LIMITES ? "newton essai-limites" : "newton" ) : o.po.nom();
    int diag_avant = 0;                                  // les diagrammes des etapes precedentes
    TF lam_courant = 0;
    const double t_courbe = now();
    if ( courbe )
        nw.o.apres_pas = [ & ]( int it, TF, int ) {
            TF p = 0;
            for ( SI i = 0; i < n; ++i ) p = std::max( p, std::fabs( nw.nu[ i ] - nw.a[ i ] ) / nw.nu[ i ] );
            std::fprintf( courbe, "%s;densite sigma %g s %g;2D;%d;%d;%.3f;%.6e;%.6e\n", etiquette.c_str(), o.sigma, lam_courant, it + 1, diag_avant + nw.st.nb_diag, now() - t_courbe, double( p ), double( nw.merite( nw.a ) ) );
        };

    std::vector<TF> w( n, TF( 0 ) ), w1, w2, dw, b, a_p, da_p, w_try, a_try, da_try, a_pl, a_mi;
    std::vector<SI> rang( n );                           // identifiant -> rang dans l'arbre
    for ( SI k = 0; k < n; ++k ) rang[ pd.ids[ k ] ] = k;
    const TF h2 = TF( 1 ) / n;                           // l'echelle des poids : h^2
    SI tot_releves = 0, tot_cel_lim = 0;
    int tot_tours = 0;
    double t_lim = 0;
    std::vector<Facette> fa_p, fa_try, fa_tmp;
    const double debut = now();
    int tot_it = 0, tot_diag = 0, tot_recul = 0, tot_extra = 0;
    bool ok = true;
    std::vector<std::string> lignes;
    auto norme_res = [ & ]( const std::vector<TF> &a ) { return nw.merite( a ); };   // le merite de Newton ( --residu )
    for ( SI e = 0; e < SI( o.liste.size() ); ++e ) {
        const TF lam = o.liste[ e ];
        lam_courant = lam;
        regle( lam );
        const TF M = rho.masse_carre();
        nw.nu.assign( n, M / n );
        nw.st = NewtonStats{};
        lin.st = StatsLin{};
        std::printf( "-- %s = %g : masse sur le carre %.6f, rho max %.4g, %s\n", melange ? "t" : "s", lam, M, rho.max_rho(),
                     e == 0 ? "depuis Voronoi" : dw.empty() ? "depuis les poids precedents" : "depuis les poids extrapoles" );
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
            SI releves = 0;
            for ( theta = 1; theta >= TF( 1. / 16 ); theta /= 2 ) {
                ++nb_essais;
                for ( SI i = 0; i < n; ++i ) w_try[ i ] = w[ i ] + theta * dw[ i ];
                nw.mesures_et_facettes( w_try, a_try, fa_try, pdt );
                // LA GARDE PAR CELLULE ( `--garde cellule` ) : les cellules que la tangente a pincees
                // sous le plancher sont RELEVEES seules -- bissection sur leur poids, les autres
                // poids et l'arbre inchanges -- jusqu'a leur masse cible ( entre nu/2 et 3nu/2 ),
                // toutes sur le meme diagramme ; puis on re-mesure, et on recommence tant qu'il en
                // reste ( les voisines peuvent l'etre a leur tour ), `passes` fois au plus. Le
                // theta global ne recule que si ca ne suffit pas.
                if ( o.garde == "cellule" ) {
                    for ( int passe = 0; passe < o.passes; ++passe ) {
                        std::vector<SI> pincees;
                        for ( SI i = 0; i < n; ++i ) if ( a_try[ i ] < plancher ) pincees.push_back( i );
                        if ( pincees.empty() ) break;
                        std::vector<TF> neuf( pincees.size() );
                        std::atomic<SI> nb_cel{ 0 };
                        parallel_for( SI( pincees.size() ), a.par, [ & ]( SI q, int ) {
                            const SI i = pincees[ q ], k = rang[ i ];
                            const TF cible = nw.nu[ i ];
                            typename PD::Cell cel;
                            auto masse = [ & ]( TF wk ) { ++nb_cel; return pd.cellule_avec_poids( k, wk, cel ) ? rho.mesure( cel, []( int, TF ) {} ) : std::numeric_limits<TF>::infinity(); };
                            TF lo = w_try[ i ], pas = std::max( std::fabs( dw[ i ] ), TF( 1e-3 ) * h2 ), hi = lo + pas, ah = masse( hi );
                            for ( int it = 0; it < 20 && ah < cible; ++it ) { pas *= 2; hi = lo + pas; ah = masse( hi ); }
                            for ( int it = 0; it < 40; ++it ) {
                                if ( ah <= TF( 1.5 ) * cible ) break;
                                const TF mid = ( lo + hi ) / 2, am = masse( mid );
                                if ( am >= TF( 0.5 ) * cible ) { hi = mid; ah = am; } else lo = mid;
                            }
                            neuf[ q ] = hi;
                        } );
                        for ( SI q = 0; q < SI( pincees.size() ); ++q ) w_try[ pincees[ q ] ] = neuf[ q ];
                        releves += SI( pincees.size() );
                        nw.mesures_et_facettes( w_try, a_try, fa_try, pdt );
                        std::printf( "   garde par cellule, passe %d : %d cellules pincees relevees ( %d cellules calculees ), |r|_2 %.3e\n",
                                     passe, int( pincees.size() ), int( nb_cel.load() ), norme_res( a_try ) );
                    }
                }
                TF amin = a_try[ 0 ];
                for ( SI i = 0; i < n; ++i ) amin = std::min( amin, a_try[ i ] );
                const TF r_t = norme_res( a_try );
                if ( amin >= plancher && r_t < r_p ) break;
                std::printf( "   theta %.4f refuse : plus petite masse %.2e ( plancher %.2e ), |r|_2 %.3e ( sans : %.3e )\n",
                             theta, amin, plancher, r_t, r_p );
            }
            tot_releves += releves;
            if ( theta >= TF( 1. / 16 ) ) {
                w.swap( w_try ); nw.a.swap( a_try ); nw.fa.swap( fa_try ); if ( nw.derivee ) nw.da.swap( da_try );
                std::printf( "   extrapolation retenue : theta %.4f ( %d essais ), |r|_2 %.3e contre %.3e sans\n", theta, nb_essais, norme_res( nw.a ), r_p );
            } else {
                theta = 0;
                nw.a.swap( a_p ); nw.fa.swap( fa_p ); if ( nw.derivee ) nw.da.swap( da_p );
                std::printf( "   extrapolation REFUSEE ( %d essais ) : depart sans\n", nb_essais );
            }
            tot_extra += nb_essais;                      // le depart nu remplace le diagramme de depart de Newton
            mesure = true;
        }
        po.st = PremierOrdreStats{};
        const bool fini = o.methode == "newton" ? nw.resout( w, mesure ) : po.resout( w, mesure );
        const double dt = now() - te;
        NewtonStats &st = nw.st;
        if ( o.methode != "newton" ) {                   // les compteurs du premier ordre rejoignent ceux de Newton
            st.nb_iter += po.st.nb_iter;
            st.reste0 = po.st.reste0;
            st.reste = po.st.reste;
            st.fin = po.st.fin;
            std::printf( "   %s : %d iterations, %d diagrammes ( %d refuses par le plancher, %d factorisations )%s\n",
                         o.methode == "cg" ? "gradient conjugue" : "L-BFGS", po.st.nb_iter, po.st.nb_diag, po.st.nb_plancher, po.st.nb_facto,
                         po.st.it_bascule >= 0 ? ( ", puis Newton depuis l'iteration " + std::to_string( po.st.it_bascule ) ).c_str() : "" );
        }
        ok = ok && fini;
        tot_it += st.nb_iter; tot_recul += st.nb_recul; tot_cel_lim += st.nb_cell_lim; tot_tours += st.nb_tours_essai; t_lim += st.t_lim;
        const int diag_newton = st.nb_diag;              // les essais d'extrapolation compris ( `nb_diag` compte tout )
        char buf[ 512 ];
        std::snprintf( buf, sizeof( buf ), "| %-8g | %.2f (%d) | %.2e | %d | %d (%d) | %.2e | %.2f s | %s |",
                       lam, theta, nb_essais, double( st.reste0 ), st.nb_iter, st.nb_diag, st.nb_recul, double( st.reste ), dt, st.fin );
        lignes.push_back( buf );
        std::printf( "   newton %s : depart %.2e, %d iterations, %d diagrammes ( %d reculs ), reste %.2e, %.2f s"
                     " [ diag %.2f  asm %.2f  lin %.2f ]%s\n",
                     st.fin, double( st.reste0 ), st.nb_iter, st.nb_diag, st.nb_recul, double( st.reste ), dt,
                     st.t_diag, st.t_asm, st.t_lin, st.nb_deborde ? ( "  DEBORDEMENTS : " + std::to_string( st.nb_deborde ) + " cellules ( --maxnv )" ).c_str() : "" );
        w = nw.w;
        dw.clear();
        tot_diag += diag_newton;
        diag_avant += st.nb_diag;
        if ( dump ) {
            pd.set_weights( w.data(), a.par );
            trames.ecrit( pd, nw.nu, a.par, "densite", double( lam ), st.nb_iter, double( st.reste ), st.nb_recul );
        }
        if ( ! fini && st.fin != std::string( "STAGNATION" ) ) break;

        // L'EXTRAPOLATION vers l'etape suivante, dans la variable `v` ( `s`, `s^2` ou `t` ) :
        //   ordre 1 : `L w' = nu' - da/dv` ;
        //   ordre 2 : `L w'' = nu'' - phi''`, `phi( eps ) = a( w + eps w', v + eps )` par differences
        //             finies le long de la tangente ( deux diagrammes ) -- toutes les derivees secondes
        //             de `a` dans la direction `( w', 1 )`, sans tenseur.
        if ( e + 1 < SI( o.liste.size() ) && o.ordre > 0 ) {
            const TF v = melange ? lam : var_de( o, lam ), v2 = melange ? o.liste[ e + 1 ] : var_de( o, o.liste[ e + 1 ] );
            const TF dv = v2 - v;
            const TF dl_dv = melange ? 1 : ( o.variable == "s2" ? 1 / ( 2 * lam ) : 1 );   // `d lam / d v`
            Laplacien L;
            L.assemble( n, nw.fa );
            b.resize( n );
            TF sda = 0;
            for ( SI i = 0; i < n; ++i ) sda += nw.da[ i ];
            for ( SI i = 0; i < n; ++i ) b[ i ] = ( sda / n - nw.da[ i ] ) * dl_dv;
            if ( ! lin.resout( L, b, w1 ) ) { std::printf( "   extrapolation : solveur lineaire en echec\n" ); continue; }
            dw.resize( n );
            for ( SI i = 0; i < n; ++i ) dw[ i ] = dv * w1[ i ];
            TF amp2 = 0;
            if ( o.ordre >= 2 ) {
                const TF h = o.fd * std::fabs( dv );
                auto phi = [ & ]( TF eps, std::vector<TF> &res ) {
                    regle( melange ? lam + eps : lam_de( o, v + eps ) );
                    w_try.resize( n );
                    for ( SI i = 0; i < n; ++i ) w_try[ i ] = w[ i ] + eps * w1[ i ];
                    nw.mesures_et_facettes( w_try, res, fa_tmp );
                };
                phi( +h, a_pl );
                phi( -h, a_mi );
                regle( lam );
                tot_extra += 2; tot_diag += 2;
                TF sb = 0;
                for ( SI i = 0; i < n; ++i ) { b[ i ] = -( a_pl[ i ] - 2 * nw.a[ i ] + a_mi[ i ] ) / ( h * h ); sb += b[ i ]; }
                for ( SI i = 0; i < n; ++i ) b[ i ] -= sb / n;           // `nu''` : la moyenne, la somme est nulle
                if ( ! lin.resout( L, b, w2 ) ) { std::printf( "   extrapolation : solveur lineaire en echec ( ordre 2 )\n" ); dw.clear(); continue; }
                for ( SI i = 0; i < n; ++i ) { const TF c = TF( 0.5 ) * dv * dv * w2[ i ]; amp2 = std::max( amp2, std::fabs( c ) ); dw[ i ] += c; }
            }
            TF amp = 0, ampw = 0;
            for ( SI i = 0; i < n; ++i ) { amp = std::max( amp, std::fabs( dw[ i ] ) ); ampw = std::max( ampw, std::fabs( w[ i ] ) ); }
            std::printf( "   extrapolation vers %g ( ordre %d ) : |dw|max %.3e%s sur des poids d'amplitude %.3e\n",
                         o.liste[ e + 1 ], o.ordre, amp, o.ordre >= 2 ? ( " ( terme d'ordre 2 : " + std::to_string( amp2 ) + " )" ).c_str() : "", ampw );
        }
    }
    const double total = now() - debut + t_arbre;
    std::printf( "  TOTAL : %d iterations, %d diagrammes dont %d pour l'extrapolation ( %d reculs, %d cellules relevees ), %.2f s ( arbre %.3f )  --  %s\n",
                 tot_it, tot_diag, tot_extra, tot_recul, int( tot_releves ), total, t_arbre, ok ? "converge" : "PAS CONVERGE" );
    if ( tot_tours )
        std::printf( "  limites en masse : %d essais corriges, %d cellules calculees, %.2f s\n", tot_tours, int( tot_cel_lim ), t_lim );
    if ( courbe ) std::fclose( courbe );
    std::printf( "  | %s | theta (essais) | depart | it | diag (reculs) | reste | temps | fin |\n  |---|---|---|---|---|---|---|---|\n", melange ? "t" : "s" );
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
        else if ( s == "--conv" )       { o.conv0 = std::atof( val() ); o.chemin = "conv"; }
        else if ( s == "--conv-ratio" ) o.conv_ratio = std::atof( val() );
        else if ( s == "--conv-min" )   o.conv_min = std::atof( val() );
        else if ( s == "--melange" )    { o.mel0 = std::atof( val() ); o.chemin = "melange"; }
        else if ( s == "--melange-ratio" ) o.mel_ratio = std::atof( val() );
        else if ( s == "--melange-min" ) o.mel_min = std::atof( val() );
        else if ( s == "--liste" ) {
            std::stringstream ss( val() ); std::string it;
            while ( std::getline( ss, it, ',' ) ) o.liste.push_back( std::atof( it.c_str() ) );
        }
        else if ( s == "--chemin" )     o.chemin = val();
        else if ( s == "--ordre" )      o.ordre = std::atoi( val() );
        else if ( s == "--variable" )   o.variable = val();
        else if ( s == "--fd" )         o.fd = std::atof( val() );
        else if ( s == "--garde" )      o.garde = val();
        else if ( s == "--residu" ) {
            const std::string v = val();
            o.newton.residu = v == "barriere" ? NewtonOptions::BARRIERE : v == "log" ? NewtonOptions::LOG : NewtonOptions::LIN;
        }
        else if ( s == "--passes" )     o.passes = std::atoi( val() );
        else if ( s == "--check" )      o.check = true;
        else if ( s == "--solver" )     o.solver = val();
        else if ( s == "--amg-var" )    o.amgvar = std::atoi( val() );
        else if ( s == "--newton-tol" ) o.newton.tol = std::atof( val() );
        else if ( s == "--newton-max" ) o.newton.maxit = std::atoi( val() );
        else if ( s == "--t-min" )      o.newton.t_min = std::atof( val() );
        else if ( s == "--pas" )        o.newton.pas = std::string( val() ) == "essai-limites" ? NewtonOptions::ESSAI_LIMITES : NewtonOptions::ESSAIS;
        else if ( s == "--beta0" )      o.newton.beta0 = std::atof( val() );
        else if ( s == "--mult-ok" )    o.newton.mult_ok = std::atof( val() );
        else if ( s == "--facteur" )    o.newton.facteur = std::atof( val() );
        else if ( s == "--lim-tol" )    o.newton.lim.tol = std::atof( val() );
        else if ( s == "--quiet" )      o.newton.trace = false;
        else if ( s == "--ecrire" )     o.ecrire = val();
        else if ( s == "--dump" )       o.dump = val();
        else if ( s == "--methode" )    o.methode = val();
        else if ( s == "--memoire" )    o.po.memoire = std::atoi( val() );
        else if ( s == "--c2" )         o.po.c2 = std::atof( val() );
        else if ( s == "--precond" )    o.po.precond = std::atoi( val() );
        else if ( s == "--refacto" )    o.po.refacto = std::atoi( val() );
        else if ( s == "--refacto-borne" ) o.po.refacto_borne = std::atof( val() );
        else if ( s == "--refacto-taux" ) o.po.refacto_taux = std::atof( val() );
        else if ( s == "--plancher-po" ) o.po.plancher = std::atoi( val() );
        else if ( s == "--max-ls" )     o.po.max_ls = std::atoi( val() );
        else if ( s == "--bascule" )    o.po.bascule = std::atof( val() );
        else if ( s == "--bascule-it" ) o.po.bascule_it = std::atoi( val() );
        else if ( s == "--courbe" )     o.courbe = val();
        else {
            std::printf( "usage: densite [options]\n" );
            Args::usage();
            std::printf(
                "  --sigma S       l'echelle des largeurs du jeu de gaussiennes         (0.05)\n"
                "  --nb-gauss N    le jeu : 3 ou 4 gaussiennes                            (4)\n"
                "  --gauss SPEC    \"cx,cy,sigma,masse;...\" a la place du jeu\n"
                "  --plancher F    la fraction de la masse dans un plancher uniforme    (0)\n"
                "  --diracs D      uniforme | rho ( germes tires selon la densite )    (uniforme)\n"
                "  --conv S0       la continuation en CONVOLUTION : largeurs S0, S0/R, ... >= Smin, puis 0   (0 : direct)\n"
                "  --conv-ratio R                                                           (2)\n"
                "  --conv-min S                                                             (0 : jusqu'a sigma/4)\n"
                "  --melange F0    la continuation en MELANGE ( 1 - t ) + t rho : planchers F0, F0/R, ... >= Fmin, puis 0\n"
                "  --melange-ratio R                                                        (2)\n"
                "  --melange-min F                                                          (1e-3)\n"
                "  --liste L       les valeurs explicites du parametre ( s, ou t avec --chemin melange ), \"0.5,0.2,0.1,0\"\n"
                "  --ordre K       l'extrapolation vers l'etape suivante : 0 | 1 ( tangente ) | 2 ( + derivee seconde, 2 diagrammes )  (0)\n"
                "  --variable V    s | s2 : la variable de l'extrapolation ( chemin conv )     (s)\n"
                "  --fd F          ordre 2 : le pas des differences finies, en fraction du pas   (0.25)\n"
                "  --pas P         essais ( KMT, defaut ) | essai-limites ( l'essai, puis les limites EN MASSE des cellules sous eps, par bissection )\n"
                "  --beta0 B       essai-limites : le premier essai                          (0.25)\n"
                "  --mult-ok M     essai-limites : apres un essai passe direct, beta *= M     (2)\n"
                "  --facteur F     essai-limites : t = F * limite                            (0.9)\n"
                "  --lim-tol T     precision relative des limites                            (1e-2)\n"
                "  --residu R      lin ( a - nu ) | barriere ( x - 1/x, x = a/nu ) | log : le residu de Newton et le merite  (lin)\n"
                "  --garde G       global ( theta = 1, 1/2, ... ) | cellule ( les pincees relevees seules, puis theta )  (global)\n"
                "  --passes K      garde par cellule : passes de relevement au plus              (6)\n"
                "  --check         verifier la mesure ( circulation contre surface, derivee contre differences finies )\n"
                "  --solver S      chol ( Eigen, defaut ) | amg\n"
                "  --amg-var V     0 = agregation+spai0 | 1 = agregation+GS | 2 = Ruge-Stuben+GS  (2)\n"
                "  --newton-tol T  arret sur max|a_i - nu| / nu             (1e-6)\n"
                "  --newton-max K  iterations au maximum                    (100)\n"
                "  --t-min T       sous ce pas, STAGNATION                  (1e-10)\n"
                "  --quiet         pas de trace par iteration\n"
                "  --ecrire FILE   ecrire les poids trouves au format de cases/\n"
                "  --dump FILE     les trames ( JSONL ) : le diagramme converge de chaque etape\n"
                "  --methode M     newton ( defaut ) | lbfgs | cg : le premier ordre a chaque etape ( PremierOrdre.h )\n"
                "  --memoire K --c2 C --precond P --refacto K --refacto-borne F --plancher-po 0|1 --max-ls K\n"
                "                  les reglages du premier ordre ( voir newton --help )\n"
                "  --bascule R     passer a Newton des que max|a-nu|/nu <= R ; --bascule-it K apres K iterations\n"
                "  --courbe FILE   CSV ( ajoute ) : methode;cas;dim;it;diagrammes;temps;max|a-nu|/nu;|r|_2 apres chaque pas\n" );
            return s == "--help" || s == "-h" ? 0 : 1;
        }
    }
    a.finalise();
    o.po.tol = o.newton.tol;
    o.po.trace = o.newton.trace;
    o.po.maxit = 10 * o.newton.maxit;                    // le premier ordre a besoin de bien plus d'iterations
    if ( o.methode == "cg" ) { o.po.methode = PremierOrdreOptions::CG; if ( o.po.c2 == TF( 0.5 ) ) o.po.c2 = 0.1; }
    if ( o.liste.empty() ) {
        if ( o.chemin == "melange" ) {
            for ( TF f = o.mel0; f > 0 && f >= o.mel_min * ( 1 - 1e-12 ); f /= o.mel_ratio ) o.liste.push_back( 1 - f );
            o.liste.push_back( 1 );
        } else {
            if ( o.conv0 > 0 ) {
                const TF smin = o.conv_min > 0 ? o.conv_min : o.sigma / 4;
                for ( TF s = o.conv0; s >= smin * ( 1 - 1e-12 ); s /= o.conv_ratio ) o.liste.push_back( s );
            }
            o.liste.push_back( 0 );
        }
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

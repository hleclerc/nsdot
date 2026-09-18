// =====================================================================================
// L'ECRASEMENT : jusqu'ou peut-on suivre une direction de Newton avant qu'une cellule se vide,
// et sait-on le predire sans recalculer le diagramme ?
//
// Deux temps. D'abord une DIRECTION : celle que Newton propose a l'iteration `--it K` sur un
// nuage ( la suite 2D, ou `--load FILE` ), ou une direction deja sauvee ( `--direction FILE` ) ;
// `--ecrire FILE` la garde pour y revenir. Ensuite l'ANALYSE, le long de `w + alpha d` :
//
//   * le polynome de chaque cellule a combinatoire figee ( `solver/Ecrasement.h` ), sa premiere
//     racine, et la premiere arete qui s'annule ;
//   * le diagramme recalcule sur une grille d'`alpha`, et l'ecart au polynome, cellule par
//     cellule ;
//   * la premiere cellule vide selon les polynomes, contre la premiere cellule vide selon le
//     diagramme ( bissection ) -- et pareil pour le plancher `eps` de l'amortissement.
//
//   xmake run ecrasement --load ../2d_des_familles/cases/lines5_n2000_s0.005_equal.txt --ecrire directions/lignes2000_it0.txt
//   xmake run ecrasement --direction directions/lignes2000_it0.txt --csv /tmp/lignes2000
// =====================================================================================

#include "bench/Direction.h"
#include "bench/Dispatch.h"
#include "solver/Ecrasement.h"
#include "solver/Laplacien.h"
#include "solver/Lineaire.h"
#include "solver/Newton.h"
#ifdef _OPENMP
#include <omp.h>
#endif
#include <atomic>
#include <cstdio>
#include <string>

using namespace sf;

namespace {

struct Opts {
    int         it        = 0;         ///< l'iteration de Newton dont on prend la direction
    std::string direction;             ///< une direction deja sauvee
    std::string ecrire;                ///< ou sauver la direction extraite
    std::string csv;                   ///< prefixe des fichiers CSV ( grille, cellules suivies )
    TF          alpha_min = 1e-4, alpha_max = 1;
    int         alpha_nb  = 41;        ///< points de la grille geometrique
    int         suivre    = 6;         ///< cellules suivies en detail
    int         bissec    = 40;        ///< pas de bissection
    OptionsLimites lim;                ///< la passe « predire, verifier, corriger »
    bool        verifie   = true;
    bool        chrono    = false;     ///< chronometrer les composantes de la passe
    bool        tenseur   = false;     ///< le pas tensoriel
    std::vector<TF> thetas;            ///< les cibles partielles
    int         picard    = 30;
    TF          amort     = 1;
    bool        newton_modele = false; ///< le vrai jacobien du modele, refactorise, au lieu de L fige
    int         man       = 0;         ///< l'ordre de la serie MAN ( 0 : pas de serie )
    std::vector<TF> man_s;             ///< ou evaluer la serie
    std::vector<TF> check_alpha;       ///< verifier le fournisseur a alpha contre le diagramme
    std::string solver    = "amg";
    int         amgvar    = Amg::SA_SPAI0;
};

TF norme2( const std::vector<TF> &v ) { TF s = 0; for ( TF x : v ) s += x * x; return std::sqrt( s ); }

/// LA DIRECTION DE NEWTON a l'iteration `K`, telle que `Newton.h` la calcule.
template<class PD, class Lin>
bool extrait( const Args &a, const Opts &o, const Nuage<2> &nu, Lin &lin, Direction<2> &dir ) {
    const SI n = nu.n;
    PD pd;
    pd.build( nu.P, nullptr, n, a.leaf );
#ifdef _OPENMP
    omp_set_num_threads( a.par.threads );
#endif
    NewtonOptions no;
    no.extraire = o.it;
    Newton<PD,Lin> nw( pd, lin, nu.P, a.par, no );
    nw.nu.assign( n, TF( 1 ) / n );
    nw.resout( std::vector<TF>( n, TF( 0 ) ) );
    if ( std::string( nw.st.fin ) != "DIRECTION EXTRAITE" ) {
        std::printf( "  pas de direction a l'iteration %d : newton a fini avant ( %s )\n", o.it, nw.st.fin );
        return false;
    }
    dir.nuage = nu;
    dir.w = nw.w;
    dir.d = nw.d;
    char e[ 512 ];
    std::snprintf( e, sizeof( e ),
                   "# direction de newton a l'iteration %d sur '%s' ( n = %d, %s, %d reculs jusque la )\n"
                   "# format : n, puis n lignes « x y w d »\n",
                   o.it, nu.nom.c_str(), int( n ), lin.nom(), nw.st.nb_recul );
    dir.entete = e;
    return true;
}

/// L'ANALYSE le long de `w + alpha d`.
template<class PD>
void analyse( const Args &a, const Opts &o, const Direction<2> &dir ) {
    const SI n = dir.n();
    const Parallel &par = a.par;
    const TF *const *P = dir.nuage.P;
    const std::vector<TF> &w = dir.w, &d = dir.d;
    const TF nu = TF( 1 ) / n;

    PD pd;
    pd.build( P, nullptr, n, a.leaf );
    std::vector<TF> w2, a0;
    mesures_en( pd, w, d, TF( 0 ), par, w2, a0 );
    TF amin = a0[ 0 ], dmax = 0;
    for ( SI i = 0; i < n; ++i ) { amin = std::min( amin, a0[ i ] ); dmax = std::max( dmax, std::fabs( d[ i ] ) ); }
    const TF eps = TF( 0.5 ) * std::min( nu, amin );  // le plancher de l'amortissement, comme Newton
    TF r0 = 0;                                          // le residu au depart, pour la decroissance
    for ( SI i = 0; i < n; ++i ) r0 += ( nu - a0[ i ] ) * ( nu - a0[ i ] );
    r0 = std::sqrt( r0 );
    std::printf( "  n = %d, nu = %.3e, aire min en alpha = 0 : %.3e ( eps = %.3e ), |d|_inf = %.3e, |r0|_2 = %.3e\n",
                 int( n ), double( nu ), double( amin ), double( eps ), double( dmax ), double( r0 ) );

    // ---- les polynomes
    double t0 = now();
    std::vector<PolyCellule> poly;
    polynomes( pd, P, w, d, par, poly );
    const double t_poly = now() - t0;
    SI n_ok = 0, n_vide = 0, n_deb = 0, n_deg = 0;
    TF ecart0 = 0;
    Premier pz, pe, pa;                                  // premiere racine ( 0 ), ( eps ), premiere arete
    for ( SI i = 0; i < n; ++i ) {
        const PolyCellule &q = poly[ i ];
        switch ( q.etat ) {
            case PolyCellule::OK:             ++n_ok; break;
            case PolyCellule::VIDE_AU_DEPART: ++n_vide; continue;
            case PolyCellule::DEBORDE:        ++n_deb; continue;
            default:                          ++n_deg; continue;
        }
        ecart0 = std::max( ecart0, std::fabs( q.a0 - a0[ i ] ) / nu );
        pz.propose( q.premiere_racine( 0 ), i );
        pe.propose( q.premiere_racine( eps ), i );
        pa.propose( q.alpha_arete, i );
    }
    std::printf( "  polynomes : %d cellules en %.3f s ( %d vides au depart, %d debordees, %d degenerees ),"
                 " ecart max a l'aire du diagramme en alpha = 0 : %.2e nu\n",
                 int( n_ok ), t_poly, int( n_vide ), int( n_deb ), int( n_deg ), double( ecart0 ) );
    std::printf( "  PREDIT : premiere cellule vide alpha = %.6e ( cellule %d ), sous eps alpha = %.6e ( %d ),"
                 " premiere arete annulee alpha = %.6e ( %d )\n",
                 double( pz.alpha ), int( pz.cellule ), double( pe.alpha ), int( pe.cellule ),
                 double( pa.alpha ), int( pa.cellule ) );

    // ---- la decomposition en triangles, pour les deux estimateurs a comparer au polynome
    std::vector<SI> offs;
    std::vector<Triangle> tris;
    decomposition( pd, P, w, d, par, offs, tris );
    auto est_pos = [ & ]( SI i, TF al ) {              // somme des parties positives
        TF s = 0;
        for ( SI t = offs[ i ]; t < offs[ i + 1 ]; ++t ) s += std::max( tris[ t ]( al ), TF( 0 ) );
        return s;
    };
    auto est_zero = [ & ]( SI i, TF al ) {             // zero passe le premier changement de signe
        TF s = 0;
        for ( SI t = offs[ i ]; t < offs[ i + 1 ]; ++t ) if ( al < tris[ t ].alpha_signe ) s += tris[ t ]( al );
        return s;
    };
    {
        TF e0 = 0;
        for ( SI i = 0; i < n; ++i ) if ( offs[ i + 1 ] > offs[ i ] ) e0 = std::max( e0, std::fabs( est_pos( i, 0 ) - a0[ i ] ) / nu );
        std::printf( "  triangles : %d, ecart max de leur somme a l'aire en alpha = 0 : %.2e nu\n", int( tris.size() ), double( e0 ) );
    }

    // ---- la grille, et le diagramme sur chacun de ses points
    std::vector<TF> grille( o.alpha_nb );
    for ( int k = 0; k < o.alpha_nb; ++k )
        grille[ k ] = o.alpha_nb > 1
                    ? o.alpha_min * std::pow( o.alpha_max / o.alpha_min, TF( k ) / ( o.alpha_nb - 1 ) )
                    : o.alpha_max;
    std::vector<std::vector<TF>> mes( o.alpha_nb );
    std::printf( "\n  %-11s %7s %7s %7s | %10s %10s | %9s %6s %6s %6s | %7s %6s | %7s %7s %7s | %6s %6s %6s %6s\n",
                 "alpha", "vides", "predit", "accord", "min exact", "min poly", "err max",
                 "<1e-8", "<1e-4", "<1e-2", "hors", "<1e-4", "|r|/r0", "poly", "borne", "pos", "pos", "zero", "zero" );
    std::printf( "  %-11s %7s %7s %7s | %10s %10s | %9s %6s %6s %6s | %7s %6s | %7s %7s %7s | %6s %6s %6s %6s\n",
                 "", "exact", "vides", "", "", "", "/ nu", "", "", "", "combin.", "", "exact", "", "1-a/2", "<1e-4", "vides", "<1e-4", "vides" );
    std::FILE *fg = o.csv.empty() ? nullptr : std::fopen( ( o.csv + "_grille.csv" ).c_str(), "w" );
    if ( fg ) std::fprintf( fg, "alpha,vides_exact,vides_poly,accord,min_exact,min_poly,err_max,"
                                "f_1e8,f_1e4,f_1e2,hors_comb,hors_comb_1e4,res_exact,res_poly,borne\n" );
    t0 = now();
    int k_vide = -1, k_eps = -1;
    for ( int k = 0; k < o.alpha_nb; ++k ) {
        const TF al = grille[ k ];
        mesures_en( pd, w, d, al, par, w2, mes[ k ] );
        const std::vector<TF> &m = mes[ k ];
        SI nv = 0, np = 0, nacc = 0, f8 = 0, f4 = 0, f2 = 0, hors = 0, hors4 = 0;
        TF mn = INFINI, mp = INFINI, err = 0, re = 0, rp = 0;
        SI fpos = 0, vpos = 0, fzero = 0, vzero = 0;
        for ( SI i = 0; i < n; ++i ) {
            if ( poly[ i ].etat != PolyCellule::OK ) continue;
            const TF q = poly[ i ]( al );
            const TF ep = est_pos( i, al ), ez = est_zero( i, al );
            fpos += std::fabs( m[ i ] - ep ) / nu < 1e-4; vpos += ! ( ep > 0 );
            fzero += std::fabs( m[ i ] - ez ) / nu < 1e-4; vzero += ! ( ez > 0 );
            re += ( nu - m[ i ] ) * ( nu - m[ i ] );
            rp += ( nu - std::max( q, TF( 0 ) ) ) * ( nu - std::max( q, TF( 0 ) ) );
            const bool ve = ! ( m[ i ] > 0 ), vp = ! ( q > 0 );
            nv += ve; np += vp; nacc += ve && vp;
            mn = std::min( mn, m[ i ] );
            mp = std::min( mp, q );
            const TF e = std::fabs( m[ i ] - std::max( q, TF( 0 ) ) ) / nu;
            err = std::max( err, e );
            f8 += e < 1e-8; f4 += e < 1e-4; f2 += e < 1e-2;
            if ( al > poly[ i ].alpha_arete ) { ++hors; hors4 += e < 1e-4; }
        }
        if ( nv && k_vide < 0 ) k_vide = k;
        if ( mn < eps && k_eps < 0 ) k_eps = k;
        re = std::sqrt( re ) / r0; rp = std::sqrt( rp ) / r0;
        std::printf( "  %-11.4e %7d %7d %7d | %10.3e %10.3e | %9.2e %6.3f %6.3f %6.3f | %7d %6.3f | %7.4f %7.4f %7.4f | %6.3f %6d %6.3f %6d\n",
                     double( al ), int( nv ), int( np ), int( nacc ), double( mn ), double( mp ), double( err ),
                     double( f8 ) / n_ok, double( f4 ) / n_ok, double( f2 ) / n_ok, int( hors ),
                     hors ? double( hors4 ) / hors : 1.0, double( re ), double( rp ), double( 1 - al / 2 ),
                     double( fpos ) / n_ok, int( vpos ), double( fzero ) / n_ok, int( vzero ) );
        if ( fg ) std::fprintf( fg, "%.10e,%d,%d,%d,%.10e,%.10e,%.10e,%.6f,%.6f,%.6f,%d,%.6f,%.10e,%.10e,%.10e\n",
                                double( al ), int( nv ), int( np ), int( nacc ), double( mn ), double( mp ),
                                double( err ), double( f8 ) / n_ok, double( f4 ) / n_ok, double( f2 ) / n_ok,
                                int( hors ), hors ? double( hors4 ) / hors : 1.0, double( re ), double( rp ),
                                double( 1 - al / 2 ) );
    }
    if ( fg ) std::fclose( fg );
    const double t_grille = now() - t0;
    std::printf( "  ( %d diagrammes en %.3f s )\n", o.alpha_nb, t_grille );

    // ---- la verite par bissection : premiere cellule vide, premiere sous eps
    auto vide = [ & ]( const std::vector<TF> &m, SI &c ) {
        for ( SI i = 0; i < n; ++i ) if ( ! ( m[ i ] > 0 ) ) { c = i; return true; }
        return false;
    };
    auto sous_eps = [ & ]( const std::vector<TF> &m, SI &c ) {
        for ( SI i = 0; i < n; ++i ) if ( m[ i ] < eps ) { c = i; return true; }
        return false;
    };
    auto borne = [ & ]( int k, auto &&crit, const char *quoi, TF predit, SI cpred, SI &c ) {
        c = -1;
        if ( k < 0 ) {
            std::printf( "  EXACT  : %s -- aucune sur la grille ( jusqu'a alpha = %.3e ) ; predit %.6e ( %d )\n",
                         quoi, double( grille.back() ), double( predit ), int( cpred ) );
            return INFINI;
        }
        const TF lo = k ? grille[ k - 1 ] : TF( 0 );
        const TF al = bissection( pd, w, d, lo, grille[ k ], par, o.bissec, crit, c );
        std::printf( "  EXACT  : %s alpha = %.6e ( cellule %d )  --  predit %.6e ( cellule %d ), rapport %.4f\n",
                     quoi, double( al ), int( c ), double( predit ), int( cpred ), double( predit / al ) );
        return al;
    };
    SI c_vide, c_eps;
    borne( k_vide, vide, "premiere cellule vide", pz.alpha, pz.cellule, c_vide );
    const TF a_eps = borne( k_eps, sous_eps, "premiere sous eps  ", pe.alpha, pe.cellule, c_eps );

    // ---- LE PAS TENSORIEL : le modele quadratique de chaque cellule, resolu par Picard pour une
    // cible partielle `a + theta ( nu - a )`, contre le pas droit `theta d`
    if ( o.tenseur ) {
#ifndef SF_EIGEN
        std::printf( "  tenseur : Eigen absent\n" );
#else
        using Cell = typename PD::Cell;
        pd.set_weights( w.data(), par );
        std::vector<ModeleCellule> mod( n );
        std::vector<TF> res0;
        std::vector<std::vector<Facette>> par_th( std::max( par.threads, 1 ) );
        pd.measures_and_facets( res0, par, [ & ]( int t, SI i, SI j, TF mes ) {
            TF d2 = 0;
            for ( int k = 0; k < 2; ++k ) { const TF e = P[ k ][ j ] - P[ k ][ i ]; d2 += e * e; }
            if ( d2 > 0 ) par_th[ t ].push_back( Facette{ i, j, mes / ( 2 * std::sqrt( d2 ) ) } );
        } );
        std::vector<Facette> fa;
        for ( auto &v : par_th ) fa.insert( fa.end(), v.begin(), v.end() );
        parallel_for( n, par, [ & ]( SI k, int ) { Cell cel; pd.cellule( k, cel ); mod[ pd.ids[ k ] ].depuis( cel, pd.ids[ k ], P, w.data() ); } );
        TF e0 = 0;
        for ( SI i = 0; i < n; ++i ) e0 = std::max( e0, std::fabs( mod[ i ].aire( nullptr ) - a0[ i ] ) / nu );
        Laplacien L;
        L.assemble( n, fa );
        Cholesky chol;
        std::vector<TF> b( n ), dn, delta, corr, r( n ), res;
        for ( SI i = 0; i < n; ++i ) b[ i ] = nu - a0[ i ];
        chol.resout( L, b, dn );
        TF ecart_d = 0, nd = 0;
        for ( SI i = 0; i < n; ++i ) { ecart_d = std::max( ecart_d, std::fabs( dn[ i ] - d[ i ] ) ); nd = std::max( nd, std::fabs( d[ i ] ) ); }
        std::printf( "\n  TENSEUR : modele contre aire en 0 : %.1e nu ; direction de Newton recalculee contre celle du fichier : %.1e / %.1e\n",
                     double( e0 ), double( ecart_d ), double( nd ) );
        auto modele = [ & ]( const std::vector<TF> &dl, std::vector<TF> &out ) {
            out.resize( n );
            parallel_for( n, par, [ & ]( SI i, int ) { out[ i ] = mod[ i ].aire( dl.data() ); } );
        };
        // le jacobien du modele en `dl` : le laplacien aux longueurs signees ( non symetrique en
        // general -- Cholesky lit le triangle inferieur, on prend ce qu'il prend )
        auto jacobien = [ & ]( const std::vector<TF> &dl, std::vector<TF> &out, Laplacien &J ) {
            out.resize( n );
            for ( auto &v : par_th ) v.clear();
            parallel_for( n, par, [ & ]( SI i, int t ) {
                out[ i ] = mod[ i ].aire( dl.data(), [ & ]( d2::SI32 j, TF c ) { par_th[ t ].push_back( Facette{ i, j, c } ); } );
            } );
            fa.clear();
            for ( auto &v : par_th ) fa.insert( fa.end(), v.begin(), v.end() );
            J.assemble( n, fa );
        };
        auto verite = [ & ]( const std::vector<TF> &dl, const char *quoi ) {
            std::vector<TF> un( n, TF( 1 ) );
            mesures_en( pd, w, dl, TF( 1 ), par, w2, res );  // les poids `w + dl`
            SI nv = 0; TF mn = INFINI, rr = 0;
            for ( SI i = 0; i < n; ++i ) { nv += ! ( res[ i ] > 0 ); mn = std::min( mn, res[ i ] ); rr += ( nu - res[ i ] ) * ( nu - res[ i ] ); }
            std::printf( "      %-22s : %6d vides, aire min %.2e ( eps %.2e ), |r|/|r0| = %.4f\n", quoi, int( nv ), double( mn ), double( eps ), double( std::sqrt( rr ) / r0 ) );
        };
        std::vector<TF> am;
        for ( TF theta : o.thetas ) {
            std::printf( "    theta = %.3f :\n", double( theta ) );
            for ( SI i = 0; i < n; ++i ) b[ i ] = a0[ i ] + theta * ( nu - a0[ i ] );
            delta.resize( n );
            for ( SI i = 0; i < n; ++i ) delta[ i ] = theta * dn[ i ];
            TF rn0 = 0, rn = 0;
            int k = 0;
            Laplacien J;
            Cholesky cj;
            const double tk0 = now();
            for ( ; k < o.picard; ++k ) {
                if ( o.newton_modele ) jacobien( delta, am, J );
                else                   modele( delta, am );
                rn = 0;
                SI neg = 0;
                for ( SI i = 0; i < n; ++i ) { r[ i ] = b[ i ] - am[ i ]; rn += r[ i ] * r[ i ]; neg += ! ( am[ i ] > 0 ); }
                rn = std::sqrt( rn );
                if ( k == 0 ) rn0 = rn;
                if ( k < 8 || k % 5 == 0 )
                    std::printf( "      %s %2d : |modele - cible| / |cible - a| = %.3e, %d cellules du modele <= 0\n",
                                 o.newton_modele ? "newton" : "picard", k, double( rn / ( theta * r0 ) ), int( neg ) );
                if ( rn < 1e-8 * theta * r0 ) break;
                if ( o.newton_modele ) {
                    if ( ! cj.resout( J, r, corr ) ) { std::printf( "      factorisation en echec\n" ); break; }
                    // l'amortissement SUR LE MODELE : des polygones a evaluer, rien d'autre
                    TF tm = 1;
                    std::vector<TF> essai( n ), am2;
                    for ( int rec = 0; rec < 30; ++rec, tm /= 2 ) {
                        for ( SI i = 0; i < n; ++i ) essai[ i ] = delta[ i ] + tm * corr[ i ];
                        essai[ 0 ] = 0;
                        modele( essai, am2 );
                        TF rn2 = 0;
                        for ( SI i = 0; i < n; ++i ) rn2 += ( b[ i ] - am2[ i ] ) * ( b[ i ] - am2[ i ] );
                        if ( std::sqrt( rn2 ) < rn ) break;
                    }
                    if ( tm < 1e-8 ) { std::printf( "      le modele ne descend plus ( amortissement %.1e )\n", double( tm ) ); break; }
                    if ( tm < 1 ) std::printf( "        amorti a %.3e\n", double( tm ) );
                    delta.swap( essai );
                } else {
                    chol.resout_encore( r, corr );
                    for ( SI i = 0; i < n; ++i ) delta[ i ] += o.amort * corr[ i ];
                    delta[ 0 ] = 0;
                }
            }
            std::printf( "      fini a %d : %.3e, %.2f s\n", k, double( rn / ( theta * r0 ) ), now() - tk0 );
            std::vector<TF> droit( n );
            for ( SI i = 0; i < n; ++i ) droit[ i ] = theta * dn[ i ];
            verite( droit, "pas droit theta d" );
            verite( delta, "pas tensoriel" );
            if ( o.csv.size() ) {
                std::FILE *fd = std::fopen( ( o.csv + "_tenseur_" + std::to_string( double( theta ) ) + ".txt" ).c_str(), "w" );
                if ( fd ) { for ( SI i = 0; i < n; ++i ) std::fprintf( fd, "%.17g\n", double( delta[ i ] ) ); std::fclose( fd ); }
            }
        }
#endif
    }

    // ---- LA SERIE ( Methode Asymptotique Numerique ) sur le modele quadratique : la cible glisse
    // `a + s ( nu - a )`, `delta( s ) = sum s^k w_k`, avec `L` factorise UNE fois et la forme
    // bilineaire `Q` par polarisation du modele. Puis la verite le long de la serie.
    if ( o.man > 0 ) {
#ifdef SF_EIGEN
        using Cell = typename PD::Cell;
        pd.set_weights( w.data(), par );
        std::vector<ModeleCellule> mod( n );
        std::vector<TF> res0;
        std::vector<std::vector<Facette>> par_th( std::max( par.threads, 1 ) );
        pd.measures_and_facets( res0, par, [ & ]( int t, SI i, SI j, TF mes ) {
            TF d2 = 0;
            for ( int k = 0; k < 2; ++k ) { const TF e = P[ k ][ j ] - P[ k ][ i ]; d2 += e * e; }
            if ( d2 > 0 ) par_th[ t ].push_back( Facette{ i, j, mes / ( 2 * std::sqrt( d2 ) ) } );
        } );
        std::vector<Facette> fa;
        for ( auto &v : par_th ) fa.insert( fa.end(), v.begin(), v.end() );
        parallel_for( n, par, [ & ]( SI k, int ) { Cell cel; pd.cellule( k, cel ); mod[ pd.ids[ k ] ].depuis( cel, pd.ids[ k ], P, w.data() ); } );
        Laplacien L;
        L.assemble( n, fa );
        Cholesky chol;
        const int N = o.man;
        std::vector<std::vector<TF>> W( N + 1 ), Q2( N + 1 );   // `W[ k ]` = w_k, `Q2[ k ]` = q2( w_k )
        std::vector<TF> r( n );
        for ( SI i = 0; i < n; ++i ) r[ i ] = nu - a0[ i ];
        double t0 = now();
        chol.resout( L, r, W[ 1 ] );
        // q2( u ) = a_modele( u ) - a0 - L u = Q( u, u ) / 2
        auto q2 = [ & ]( const std::vector<TF> &u, std::vector<TF> &out ) {
            out.resize( n );
            parallel_for( n, par, [ & ]( SI i, int ) {
                TF lu = L.dia[ i ] * u[ i ];
                for ( SI k = L.row[ i ]; k < L.row[ i + 1 ]; ++k ) lu -= L.c[ k ] * u[ L.col[ k ] ];
                out[ i ] = mod[ i ].aire( u.data() ) - a0[ i ] - lu;
            } );
        };
        q2( W[ 1 ], Q2[ 1 ] );
        std::vector<TF> rhs( n ), tmp( n ), qq;
        std::printf( "\n  MAN : ordre %d, |w_1| = %.3e\n", N, double( norme2( W[ 1 ] ) ) );
        for ( int k = 2; k <= N; ++k ) {
            // sum_{ i + j = k } Q( w_i, w_j ) / 2 = sum_{ i < j } [ q2( w_i + w_j ) - q2( w_i ) - q2( w_j ) ] + [ k pair ] q2( w_k/2 )
            std::fill( rhs.begin(), rhs.end(), TF( 0 ) );
            for ( int i = 1; i < k - i; ++i ) {
                const int j = k - i;
                for ( SI m = 0; m < n; ++m ) tmp[ m ] = W[ i ][ m ] + W[ j ][ m ];
                q2( tmp, qq );
                for ( SI m = 0; m < n; ++m ) rhs[ m ] += qq[ m ] - Q2[ i ][ m ] - Q2[ j ][ m ];
            }
            if ( k % 2 == 0 ) for ( SI m = 0; m < n; ++m ) rhs[ m ] += Q2[ k / 2 ][ m ];
            for ( SI m = 0; m < n; ++m ) rhs[ m ] = -rhs[ m ];
            chol.resout_encore( rhs, W[ k ] );
            q2( W[ k ], Q2[ k ] );
            const TF nk = norme2( W[ k ] ), nk1 = norme2( W[ k - 1 ] );
            std::printf( "    w_%-2d : |w_k| = %.3e, |w_k-1| / |w_k| = %.4e\n", k, double( nk ), double( nk1 / nk ) );
        }
        const TF s_man = std::pow( TF( 1e-6 ) * norme2( W[ 1 ] ) / norme2( W[ N ] ), TF( 1 ) / ( N - 1 ) );
        std::printf( "    serie en %.2f s ; rayon MAN ( tol 1e-6 ) : s_max = %.4e\n", now() - t0, double( s_man ) );

        // LE TEST : pour chaque ordre, `s = 1` puis `s / 2` tant qu'une cellule passe sous `eps`
        std::vector<TF> delta( n ), am( n ), res;
        std::printf( "  essais dyadiques par ordre ( eps = %.2e ) :\n", double( eps ) );
        for ( int ordre = 1; ordre <= N; ordre = ordre < 4 ? ordre + 1 : 2 * ordre ) {
            TF sv = 1;
            int essais = 0;
            for ( ;; ++essais ) {
                for ( SI m = 0; m < n; ++m ) { TF acc = 0, p = 1; for ( int k = 1; k <= ordre; ++k ) { p *= sv; acc += p * W[ k ][ m ]; } delta[ m ] = acc; }
                mesures_en( pd, w, delta, TF( 1 ), par, w2, res );
                TF mn = INFINI, rr = 0;
                for ( SI i = 0; i < n; ++i ) { mn = std::min( mn, res[ i ] ); rr += ( nu - res[ i ] ) * ( nu - res[ i ] ); }
                if ( mn >= eps || sv < 1e-12 ) {
                    std::printf( "    ordre %2d : s = %.3e accepte apres %d essais, aire min %.2e, |r|/r0 = %.6f ( cible %.6f )\n",
                                 ordre, double( sv ), essais, double( mn ), double( std::sqrt( rr ) / r0 ), double( 1 - sv ) );
                    break;
                }
                sv /= 2;
            }
        }

        // la verite le long de la serie
        std::printf( "  %-10s | %10s | %6s %10s %8s | %8s %8s\n", "s", "modele-cible", "vides", "aire min", "|r|/r0", "cible", "droit" );
        for ( TF sv : o.man_s ) {
            for ( SI m = 0; m < n; ++m ) { TF acc = 0, p = 1; for ( int k = 1; k <= N; ++k ) { p *= sv; acc += p * W[ k ][ m ]; } delta[ m ] = acc; }
            parallel_for( n, par, [ & ]( SI i, int ) { am[ i ] = mod[ i ].aire( delta.data() ); } );
            TF em = 0;
            for ( SI i = 0; i < n; ++i ) em += ( am[ i ] - a0[ i ] - sv * r[ i ] ) * ( am[ i ] - a0[ i ] - sv * r[ i ] );
            mesures_en( pd, w, delta, TF( 1 ), par, w2, res );
            SI nv = 0; TF mn = INFINI, rr = 0;
            for ( SI i = 0; i < n; ++i ) { nv += ! ( res[ i ] > 0 ); mn = std::min( mn, res[ i ] ); rr += ( nu - res[ i ] ) * ( nu - res[ i ] ); }
            for ( SI m = 0; m < n; ++m ) delta[ m ] = sv * W[ 1 ][ m ];
            mesures_en( pd, w, delta, TF( 1 ), par, w2, res );
            SI nvd = 0;
            for ( SI i = 0; i < n; ++i ) nvd += ! ( res[ i ] > 0 );
            std::printf( "  %-10.4e | %10.2e | %6d %10.2e %8.4f | %8.4f %8d\n", double( sv ), double( std::sqrt( em ) / ( sv * r0 ) ),
                         int( nv ), double( mn ), double( std::sqrt( rr ) / r0 ), double( 1 - sv ), int( nvd ) );
        }
#endif
    }

    // ---- le chronometre des composantes de la passe
    if ( o.chrono ) {
        using Cell = typename PD::Cell;
        using TK   = typename PD::TKernel;
        pd.set_weights( w.data(), par );
        std::vector<TF> dt( n );
        for ( SI k = 0; k < n; ++k ) dt[ k ] = d[ pd.arbre.order[ k ] ];
        std::vector<WMajT<2>> dm( pd.arbre.nodes.size() );
        double t0 = now();
        parallel_for( SI( dm.size() ), par, [ & ]( SI m, int ) {
            const auto &nd = pd.arbre.nodes[ m ];
            dm[ m ] = weight_majorant<2>( nd.beg, nd.end, [ & ]( SI k, Vec<2> &q, TF &v ) {
                q[ 0 ] = pd.arbre.p[ 0 ][ k ]; q[ 1 ] = pd.arbre.p[ 1 ][ k ]; v = dt[ k ];
            } );
        } );
        const double t_dm = now() - t0;
        auto chrono = [ & ]( const char *quoi, auto &&f ) {
            double best = 1e9;
            for ( int r = 0; r < 3; ++r ) { const double t = now(); parallel_for( n, par, f ); best = std::min( best, now() - t ); }
            std::printf( "  CHRONO %-44s %.4f s\n", quoi, best );
            return best;
        };
        std::printf( "  CHRONO %-44s %.4f s\n", "majorant de d", t_dm );
        chrono( "cellule en 0 ( = un diagramme )", [ & ]( SI k, int ) { Cell cel; pd.cellule( k, cel ); } );
        {                                                // le voisinage, comme Newton l'a
            std::vector<Facette> fa;
            std::vector<TF> res;
            std::vector<std::vector<Facette>> par_th( std::max( par.threads, 1 ) );
            pd.measures_and_facets( res, par, [ & ]( int t, SI i, SI j, TF ) { par_th[ t ].push_back( Facette{ i, j, 1 } ); } );
            for ( auto &v : par_th ) fa.insert( fa.end(), v.begin(), v.end() );
            Laplacien L;
            L.assemble( n, fa );
            chrono( "cellule en 0 par ses voisins ( CSR )", [ & ]( SI k, int ) {
                const SI i = pd.ids[ k ];
                Cell cel;
                std::vector<d2::SI32> vs( L.col.data() + L.row[ i ], L.col.data() + L.row[ i + 1 ] );
                d2::FournisseurAlpha<TK> f( &pd.arbre, dm.data(), dt.data(), P, w.data(), d.data(), TF( 0 ),
                                            d2::SI32( i ), vs.data(), int( vs.size() ) );
                f.parcours = false;
                d2::moteur<TK>( &f, &cel ); } );
        }
        chrono( "cellule en 0 + polynome", [ & ]( SI k, int ) {
            Cell cel; pd.cellule( k, cel ); polynome_cellule( cel, pd.ids[ k ], P, w.data(), d.data() ); } );
        for ( TF al : { TF( 1e-3 ), TF( 4e-3 ), TF( 1e-1 ), TF( 1 ) } ) {
            char nom[ 64 ];
            std::snprintf( nom, sizeof( nom ), "cellule en 0 + a chaud en %.0e", double( al ) );
            chrono( nom, [ & ]( SI k, int ) {
                Cell cel; pd.cellule( k, cel );
                d2::FournisseurAlpha<TK> f( &pd.arbre, dm.data(), dt.data(), P, w.data(), d.data(), al,
                                            d2::SI32( pd.ids[ k ] ), cel.cid, std::max( cel.nb, 0 ) );
                d2::moteur<TK>( &f, &cel ); } );
            std::snprintf( nom, sizeof( nom ), "cellule a froid en %.0e ( fournisseur alpha )", double( al ) );
            chrono( nom, [ & ]( SI k, int ) {
                Cell cel;
                d2::FournisseurAlpha<TK> f( &pd.arbre, dm.data(), dt.data(), P, w.data(), d.data(), al,
                                            d2::SI32( pd.ids[ k ] ), nullptr, 0 );
                d2::moteur<TK>( &f, &cel ); } );
        }
    }

    // ---- le fournisseur a alpha contre le diagramme rafraichi, cellule par cellule
    for ( TF ac : o.check_alpha ) {
        using Cell = typename PD::Cell;
        using TK   = typename PD::TKernel;
        std::vector<TF> exact;
        mesures_en( pd, w, d, ac, par, w2, exact );      // l'arbre porte `w + ac d`
        std::vector<Cell> cel0( n );
        pd.set_weights( w.data(), par );                  // puis `w`, pour le fournisseur
        parallel_for( n, par, [ & ]( SI k, int ) { pd.cellule( k, cel0[ pd.ids[ k ] ] ); } );
        std::vector<TF> dt( n );
        for ( SI k = 0; k < n; ++k ) dt[ k ] = d[ pd.arbre.order[ k ] ];
        std::vector<WMajT<2>> dm( pd.arbre.nodes.size() );
        for ( size_t m = 0; m < dm.size(); ++m ) {
            const auto &nd = pd.arbre.nodes[ m ];
            dm[ m ] = weight_majorant<2>( nd.beg, nd.end, [ & ]( SI k, Vec<2> &q, TF &v ) {
                q[ 0 ] = pd.arbre.p[ 0 ][ k ]; q[ 1 ] = pd.arbre.p[ 1 ][ k ]; v = dt[ k ];
            } );
        }
        std::atomic<SI> nb_diff{ 0 };
        std::vector<TF> ecart_th( std::max( par.threads, 1 ), TF( 0 ) );
        parallel_for( n, par, [ & ]( SI k, int t ) {
            const SI i = pd.ids[ k ];
            Cell cel;
            d2::FournisseurAlpha<TK> f( &pd.arbre, dm.data(), dt.data(), P, w.data(), d.data(), ac,
                                        d2::SI32( i ), cel0[ i ].cid, std::max( cel0[ i ].nb, 0 ) );
            d2::moteur<TK>( &f, &cel );
            const TF m = cel.nb > 0 ? PD::mesure( cel ) : TF( 0 );
            const TF e = std::fabs( m - exact[ i ] ) / nu;
            ecart_th[ t ] = std::max( ecart_th[ t ], e );
            if ( e > 1e-9 ) {
                if ( nb_diff++ < 5 )
                    std::printf( "    alpha %.3e cellule %d : fournisseur %.6e ( nb %d ), diagramme %.6e\n",
                                 double( ac ), int( i ), double( m ), cel.nb, double( exact[ i ] ) );
            }
        } );
        TF ecart = 0;
        for ( TF e : ecart_th ) ecart = std::max( ecart, e );
        std::printf( "  CHECK alpha = %.3e : fournisseur a alpha contre diagramme, %d cellules differentes, ecart max %.2e nu\n",
                     double( ac ), int( nb_diff.load() ), double( ecart ) );
    }

    // ---- la passe « predire, verifier, corriger », contre la bissection et contre la grille
    if ( o.verifie ) {
        OptionsLimites ol = o.lim;
        ol.niveau = eps;
        pd.set_weights( w.data(), par );
        std::vector<LimiteCellule> lim;
        t0 = now();
        limites( pd, P, w, d, par, ol, lim );
        const double t_lim = now() - t0;

        SI cnt[ 5 ] = {}, tours = 0, tmax = 0;
        Premier pl;
        for ( SI i = 0; i < n; ++i ) {
            ++cnt[ lim[ i ].etat ];
            tours += lim[ i ].tours;
            tmax = std::max( tmax, lim[ i ].tours );
            if ( lim[ i ].etat != LimiteCellule::VIDE_AU_DEPART ) pl.propose( lim[ i ].alpha, i );
        }
        std::printf( "\n  VERIFIE ( coeff %.2f, horizon %.2f, tol %.0e ) : %.3f s, soit %.2f diagrammes de la grille ;"
                     " %d cellules calculees ( %.2f par cellule, %d au plus )\n",
                     double( ol.coeff ), double( ol.horizon ), double( ol.tol ), t_lim,
                     t_lim / ( t_grille / o.alpha_nb ), int( tours ), double( tours ) / n, int( tmax ) );
        std::printf( "           %d confirmees, %d corrigees, %d au-dela de l'horizon, %d vides au depart, %d en echec\n",
                     int( cnt[ 0 ] ), int( cnt[ 1 ] ), int( cnt[ 2 ] ), int( cnt[ 3 ] ), int( cnt[ 4 ] ) );
        std::printf( "           limite globale alpha = %.6e ( cellule %d )  --  exact %.6e ( cellule %d ), rapport %.6f\n",
                     double( pl.alpha ), int( pl.cellule ), double( a_eps ), int( c_eps ), double( pl.alpha / a_eps ) );

        // chaque limite contre l'encadrement que donne la grille : `mes[ k ][ i ] < eps` pour la
        // premiere fois en `k` veut dire une limite exacte dans `( grille[ k - 1 ], grille[ k ] ]`
        SI coherent = 0, incoh = 0, montre = 0;
        for ( SI i = 0; i < n; ++i ) {
            if ( lim[ i ].etat == LimiteCellule::VIDE_AU_DEPART ) continue;
            int kk = -1;
            for ( int k = 0; k < o.alpha_nb && kk < 0; ++k ) if ( mes[ k ][ i ] < eps ) kk = k;
            const TF lo = kk < 0 ? grille.back() : kk ? grille[ kk - 1 ] : TF( 0 ), hi = kk < 0 ? INFINI : grille[ kk ];
            const TF al = lim[ i ].alpha;
            const bool ok = al >= lo * ( 1 - 2 * ol.tol ) && al <= hi * ( 1 + 2 * ol.tol );
            coherent += ok; incoh += ! ok;
            if ( ! ok && montre++ < 8 )
                std::printf( "           incoherente : cellule %d, limite %.4e ( etat %d, %d tours, poly %.4e ), grille dit ( %.4e, %.4e ]\n",
                             int( i ), double( al ), lim[ i ].etat, lim[ i ].tours, double( lim[ i ].alpha_poly ),
                             double( lo ), double( hi ) );
        }
        std::printf( "           contre la grille : %d limites coherentes, %d incoherentes\n", int( coherent ), int( incoh ) );
    }

    // ---- les cellules suivies : les premieres racines predites, et la premiere vide exacte
    std::vector<SI> suivies;
    {
        std::vector<SI> ord;
        for ( SI i = 0; i < n; ++i ) if ( poly[ i ].etat == PolyCellule::OK ) ord.push_back( i );
        std::vector<TF> rac( n, INFINI );
        for ( SI i : ord ) rac[ i ] = poly[ i ].premiere_racine( 0 );
        std::partial_sort( ord.begin(), ord.begin() + std::min<size_t>( o.suivre, ord.size() ), ord.end(),
                           [ & ]( SI x, SI y ) { return rac[ x ] < rac[ y ]; } );
        for ( int j = 0; j < o.suivre && j < int( ord.size() ); ++j ) suivies.push_back( ord[ j ] );
        if ( k_vide >= 0 )                               // les cellules vides au premier point de grille
            for ( SI i = 0; i < n && suivies.size() < size_t( 2 * o.suivre ); ++i )
                if ( ! ( mes[ k_vide ][ i ] > 0 ) && std::find( suivies.begin(), suivies.end(), i ) == suivies.end() )
                    suivies.push_back( i );
    }
    std::printf( "\n  cellules suivies ( les %d premieres racines predites, puis les vides au point de grille %d ) :\n",
                 o.suivre, k_vide );
    std::printf( "  %8s %4s %10s %10s %10s %10s | %10s %10s | %10s %10s\n", "cellule", "nb", "a0/nu", "a1/nu", "a2/nu",
                 "racine", "arete", "exact vide", "q(a_vide)", "q(arete)" );
    std::FILE *fc = o.csv.empty() ? nullptr : std::fopen( ( o.csv + "_cellules.csv" ).c_str(), "w" );
    if ( fc ) {
        std::fprintf( fc, "alpha" );
        for ( SI i : suivies ) std::fprintf( fc, ",exact_%d,poly_%d", int( i ), int( i ) );
        std::fprintf( fc, "\n" );
        for ( int k = 0; k < o.alpha_nb; ++k ) {
            std::fprintf( fc, "%.10e", double( grille[ k ] ) );
            for ( SI i : suivies ) std::fprintf( fc, ",%.10e,%.10e", double( mes[ k ][ i ] ), double( poly[ i ]( grille[ k ] ) ) );
            std::fprintf( fc, "\n" );
        }
        std::fclose( fc );
    }
    for ( SI i : suivies ) {
        const PolyCellule &q = poly[ i ];
        int kv = -1;
        for ( int k = 0; k < o.alpha_nb && kv < 0; ++k ) if ( ! ( mes[ k ][ i ] > 0 ) ) kv = k;
        TF av = INFINI;
        if ( kv >= 0 ) {
            SI c;
            av = bissection( pd, w, d, kv ? grille[ kv - 1 ] : TF( 0 ), grille[ kv ], par, o.bissec,
                             [ & ]( const std::vector<TF> &m, SI &cc ) { cc = i; return ! ( m[ i ] > 0 ); }, c );
        }
        const TF rac = q.premiere_racine( 0 );
        std::printf( "  %8d %4d %10.3e %10.3e %10.3e %10.4e | %10.4e %10.4e | %10.3e %10.3e\n",
                     int( i ), q.nb_aretes, double( q.a0 / nu ), double( q.a1 / nu ), double( q.a2 / nu ),
                     double( rac ), double( q.alpha_arete ), double( av ),
                     double( av < INFINI ? q( av ) / nu : 0 ),
                     double( q.alpha_arete < INFINI ? q( q.alpha_arete ) / nu : 0 ) );
    }
    if ( ! o.csv.empty() )
        std::printf( "  csv : %s_grille.csv, %s_cellules.csv\n", o.csv.c_str(), o.csv.c_str() );
}

template<class PD>
int deroule_nuage( const Args &a, const Opts &o, const Nuage<2> &nu ) {
    Direction<2> dir;
    bool ok = false;
#ifdef SF_EIGEN
    if ( o.solver == "chol" ) {
        Cholesky lin;
        ok = extrait<PD>( a, o, nu, lin, dir );
    } else
#endif
    {
#ifdef SF_AMGCL
        Amg lin;
        lin.variante = o.amgvar;
        ok = extrait<PD>( a, o, nu, lin, dir );
#else
        std::printf( "  AMGCL absent : --solver chol\n" );
#endif
    }
    if ( ! ok ) return 1;
    if ( ! o.ecrire.empty() )
        std::printf( ecrit_direction( o.ecrire, dir ) ? "  direction ecrite dans '%s'\n"
                                                      : "  impossible d'ecrire '%s'\n", o.ecrire.c_str() );
    analyse<PD>( a, o, dir );
    return 0;
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
        else if ( s == "--it" )        o.it = std::atoi( val() );
        else if ( s == "--direction" ) o.direction = val();
        else if ( s == "--ecrire" )    o.ecrire = val();
        else if ( s == "--csv" )       o.csv = val();
        else if ( s == "--alpha-min" ) o.alpha_min = std::atof( val() );
        else if ( s == "--alpha-max" ) o.alpha_max = std::atof( val() );
        else if ( s == "--alpha-nb" )  o.alpha_nb = std::atoi( val() );
        else if ( s == "--suivre" )    o.suivre = std::atoi( val() );
        else if ( s == "--bissec" )    o.bissec = std::atoi( val() );
        else if ( s == "--coeff" )     o.lim.coeff = std::atof( val() );
        else if ( s == "--horizon" )   o.lim.horizon = std::atof( val() );
        else if ( s == "--tol" )       o.lim.tol = std::atof( val() );
        else if ( s == "--tours" )     o.lim.max_tours = std::atoi( val() );
        else if ( s == "--sans-verif" ) o.verifie = false;
        else if ( s == "--trace" )     o.lim.trace = std::atoi( val() );
        else if ( s == "--check" )     o.check_alpha.push_back( std::atof( val() ) );
        else if ( s == "--chrono" )    o.chrono = true;
        else if ( s == "--tenseur" )   o.tenseur = true;
        else if ( s == "--theta" )     o.thetas.push_back( std::atof( val() ) );
        else if ( s == "--picard" )    o.picard = std::atoi( val() );
        else if ( s == "--amort" )     o.amort = std::atof( val() );
        else if ( s == "--newton-modele" ) o.newton_modele = true;
        else if ( s == "--man" )       o.man = std::atoi( val() );
        else if ( s == "--s" )         o.man_s.push_back( std::atof( val() ) );
        else if ( s == "--solver" )    o.solver = val();
        else if ( s == "--amg-var" )   o.amgvar = std::atoi( val() );
        else {
            std::printf( "usage: ecrasement [options]   ( 2D seulement pour l'instant )\n" );
            Args::usage();
            std::printf(
                "  --it K            la direction de Newton a l'iteration K        (0)\n"
                "  --direction FILE  une direction sauvee, au lieu d'en extraire une\n"
                "  --ecrire FILE     sauver la direction extraite\n"
                "  --csv PREFIX      ecrire PREFIX_grille.csv et PREFIX_cellules.csv\n"
                "  --alpha-min A     debut de la grille geometrique                (1e-4)\n"
                "  --alpha-max A     fin de la grille                              (1)\n"
                "  --alpha-nb K      points de la grille                           (41)\n"
                "  --suivre K        cellules suivies en detail                    (6)\n"
                "  --bissec K        pas de bissection                             (40)\n"
                "  --coeff C         verifier en a_ok + C ( predit - a_ok )          (0.99)\n"
                "  --horizon A       au-dela, ne verifier qu'a l'horizon             (1)\n"
                "  --tol T           precision relative des limites                  (1e-2)\n"
                "  --tours K         cellules calculees au plus, par cellule         (12)\n"
                "  --sans-verif      sauter la passe « predire, verifier, corriger »\n"
                "  --trace I         imprimer chaque tour de la cellule I\n"
                "  --check A         le fournisseur a alpha contre le diagramme rafraichi, en A ( repetable )\n"
                "  --chrono          chronometrer les composantes de la passe\n"
                "  --tenseur         le pas tensoriel ( modele quadratique de chaque cellule, Picard )\n"
                "  --theta T         cible partielle a + T ( nu - a ), repetable      (0.01 0.02 0.05 0.1 0.2)\n"
                "  --picard K        iterations de Picard au plus                   (30)\n"
                "  --amort A         amortissement de Picard                        (1)\n"
                "  --newton-modele   Newton sur le modele ( jacobien signe refactorise ) au lieu de Picard\n"
                "  --man N           la serie MAN a l'ordre N sur le modele, et la verite le long\n"
                "  --s S             ou evaluer la serie ( repetable )\n"
                "  --solver S        amg | chol, pour la direction                 (amg)\n"
                "  --amg-var V       la hierarchie AMGCL                           (0)\n" );
            return s == "--help" || s == "-h" ? 0 : 1;
        }
    }
    a.finalise();
    if ( o.thetas.empty() ) o.thetas = { 0.01, 0.02, 0.05, 0.1, 0.2 };
    if ( o.man_s.empty() ) o.man_s = { 0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2 };
    if ( a.dims != 2 ) { std::printf( "2D seulement pour l'instant\n" ); return 1; }

    return dispatch<2>( a, [ & ]( auto tag ) {
        using PD = typename decltype( tag )::type;
        if ( ! o.direction.empty() ) {
            Direction<2> dir;
            if ( ! charge_direction<2>( o.direction, dir ) ) return 1;
            std::printf( "-- %s\n%s", o.direction.c_str(), dir.entete.c_str() );
            analyse<PD>( a, o, dir );
            return 0;
        }
        int bad = 0;
        for ( const Nuage<2> &nu : a.nuages<2>() ) {
            if ( nu.absent ) { std::printf( "  %-28s : ABSENT\n", nu.nom.c_str() ); continue; }
            std::printf( "-- %s, direction de l'iteration %d\n", nu.nom.c_str(), o.it );
            bad += deroule_nuage<PD>( a, o, nu );
        }
        return bad;
    } );
}

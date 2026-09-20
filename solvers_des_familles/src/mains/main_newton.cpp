// =====================================================================================
// LE SOLVEUR : un probleme de transport semi-discret resolu de bout en bout, chronometre par poste,
// sur la suite ( uniforme, nuage dur / Voronoi, nuage dur / masses egales ), en 2D puis en 3D.
//
// Ce qu'on veut y lire n'est pas le temps total mais sa REPARTITION : sur l'uniforme l'algebre
// lineaire pese les deux tiers, sur les lignes c'est le diagramme, et les deux regimes n'appellent
// pas le meme travail ( `2d_des_familles/README.md`, § 1.4 ).
//
//   xmake run newton --help
//   xmake run newton --2d -n 100000 --solver amg --amg-var 2
//   xmake run newton --3d --kernel float
// =====================================================================================

#include "bench/Dispatch.h"
#include "bench/Trames.h"
#include "solver/Lineaire.h"
#include "solver/Newton.h"
#ifdef _OPENMP
#include <omp.h>
#endif
#include <cstdio>
#include <string>

using namespace sf;

namespace {

/// les options PROPRES au solveur.
struct Opts {
    NewtonOptions newton;
    std::string   solver = "amg";  ///< amg | chol
    int           amgvar = Amg::SA_SPAI0;
    TF            lintol = 1e-10;
    int           linmax = 20000;
    std::string   ecrire;          ///< ou ecrire les poids trouves, au format de `cases/`
    std::string   dump;            ///< les trames de l'animation ( JSONL )
};

template<class PD, class Lin>
int lance( const Args &a, const Opts &o, const Nuage<PD::dim> &nu, Lin &lin ) {
    constexpr int D = PD::dim;
    const SI n = nu.n;

    // l'arbre est bati une fois pour toutes, sur des poids nuls, et ne sera plus que rafraichi
    double t0 = now();
    PD pd;
    pd.build( nu.P, nullptr, n, a.leaf );
    const double t_arbre = now() - t0;

#ifdef _OPENMP
    // AMGCL est parallelise en OpenMP, le diagramme en `std::thread` : sans ca les deux moities
    // du chronometre ne tourneraient pas sur le meme nombre de coeurs.
    omp_set_num_threads( a.par.threads );
#endif

    Newton<PD,Lin> nw( pd, lin, nu.P, a.par, o.newton );
    nw.nu.assign( n, TF( 1 ) / n );
    Trames trames;
    if ( ! o.dump.empty() && trames.ouvre( o.dump ) )
        nw.o.apres_pas = [ & ]( int it, TF t, int reculs ) { trames.ecrit( pd, nw.nu, a.par, "newton", 0, it, double( t ), reculs ); };
    t0 = now();
    const bool ok = nw.resout( std::vector<TF>( n, TF( 0 ) ) );
    const double total = now() - t0 + t_arbre;
    const NewtonStats &st = nw.st;
    const StatsLin &sl = lin.st;
    const double autre = total - t_arbre - st.t_maj - st.t_diag - st.t_asm - st.t_lin - st.t_lim;

    std::printf( "  newton %s ( max|a-nu|/nu = %.2e ) : %dD n=%d threads=%d kernel=%s maxnv=%d leaf=%d"
                 " -- %d iterations, %d diagrammes ( %d reculs ), %s%s\n",
                 st.fin, double( st.reste ), D, int( n ), a.par.threads, a.kernel.c_str(), PD::max_nv,
                 int( a.leaf ), st.nb_iter, st.nb_diag, st.nb_recul, lin.nom(),
                 sl.nb_iter ? ( " ( " + std::to_string( sl.nb_iter ) + " iterations )" ).c_str() : "" );
    std::printf( "         arbre %.3f | majorants %.3f | diagrammes %.3f | assemblage %.3f"
                 " | resolution %.3f | limites %.3f | reste %.3f | TOTAL %.3f s\n",
                 t_arbre, st.t_maj, st.t_diag, st.t_asm, st.t_lin, st.t_lim, autre, total );
    if ( st.nb_cell_lim )
        std::printf( "         limites : %d cellules calculees ( %.2f par germe et par iteration ), %d pas refuses par le diagramme\n",
                     int( st.nb_cell_lim ), double( st.nb_cell_lim ) / n / std::max( st.nb_iter, 1 ), st.nb_lim_refus );
    if ( st.nb_tours_essai )
        std::printf( "         essai-limites : %d essais corriges, %d cellules sous eps en tout\n", st.nb_tours_essai, int( st.nb_cell_mauvaises ) );
    if ( st.nb_tenseur )
        std::printf( "         tenseur : %d pas tensoriels, %.3f s ( compris dans limites )\n", st.nb_tenseur, st.t_tenseur );
    std::printf( "         soit %.0f %% de diagramme, %.3f s par diagramme, %.1f us/germe en tout\n",
                 100 * st.t_diag / total, st.t_diag / std::max( st.nb_diag, 1 ), 1e6 * total / n );
    std::printf( "         resolution en detail : mise en forme %.3f | hierarchie/analyse %.3f ( %d )"
                 " | resolution %.3f | pire residu lineaire %.2e\n",
                 sl.t_forme, sl.t_hier, sl.nb_hier, sl.t_res, double( sl.pire ) );
    if ( st.nb_deborde )
        std::printf( "  ATTENTION : %d cellules ont DEBORDE %d sommets pendant la resolution --"
                     " relancer avec --maxnv %d.\n", int( st.nb_deborde ), PD::max_nv, 2 * PD::max_nv );

    if ( ! o.ecrire.empty() ) {
        char entete[ 512 ];
        std::snprintf( entete, sizeof( entete ),
                       "# nuage %dD a masses egales, poids obtenus par newton ( %s, max|a-nu|/nu = %.3e )\n"
                       "# ATTENTION : produits par le banc lui-meme, donc PAS un temoin independant\n"
                       "# pour Newton -- seulement un cas de chronometrage.\n", D, st.fin, double( st.reste ) );
        if ( ecrit_nuage<D>( o.ecrire, nu, nw.w.data(), entete ) )
            std::printf( "  poids ecrits dans '%s'\n", o.ecrire.c_str() );
        else
            std::printf( "  impossible d'ecrire '%s'\n", o.ecrire.c_str() );
    }

    // La verification qui ne coute rien : le nuage `_equal` PORTE deja la solution, obtenue par
    // un tout autre chemin ( L-BFGS, pysdot ). Elle n'est definie qu'a une constante pres, donc on
    // recale sur le germe 0 -- exactement la jauge que Newton impose.
    if ( nu.temoin && nu.W ) {
        TF m = 0, ampl = 0;
        for ( SI i = 0; i < n; ++i ) {
            m = std::max( m, std::fabs( ( nw.w[ i ] - nw.w[ 0 ] ) - ( nu.W[ i ] - nu.W[ 0 ] ) ) );
            ampl = std::max( ampl, std::fabs( nu.W[ i ] - nu.W[ 0 ] ) );
        }
        std::printf( "  contre les poids du fichier : ecart max %.3e sur une amplitude %.3e\n",
                     double( m ), double( ampl ) );
    }
    return ok && st.nb_deborde == 0 ? 0 : 1;
}

template<int D>
int deroule( const Args &a, const Opts &o ) {
    int bad = 0;
    std::printf( "=== %dD\n", D );
    for ( const Nuage<D> &nu : a.nuages<D>() ) {
        if ( nu.absent ) {
            std::printf( "  %-28s : ABSENT ( --cases DIR, ou lancer 2d_des_familles/cases/gen_cases.py )\n",
                         nu.nom.c_str() );
            continue;
        }
        std::printf( "-- %s\n", nu.nom.c_str() );
        bad += dispatch<D>( a, [ & ]( auto tag ) {
            using PD = typename decltype( tag )::type;
#ifdef SF_EIGEN
            if ( o.solver == "chol" ) {
                Cholesky lin;
                return lance<PD>( a, o, nu, lin );
            }
#endif
#ifdef SF_AMGCL
            Amg lin;
            lin.variante = o.amgvar;
            lin.tol = o.lintol;
            lin.maxit = o.linmax;
            return lance<PD>( a, o, nu, lin );
#else
            std::printf( "  AMGCL absent : --solver chol\n" );
            return 1;
#endif
        } );
    }
    return bad;
}

} // namespace

int main( int argc, char **argv ) {
    Args a;
    a.n = 100000;
    Opts o;
    for ( int i = 1; i < argc; ++i ) {
        const std::string s = argv[ i ];
        auto val = [ & ]() { return i + 1 < argc ? argv[ ++i ] : ""; };
        if ( a.parse( s, i, argc, argv ) ) continue;
        else if ( s == "--newton-tol" ) o.newton.tol = std::atof( val() );
        else if ( s == "--newton-max" ) o.newton.maxit = std::atoi( val() );
        else if ( s == "--lin-tol" )    o.lintol = std::atof( val() );
        else if ( s == "--lin-max" )    o.linmax = std::atoi( val() );
        else if ( s == "--solver" )     o.solver = val();
        else if ( s == "--amg-var" )    o.amgvar = std::atoi( val() );
        else if ( s == "--ecrire" )     o.ecrire = val();
        else if ( s == "--quiet" )      o.newton.trace = false;
        else if ( s == "--pas" ) {
            const std::string v = val();
            o.newton.pas = v == "dyadique" ? NewtonOptions::DYADIQUE : v == "facteur" ? NewtonOptions::FACTEUR
                         : v == "tenseur" ? NewtonOptions::TENSEUR : v == "essai-limites" ? NewtonOptions::ESSAI_LIMITES
                         : NewtonOptions::ESSAIS;
        }
        else if ( s == "--facteur" )    o.newton.facteur = std::atof( val() );
        else if ( s == "--theta-mult" ) o.newton.theta_mult = std::atof( val() );
        else if ( s == "--confiance" )  o.newton.confiance = std::atof( val() );
        else if ( s == "--beta0" )      o.newton.beta0 = std::atof( val() );
        else if ( s == "--mult-ok" )    o.newton.mult_ok = std::atof( val() );
        else if ( s == "--dump" )       o.dump = val();
        else if ( s == "--lim-tol" )    o.newton.lim.tol = std::atof( val() );
        else if ( s == "--lim-coeff" )  o.newton.lim.coeff = std::atof( val() );
        else if ( s == "--t-min" )      o.newton.t_min = std::atof( val() );
        else if ( s == "--residu" ) {
            const std::string v = val();
            o.newton.residu = v == "barriere" ? NewtonOptions::BARRIERE : v == "log" ? NewtonOptions::LOG : NewtonOptions::LIN;
        }
        else {
            std::printf( "usage: newton [options]\n" );
            Args::usage();
            std::printf(
                "  --solver S      amg ( AMGCL, defaut ) | chol ( Eigen )\n"
                "  --amg-var V     0 = agregation+spai0 | 1 = agregation+GS | 2 = Ruge-Stuben+GS  (0)\n"
                "  --newton-tol T  arret sur max|a_i - nu| / nu             (1e-6)\n"
                "  --newton-max K  iterations au maximum                    (100)\n"
                "  --lin-tol T     arret du solveur lineaire, relatif       (1e-10)\n"
                "  --lin-max K     iterations du solveur lineaire au plus   (20000)\n"
                "  --ecrire FILE   ecrire les poids trouves au format de cases/ ( le dernier nuage deroule )\n"
                "  --quiet         pas de trace par iteration\n"
                "  --pas P         essais ( KMT, defaut ) | dyadique | facteur | tenseur | essai-limites ( 2D )\n"
                "  --beta0 B       essai-limites : le premier essai                          (0.25)\n"
                "  --mult-ok M     essai-limites : apres un essai passe direct, beta *= M     (2)\n"
                "  --confiance C   essai-limites : apres un pas corrige, au moins C * t       (0 = beta inchange)\n"
                "  --theta-mult M  tenseur : cible partielle theta = M * alpha*        (5)\n"
                "  --facteur F     t = F * alpha* en mode facteur                (0.9)\n"
                "  --lim-tol T     precision relative des limites               (1e-2)\n"
                "  --lim-coeff C   ou verifier la prediction                    (0.99)\n"
                "  --t-min T       sous ce pas, STAGNATION                      (1e-10)\n"
                "  --residu R      lin ( a - nu ) | barriere ( x - 1/x, x = a/nu ) | log  (lin)\n" );
            return s == "--help" || s == "-h" ? 0 : 1;
        }
    }
    a.finalise();

    int bad = 0;
    if ( a.dims != 3 ) bad += deroule<2>( a, o );
    if ( a.dims != 2 ) bad += deroule<3>( a, o );
    return bad ? 1 : 0;
}

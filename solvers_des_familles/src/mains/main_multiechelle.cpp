// =====================================================================================
// LE MULTI-ECHELLE : resoudre sur des representants, prolonger, resoudre en dessous. La question
// est celle de la PROLONGATION ( `solver/Prolongation.h` ) : quels poids donner aux germes qui
// n'etaient pas au niveau grossier, pour que le diagramme fin soit admissible ( aucune cellule
// vide ) et que Newton y ait moins a faire que depuis Voronoi.
//
// Ce qui est compte : par niveau, la prolongation ( graphes, systemes ), les diagrammes d'essai
// ( un par `t` tente ), Newton ( iterations, diagrammes, temps ) ; et le TOTAL contre Newton
// depuis `w = 0` sur le niveau fin ( `--reference` ), a options egales.
//
//   xmake run multiechelle --load ../2d_des_familles/cases/lines5_n100000_s0.005_voronoi.txt --2d --reference
//   xmake run multiechelle -n 100000 --2d --prol mls --corr retrait
// =====================================================================================

#include "bench/Dispatch.h"
#include "solver/Lineaire.h"
#include "solver/Newton.h"
#include "solver/Prolongation.h"
#ifdef _OPENMP
#include <omp.h>
#endif
#include <cstdio>
#include <memory>
#include <string>

using namespace sf;

namespace {

struct Opts {
    NewtonOptions newton;
    std::string   prol   = "harmonique";  ///< copie | harmonique | mls | ctransf
    std::string   corr   = "penal";       ///< penal | retrait | jacobi | rattrape | aucune
    SI            R      = 8;             ///< germes par paquet
    SI            nmin   = 2000;          ///< en dessous, on part de Voronoi
    TF            seuil  = 0.5;           ///< admissible : toute aire >= seuil * min( nu_i, aire min de Voronoi )
                                          ///< ( le plancher de l'amortissement, s'il partait de Voronoi )
    int           lisse_solution = -1;    ///< >= 0 : partir de la SOLUTION du fichier, lissee par K balayages de Jacobi
    int           mls_anneaux = 2;        ///< MLS : anneaux du pochoir
    TF            mls_largeur = 1;        ///< MLS : largeur de la gaussienne, en unites de H
    TF            tolg   = 1e-3;          ///< la tolerance de Newton aux niveaux grossiers
    int           essais = 12;            ///< divisions de `t` au plus
    int           passes = 20;            ///< rattrape : passes au plus
    TF            marge  = 0.1;           ///< rattrape : l'air donne a la cellule relevee, en h_i^2
    bool          reference = false;
};

template<int D>
struct Niveau {
    SI              n = 0;
    std::vector<TF> c[ D ];
    const TF       *P[ D ] = {};
    std::vector<TF> nu, w;
    Paquets         pq;                   ///< vers le niveau grossier ( vide au plus grossier )
    Laplacien       Lvor, Llag;           ///< les graphes de ce niveau : Voronoi, et Laguerre resolu
    // le bilan
    int    it = 0, diag = 0, essais = 0;
    double t_prol = 0, t_test = 0, t_newton = 0;
    TF     t_final = 1, pire_depart = 0;
    const char *fin = "";
};

/// LA BORNE : Newton depuis la SOLUTION ( les poids du fichier ) lissee par `K` balayages de Jacobi
/// sur le graphe de Voronoi. Si meme ce depart-la coute autant que Voronoi, aucune prolongation
/// ne fera mieux.
template<class PD>
int depuis_solution( const Args &a, const Opts &o, const Nuage<PD::dim> &nu0 ) {
    if ( ! nu0.W ) { std::printf( "  pas de poids dans le fichier\n" ); return 1; }
    const SI n = nu0.n;
    PD pd;
    pd.build( nu0.P, nullptr, n, a.leaf );
    Laplacien Lvor;
    std::vector<TF> avor, nu( n, TF( 1 ) / n ), w( nu0.W, nu0.W + n );
    laplacien_de( pd, nu0.P, nullptr, a.par, Lvor, &avor );
    const TF amin_vor = *std::min_element( avor.begin(), avor.end() );
    lisse_jacobi( Lvor, o.lisse_solution, w );
    TF amin, pire;
    SI mauv = teste_admissible( pd, w, nu, o.seuil * amin_vor, a.par, amin, pire );
    std::printf( "  solution lissee par %d balayages : %d cellules sous le plancher, aire min %.2e nu, max|a-nu|/nu %.2e\n",
                 o.lisse_solution, int( mauv ), double( amin ), double( pire ) );
    if ( o.corr == "rattrape" && mauv ) {
        SI releves; int faites;
        const SI reste = rattrape( pd, nu0.P, Lvor, w, o.seuil * amin_vor, o.marge, o.passes, a.par, releves, faites );
        mauv = teste_admissible( pd, w, nu, o.seuil * amin_vor, a.par, amin, pire );
        std::printf( "  rattrapage : %d passes, %d relevements, %d cellules encore sous le plancher, max|a-nu|/nu %.2e\n",
                     faites, int( releves ), int( reste ), double( pire ) );
    }
    Cholesky lin;
    Newton<PD,Cholesky> nw( pd, lin, nu0.P, a.par, o.newton );
    nw.nu = nu;
    const double t0 = now();
    nw.resout( w );
    std::printf( "  newton depuis la solution lissee : %s, %d iterations, %d diagrammes, %.3f s\n",
                 nw.st.fin, nw.st.nb_iter, nw.st.nb_diag, now() - t0 );
    return 0;
}

template<class PD>
int lance( const Args &a, const Opts &o, const Nuage<PD::dim> &nu0 ) {
    constexpr int D = PD::dim;
    const double T0 = now();

    if ( o.lisse_solution >= 0 ) return depuis_solution<PD>( a, o, nu0 );

    // ---- LES NIVEAUX
    std::vector<Niveau<D>> niv( 1 );
    niv[ 0 ].n = nu0.n;
    for ( int d = 0; d < D; ++d ) { niv[ 0 ].c[ d ].assign( nu0.P[ d ], nu0.P[ d ] + nu0.n ); niv[ 0 ].P[ d ] = niv[ 0 ].c[ d ].data(); }
    niv[ 0 ].nu.assign( nu0.n, TF( 1 ) / nu0.n );
    while ( niv.back().n > o.nmin ) {
        Niveau<D> &f = niv.back();
        f.pq = fait_paquets<D>( f.P, f.n, o.R );
        Niveau<D> g;
        g.n = f.pq.nb;
        for ( int d = 0; d < D; ++d ) { g.c[ d ].resize( g.n ); for ( SI k = 0; k < g.n; ++k ) g.c[ d ][ k ] = f.P[ d ][ f.pq.rep[ k ] ]; g.P[ d ] = g.c[ d ].data(); }
        g.nu.assign( g.n, TF( 0 ) );
        for ( SI i = 0; i < f.n; ++i ) g.nu[ f.pq.paquet[ i ] ] += f.nu[ i ];
        niv.push_back( std::move( g ) );
    }
    for ( auto &l : niv )                                // les pointeurs, une fois le tableau stable
        for ( int d = 0; d < D; ++d ) l.P[ d ] = l.c[ d ].data();
    const int NL = int( niv.size() );
    std::printf( "  niveaux :" );
    for ( const auto &l : niv ) std::printf( " %d", int( l.n ) );
    std::printf( "  ( R = %d, prolongation %s, correction %s, seuil %.2g x plancher )\n", int( o.R ), o.prol.c_str(), o.corr.c_str(), double( o.seuil ) );

#ifdef _OPENMP
    omp_set_num_threads( a.par.threads );
#endif

    // ---- DU PLUS GROSSIER AU PLUS FIN
    for ( int l = NL - 1; l >= 0; --l ) {
        Niveau<D> &L = niv[ l ];
        PD pd;
        pd.build( L.P, nullptr, L.n, a.leaf );
        std::vector<TF> w0( L.n, TF( 0 ) );
        double t0 = now();
        if ( l < NL - 1 ) {
            const Niveau<D> &G = niv[ l + 1 ];
            std::vector<TF> base;
            std::vector<TF> avor;
            laplacien_de( pd, L.P, nullptr, a.par, L.Lvor, &avor );
            const TF amin_vor = *std::min_element( avor.begin(), avor.end() );
            SI retombes = 0;
            if ( o.prol == "copie" )           prolonge_copie( L.pq, G.w, base );
            else if ( o.prol == "harmonique" ) prolonge_harmonique( L.Lvor, L.pq, G.w, std::numeric_limits<TF>::infinity(), base );
            else if ( o.prol == "mls" )        retombes = prolonge_mls<D>( L.P, L.pq, G.P, G.Lvor, G.w, a.par, base, o.mls_anneaux, o.mls_largeur );
            else if ( o.prol == "ctransf" )    prolonge_ctransf<D>( L.P, L.pq, G.P, G.Llag, G.w, base );
            else { std::printf( "  prolongation inconnue : %s\n", o.prol.c_str() ); return 1; }
            L.t_prol += now() - t0;
            if ( retombes ) std::printf( "    mls : %d germes retombes sur le lineaire / la copie\n", int( retombes ) );

            // les essais
            TF t = 1; int k = 0;
            for ( int e = 0; e <= o.essais; ++e ) {
                t0 = now();
                if ( o.corr == "rattrape" ) {
                    w0 = base;
                    SI releves; int faites;
                    const SI reste = rattrape( pd, L.P, L.Lvor, w0, o.seuil * amin_vor, o.marge, o.passes, a.par, releves, faites );
                    std::printf( "    rattrapage : %d passes, %d relevements, %d cellules encore sous le plancher\n", faites, int( releves ), int( reste ) );
                    L.essais += faites;
                }
                else if ( e == 0 || o.corr == "aucune" ) w0 = base;
                else if ( o.corr == "penal" ) {
                    if ( o.prol != "harmonique" ) { std::printf( "  --corr penal demande --prol harmonique\n" ); return 1; }
                    prolonge_harmonique( L.Lvor, L.pq, G.w, t / ( 1 - t ), w0 );
                } else if ( o.corr == "retrait" ) { w0 = base; for ( TF &x : w0 ) x *= t; }
                else if ( o.corr == "jacobi" ) { w0 = base; lisse_jacobi( L.Lvor, k, w0 ); }
                else { std::printf( "  correction inconnue : %s\n", o.corr.c_str() ); return 1; }
                L.t_prol += now() - t0;
                t0 = now();
                TF amin, pire;
                const SI mauv = teste_admissible( pd, w0, L.nu, o.seuil * amin_vor, a.par, amin, pire );
                L.t_test += now() - t0;
                ++L.essais;
                std::printf( "    essai %2d  t %.4g%s : %d cellules sous seuil, aire min %.2e nu, max|a-nu|/nu %.2e\n",
                             e, double( t ), o.corr == "jacobi" ? ( " ( " + std::to_string( k ) + " balayages )" ).c_str() : "",
                             int( mauv ), double( amin ), double( pire ) );
                L.t_final = t; L.pire_depart = pire;
                if ( ! mauv || o.corr == "aucune" || o.corr == "rattrape" ) break;
                if ( o.corr == "jacobi" ) k = k ? 2 * k : 1; else t /= 2;
            }
        }

        // Newton
        NewtonOptions no = o.newton;
        no.tol = l ? o.tolg : o.newton.tol;
        Cholesky lin;
        Newton<PD,Cholesky> nw( pd, lin, L.P, a.par, no );
        nw.nu = L.nu;
        t0 = now();
        nw.resout( w0 );
        L.t_newton = now() - t0;
        L.it = nw.st.nb_iter; L.diag = nw.st.nb_diag; L.fin = nw.st.fin;
        L.w = nw.w;
        std::printf( "  niveau %d  n=%-7d %s : %d iterations, %d diagrammes ( reste %.2e ) -- prolongation %.3f | essais %d ( %.3f s ) | newton %.3f s\n",
                     l, int( L.n ), nw.st.fin, L.it, L.diag, double( nw.st.reste ), L.t_prol, L.essais, L.t_test, L.t_newton );
        if ( l > 0 ) {                                   // les graphes dont la prolongation aura besoin
            if ( o.prol == "mls" )     { std::vector<TF> z( L.n, TF( 0 ) ); laplacien_de( pd, L.P, z.data(), a.par, L.Lvor ); }
            if ( o.prol == "ctransf" ) laplacien_de( pd, L.P, nw.w.data(), a.par, L.Llag );
        }
    }
    const double total = now() - T0;
    int it = 0, diag = 0;
    for ( const auto &l : niv ) { it += l.it; diag += l.diag; }
    std::printf( "  MULTI-ECHELLE : %d iterations, %d diagrammes en tout, TOTAL %.3f s ( niveau fin : %d it, %d diag, depart max|a-nu|/nu %.2e a t = %.3g )\n",
                 it, diag, total, niv[ 0 ].it, niv[ 0 ].diag, double( niv[ 0 ].pire_depart ), double( niv[ 0 ].t_final ) );

    if ( o.reference ) {
        const double t0 = now();
        PD pd;
        pd.build( niv[ 0 ].P, nullptr, niv[ 0 ].n, a.leaf );
        Cholesky lin;
        NewtonOptions no = o.newton;
        no.trace = false;
        Newton<PD,Cholesky> nw( pd, lin, niv[ 0 ].P, a.par, no );
        nw.nu = niv[ 0 ].nu;
        nw.resout( std::vector<TF>( niv[ 0 ].n, TF( 0 ) ) );
        std::printf( "  REFERENCE depuis w = 0 : %s, %d iterations, %d diagrammes, TOTAL %.3f s\n",
                     nw.st.fin, nw.st.nb_iter, nw.st.nb_diag, now() - t0 );
    }
    return 0;
}

} // namespace

int main( int argc, char **argv ) {
    Args a;
    a.n = 100000;
    a.dims = 2;
    Opts o;
    o.newton.pas = NewtonOptions::ESSAI_LIMITES;
    o.newton.t_min = 1e-3;
    for ( int i = 1; i < argc; ++i ) {
        const std::string s = argv[ i ];
        auto val = [ & ]() { return i + 1 < argc ? argv[ ++i ] : ""; };
        if ( a.parse( s, i, argc, argv ) ) continue;
        else if ( s == "--prol" )       o.prol = val();
        else if ( s == "--corr" )       o.corr = val();
        else if ( s == "-r" )           o.R = std::atoi( val() );
        else if ( s == "--n-min" )      o.nmin = std::atoi( val() );
        else if ( s == "--seuil" )      o.seuil = std::atof( val() );
        else if ( s == "--tol-grossier" ) o.tolg = std::atof( val() );
        else if ( s == "--essais" )     o.essais = std::atoi( val() );
        else if ( s == "--passes" )     o.passes = std::atoi( val() );
        else if ( s == "--marge" )      o.marge = std::atof( val() );
        else if ( s == "--reference" )  o.reference = true;
        else if ( s == "--lisse-solution" ) o.lisse_solution = std::atoi( val() );
        else if ( s == "--mls-anneaux" ) o.mls_anneaux = std::atoi( val() );
        else if ( s == "--mls-largeur" ) o.mls_largeur = std::atof( val() );
        else if ( s == "--newton-tol" ) o.newton.tol = std::atof( val() );
        else if ( s == "--newton-max" ) o.newton.maxit = std::atoi( val() );
        else if ( s == "--quiet" )      o.newton.trace = false;
        else if ( s == "--t-min" )      o.newton.t_min = std::atof( val() );
        else if ( s == "--pas" ) {
            const std::string v = val();
            o.newton.pas = v == "essais" ? NewtonOptions::ESSAIS : NewtonOptions::ESSAI_LIMITES;
        }
        else {
            std::printf( "usage: multiechelle [options]   ( 2D, Cholesky )\n" );
            Args::usage();
            std::printf(
                "  --prol P        copie | harmonique | mls | ctransf                (harmonique)\n"
                "  --corr C        penal | retrait | jacobi | rattrape | aucune       (penal)\n"
                "  --passes K      rattrape : passes au plus                          (20)\n"
                "  --marge M       rattrape : l'air donne a la cellule relevee, en h_i^2  (0.1)\n"
                "  -r R            germes par paquet                                 (8)\n"
                "  --n-min N       taille du niveau le plus grossier, au plus        (2000)\n"
                "  --seuil S       admissible : toute aire >= S * min( nu_i, aire min de Voronoi )  (0.5)\n"
                "  --mls-anneaux A --mls-largeur W   le pochoir et la gaussienne du MLS    (2, 1)\n"
                "  --lisse-solution K   Newton depuis les poids du fichier lisses par K balayages de Jacobi ( la borne )\n"
                "  --tol-grossier  tolerance de Newton aux niveaux grossiers         (1e-3)\n"
                "  --essais K      divisions de t au plus                            (12)\n"
                "  --reference     Newton depuis w = 0 sur le niveau fin, a options egales\n"
                "  --pas P         essais | essai-limites                            (essai-limites)\n"
                "  --newton-tol T  --newton-max K  --t-min T  --quiet\n" );
            return s == "--help" || s == "-h" ? 0 : 1;
        }
    }
    a.finalise();
    if ( a.dims != 2 ) { std::printf( "2D seulement\n" ); return 1; }

    int bad = 0;
    std::printf( "=== 2D\n" );
    for ( const Nuage<2> &nu : a.nuages<2>() ) {
        if ( nu.absent ) { std::printf( "  %s : ABSENT\n", nu.nom.c_str() ); continue; }
        std::printf( "-- %s\n", nu.nom.c_str() );
        bad += dispatch<2>( a, [ & ]( auto tag ) { return lance<typename decltype( tag )::type>( a, o, nu ); } );
    }
    return bad ? 1 : 0;
}

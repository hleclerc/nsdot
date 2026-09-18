// =====================================================================================
// L'HOMOTOPIE SUR LA PRESCRIPTION, avec de vrais diagrammes : la cible glisse des aires de
// Voronoi vers `1/n`, `nu_s = a0 + s ( nu - a0 )`, et on CONVERGE sur chaque cible intermediaire
// ( tolerance lache ) avant de doubler `s` -- divise par deux si Newton peine. Ni l'amortissement
// ( qui vise toujours la cible finale ), ni la serie ( combinatoire figee ) : le chemin des
// solutions des cibles partielles, suivi a combinatoire vivante. Compte : les iterations de
// Newton en tout, contre la resolution directe.
//
//   xmake run homotopie --load ../2d_des_familles/cases/lines5_n100000_s0.005_equal.txt --solver chol
// =====================================================================================

#include "bench/Dispatch.h"
#include "solver/Lineaire.h"
#include "solver/Newton.h"
#ifdef _OPENMP
#include <omp.h>
#endif
#include <cstdio>
#include <string>

using namespace sf;

namespace {

struct Opts {
    TF  s0     = 1.0 / 64;     ///< le premier `s`
    int it_max = 4;            ///< au-dela, le pas en `s` est divise par deux
    TF  tol_inter = 1e-3;      ///< la tolerance sur les cibles intermediaires
    NewtonOptions newton;
};

template<class PD, class Lin>
int lance( const Args &a, const Opts &o, const Nuage<2> &nu, Lin &lin ) {
    const SI n = nu.n;
    const TF nuv = TF( 1 ) / n;
#ifdef _OPENMP
    omp_set_num_threads( a.par.threads );
#endif
    PD pd;
    pd.build( nu.P, nullptr, n, a.leaf );
    std::vector<TF> a0;
    pd.measures( a0, a.par );                            // les aires de Voronoi : la cible en s = 0

    const double t0 = now();
    std::vector<TF> w( n, TF( 0 ) );
    TF s = 0, ds = o.s0;
    int it_total = 0, diag_total = 0, pas = 0, refus = 0;
    std::printf( "  %4s %9s %9s | %4s %5s %6s | %s\n", "pas", "s", "ds", "it", "diag", "reculs", "fin" );
    while ( s < 1 ) {
        const TF s2 = std::min( TF( 1 ), s + ds );
        NewtonOptions no = o.newton;
        no.tol = s2 < 1 ? o.tol_inter : o.newton.tol;
        Newton<PD,Lin> nw( pd, lin, nu.P, a.par, no );
        nw.nu.resize( n );
        for ( SI i = 0; i < n; ++i ) nw.nu[ i ] = a0[ i ] + s2 * ( nuv - a0[ i ] );
        const bool ok = nw.resout( w );
        it_total += nw.st.nb_iter; diag_total += nw.st.nb_diag;
        const bool pris = ok && nw.st.nb_iter <= o.it_max;
        std::printf( "  %4d %9.6f %9.2e | %4d %5d %6d | %s%s\n", pas, double( s2 ), double( ds ), nw.st.nb_iter, nw.st.nb_diag,
                     nw.st.nb_recul, nw.st.fin, pris ? "" : "  REFUSE" );
        if ( ! pris ) {
            ++refus; ds /= 2;
            if ( ds < 1e-6 ) { std::printf( "  pas trop petit\n" ); return 1; }
            if ( ok ) { w = nw.w; s = s2; ++pas; }        // converge mais cher : on garde, et on ralentit
            continue;
        }
        w = nw.w; s = s2; ++pas;
        if ( nw.st.nb_iter <= 2 ) ds *= 2;
    }
    std::printf( "  TOTAL : %d pas ( %d refus ), %d iterations de Newton, %d diagrammes, %.3f s\n",
                 pas, refus, it_total, diag_total, now() - t0 );
    if ( nu.W ) {
        TF m = 0;
        for ( SI i = 0; i < n; ++i ) m = std::max( m, std::fabs( ( w[ i ] - w[ 0 ] ) - ( nu.W[ i ] - nu.W[ 0 ] ) ) );
        std::printf( "  contre les poids du fichier : ecart max %.3e\n", double( m ) );
    }
    return 0;
}

} // namespace

int main( int argc, char **argv ) {
    Args a;
    a.dims = 2;
    Opts o;
    o.newton.trace = false;
    std::string solver = "chol";
    for ( int i = 1; i < argc; ++i ) {
        const std::string s = argv[ i ];
        auto val = [ & ]() { return i + 1 < argc ? argv[ ++i ] : ""; };
        if ( a.parse( s, i, argc, argv ) ) continue;
        else if ( s == "--s0" )        o.s0 = std::atof( val() );
        else if ( s == "--it-max" )    o.it_max = std::atoi( val() );
        else if ( s == "--tol-inter" ) o.tol_inter = std::atof( val() );
        else if ( s == "--newton-tol" ) o.newton.tol = std::atof( val() );
        else if ( s == "--t-min" )     o.newton.t_min = std::atof( val() );
        else if ( s == "--solver" )    solver = val();
        else if ( s == "--pas" )       o.newton.pas = std::string( val() ) == "facteur" ? NewtonOptions::FACTEUR : NewtonOptions::ESSAIS;
        else if ( s == "--trace" )     o.newton.trace = true;
        else { std::printf( "usage: homotopie --load FILE [--s0 S] [--it-max K] [--tol-inter T] [--solver chol|amg] [--pas essais|facteur] [--t-min T]\n" ); return 1; }
    }
    a.finalise();
    if ( a.load.empty() ) { std::printf( "--load FILE\n" ); return 1; }
    return dispatch<2>( a, [ & ]( auto tag ) {
        using PD = typename decltype( tag )::type;
        for ( const Nuage<2> &nu : a.nuages<2>() ) {
            if ( nu.absent ) return 1;
            std::printf( "-- %s\n", nu.nom.c_str() );
#ifdef SF_EIGEN
            if ( solver == "chol" ) { Cholesky lin; return lance<PD>( a, o, nu, lin ); }
#endif
#ifdef SF_AMGCL
            Amg lin; lin.variante = Amg::RS_GS; return lance<PD>( a, o, nu, lin );
#else
            return 1;
#endif
        }
        return 1;
    } );
}

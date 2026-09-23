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
#include "bench/Trames.h"
#include "solver/Ecrasement.h"
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
    int ordre  = 0;            ///< le predicteur : 0 = `w( s )` tel quel, 2 = `w + w_1 + w_2` ( serie en s )
    std::string dump;          ///< les trames de l'animation ( JSONL )
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
    std::vector<TF> w_pred;
    Trames trames;
    if ( ! o.dump.empty() ) trames.ouvre( o.dump );
    {
        std::vector<TF> nuv0( n, nuv );
        pd.set_weights( w.data(), a.par );
        trames.ecrit( pd, nuv0, a.par, "voronoi", 0, -1, 0, 0 );
    }
    while ( s < 1 ) {
        const TF s2 = std::min( TF( 1 ), s + ds );
        w_pred = w;
        if ( o.ordre >= 2 ) {
            // LE PREDICTEUR D'ORDRE 2 : `w_1 = L^-1 ( nu_s2 - a )`, `w_2 = -L^-1 q2( w_1 )` avec `q2( u ) =
            // a_modele( u ) - a - L u` ( le modele quadratique de chaque cellule, `Ecrasement.h` )
            pd.set_weights( w.data(), a.par );
            std::vector<TF> mes;
            std::vector<std::vector<Facette>> par_th( std::max( a.par.threads, 1 ) );
            pd.measures_and_facets( mes, a.par, [ & ]( int t, SI i, SI j, TF m ) {
                TF d2 = 0; for ( int d = 0; d < 2; ++d ) { const TF e = nu.P[ d ][ j ] - nu.P[ d ][ i ]; d2 += e * e; }
                if ( d2 > 0 ) par_th[ t ].push_back( Facette{ i, j, m / ( 2 * std::sqrt( d2 ) ) } );
            } );
            std::vector<Facette> fa;
            for ( auto &v : par_th ) fa.insert( fa.end(), v.begin(), v.end() );
            Laplacien L; L.assemble( n, fa );
            std::vector<ModeleCellule> mod( n );
            parallel_for( n, a.par, [ & ]( SI k, int ) { typename PD::Cell cel; pd.cellule( k, cel ); mod[ pd.ids[ k ] ].depuis( cel, pd.ids[ k ], nu.P, w.data() ); } );
            std::vector<TF> r( n ), w1, q( n ), w2;
            for ( SI i = 0; i < n; ++i ) r[ i ] = a0[ i ] + s2 * ( nuv - a0[ i ] ) - mes[ i ];
            if ( ! lin.resout( L, r, w1 ) ) return 1;
            parallel_for( n, a.par, [ & ]( SI i, int ) {
                TF lu = L.dia[ i ] * w1[ i ];
                for ( SI k = L.row[ i ]; k < L.row[ i + 1 ]; ++k ) lu -= L.c[ k ] * w1[ L.col[ k ] ];
                q[ i ] = -( mod[ i ].aire( w1.data() ) - mes[ i ] - lu );
            } );
            if ( ! lin.resout( L, q, w2 ) ) return 1;
            // LES TROIS CANDIDATS -- `w`, `w + w_1`, `w + w_1 + w_2` -- juges par un diagramme chacun :
            // l'ordre 2 est excellent quand la combinatoire est calme et toxique devant un pli
            std::vector<TF> cand( n ), res;
            TF meilleur = INFINI;
            int choix = -1;
            for ( int ordre = 0; ordre <= 2; ++ordre ) {
                for ( SI i = 0; i < n; ++i ) cand[ i ] = w[ i ] + ( ordre >= 1 ? w1[ i ] : 0 ) + ( ordre >= 2 ? w2[ i ] : 0 );
                pd.set_weights( cand.data(), a.par );
                pd.measures( res, a.par );
                TF pire = 0;
                for ( SI i = 0; i < n; ++i ) pire = std::max( pire, std::fabs( res[ i ] - ( a0[ i ] + s2 * ( nuv - a0[ i ] ) ) ) / nuv );
                if ( pire < meilleur ) { meilleur = pire; choix = ordre; w_pred = cand; }
            }
            diag_total += 3;
            std::printf( "      predicteur : ordre %d retenu ( max|a-nu_s|/nu = %.2e )\n", choix, double( meilleur ) );
        }
        NewtonOptions no = o.newton;
        no.tol = s2 < 1 ? o.tol_inter : o.newton.tol;
        Newton<PD,Lin> nw( pd, lin, nu.P, a.par, no );
        nw.nu.resize( n );
        for ( SI i = 0; i < n; ++i ) nw.nu[ i ] = a0[ i ] + s2 * ( nuv - a0[ i ] );
        std::vector<TF> nuf( n, nuv );
        nw.o.apres_pas = [ & ]( int it, TF t, int reculs ) {
            trames.ecrit( pd, nuf, a.par, it < 0 ? "palier" : "correcteur", double( s2 ), it, double( t ), reculs );
        };
        const bool ok = nw.resout( w_pred );
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
        else if ( s == "--ordre" )     o.ordre = std::atoi( val() );
        else if ( s == "--dump" )      o.dump = val();
        else if ( s == "--it-max" )    o.it_max = std::atoi( val() );
        else if ( s == "--tol-inter" ) o.tol_inter = std::atof( val() );
        else if ( s == "--newton-tol" ) o.newton.tol = std::atof( val() );
        else if ( s == "--t-min" )     o.newton.t_min = std::atof( val() );
        else if ( s == "--solver" )    solver = val();
        else if ( s == "--pas" ) { const std::string v = val(); o.newton.pas = v == "facteur" ? NewtonOptions::FACTEUR : v == "essai-limites" ? NewtonOptions::ESSAI_LIMITES : NewtonOptions::ESSAIS; }
        else if ( s == "--confiance" ) o.newton.confiance = std::atof( val() );
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

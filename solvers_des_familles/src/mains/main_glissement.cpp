// =====================================================================================
// LE GLISSEMENT : une continuation en POSITIONS, dans le meilleur cas possible.
//
// On connait `w*` ( le fichier `_equal` ). On prend `c` = les barycentres des cellules de la
// solution, on resout en `c` depuis `w = 0` ( le Voronoi des barycentres est-il deja presque
// bon ? ), puis on fait glisser les diracs de `c` vers `p` le long de `p( tau ) = ( 1 - tau ) c
// + tau p`, en suivant `w*( tau )` : prediction tangente
//
//     L dw = nu - a - ( da/dp ) dp,     da_i = sum_j l_ij / |p_j - p_i| [ dp_j . ( p_j - m_ij ) + dp_i . ( m_ij - p_i ) ]
//
// ( la vitesse normale du plan `ij` quand un site bouge, integree sur l'arete de longueur
// `l_ij` et de milieu `m_ij` ), puis Newton en correcteur. On compte les iterations de Newton
// et les cellules qui changent de voisins a chaque pas : c'est le nombre d'epoques
// combinatoires d'un chemin en positions, la seule chose qui decide si la piste vaut.
//
//   xmake run glissement --load ../2d_des_familles/cases/lines5_n2000_s0.005_equal.txt
// =====================================================================================

#include "bench/Dispatch.h"
#include "solver/Lineaire.h"
#include "solver/Newton.h"
#ifdef _OPENMP
#include <omp.h>
#endif
#include <algorithm>
#include <cstdio>
#include <string>

using namespace sf;

namespace {

struct Opts {
    TF  dtau      = 0.1;       ///< le pas initial en tau
    TF  dtau_min  = 1e-4;
    int it_max    = 4;         ///< au-dela, on divise le pas par deux
    int it_double = 1;         ///< en deca, on le double
    std::string predicteur = "tangent";  ///< tangent | secant | aucun
    NewtonOptions newton;
};

/// les barycentres et, pour le predicteur, `da_i` pour un deplacement `dp` des sites
template<class PD>
void barycentres( const PD &pd, const Parallel &par, std::vector<TF> cx[ 2 ] ) {
    const SI n = pd.n;
    cx[ 0 ].assign( n, 0 ); cx[ 1 ].assign( n, 0 );
    parallel_for( n, par, [ & ]( SI k, int ) {
        typename PD::Cell cel;
        pd.cellule( k, cel );
        const SI i = pd.ids[ k ];
        TF a = 0, sx = 0, sy = 0;
        for ( int j = 0, l = cel.nb - 1; j < cel.nb; l = j++ ) {
            const TF cr = TF( cel.vx[ l ] ) * TF( cel.vy[ j ] ) - TF( cel.vx[ j ] ) * TF( cel.vy[ l ] );
            a += cr; sx += ( TF( cel.vx[ l ] ) + TF( cel.vx[ j ] ) ) * cr; sy += ( TF( cel.vy[ l ] ) + TF( cel.vy[ j ] ) ) * cr;
        }
        if ( a != 0 ) { cx[ 0 ][ i ] = sx / ( 3 * a ); cx[ 1 ][ i ] = sy / ( 3 * a ); }
    } );
}

template<class PD>
void da_dp( const PD &pd, const TF *const *X, const std::vector<TF> dp[ 2 ], const Parallel &par, std::vector<TF> &da ) {
    const SI n = pd.n;
    da.assign( n, 0 );
    parallel_for( n, par, [ & ]( SI k, int ) {
        typename PD::Cell cel;
        pd.cellule( k, cel );
        const SI i = pd.ids[ k ];
        TF s = 0;
        for ( int j = 0, l = cel.nb - 1; j < cel.nb; l = j++ ) {  // l'arete [ v_l, v_j ] portee par cid[ l ]
            const SI id = cel.cid[ l ];
            if ( id < 0 ) continue;
            const TF ex = TF( cel.vx[ j ] ) - TF( cel.vx[ l ] ), ey = TF( cel.vy[ j ] ) - TF( cel.vy[ l ] );
            const TF lg = std::sqrt( ex * ex + ey * ey );
            const TF mx = TF( 0.5 ) * ( TF( cel.vx[ j ] ) + TF( cel.vx[ l ] ) ), my = TF( 0.5 ) * ( TF( cel.vy[ j ] ) + TF( cel.vy[ l ] ) );
            const TF nx = X[ 0 ][ id ] - X[ 0 ][ i ], ny = X[ 1 ][ id ] - X[ 1 ][ i ];
            const TF nn = std::sqrt( nx * nx + ny * ny );
            if ( ! ( nn > 0 ) ) continue;
            s += lg / nn * ( dp[ 0 ][ id ] * ( X[ 0 ][ id ] - mx ) + dp[ 1 ][ id ] * ( X[ 1 ][ id ] - my )
                           + dp[ 0 ][ i ] * ( mx - X[ 0 ][ i ] ) + dp[ 1 ][ i ] * ( my - X[ 1 ][ i ] ) );
        }
        da[ i ] = s;
    } );
}

/// les voisinages ( tries ), pour compter les cellules qui changent de voisins
template<class PD>
void voisinages( const PD &pd, const Parallel &par, std::vector<std::vector<SI>> &v ) {
    const SI n = pd.n;
    v.assign( n, {} );
    parallel_for( n, par, [ & ]( SI k, int ) {
        typename PD::Cell cel;
        pd.cellule( k, cel );
        auto &o = v[ pd.ids[ k ] ];
        for ( int j = 0; j < cel.nb; ++j ) o.push_back( cel.cid[ j ] );
        std::sort( o.begin(), o.end() );
    } );
}

template<class PD, class Lin>
int lance( const Args &a, const Opts &o, const Nuage<2> &nu, Lin &lin ) {
    const SI n = nu.n;
    const Parallel &par = a.par;
    const TF nuv = TF( 1 ) / n;
#ifdef _OPENMP
    omp_set_num_threads( par.threads );
#endif
    if ( ! nu.W ) { std::printf( "  il faut un nuage qui porte sa solution ( _equal )\n" ); return 1; }

    // ---- la solution, et ses barycentres
    PD pd;
    pd.build( nu.P, nullptr, n, a.leaf );
    pd.set_weights( nu.W, par );
    std::vector<TF> mes;
    pd.measures( mes, par );
    TF pire = 0;
    for ( SI i = 0; i < n; ++i ) pire = std::max( pire, std::fabs( mes[ i ] - nuv ) / nuv );
    std::vector<TF> c[ 2 ];
    barycentres( pd, par, c );
    TF dmax = 0, dmoy = 0;
    for ( SI i = 0; i < n; ++i ) {
        const TF dx = c[ 0 ][ i ] - nu.P[ 0 ][ i ], dy = c[ 1 ][ i ] - nu.P[ 1 ][ i ], dd = std::sqrt( dx * dx + dy * dy );
        dmax = std::max( dmax, dd ); dmoy += dd;
    }
    std::printf( "  solution du fichier : max|a-nu|/nu = %.2e ; barycentre -> dirac : moyenne %.3e, max %.3e ( h = %.3e )\n",
                 double( pire ), double( dmoy / n ), double( dmax ), double( std::sqrt( nuv ) ) );

    // ---- tau = 0 : les barycentres, depuis w = 0
    std::vector<TF> X[ 2 ] = { c[ 0 ], c[ 1 ] };
    const TF *XP[ 2 ] = { X[ 0 ].data(), X[ 1 ].data() };
    double t_total = now();
    int it_total = 0, diag_total = 0;
    std::vector<TF> w;
    {
        PD pd0;
        pd0.build( XP, nullptr, n, a.leaf );
        Newton<PD,Lin> nw( pd0, lin, XP, par, o.newton );
        nw.nu.assign( n, nuv );
        const bool ok = nw.resout( std::vector<TF>( n, TF( 0 ) ) );
        std::printf( "  tau = 0 ( barycentres ), depuis Voronoi : %s, %d iterations, %d diagrammes ( %d reculs )\n",
                     nw.st.fin, nw.st.nb_iter, nw.st.nb_diag, nw.st.nb_recul );
        if ( ! ok ) return 1;
        w = nw.w;
        it_total += nw.st.nb_iter; diag_total += nw.st.nb_diag;
    }

    // ---- la continuation
    std::vector<std::vector<SI>> vois_prec, vois;
    {
        PD pdc; pdc.build( XP, nullptr, n, a.leaf ); pdc.set_weights( w.data(), par );
        voisinages( pdc, par, vois_prec );
    }
    TF tau = 0, dtau = o.dtau;
    int pas = 0, refus = 0;
    SI chang_total = 0;
    std::vector<TF> dp[ 2 ], da, b, dw, dw_prec, w_pred( n ), Xn[ 2 ];
    TF dtau_prec = 1;
    std::vector<Facette> fa;
    Laplacien L;
    std::printf( "  %6s %9s %9s | %4s %5s %6s | %8s %8s %9s\n", "pas", "tau", "dtau", "it", "diag", "reculs", "voisins", "|dw|inf", "res.pred" );
    while ( tau < 1 ) {
        const TF tau2 = std::min( TF( 1 ), tau + dtau );
        for ( int d = 0; d < 2; ++d ) {
            Xn[ d ].resize( n ); dp[ d ].resize( n );
            for ( SI i = 0; i < n; ++i ) { Xn[ d ][ i ] = ( 1 - tau2 ) * c[ d ][ i ] + tau2 * nu.P[ d ][ i ]; dp[ d ][ i ] = Xn[ d ][ i ] - X[ d ][ i ]; }
        }
        // le predicteur, sur l'etat courant ( X, w )
        PD pdx;
        pdx.build( XP, nullptr, n, a.leaf );
        pdx.set_weights( w.data(), par );
        std::vector<std::vector<Facette>> par_th( std::max( par.threads, 1 ) );
        pdx.measures_and_facets( mes, par, [ & ]( int t, SI i, SI j, TF m ) {
            TF d2 = 0; for ( int d = 0; d < 2; ++d ) { const TF e = X[ d ][ j ] - X[ d ][ i ]; d2 += e * e; }
            if ( d2 > 0 ) par_th[ t ].push_back( Facette{ i, j, m / ( 2 * std::sqrt( d2 ) ) } );
        } );
        fa.clear(); for ( auto &v : par_th ) fa.insert( fa.end(), v.begin(), v.end() );
        L.assemble( n, fa );
        da_dp( pdx, XP, dp, par, da );
        b.resize( n );
        for ( SI i = 0; i < n; ++i ) b[ i ] = nuv - mes[ i ] - da[ i ];
        if ( ! lin.resout( L, b, dw ) ) { std::printf( "  predicteur : solveur en echec\n" ); return 1; }
        if ( o.predicteur == "aucun" ) std::fill( dw.begin(), dw.end(), TF( 0 ) );
        else if ( o.predicteur == "secant" ) {
            if ( dw_prec.empty() ) std::fill( dw.begin(), dw.end(), TF( 0 ) );
            else for ( SI i = 0; i < n; ++i ) dw[ i ] = dw_prec[ i ] * ( tau2 - tau ) / dtau_prec;
        }
        TF dwmax = 0;
        for ( SI i = 0; i < n; ++i ) { w_pred[ i ] = w[ i ] + dw[ i ]; dwmax = std::max( dwmax, std::fabs( dw[ i ] ) ); }
        // la qualite du predicteur : le residu aux positions nouvelles, avant correction
        TF res_pred = 0;
        {
            const TF *XnP[ 2 ] = { Xn[ 0 ].data(), Xn[ 1 ].data() };
            PD pdp; pdp.build( XnP, nullptr, n, a.leaf ); pdp.set_weights( w_pred.data(), par );
            std::vector<TF> mp; pdp.measures( mp, par );
            for ( SI i = 0; i < n; ++i ) res_pred = std::max( res_pred, std::fabs( mp[ i ] - nuv ) / nuv );
        }

        // le correcteur, aux positions nouvelles
        const TF *XnP[ 2 ] = { Xn[ 0 ].data(), Xn[ 1 ].data() };
        PD pdn;
        pdn.build( XnP, nullptr, n, a.leaf );
        Newton<PD,Lin> nw( pdn, lin, XnP, par, o.newton );
        nw.nu.assign( n, nuv );
        const bool ok = nw.resout( w_pred );
        it_total += nw.st.nb_iter; diag_total += nw.st.nb_diag;
        if ( ! ok || nw.st.nb_iter > o.it_max ) {
            ++refus;
            std::printf( "  %6d %9.5f %9.2e | %4d %5d %6d | REFUSE ( %s )\n", pas, double( tau2 ), double( dtau ), nw.st.nb_iter, nw.st.nb_diag, nw.st.nb_recul, nw.st.fin );
            dtau /= 2;
            if ( dtau < o.dtau_min ) { std::printf( "  pas trop petit\n" ); return 1; }
            continue;
        }
        pdn.set_weights( nw.w.data(), par );
        voisinages( pdn, par, vois );
        SI chang = 0;
        for ( SI i = 0; i < n; ++i ) chang += vois[ i ] != vois_prec[ i ];
        chang_total += chang;
        std::printf( "  %6d %9.5f %9.2e | %4d %5d %6d | %8d %8.2e %9.2e\n", pas, double( tau2 ), double( dtau ), nw.st.nb_iter, nw.st.nb_diag, nw.st.nb_recul, int( chang ), double( dwmax ), double( res_pred ) );
        ++pas;
        dw_prec.resize( n );
        for ( SI i = 0; i < n; ++i ) dw_prec[ i ] = nw.w[ i ] - w[ i ];
        dtau_prec = tau2 - tau;
        tau = tau2;
        w = nw.w;
        X[ 0 ] = Xn[ 0 ]; X[ 1 ] = Xn[ 1 ];
        vois_prec.swap( vois );
        if ( nw.st.nb_iter <= o.it_double ) dtau *= 2;
    }
    std::printf( "  TOTAL : %d pas ( %d refus ), %d iterations de Newton, %d diagrammes, %d changements de voisinage ( %.2f par cellule ), %.3f s\n",
                 pas, refus, it_total, diag_total, int( chang_total ), double( chang_total ) / n, now() - t_total );
    TF ec = 0;
    for ( SI i = 0; i < n; ++i ) ec = std::max( ec, std::fabs( ( w[ i ] - w[ 0 ] ) - ( nu.W[ i ] - nu.W[ 0 ] ) ) );
    std::printf( "  contre les poids du fichier : ecart max %.3e\n", double( ec ) );
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
        else if ( s == "--dtau" )      o.dtau = std::atof( val() );
        else if ( s == "--predicteur" ) o.predicteur = val();
        else if ( s == "--it-max" )    o.it_max = std::atoi( val() );
        else if ( s == "--solver" )    solver = val();
        else if ( s == "--trace" )     o.newton.trace = true;
        else if ( s == "--pas" )       o.newton.pas = std::string( val() ) == "facteur" ? NewtonOptions::FACTEUR : NewtonOptions::ESSAIS;
        else if ( s == "--t-min" )     o.newton.t_min = std::atof( val() );
        else if ( s == "--newton-tol" ) o.newton.tol = std::atof( val() );
        else {
            std::printf( "usage: glissement --load FILE_equal [options]\n" );
            Args::usage();
            std::printf( "  --dtau D        pas initial en tau ( 0.1 )\n  --it-max K      au-dela, le pas est refuse ( 4 )\n"
                         "  --solver S      chol ( defaut ) | amg\n  --pas P         essais | facteur\n  --trace         la trace de Newton\n" );
            return 1;
        }
    }
    a.finalise();
    if ( a.load.empty() ) { std::printf( "--load FILE_equal\n" ); return 1; }
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

#pragma once

// =====================================================================================
// L'ENTREE DU SOLVEUR ( ce que `OtPlan.py` appelle, en UN `driver.call` ) : le point de depart,
// Newton, et ce qui en sort.
//
// = Le point de depart
//
// Newton ( KMT ) demande un depart ADMISSIBLE : aucune cellule vide. On part des poids donnes
// ( `w0` : les poids d'un ajustement voisin, ce dont vit une reconstruction ) ou de zero -- le
// Voronoi, dont aucune cellule n'est vide tant que chaque germe est dans le domaine. Un depart
// chaud qui VIDE une cellule ( des poids herites d'autres positions ) est pire que le Voronoi : la
// theorie part d'un plancher strictement positif, et une cellule vide y revient en rampant ( son
// gradient est constant, sa ligne de hessienne nulle ). On repart alors de zero si c'est mieux.
// ( essaye, et rejete, deux facons de faire mieux que le Voronoi : ne « ranimer » que les cellules
// vides, et un depart multi-echelle -- les deux cascadent, voir `solvers_des_familles` README § 8. )
//
// Et le Voronoi lui-meme peut laisser des cellules vides ( des germes HORS du domaine ) : alors les
// poids d'une SIMILITUDE qui ramene le nuage dans le domaine -- le Voronoi du nuage translate et
// contracte s'ecrit comme un diagramme de puissance du nuage d'origine, `w_i = |p_i|^2 - |a p_i +
// b|^2 / a`, et ses cellules sont toutes nourries.
//
// = La masse cible
//
// Les cellules PARTITIONNENT le domaine : leurs masses somment a la masse de la densite dans le
// domaine, quels que soient les poids. Une densite que le domaine tronque ( des gaussiennes dans
// une boite ) n'y pese pas 1 : les masses cibles `nu` sont remises a cette echelle, sans quoi le
// residu ne peut jamais s'annuler. Ce qui sort est le transport vers la densite RESTREINTE au
// domaine, normalisee -- ce qu'on veut dire quand on donne un domaine.
// =====================================================================================

#include "Limites.h"
#include "Newton.h"

namespace sdot {
namespace otplan {

/// ce que l'appelant lit dans `stats( . )` -- meme liste cote python ( `OtPlan._STATS` )
enum Stat : int {
    FIN = 0, RESTE, RESTE0, NB_ITER, NB_DIAG, NB_RECUL, T_MAJ, T_DIAG, T_ASM, T_LIN, T_LIM, EPS,
    MASSE_DOMAINE, NB_DEBORDE, NB_CELL_LIM, NB_TOURS_ESSAI, LIN_NB_HIER, LIN_NB_ITER, LIN_PIRE, DEPART, T_TOTAL,
    NB_STATS
};
enum Depart : int { DEPART_DONNE = 0, DEPART_VORONOI = 1, DEPART_SIMILITUDE = 2 };

/// ce que chaque ligne de l'historique porte -- meme liste cote python ( `OtPlan._HISTORY` )
enum Hist : int { H_STEP = 0, H_T, H_RESIDU_L2, H_MIN_MASSE, H_MAX_RESIDU, H_NB_DIAG, H_NB_EVALS, NB_HIST };

/// les poids du Voronoi d'une SIMILITUDE du nuage qui le loge dans le pave `[ lo, hi ]` : la boite
/// du nuage est contractee ( jamais dilatee ) et translatee dans le pave reduit d'une marge
template<int D>
void similitude( const auto &pd, const double *lo, const double *hi, std::vector<double> &w, double marge = 0.1 ) {
    const SI n = pd.nb_seeds();
    double p_lo[ D ], p_hi[ D ];
    for ( int d = 0; d < D; ++d ) { p_lo[ d ] = 1e300; p_hi[ d ] = -1e300; }
    for ( SI k = 0; k < n; ++k ) {
        const auto p = pd.point( k );
        for ( int d = 0; d < D; ++d ) { p_lo[ d ] = std::min( p_lo[ d ], double( p[ d ] ) ); p_hi[ d ] = std::max( p_hi[ d ], double( p[ d ] ) ); }
    }
    double a = 1;
    for ( int d = 0; d < D; ++d ) {
        const double span_dom = ( hi[ d ] - lo[ d ] ) * ( 1 - 2 * marge );
        const double span_pts = std::max( p_hi[ d ] - p_lo[ d ], 1e-300 );
        a = std::min( a, span_dom / span_pts );
    }
    double b[ D ];
    for ( int d = 0; d < D; ++d )                        // le centre du nuage contracte sur le centre du domaine
        b[ d ] = ( lo[ d ] + hi[ d ] ) / 2 - a * ( p_lo[ d ] + p_hi[ d ] ) / 2;
    w.assign( n, 0.0 );
    for ( SI k = 0; k < n; ++k ) {
        const auto p = pd.point( k );
        double pp = 0, qq = 0;
        for ( int d = 0; d < D; ++d ) {
            const double q = a * double( p[ d ] ) + b[ d ];
            pp += double( p[ d ] ) * double( p[ d ] );
            qq += q * q;
        }
        w[ pd.user_id( k ) ] = pp - qq / a;
    }
}

/// LE SOLVEUR. `pd` porte des poids et des majorants INSCRIPTIBLES ( `with_weights` ) ; `nu` et `w0`
/// sont dans l'ordre utilisateur ( `w0` peut etre `NoneTensor` : zero ). `weights` ( ordre
/// utilisateur ), `hist` ( `nb_steps`, `rows [ step, NB_HIST ]`, `weights [ step, n ]` facultatif ) et
/// `stats` sont les sorties.
template<class TK>
void resoudre( const CpuQueue &queue, auto &pd, const auto &dom, const auto &dist, const auto &nu_in, const auto &w0_in,
               const NewtonOptions &o, Lin lin_methode, SI cap0, auto &&weights, auto &&hist, auto &&stats ) {
    using PD = DECAYED_TYPE_OF( pd );
    constexpr int D = PD::ct_dim;
    const SI n = pd.nb_seeds();
    const double t_debut = now();

    Balayage<PD,DECAYED_TYPE_OF( dom ),DECAYED_TYPE_OF( dist ),TK> bal( queue, pd, dom, dist, cap0 );
    auto lin = solveur_lineaire( lin_methode, n, D );
    Newton<decltype( bal )> newton( bal, *lin, o );

    // ---- la cible, et le depart
    std::vector<double> nu( n ), w( n, 0.0 );
    for ( SI i = 0; i < n; ++i ) nu[ i ] = double( nu_in( i ) );
    bool donne = false;
    if constexpr ( CT_VALUE( w0_in.is_valid() ) ) {
        for ( SI i = 0; i < n; ++i ) { w[ i ] = double( w0_in( i ) ); donne |= w[ i ] != 0; }
    }
    int depart = donne ? DEPART_DONNE : DEPART_VORONOI;
    const double jauge = w[ 0 ];
    for ( SI i = 0; i < n; ++i ) w[ i ] -= jauge;

    std::vector<double> a;
    std::vector<Facette> fa;
    newton.mesures_et_facettes( w, a, fa );
    auto minimum = []( const std::vector<double> &v ) { double m = v.empty() ? 0 : v[ 0 ]; for ( double x : v ) m = std::min( m, x ); return m; };
    double nu_min = minimum( nu );
    if ( donne && minimum( a ) < 1e-3 * nu_min ) {   // un depart chaud qui vide une cellule : le Voronoi, s'il fait mieux
        std::vector<double> w0( n, 0.0 ), a0;
        std::vector<Facette> fa0;
        newton.mesures_et_facettes( w0, a0, fa0 );
        if ( minimum( a0 ) > minimum( a ) ) { w.swap( w0 ); a.swap( a0 ); fa.swap( fa0 ); depart = DEPART_VORONOI; }
        else newton.bal.set_weights( w );
    }
    if constexpr ( CT_VALUE( pd.box_min.is_valid() ) ) {
        if ( minimum( a ) <= 0 ) {                   // des germes hors du domaine : la similitude
            double lo[ D ], hi[ D ];
            for ( int d = 0; d < D; ++d ) { lo[ d ] = double( pd.box_min( d ) ); hi[ d ] = double( pd.box_max( d ) ); }
            std::vector<double> w1, a1;
            std::vector<Facette> fa1;
            similitude<D>( pd, lo, hi, w1 );
            newton.mesures_et_facettes( w1, a1, fa1 );
            if ( minimum( a1 ) > minimum( a ) ) { w.swap( w1 ); a.swap( a1 ); fa.swap( fa1 ); depart = DEPART_SIMILITUDE; }
            else newton.bal.set_weights( w );
        }
    }

    // ---- la masse cible, a l'echelle de ce que le domaine contient
    double masse_dom = 0, masse_nu = 0;
    for ( SI i = 0; i < n; ++i ) { masse_dom += a[ i ]; masse_nu += nu[ i ]; }
    if ( masse_dom > 0 && masse_nu > 0 && masse_dom != masse_nu )
        for ( SI i = 0; i < n; ++i ) nu[ i ] *= masse_dom / masse_nu;
    newton.nu = nu;
    newton.a = a;
    newton.fa = fa;

    // ---- l'historique, une ligne par pas accepte
    SI nb_steps = 0;
    const SI cap_steps = SI( hist.rows.shape( 0 ) );
    newton.o.apres_pas = [&]( int it, double t, int nb_evals ) {
        if ( nb_steps >= cap_steps ) return;
        const auto &A = newton.a;
        double mn = A.empty() ? 0 : A[ 0 ], mx = 0, l2 = 0;
        for ( SI i = 0; i < n; ++i ) {
            mn = std::min( mn, A[ i ] );
            mx = std::max( mx, std::fabs( A[ i ] - nu[ i ] ) );
            l2 += ( A[ i ] - nu[ i ] ) * ( A[ i ] - nu[ i ] );
        }
        hist.rows( nb_steps, int( H_STEP ) ) = double( it );
        hist.rows( nb_steps, int( H_T ) ) = t;
        hist.rows( nb_steps, int( H_RESIDU_L2 ) ) = std::sqrt( l2 );
        hist.rows( nb_steps, int( H_MIN_MASSE ) ) = mn;
        hist.rows( nb_steps, int( H_MAX_RESIDU ) ) = mx;
        hist.rows( nb_steps, int( H_NB_DIAG ) ) = double( bal.nb_diag );
        hist.rows( nb_steps, int( H_NB_EVALS ) ) = double( nb_evals );
        if constexpr ( CT_VALUE( hist.weights.is_valid() ) )
            for ( SI i = 0; i < n; ++i )
                hist.weights( nb_steps, i ) = newton.w[ i ];
        ++nb_steps;
    };

    // ---- le pas par les limites ( 2D ) : la passe est branchee sur Newton quand elle est demandee
    if constexpr ( D == 2 ) {
        Limites2D<decltype( bal )> lim( bal );
        LimitesLocales ll;
        ll.alpha_min = [&]( const std::vector<double> &W, const std::vector<double> &Dd, const std::vector<SI> &mauvaises,
                            double horizon, double eps, const Laplacien &L, SI &nb_cellules ) {
            return lim.alpha_min( W, Dd, mauvaises, horizon, eps, L, nb_cellules );
        };
        if ( o.pas == NewtonOptions::ESSAI_LIMITES )
            newton.limites = &ll;
        newton.resout( w, true );
    } else
        newton.resout( w, true );
    hist.nb_steps.set( nb_steps );

    for ( SI i = 0; i < n; ++i )
        weights( i ) = newton.w[ i ];

    auto put = [&]( int i, double v ) { stats( i ) = v; };
    put( FIN, double( newton.st.fin ) );
    put( RESTE, newton.st.reste );
    put( RESTE0, newton.st.reste0 );
    put( NB_ITER, double( newton.st.nb_iter ) );
    put( NB_DIAG, double( bal.nb_diag ) );
    put( NB_RECUL, double( newton.st.nb_recul ) );
    put( T_MAJ, bal.t_maj );
    put( T_DIAG, bal.t_diag );
    put( T_ASM, newton.st.t_asm );
    put( T_LIN, lin->st.total() );
    put( T_LIM, newton.st.t_lim );
    put( EPS, newton.st.eps );
    put( MASSE_DOMAINE, masse_dom );
    put( NB_DEBORDE, double( bal.nb_deborde ) );
    put( NB_CELL_LIM, double( newton.st.nb_cell_lim ) );
    put( NB_TOURS_ESSAI, double( newton.st.nb_tours_essai ) );
    put( LIN_NB_HIER, double( lin->st.nb_hier ) );
    put( LIN_NB_ITER, double( lin->st.nb_iter ) );
    put( LIN_PIRE, lin->st.pire );
    put( DEPART, double( depart ) );
    put( T_TOTAL, now() - t_debut );
}

} // namespace otplan
} // namespace sdot

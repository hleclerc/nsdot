#pragma once

// =====================================================================================
// L'ENTREE DU SOLVEUR ( ce que `OtPlan.py` appelle, en UN `driver.call` ) : le point de depart, la
// continuation en largeur s'il en faut une, Newton a chaque etape, et ce qui en sort.
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
// = La continuation en largeur ( `Continuation.h` )
//
// Reste le cas ou des cellules n'ont pas de masse parce que la DENSITE n'en a pas la ou elles sont
// ( des bosses etroites, des deserts ) : ni un depart geometrique ni l'amortissement n'y peuvent
// rien, et Newton direct stagne ( README § 9.1 ). On resout alors d'abord pour la densite convolee
// par une gaussienne large ( positive partout ), puis de plus en plus etroite, chaque etape partant
// des poids de la precedente. `AUTO` la declenche quand le meilleur depart laisse encore une
// cellule sous `seuil_continuation` fois la plus petite masse cible.
//
// = La masse cible
//
// Les cellules PARTITIONNENT le domaine : leurs masses somment a la masse de la densite dans le
// domaine, quels que soient les poids. Une densite que le domaine tronque ( des gaussiennes dans
// une boite ) n'y pese pas 1 : les masses cibles `nu` sont remises a cette echelle, a chaque etape,
// sans quoi le residu ne peut jamais s'annuler. Ce qui sort est le transport vers la densite
// RESTREINTE au domaine, normalisee -- ce qu'on veut dire quand on donne un domaine.
// =====================================================================================

#include "Continuation.h"
#include "Limites.h"
#include "Newton.h"

namespace sdot {
namespace otplan {

/// ce que l'appelant lit dans `stats( . )` -- meme liste cote python ( `OtPlan._STATS` )
enum Stat : int {
    FIN = 0, RESTE, RESTE0, NB_ITER, NB_DIAG, NB_RECUL, T_MAJ, T_DIAG, T_ASM, T_LIN, T_LIM, EPS,
    MASSE_DOMAINE, NB_DEBORDE, NB_CELL_LIM, NB_TOURS_ESSAI, LIN_NB_HIER, LIN_NB_ITER, LIN_PIRE, DEPART, T_TOTAL,
    NB_ETAPES, MIN_MASSE_DEPART,
    NB_STATS
};
enum Depart : int { DEPART_DONNE = 0, DEPART_VORONOI = 1, DEPART_SIMILITUDE = 2 };

/// ce que chaque ligne de l'historique porte -- meme liste cote python ( `OtPlan._HISTORY` )
enum Hist : int { H_STEP = 0, H_T, H_RESIDU_L2, H_MIN_MASSE, H_MAX_RESIDU, H_NB_DIAG, H_NB_EVALS, H_S, NB_HIST };

struct OptionsSolveur {
    NewtonOptions newton;
    Lin    lin = Lin::AUTO;
    SI     cap0 = 64;                ///< sommets par cellule locale, au depart
    enum Continuation : int { JAMAIS = 0, AUTO = 1, TOUJOURS = 2 };
    int    continuation = AUTO;
    double seuil_continuation = 1e-2; ///< AUTO : une cellule sous ce facteur de la plus petite cible la declenche
    double conv_s0 = 0;              ///< la premiere largeur ( 0 : la moitie du diametre du domaine )
    double conv_ratio = 1.4142135623730951;
    double conv_min = 0;             ///< la derniere largeur avant 0 ( 0 : l'echelle de la distribution )
};

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

inline double minimum( const std::vector<double> &v ) {
    double m = v.empty() ? 0 : v[ 0 ];
    for ( double x : v ) m = std::min( m, x );
    return m;
}

/// LE SOLVEUR. `pd` porte des poids et des majorants INSCRIPTIBLES ( `with_weights` ) ; `nu` et `w0`
/// sont dans l'ordre utilisateur. `weights` ( ordre utilisateur ), `hist` ( `nb_steps`, `rows [ step,
/// NB_HIST ]`, `weights [ step, n ]` facultatif ) et `stats` sont les sorties.
template<class TK>
void resoudre( const CpuQueue &queue, auto &pd, const auto &dom, const auto &dist, const auto &nu_in, const auto &w0_in,
               const OptionsSolveur &o, auto &&weights, auto &&hist, auto &&stats ) {
    using PD = DECAYED_TYPE_OF( pd );
    using Dist = DECAYED_TYPE_OF( dist );
    constexpr int D = PD::ct_dim;
    const SI n = pd.nb_seeds();
    const double t_debut = now();

    Convolee<Dist> conv( dist );
    Balayage<PD,DECAYED_TYPE_OF( dom ),Dist,TK> bal( queue, pd, dom, dist, o.cap0 );
    auto lin = solveur_lineaire( o.lin, n, D );
    Newton<decltype( bal )> newton( bal, *lin, o.newton );

    // ---- l'historique, une ligne par pas accepte
    SI nb_steps = 0;
    int nb_steps_avant = 0;                              // les pas des etapes precedentes
    double s_courant = 0;
    const SI cap_steps = SI( hist.rows.shape( 0 ) );
    newton.o.apres_pas = [&]( int it, double t, int nb_evals ) {
        if ( nb_steps >= cap_steps ) return;
        const auto &A = newton.a;
        double mn = A.empty() ? 0 : A[ 0 ], mx = 0, l2 = 0;
        for ( SI i = 0; i < n; ++i ) {
            mn = std::min( mn, A[ i ] );
            mx = std::max( mx, std::fabs( A[ i ] - newton.nu[ i ] ) );
            l2 += ( A[ i ] - newton.nu[ i ] ) * ( A[ i ] - newton.nu[ i ] );
        }
        hist.rows( nb_steps, int( H_STEP ) ) = double( nb_steps_avant + it );
        hist.rows( nb_steps, int( H_T ) ) = t;
        hist.rows( nb_steps, int( H_RESIDU_L2 ) ) = std::sqrt( l2 );
        hist.rows( nb_steps, int( H_MIN_MASSE ) ) = mn;
        hist.rows( nb_steps, int( H_MAX_RESIDU ) ) = mx;
        hist.rows( nb_steps, int( H_NB_DIAG ) ) = double( bal.nb_diag );
        hist.rows( nb_steps, int( H_NB_EVALS ) ) = double( nb_evals );
        hist.rows( nb_steps, int( H_S ) ) = s_courant;
        if constexpr ( CT_VALUE( hist.weights.is_valid() ) )
            for ( SI i = 0; i < n; ++i )
                hist.weights( nb_steps, i ) = newton.w[ i ];
        ++nb_steps;
    };

    // ---- la cible, et le depart ( sur la densite la plus large si la continuation est imposee )
    std::vector<double> nu( n ), w( n, 0.0 );
    for ( SI i = 0; i < n; ++i ) nu[ i ] = double( nu_in( i ) );
    bool donne = false;
    if constexpr ( CT_VALUE( w0_in.is_valid() ) ) {
        for ( SI i = 0; i < n; ++i ) { w[ i ] = double( w0_in( i ) ); donne |= w[ i ] != 0; }
    }
    int depart = donne ? DEPART_DONNE : DEPART_VORONOI;
    const double jauge = w[ 0 ];
    for ( SI i = 0; i < n; ++i ) w[ i ] -= jauge;

    // les etapes de la continuation : la liste des largeurs, `0` en dernier
    double s0 = o.conv_s0;
    if ( s0 <= 0 ) {                                     // la moitie du diametre du domaine, ou du nuage
        double lo[ D ], hi[ D ];
        for ( int d = 0; d < D; ++d ) { lo[ d ] = 1e300; hi[ d ] = -1e300; }
        if constexpr ( CT_VALUE( pd.box_min.is_valid() ) ) {
            for ( int d = 0; d < D; ++d ) { lo[ d ] = double( pd.box_min( d ) ); hi[ d ] = double( pd.box_max( d ) ); }
        } else {
            for ( SI k = 0; k < n; ++k ) {
                const auto p = pd.point( k );
                for ( int d = 0; d < D; ++d ) { lo[ d ] = std::min( lo[ d ], double( p[ d ] ) ); hi[ d ] = std::max( hi[ d ], double( p[ d ] ) ); }
            }
        }
        double diam2 = 0;
        for ( int d = 0; d < D; ++d ) diam2 += ( hi[ d ] - lo[ d ] ) * ( hi[ d ] - lo[ d ] );
        s0 = 0.5 * std::sqrt( diam2 );
    }
    const double s_min = o.conv_min > 0 ? o.conv_min : conv.echelle_min( s0 );
    std::vector<double> liste = ( o.continuation == OptionsSolveur::TOUJOURS && Convolee<Dist>::possible ) ? etapes( s0, o.conv_ratio, s_min )
                                                                                                          : std::vector<double>{ 0.0 };

    std::vector<double> a;
    std::vector<Facette> fa;
    s_courant = liste[ 0 ];
    bal.dist = &conv.at( s_courant );
    newton.mesures_et_facettes( w, a, fa );
    const double nu_min = minimum( nu );
    if ( donne && minimum( a ) < 1e-3 * nu_min ) {       // un depart chaud qui vide une cellule : le Voronoi, s'il fait mieux
        std::vector<double> w0( n, 0.0 ), a0;
        std::vector<Facette> fa0;
        newton.mesures_et_facettes( w0, a0, fa0 );
        if ( minimum( a0 ) > minimum( a ) ) { w.swap( w0 ); a.swap( a0 ); fa.swap( fa0 ); depart = DEPART_VORONOI; }
        else newton.bal.set_weights( w );
    }
    if constexpr ( CT_VALUE( pd.box_min.is_valid() ) ) {
        if ( minimum( a ) <= 0 ) {                       // des germes hors du domaine : la similitude
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
    const double min_masse_depart = minimum( a );
    // AUTO : la densite manque la ou sont des cellules -> la continuation, depuis le meme depart
    if ( o.continuation == OptionsSolveur::AUTO && Convolee<Dist>::possible && liste.size() == 1
      && ( min_masse_depart < o.seuil_continuation * nu_min ) ) {
        liste = etapes( s0, o.conv_ratio, s_min );
        s_courant = liste[ 0 ];
        bal.dist = &conv.at( s_courant );
        newton.mesures_et_facettes( w, a, fa );
    }

    // ---- les etapes
    NewtonStats total;
    double masse_dom = 0;
    for ( PI etape = 0; etape < liste.size(); ++etape ) {
        s_courant = liste[ etape ];
        if ( etape > 0 ) {                               // la densite suivante : les mesures du depart sont a refaire
            bal.dist = &conv.at( s_courant );
            newton.mesures_et_facettes( w, a, fa );
        }
        // la masse cible, a l'echelle de ce que le domaine contient de CETTE densite
        std::vector<double> nu_s = nu;
        double masse_nu = 0;
        masse_dom = 0;
        for ( SI i = 0; i < n; ++i ) { masse_dom += a[ i ]; masse_nu += nu[ i ]; }
        if ( masse_dom > 0 && masse_nu > 0 && masse_dom != masse_nu )
            for ( SI i = 0; i < n; ++i ) nu_s[ i ] *= masse_dom / masse_nu;
        newton.nu = nu_s;
        newton.a = a;
        newton.fa = fa;
        if ( o.newton.trace )
            std::printf( "  etape %d / %d : s = %.4e, masse du domaine %.6f, plus petite masse %.3e\n",
                         int( etape + 1 ), int( liste.size() ), s_courant, masse_dom, minimum( a ) );

        // le pas par les limites ( 2D ) : la passe est branchee sur Newton quand elle est demandee
        if constexpr ( D == 2 ) {
            Limites2D<decltype( bal )> lim( bal );
            LimitesLocales ll;
            ll.alpha_min = [&]( const std::vector<double> &W, const std::vector<double> &Dd, const std::vector<SI> &mauvaises,
                                double horizon, double eps, const Laplacien &L, SI &nb_cellules ) {
                return lim.alpha_min( W, Dd, mauvaises, horizon, eps, L, nb_cellules );
            };
            if ( o.newton.pas == NewtonOptions::ESSAI_LIMITES )
                newton.limites = &ll;
            newton.resout( w, true );
            newton.limites = nullptr;
        } else
            newton.resout( w, true );

        // ce que l'etape laisse : ses poids, ses mesures ( pour l'etape suivante ), ses compteurs
        w = newton.w;
        a = newton.a;
        fa = newton.fa;
        nb_steps_avant = nb_steps > 0 ? int( double( hist.rows( nb_steps - 1, int( H_STEP ) ) ) ) + 1 : 0;
        total.nb_iter += newton.st.nb_iter;
        total.nb_recul += newton.st.nb_recul;
        total.nb_cell_lim += newton.st.nb_cell_lim;
        total.nb_tours_essai += newton.st.nb_tours_essai;
        total.t_asm += newton.st.t_asm;
        total.t_lim += newton.st.t_lim;
        if ( etape == 0 ) { total.reste0 = newton.st.reste0; total.eps = newton.st.eps; }
        total.fin = newton.st.fin;
        total.reste = newton.st.reste;
        if ( newton.st.fin == NewtonStats::ECHEC_LINEAIRE )
            break;
        newton.st = NewtonStats{};
    }
    hist.nb_steps.set( nb_steps );

    for ( SI i = 0; i < n; ++i )
        weights( i ) = w[ i ];

    auto put = [&]( int i, double v ) { stats( i ) = v; };
    put( FIN, double( total.fin ) );
    put( RESTE, total.reste );
    put( RESTE0, total.reste0 );
    put( NB_ITER, double( total.nb_iter ) );
    put( NB_DIAG, double( bal.nb_diag ) );
    put( NB_RECUL, double( total.nb_recul ) );
    put( T_MAJ, bal.t_maj );
    put( T_DIAG, bal.t_diag );
    put( T_ASM, total.t_asm );
    put( T_LIN, lin->st.total() );
    put( T_LIM, total.t_lim );
    put( EPS, total.eps );
    put( MASSE_DOMAINE, masse_dom );
    put( NB_DEBORDE, double( bal.nb_deborde ) );
    put( NB_CELL_LIM, double( total.nb_cell_lim ) );
    put( NB_TOURS_ESSAI, double( total.nb_tours_essai ) );
    put( LIN_NB_HIER, double( lin->st.nb_hier ) );
    put( LIN_NB_ITER, double( lin->st.nb_iter ) );
    put( LIN_PIRE, lin->st.pire );
    put( DEPART, double( depart ) );
    put( T_TOTAL, now() - t_debut );
    put( NB_ETAPES, double( liste.size() ) );
    put( MIN_MASSE_DEPART, min_masse_depart );
}

} // namespace otplan
} // namespace sdot

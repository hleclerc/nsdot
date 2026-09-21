#pragma once

// =====================================================================================
// LE TRANSPORT SEMI-DISCRET, RESOLU : trouver `w` tel que `masse( Lag_i( w ) ) = nu_i` pour tout `i`.
// Le Newton amorti de `solvers_des_familles` ( `src/solver/Newton.h`, README § 3, § 7, § 10 ), qui a
// gagne contre tout ce qui a ete essaye -- L-BFGS et gradient conjugue sur le dual ( 2 a 4x plus de
// diagrammes ), la continuation par serie, le multi-echelle, la barriere -- repris ici sur les
// diagrammes de `Balayage.h`.
//
// = Le dual, et pourquoi Newton
//
//     Phi( w ) = integrale min_i ( |x - p_i|^2 - w_i ) rho( x ) dx + sum_i w_i nu_i
//
// est CONCAVE, de gradient `nu_i - masse( Lag_i( w ) )`, et sa hessienne est au signe pres le
// laplacien du graphe de Laguerre ( `Laplacien.h` ). Son noyau est les constantes : on fixe `w_0 = 0`.
//
// = L'amortissement ( Kitagawa-Merigot-Thibert ), et ce qu'il protege
//
// La hessienne n'est definie que tant qu'aucune cellule n'est vide. Le pas essaye doit donc
// garder toute masse au-dessus d'un plancher `eps` fixe au depart ( la moitie de la plus petite
// masse, de depart ou cible ), et faire decroitre le residu d'au moins `1 - t / 2`. Le depart doit
// etre admissible ( aucune cellule vide ) : c'est l'affaire de l'appelant ( `Solve.h` ).
//
// La decroissance est demandee STRICTE ( `n2 < nr` ) : sans cela un pas qui tend vers zero passe
// le test par egalite des que `t` est negligeable, et Newton tourne sur place indefiniment
// ( mesure : 54 diagrammes par iteration a residu constant ). On sort alors en STAGNATION -- le
// plancher numerique, pas un echec, et la difference se lit sur `reste`.
//
// = Le pas d'essai
//
// ESSAIS : `t` repart de `mult_ok` fois le dernier pas accepte ( plafonne a 1 ), et se divise par
// deux tant que le pas est refuse. Repartir de 1 a chaque iteration coutait dix diagrammes par
// pas dans la phase lineaire ( `t ~ 1e-3`, des cellules presque vides ) pour retomber au meme `t`.
//
// ESSAI_LIMITES ( 2D ) : l'essai `t = beta` d'abord ; si des cellules y passent sous `eps`, leurs
// LIMITES le long de `d` ( `Limites.h` : une cellule exacte par tour, a chaud, le polynome ou la
// bissection en masse ), le pas ramene sous la plus petite, et on recommence. Le meilleur pas
// mesure ( -37 % de diagrammes sur les lignes, les reculs disparaissent sur les densites ) ; il
// demande une passe de limites que seul le 2D sait faire aujourd'hui.
//
// = Ce que coute une iteration
//
// UN diagramme par pas essaye, et rien de plus : le pas accepte livre a la fois les mesures ( le
// residu ) et les facettes ( la hessienne suivante ). Le temps est compte par poste -- majorants,
// diagrammes, assemblage, resolution -- parce que c'est la REPARTITION qu'on veut lire.
// =====================================================================================

#include "Balayage.h"
#include "Lineaire.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <vector>

namespace sdot {
namespace otplan {

struct NewtonOptions {
    double tol_abs    = 1e-8;    ///< arret : `max_i |a_i - nu_i| <= tol_abs` ...
    double tol_rel    = 0;       ///< ... ou `max_i |a_i - nu_i| / nu_i <= tol_rel` ( 0 : jamais )
    int    maxit      = 100;
    int    max_reculs = 60;      ///< divisions par deux du pas, au plus, par iteration
    double t_min      = 1e-10;   ///< en dessous, on declare la STAGNATION
    double mult_ok    = 4;       ///< ESSAIS : le prochain essai part de `mult_ok * t` ( plafonne a 1 )
    bool   trace      = false;
    enum Pas : int { ESSAIS = 0, ESSAI_LIMITES = 1 };
    int    pas        = ESSAIS;
    double facteur    = 0.9;     ///< ESSAI_LIMITES : `t = facteur * alpha*`
    double beta0      = 0.25;    ///< ESSAI_LIMITES : le tout premier essai
    double mult_lim   = 2;       ///< ESSAI_LIMITES : apres un essai passe DIRECT, `beta *= mult_lim`
    double confiance  = 0;       ///< ESSAI_LIMITES : apres un pas CORRIGE, le prochain essai est au moins `confiance * t`
    /// appele apres chaque pas ACCEPTE ( et au depart, `it = 0` ) : `w` et `a` sont ceux du pas
    std::function<void( int it, double t, int nb_evals )> apres_pas;
};

struct NewtonStats {
    int    fin = 0;              ///< pourquoi la boucle s'est arretee ( `Fin` )
    enum Fin : int { EN_COURS = 0, CONVERGE = 1, MAX_ITERATIONS = 2, STAGNATION = 3, ECHEC_LINEAIRE = 4 };
    double reste  = 0;           ///< le `max_i |a_i - nu_i|` atteint
    double reste0 = 0;           ///< le meme AU DEPART
    double eps    = 0;           ///< le plancher de masse de l'amortissement
    int    nb_iter = 0, nb_recul = 0;
    SI     nb_cell_lim = 0;      ///< cellules calculees par les passes de limites, en tout
    int    nb_tours_essai = 0;   ///< ESSAI_LIMITES : essais corriges par des limites locales
    double t_asm = 0, t_lin = 0, t_lim = 0;
    static const char *texte( int fin ) {
        switch ( fin ) {
            case CONVERGE:       return "CONVERGE";
            case MAX_ITERATIONS: return "MAX ITERATIONS";
            case STAGNATION:     return "STAGNATION";
            case ECHEC_LINEAIRE: return "SOLVEUR LINEAIRE EN ECHEC";
            default:             return "?";
        }
    }
};

/// CE QU'UNE PASSE DE LIMITES rend a Newton ( `Limites.h`, 2D ) : `alpha` par cellule demandee.
/// Un `Balayage` qui n'en a pas ( `limites == nullptr` ) prend le pas par ESSAIS.
struct LimitesLocales {
    /// les limites des cellules `mauvaises` ( identifiants ) le long de `d` depuis `w`, sous
    /// `horizon`, au niveau `eps` ; rend `min_i alpha_i` et le nombre de cellules calculees
    std::function<double( const std::vector<double> &w, const std::vector<double> &d, const std::vector<SI> &mauvaises,
                          double horizon, double eps, const Laplacien &L, SI &nb_cellules )> alpha_min;
};

template<class Bal>
struct Newton {
    Bal                &bal;
    SolveurLineaire    &lin;
    NewtonOptions       o;
    const LimitesLocales *limites = nullptr;

    std::vector<double> nu;      ///< la masse cible, par germe
    std::vector<double> w;       ///< les poids courants, `w[ 0 ] == 0`
    std::vector<double> a;       ///< les masses courantes
    std::vector<double> d;       ///< la derniere direction de Newton ( `d[ 0 ] == 0` )
    std::vector<Facette> fa;     ///< les facettes du diagramme courant ( celui de `w` )
    NewtonStats         st;
    double              t_dernier = 1;   ///< le dernier pas accepte
    int                 nb_evals_dernier = 0;
    double              beta;            ///< ESSAI_LIMITES : le prochain essai -- GARDE d'un `resout` a l'autre ( les
                                         ///< etapes d'une continuation : un depart proche accepte `t = 1` d'emblee )

    Newton( Bal &bal, SolveurLineaire &lin, NewtonOptions o = {} ) : bal( bal ), lin( lin ), o( o ), beta( o.beta0 ) {}

    static double norme2( const std::vector<double> &v ) {
        double s = 0;
        for ( double x : v ) s += x * x;
        return std::sqrt( s );
    }

    /// `| a - nu |_2`, le merite de l'amortissement
    double merite( const std::vector<double> &A ) const {
        double s = 0;
        for ( SI i = 0; i < SI( A.size() ); ++i ) s += ( nu[ i ] - A[ i ] ) * ( nu[ i ] - A[ i ] );
        return std::sqrt( s );
    }

    /// LES MESURES ET LES FACETTES pour les poids `W`
    void mesures_et_facettes( const std::vector<double> &W, std::vector<double> &res, std::vector<Facette> &f ) {
        bal.set_weights( W );
        bal.mesures( res, &f );
    }

    /// LA BOUCLE, depuis `w_init` ( `a` et `fa` DEJA calcules pour `w_init` si `deja_mesure` ).
    /// Rend `true` si le critere d'arret est atteint. Le diagramme porte les poids ACCEPTES en sortie.
    bool resout( const std::vector<double> &w_init, bool deja_mesure = false ) {
        const SI n = bal.n();
        std::vector<double> a2, b, w2;
        std::vector<Facette> fa2;
        Laplacien L;

        w = w_init;
        const double jauge = w[ 0 ];
        for ( SI i = 0; i < n; ++i )                     // la jauge, imposee ici et maintenue par
            w[ i ] -= jauge;                             // `d[ 0 ] = 0` ensuite
        if ( ! deja_mesure )
            mesures_et_facettes( w, a, fa );
        t_dernier = 1;
        nb_evals_dernier = 1;
        if ( o.apres_pas ) o.apres_pas( 0, 0, 1 );

        double eps = 0;
        for ( int it = 0; it < o.maxit; ++it ) {
            double pire = 0, pire_rel = 0;
            SI nvide = 0;
            b.assign( n, 0.0 );
            for ( SI i = 0; i < n; ++i ) {
                nvide += ! ( a[ i ] > 0 );
                pire = std::max( pire, std::fabs( nu[ i ] - a[ i ] ) );
                pire_rel = std::max( pire_rel, std::fabs( nu[ i ] - a[ i ] ) / nu[ i ] );
                b[ i ] = nu[ i ] - a[ i ];               // `-r`, le second membre de Newton
            }
            if ( it == 0 ) {                             // le plancher de masse de l'amortissement
                double am = a[ 0 ], nm = nu[ 0 ];
                for ( SI i = 0; i < n; ++i ) { am = std::min( am, a[ i ] ); nm = std::min( nm, nu[ i ] ); }
                eps = 0.5 * std::min( nm, am );
                st.eps = eps;
                st.reste0 = pire;
            }
            const double nr = merite( a );
            st.reste = pire;

            if ( pire <= o.tol_abs || ( o.tol_rel > 0 && pire_rel <= o.tol_rel ) ) {
                if ( o.trace )
                    std::printf( "    it %2d  |r|_2 %.3e  max|a-nu| %.3e  CONVERGE\n", it, nr, pire );
                st.fin = NewtonStats::CONVERGE;
                return true;
            }
            ++st.nb_iter;
            const int g0 = bal.nb_diag;

            double t0 = now();
            L.assemble( n, fa );
            st.t_asm += now() - t0;
            t0 = now();
            const bool fait = lin.resout( L, b, d );
            st.t_lin += now() - t0;
            if ( ! fait ) {
                st.fin = NewtonStats::ECHEC_LINEAIRE;
                return false;
            }

            // ---- L'ESSAI PUIS LES LIMITES LOCALES : le diagramme du pas d'abord, et si des cellules
            // y passent sous `eps`, leurs limites ( a elles seules ), le pas ramene sous la plus
            // petite, et on recommence -- la non-monotonie peut en reveler d'autres
            double t = std::min( 1.0, o.mult_ok * t_dernier );
            bool deja = false;                           // le diagramme en `t` est deja fait
            double alpha_lim = -1;
            int nb_evals = 0;
            if ( o.pas == NewtonOptions::ESSAI_LIMITES && limites ) {
                t = beta;
                std::vector<SI> mauvaises;
                w2.resize( n );
                double t_fait = -1;                      // le pas dont le diagramme est dans `a2`
                for ( int tour = 0; tour < 8; ++tour ) {
                    for ( SI i = 0; i < n; ++i ) w2[ i ] = w[ i ] + t * d[ i ];
                    w2[ 0 ] = 0;
                    mesures_et_facettes( w2, a2, fa2 );
                    ++nb_evals;
                    t_fait = t;
                    mauvaises.clear();
                    for ( SI i = 0; i < n; ++i ) if ( a2[ i ] < eps ) mauvaises.push_back( i );
                    if ( mauvaises.empty() ) break;
                    ++st.nb_tours_essai;
                    t0 = now();
                    bal.set_weights( w );
                    SI nb_cel = 0;
                    const double al = std::min( t, limites->alpha_min( w, d, mauvaises, t, eps, L, nb_cel ) );
                    st.nb_cell_lim += nb_cel;
                    st.t_lim += now() - t0;
                    if ( o.trace )
                        std::printf( "      essai t %.3e : %d cellules sous eps, limite locale %.3e ( %lld cellules calculees )\n",
                                     t, int( mauvaises.size() ), al, ( long long ) nb_cel );
                    t = o.facteur * al;
                    if ( t < o.t_min ) break;
                }
                // une limite nulle n'est pas une raison de stagner : on rend la main aux essais,
                // depuis la moitie du dernier pas calcule
                if ( t < o.t_min ) t = t_fait / 2;
                deja = t == t_fait;
                alpha_lim = t;
                const bool direct = t >= beta;
                beta = std::min( 1.0, std::max( direct ? o.mult_lim * beta : beta, o.confiance * t ) );
            }

            // ---- L'AMORTISSEMENT
            bool pris = false;
            const double t_lim0 = t;
            w2.resize( n );
            for ( int essai = 0; essai < o.max_reculs; ++essai ) {
                if ( ! ( essai == 0 && deja ) ) {        // sinon, deja fait en `t`
                    for ( SI i = 0; i < n; ++i ) w2[ i ] = w[ i ] + t * d[ i ];
                    w2[ 0 ] = 0;                         // la jauge, imposee et non esperee
                    mesures_et_facettes( w2, a2, fa2 );
                    ++nb_evals;
                }
                double m2 = a2[ 0 ];                     // le plancher `eps` est une masse ABSOLUE
                for ( SI i = 0; i < n; ++i ) m2 = std::min( m2, a2[ i ] );
                const double n2r = merite( a2 );
                if ( m2 >= eps && std::isfinite( n2r ) && n2r <= ( 1 - t / 2 ) * nr && n2r < nr ) { pris = true; break; }
                t /= 2;
                ++st.nb_recul;
                if ( t < o.t_min )
                    break;
            }
            if ( o.trace ) {
                std::printf( "    it %2d  |r|_2 %.3e  max|a-nu| %.3e  %lld vides  pas %.2e  %d diag  [maj %.2f  diag %.2f  asm %.2f  lin %.2f]",
                             it, nr, pire, ( long long ) nvide, t, bal.nb_diag - g0, bal.t_maj, bal.t_diag, st.t_asm, lin.st.total() );
                if ( alpha_lim >= 0 )
                    std::printf( "  alpha* %.2e%s", alpha_lim, t < t_lim0 ? " REFUSE" : "" );
                std::printf( "\n" );
                std::fflush( stdout );
            }
            if ( ! pris ) {
                bal.set_weights( w );                    // le diagramme reprend les poids acceptes
                st.fin = NewtonStats::STAGNATION;        // le plancher numerique, pas un echec
                return false;
            }
            t_dernier = t;
            nb_evals_dernier = nb_evals;
            w.swap( w2 );
            a.swap( a2 );
            fa.swap( fa2 );
            if ( o.apres_pas ) o.apres_pas( it + 1, t, nb_evals );
        }
        // le dernier point : ce qu'il vaut
        double pire = 0;
        for ( SI i = 0; i < n; ++i ) pire = std::max( pire, std::fabs( nu[ i ] - a[ i ] ) );
        st.reste = pire;
        st.fin = NewtonStats::MAX_ITERATIONS;
        return false;
    }
};

} // namespace otplan
} // namespace sdot

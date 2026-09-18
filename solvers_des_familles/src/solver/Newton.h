#pragma once

// =====================================================================================
// LE TRANSPORT SEMI-DISCRET, RESOLU : trouver `w` tel que `|Lag_i( w )| = nu_i` pour tout `i`.
//
// = Le dual, et pourquoi Newton
//
//     Phi( w ) = integrale min_i ( |x - p_i|^2 - w_i ) dx + sum_i w_i nu_i
//
// est CONCAVE, de gradient `nu_i - |Lag_i( w )|`, et sa hessienne est au signe pres le laplacien
// du graphe de Laguerre ( `Laplacien.h` ). Son noyau est les constantes : on fixe `w_0 = 0`.
//
// = L'amortissement ( Kitagawa-Merigot-Thibert ), et ce qu'il protege
//
// La hessienne n'est definie que tant qu'aucune cellule n'est vide. Le pas essaye doit donc
// garder toute aire au-dessus d'un plancher `eps` fixe au depart, et faire decroitre le residu
// d'au moins `1 - t / 2`. On part de `w = 0`, le diagramme de VORONOI, dont aucune cellule n'est
// vide : le depart est toujours admissible.
//
// La decroissance est demandee STRICTE ( `n2 < nr` ) : sans cela un pas qui tend vers zero passe
// le test par egalite des que `t` est negligeable, et Newton tourne sur place indefiniment
// ( mesure : 54 diagrammes par iteration a residu constant ). On sort alors en STAGNATION -- le
// plancher numerique, pas un echec, et la difference se lit sur `reste`.
//
// = Le pas par les limites ( `pas != ESSAIS`, 2D )
//
// Au lieu d'essayer `t = 1, 1/2, 1/4, ...` a un diagramme l'essai, on calcule la LIMITE de
// chaque cellule le long de `d` ( `Ecrasement.h` : le polynome predit, une cellule exacte
// verifie et corrige ), et `alpha* = min_i` dit jusqu'ou on peut aller sans passer sous `eps`.
// Le pas est la puissance de deux sous `alpha*` ( DYADIQUE ) ou `facteur * alpha*` ( FACTEUR ),
// et le diagramme de ce pas -- qu'il faut de toute facon pour l'iteration suivante -- confirme
// la decroissance du residu. S'il refuse ( residu, ou une cellule non monotone ), on recule
// comme avant depuis la.
//
// = Ce que coute une iteration
//
// UN diagramme par pas essaye, et rien de plus : le pas accepte livre a la fois les mesures ( le
// residu ) et les facettes ( la hessienne suivante ). Le temps est compte par poste -- majorants,
// diagrammes, assemblage, resolution -- parce que c'est la REPARTITION qu'on veut lire.
// =====================================================================================

#include "diagram/PowerDiagram.h"
#include "solver/Ecrasement.h"
#include "solver/Laplacien.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

namespace sf {

struct NewtonOptions {
    TF   tol        = 1e-6;    ///< arret : `max_i |a_i - nu_i| <= tol * nu_i`
    int  maxit      = 100;
    int  max_reculs = 60;      ///< divisions par deux du pas, au plus, par iteration
    TF   t_min      = 1e-10;   ///< en dessous, on declare la STAGNATION ( 34 diagrammes pour y
                               ///< descendre depuis 1 : `1e-3` en coute 10 )
    bool trace      = true;
    int  extraire   = -1;      ///< >= 0 : s'arreter des que la DIRECTION de cette iteration est
                               ///< calculee ( `w` et `d` sont alors ceux du pas propose )
    enum Pas : int { ESSAIS = 0, DYADIQUE, FACTEUR, TENSEUR };
    int  pas        = ESSAIS;  ///< comment choisir `t` ( voir en tete )
    TF   facteur    = 0.9;     ///< `t = facteur * alpha*` en mode FACTEUR
    TF   theta_mult = 5;       ///< TENSEUR : la cible partielle est `theta = theta_mult * alpha*`
    OptionsLimites lim;        ///< les reglages de la passe des limites ( `niveau` est mis ici )
};

struct NewtonStats {
    const char *fin = "?";     ///< pourquoi la boucle s'est arretee
    TF     reste = 0;          ///< le `max_i |a_i - nu_i| / nu_i` atteint
    int    nb_iter = 0, nb_diag = 0, nb_recul = 0;
    SI     nb_deborde = 0;     ///< cellules qui ont deborde `MaxNv`, en tout ( mesure fausse )
    SI     nb_cell_lim = 0;    ///< cellules calculees par la passe des limites, en tout
    int    nb_tenseur = 0;     ///< pas tensoriels tentes
    double t_tenseur = 0;
    int    nb_lim_refus = 0;   ///< pas proposes par les limites et refuses par le diagramme
    double t_maj = 0, t_diag = 0, t_asm = 0, t_lin = 0, t_lim = 0;
};

template<class PD, class Lin>
struct Newton {
    PD             &pd;
    Lin            &lin;
    const TF *const *P;        ///< les positions, dans l'ordre des identifiants
    Parallel        par;
    NewtonOptions   o;

    std::vector<TF> nu;        ///< la mesure cible, par germe
    std::vector<TF> w;         ///< les poids courants, `w[ 0 ] == 0`
    std::vector<TF> a;         ///< les mesures courantes
    std::vector<TF> d;         ///< la derniere direction de Newton ( `d[ 0 ] == 0` )
    NewtonStats     st;

    Newton( PD &pd, Lin &lin, const TF *const *P, Parallel par, NewtonOptions o = {} )
        : pd( pd ), lin( lin ), P( P ), par( par ), o( o ) {}

    /// LES MESURES ET LES FACETTES pour les poids `W` : un diagramme, et la conversion
    /// `c_ij = |facette| / ( 2 |p_i - p_j| )` faite par le thread qui a mesure la cellule.
    void mesures_et_facettes( const std::vector<TF> &W, std::vector<TF> &res, std::vector<Facette> &fa ) {
        constexpr int D = PD::dim;
        double t0 = now();
        pd.set_weights( W.data(), par );
        st.t_maj += now() - t0;

        t0 = now();
        std::vector<std::vector<Facette>> par_th( std::max( par.threads, 1 ) );
        st.nb_deborde += pd.measures_and_facets( res, par, [ & ]( int t, SI i, SI j, TF mes ) {
            TF d2 = 0;
            for ( int d = 0; d < D; ++d ) {
                const TF e = P[ d ][ j ] - P[ d ][ i ];
                d2 += e * e;
            }
            if ( d2 > 0 )
                par_th[ t ].push_back( Facette{ i, j, mes / ( 2 * std::sqrt( d2 ) ) } );
        } );
        fa.clear();
        for ( auto &v : par_th )
            fa.insert( fa.end(), v.begin(), v.end() );
        st.t_diag += now() - t0;
        ++st.nb_diag;
    }

    static TF norme2( const std::vector<TF> &v ) {
        TF s = 0;
        for ( TF x : v ) s += x * x;
        return std::sqrt( s );
    }

    /// LA BOUCLE, depuis `w_init` ( zero : Voronoi ). Rend `true` si le critere d'arret est atteint.
    bool resout( const std::vector<TF> &w_init ) {
        const SI n = pd.n;
        std::vector<TF> a2, b, w2;
        std::vector<Facette> fa, fa2;
        std::vector<LimiteCellule> lim;
        Laplacien L;

        w = w_init;
        const TF g = w[ 0 ];
        for ( SI i = 0; i < n; ++i )                     // la jauge, imposee ici et maintenue par
            w[ i ] -= g;                                 // `d[ 0 ] = 0` ensuite
        mesures_et_facettes( w, a, fa );

        TF eps = 0;
        for ( int it = 0; it < o.maxit; ++it ) {
            TF pire = 0;
            SI nvide = 0;
            b.assign( n, TF( 0 ) );
            for ( SI i = 0; i < n; ++i ) {
                nvide += ! ( a[ i ] > 0 );
                b[ i ] = nu[ i ] - a[ i ];               // `-r`, le second membre de Newton
                pire = std::max( pire, std::fabs( b[ i ] ) / nu[ i ] );
            }
            if ( it == 0 ) {                             // le plancher d'aire de l'amortissement
                TF am = a[ 0 ], nm = nu[ 0 ];
                for ( SI i = 0; i < n; ++i ) { am = std::min( am, a[ i ] ); nm = std::min( nm, nu[ i ] ); }
                eps = TF( 0.5 ) * std::min( nm, am );
            }
            const TF nr = norme2( b );
            st.reste = pire;

            if ( pire <= o.tol ) {
                if ( o.trace )
                    std::printf( "    it %2d  |r|_2 %.3e  max|a-nu|/nu %.3e  CONVERGE\n",
                                 it, double( nr ), double( pire ) );
                st.fin = "CONVERGE";
                return true;
            }
            ++st.nb_iter;
            const double d0 = st.t_diag, l0 = lin.st.total(), s0 = st.t_asm, m0 = st.t_maj;
            const int    g0 = st.nb_diag, i0 = lin.st.nb_iter;

            double t0 = now();
            L.assemble( n, fa );
            st.t_asm += now() - t0;
            t0 = now();
            const bool fait = lin.resout( L, b, d );
            st.t_lin += now() - t0;
            if ( ! fait ) {
                st.fin = "SOLVEUR LINEAIRE EN ECHEC";
                return false;
            }
            if ( it == o.extraire ) {                    // la direction est ce qu'on venait chercher
                st.fin = "DIRECTION EXTRAITE";
                return false;
            }

            // ---- LE PAS PAR LES LIMITES, s'il est demande
            TF t = 1, alpha_lim = -1;
            TF gain = 1;                                 // ce que `t = 1` vise : `nu` pour `d`, la cible
                                                         // partielle `theta` pour un pas tensoriel
            if ( o.pas != NewtonOptions::ESSAIS ) {
                if constexpr ( PD::dim == 2 ) {
                    t0 = now();
                    OptionsLimites ol = o.lim;
                    ol.niveau = eps;
                    ol.global = true;
                    pd.set_weights( w.data(), par );
                    limites( pd, P, w, d, par, ol, lim, Voisinage{ L.row.data(), L.col.data() } );
                    // le minimum des limites TROUVEES ; une borne ( HORIZON ) ne compte que si elle
                    // est sous ce minimum, et alors comme lui. Rien avant l'horizon : le pas plein.
                    alpha_lim = INFINI;
                    for ( SI i = 0; i < n; ++i ) {
                        st.nb_cell_lim += lim[ i ].tours;
                        if ( lim[ i ].etat != LimiteCellule::VIDE_AU_DEPART && lim[ i ].etat != LimiteCellule::HORIZON )
                            alpha_lim = std::min( alpha_lim, lim[ i ].alpha );
                    }
                    for ( SI i = 0; i < n; ++i )
                        if ( lim[ i ].etat == LimiteCellule::HORIZON && lim[ i ].alpha < ol.horizon )
                            alpha_lim = std::min( alpha_lim, lim[ i ].alpha );
                    // ---- LE PAS TENSORIEL : quand la direction de Newton est bloquee bien avant le
                    // pas plein, on demande au modele quadratique la direction de la cible partielle
                    // `theta = theta_mult * alpha*`, puis ses propres limites
                    if ( o.pas == NewtonOptions::TENSEUR && alpha_lim < TF( 0.2 ) ) {
                        const TF theta = std::min( TF( 1 ), o.theta_mult * alpha_lim );
                        std::vector<ModeleCellule> mod( n );
                        parallel_for( n, par, [ & ]( SI k, int ) {
                            typename PD::Cell cel;
                            pd.cellule( k, cel );
                            mod[ pd.ids[ k ] ].depuis( cel, pd.ids[ k ], P, w.data() );
                        } );
                        PasTensoriel<Lin> pt;
                        std::vector<TF> delta;
                        const TF rm = pt.resout( mod, a, nu, d, theta, par, lin, delta );
                        st.t_tenseur += pt.t;
                        ++st.nb_tenseur;
                        // les limites de la direction corrigee ( `delta` deja a l'echelle : horizon 1 )
                        limites( pd, P, w, delta, par, ol, lim, Voisinage{ L.row.data(), L.col.data() } );
                        TF al2 = INFINI;
                        for ( SI i = 0; i < n; ++i ) {
                            st.nb_cell_lim += lim[ i ].tours;
                            if ( lim[ i ].etat != LimiteCellule::VIDE_AU_DEPART && lim[ i ].etat != LimiteCellule::HORIZON )
                                al2 = std::min( al2, lim[ i ].alpha );
                        }
                        for ( SI i = 0; i < n; ++i )
                            if ( lim[ i ].etat == LimiteCellule::HORIZON && lim[ i ].alpha < ol.horizon )
                                al2 = std::min( al2, lim[ i ].alpha );
                        if ( o.trace )
                            std::printf( "      tenseur : theta %.3e, %d it, residu du modele %.2e, limite de delta %.3e\n",
                                         double( theta ), pt.nb_it, double( rm ), double( al2 ) );
                        // on garde `delta` si elle porte plus loin que `alpha* d` ( en unites de theta )
                        const TF portee = std::min( al2, TF( 1 ) ) * theta;
                        if ( portee > alpha_lim ) {
                            d.swap( delta );                     // `d` devient `delta`, `t` en fraction de delta
                            alpha_lim = std::min( al2, ol.horizon );
                            gain = theta;
                        }
                    }
                    if ( alpha_lim >= ol.horizon )
                        t = 1;
                    else if ( o.pas == NewtonOptions::DYADIQUE ) {
                        t = 1;
                        while ( t > alpha_lim && t > TF( 1e-10 ) ) t /= 2;
                    } else
                        t = std::min( TF( 1 ), o.facteur * alpha_lim );
                    st.t_lim += now() - t0;
                }
            }

            // ---- L'AMORTISSEMENT
            bool pris = false;
            const TF t_lim0 = t;
            w2.resize( n );
            for ( int essai = 0; essai < o.max_reculs; ++essai ) {
                for ( SI i = 0; i < n; ++i ) w2[ i ] = w[ i ] + t * d[ i ];
                w2[ 0 ] = 0;                             // la jauge, imposee et non esperee
                mesures_et_facettes( w2, a2, fa2 );
                TF m2 = a2[ 0 ], n2 = 0;                 // le plancher `eps` est une aire ABSOLUE
                for ( SI i = 0; i < n; ++i ) {
                    m2 = std::min( m2, a2[ i ] );
                    const TF e = nu[ i ] - a2[ i ];
                    n2 += e * e;
                }
                const TF n2r = std::sqrt( n2 );
                if ( m2 >= eps && n2r <= ( 1 - gain * t / 2 ) * nr && n2r < nr ) { pris = true; break; }
                t /= 2;
                ++st.nb_recul;
                if ( t < o.t_min )
                    break;
            }
            if ( pris && alpha_lim >= 0 && t < t_lim0 ) ++st.nb_lim_refus;
            if ( o.trace ) {
                std::printf( "    it %2d  |r|_2 %.3e  max|a-nu|/nu %.3e  %d vides  pas %.2e  %d diag"
                             "  [maj %.2f  diag %.2f  asm %.2f  lin %.2f%s]",
                             it, double( nr ), double( pire ), int( nvide ), double( t ),
                             st.nb_diag - g0, st.t_maj - m0, st.t_diag - d0, st.t_asm - s0,
                             lin.st.total() - l0, it_txt( lin.st.nb_iter - i0 ) );
                if ( alpha_lim >= 0 )
                    std::printf( "  alpha* %.2e%s", double( alpha_lim ), t < t_lim0 ? " REFUSE" : "" );
                std::printf( "\n" );
                std::fflush( stdout );
            }
            if ( ! pris ) {
                st.fin = "STAGNATION";                   // le plancher numerique, pas un echec
                return false;
            }
            w.swap( w2 );
            a.swap( a2 );
            fa.swap( fa2 );
        }
        st.fin = "MAX ITERATIONS";
        return false;
    }

private:
    static const char *it_txt( int nb ) {
        static thread_local char buf[ 32 ];
        if ( ! nb ) return "";
        std::snprintf( buf, sizeof( buf ), " (%d it)", nb );
        return buf;
    }
};

} // namespace sf

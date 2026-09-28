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
#include "solver/Cible.h"
#include "solver/Densite.h"
#include "solver/Ecrasement.h"
#include "solver/Laplacien.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <vector>

namespace sf {

struct NewtonOptions {
    TF   tol        = 1e-6;    ///< arret : `max_i |a_i - nu_i| <= tol * nu_i`
    int  maxit      = 100;
    int  max_reculs = 60;      ///< divisions par deux du pas, au plus, par iteration
    TF   t_min      = 1e-10;   ///< en dessous, on declare la STAGNATION ( 34 diagrammes pour y
                               ///< descendre depuis 1 : `1e-3` en coute 10 )
    /// LE PROGRES MINIMAL PAR ITERATION, en fraction de `|r|_2`. `0` : eteint.
    ///
    /// `t_min` ne suffit pas a arreter une agonie, parce qu'une iteration mourante ACCEPTE son
    /// pas -- elle gagne un pour cent sur le residu pour huit diagrammes. Mesure sur l'uniforme
    /// 3D a `n = 1e5` en `--kernel float` : les six premieres iterations sont identiques a celles
    /// du `double` et coutent NEUF diagrammes ; les quatorze suivantes en coutent CENT QUARANTE
    /// CINQ pour faire passer `|r|_2` de 4.3e-10 a 2.0e-10. En simple precision c'est le regime
    /// normal de fin de course -- le residu a atteint le bruit de la mesure -- donc il faut un
    /// critere qui le voie : une iteration qui gagne moins que `progres_min` est la derniere.
    TF   progres_min = 0;
    /// LE PLANCHER DE BRUIT DE LA MESURE, en unites de `|r|_2`. `0` : eteint.
    ///
    /// Il est PREVISIBLE et ne coute rien a poser : l'ecart de mesure d'une cellule vaut
    /// `kappa eps nu_i`, donc `|bruit|_2 = kappa eps sqrt( n ) nu = kappa eps / sqrt( n )`. A
    /// `n = 1e5` en `float` ca donne 1.9e-10 -- et la trace montre Newton bloque a 2.0e-10.
    /// S'arreter LA, plutot que d'y descendre a coups de demi-pas, est ce qui rend la bascule
    /// `--kernel mixte` rentable : c'est la difference entre rendre la main apres SIX diagrammes
    /// inutiles et apres CENT QUARANTE CINQ.
    TF   plancher = 0;
    bool trace      = true;
    int  extraire   = -1;      ///< >= 0 : s'arreter des que la DIRECTION de cette iteration est
                               ///< calculee ( `w` et `d` sont alors ceux du pas propose )
    enum Pas : int { ESSAIS = 0, DYADIQUE, FACTEUR, TENSEUR, ESSAI_LIMITES };
    int  pas        = ESSAIS;  ///< comment choisir `t` ( voir en tete )
    TF   facteur    = 0.9;     ///< `t = facteur * alpha*` en mode FACTEUR
    TF   theta_mult = 5;       ///< TENSEUR : la cible partielle est `theta = theta_mult * alpha*`
    TF   confiance  = 0;       ///< ESSAI_LIMITES : apres un pas CORRIGE, le prochain essai est au moins `confiance * t`
                               ///< ( 0 : `beta` inchange -- mesure meilleur : le pas admissible croit vite )
    TF   beta0      = 0.25;    ///< ESSAI_LIMITES : le tout premier essai ( 1 : un diagramme a moitie vide sur Voronoi )
    TF   mult_ok    = 2;       ///< ESSAI_LIMITES : apres un essai passe DIRECT, `beta *= mult_ok` ( plafonne a 1 )
    OptionsLimites lim;        ///< les reglages de la passe des limites ( `niveau` est mis ici )
    OptionsCible   cible;      ///< LA CIBLE MODIFIEE : deformer `nu` pour allonger le pas ( `Cible.h` )
    /// LE RESIDU : `g( a_i / nu_i )` au lieu de `a_i - nu_i`. Meme solution, autre Newton et autre
    /// merite pour l'amortissement. BARRIERE `g = x - 1/x` : une cellule minuscule ( `x << 1` ) recoit
    /// `x -> 2x` au lieu de sa masse entiere d'un coup, et `|g| ~ 1/x` refuse les pas qui la pincent ;
    /// LOG `g = log x` : `x -> x ( 1 - log x )`.
    enum Residu : int { LIN = 0, BARRIERE, LOG };
    int  residu     = LIN;
    bool memo       = false;   ///< 3D : les facettes du dernier diagramme ACCEPTE proposees en premier au suivant ( § 11 )
    /// appele apres chaque pas ACCEPTE ( et au depart, `it = -1` ) : `pd` porte alors `w`
    std::function<void( int it, TF t, int reculs )> apres_pas;
};

struct NewtonStats {
    const char *fin = "?";     ///< pourquoi la boucle s'est arretee
    TF     reste = 0;          ///< le `max_i |a_i - nu_i| / nu_i` atteint
    TF     reste0 = 0;         ///< le meme AU DEPART ( ce que vaut le point de depart )
    int    nb_iter = 0, nb_diag = 0, nb_recul = 0;
    SI     nb_deborde = 0;     ///< cellules qui ont deborde `MaxNv`, en tout ( mesure fausse )
    SI     nb_cell_lim = 0;    ///< cellules calculees par la passe des limites, en tout
    int    nb_tenseur = 0;     ///< pas tensoriels tentes
    double t_tenseur = 0;
    SI     nb_cell_mauvaises = 0; ///< ESSAI_LIMITES : cellules trouvees sous `eps` par les essais, en tout
    int    nb_tours_essai = 0;    ///< ESSAI_LIMITES : essais corriges par des limites locales
    int    nb_lim_refus = 0;   ///< pas proposes par les limites et refuses par le diagramme
    int    nb_cible_res = 0;   ///< CIBLE : resolutions lineaires depensees, en tout
    int    nb_cible_pris = 0;  ///< CIBLE : iterations ou la direction deformee a ete retenue
    int    nb_cible_refus = 0; ///< CIBLE : ... et ou l'amortissement l'a rejetee, filet declenche
    TF     cible_gain = 0;     ///< CIBLE : produit des `U*_apres / U*_avant`, pour la moyenne geometrique
    double t_cible = 0;
    TF     amp_d0 = 0;         ///< `| d |inf` de la PREMIERE direction resolue. Une continuation qui
                               ///< enchaine des Newton s'en sert pour choisir son pas ( README
                               ///< § 12.6.1 ) : c'est la correction de poids que l'etape a reclamee,
                               ///< et elle est deja calculee -- la relire ne coute rien.
    double t_maj = 0, t_diag = 0, t_asm = 0, t_lin = 0, t_lim = 0, t_memo = 0;
};

template<class PD, class Rho = Densite>
struct Newton {
    PD             &pd;
    Lineaire       &lin;
    const TF *const *P;        ///< les positions, dans l'ordre des identifiants
    Parallel        par;
    NewtonOptions   o;

    std::vector<TF> nu;        ///< la mesure cible, par germe
    std::vector<TF> w;         ///< les poids courants, `w[ 0 ] == 0`
    std::vector<TF> a;         ///< les mesures courantes
    std::vector<TF> d;         ///< la derniere direction de Newton ( `d[ 0 ] == 0` )
    std::vector<Facette> fa;   ///< les facettes du diagramme courant ( celui de `w` )
    NewtonStats     st;

    /// UNE DENSITE au lieu de Lebesgue ( 2D ) : la mesure d'une cellule est sa masse. Le pas par les
    /// limites ( ESSAI_LIMITES ) passe alors par `limites_masse` : la bissection, pas le polynome.
    /// `Rho` n'a qu'a offrir `mesure( cel, facette, dl )` -- `Densite.h` ( gaussiennes ) et
    /// `Image.h` ( une grille de pixels ) le font, et Newton ne distingue pas les deux.
    const Rho      *rho = nullptr;
    bool            derivee = false; ///< avec `rho` : calculer aussi `da = d a / d lambda` a chaque diagramme
    std::vector<TF> da;        ///< `d a_i / d lambda` pour `w` ( le parametre du chemin de `rho` )

    Newton( PD &pd, Lineaire &lin, const TF *const *P, Parallel par, NewtonOptions o = {} )
        : pd( pd ), lin( lin ), P( P ), par( par ), o( o ) {}

    /// LES MESURES ET LES FACETTES pour les poids `W` : un diagramme, et la conversion
    /// `c_ij = |facette| / ( 2 |p_i - p_j| )` faite par le thread qui a mesure la cellule.
    void mesures_et_facettes( const std::vector<TF> &W, std::vector<TF> &res, std::vector<Facette> &fa, std::vector<TF> *dres = nullptr ) {
        constexpr int D = PD::dim;
        double t0 = now();
        pd.set_weights( W.data(), par );
        st.t_maj += now() - t0;
        if constexpr ( D == 3 ) {                        // la memoire : les facettes du diagramme accepte
            if ( o.memo && ! this->fa.empty() ) {
                t0 = now();
                std::vector<d2::SI32> ii( this->fa.size() ), jj( this->fa.size() );
                for ( SI q = 0; q < SI( this->fa.size() ); ++q ) { ii[ q ] = d2::SI32( this->fa[ q ].i ); jj[ q ] = d2::SI32( this->fa[ q ].j ); }
                pd.memorise( ii.data(), jj.data(), SI( ii.size() ) );
                st.t_memo += now() - t0;
            } else
                pd.oublie();
        }

        t0 = now();
        std::vector<std::vector<Facette>> par_th( std::max( par.threads, 1 ) );
        auto facette = [ & ]( int t, SI i, SI j, TF mes ) {
            TF d2 = 0;
            for ( int d = 0; d < D; ++d ) {
                const TF e = P[ d ][ j ] - P[ d ][ i ];
                d2 += e * e;
            }
            if ( d2 > 0 )
                par_th[ t ].push_back( Facette{ i, j, mes / ( 2 * std::sqrt( d2 ) ) } );
        };
        if ( rho ) {
            if constexpr ( D == 2 ) {
                if ( dres ) dres->assign( pd.n, TF( 0 ) );
                st.nb_deborde += pd.measures_and_facets_avec( res, par, facette, [ & ]( const typename PD::Cell &cel, auto &&fac, SI i ) {
                    return rho->mesure( cel, fac, dres ? &( *dres )[ i ] : nullptr );
                } );
            }
        } else
            st.nb_deborde += pd.measures_and_facets( res, par, facette );
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

    /// `g( x )` et `g'( x )` du residu choisi, `x = a / nu` borne loin de zero
    TF g( TF x ) const {
        x = std::max( x, TF( 1e-8 ) );
        return o.residu == NewtonOptions::BARRIERE ? x - 1 / x : o.residu == NewtonOptions::LOG ? std::log( x ) : x - 1;
    }
    TF gp( TF x ) const {
        x = std::max( x, TF( 1e-8 ) );
        return o.residu == NewtonOptions::BARRIERE ? 1 + 1 / ( x * x ) : o.residu == NewtonOptions::LOG ? 1 / x : 1;
    }
    /// LE MERITE de l'amortissement : `| a - nu |_2` pour LIN ( les chiffres de reference ), et la norme
    /// SANS DIMENSION `| g( a / nu ) - moyenne |_2` pour les autres. La moyenne : `sum a = sum nu` est
    /// automatique, donc `g( x_i ) = 0` pour tout `i` fait `n` equations pour `n - 1` inconnues ; c'est
    /// `g( x_i ) = c` pour tout `i` qu'on resout ( qui force `x_i = 1` puisque la moyenne des `x` est 1 ),
    /// et le second membre projete somme a zero comme il faut ( sans ca, la ligne rayee par la jauge
    /// porte toute l'incoherence : mesure, le germe 0 explose et Newton stagne a la premiere etape ).
    TF merite( const std::vector<TF> &A ) const {
        const SI n = SI( A.size() );
        if ( o.residu == NewtonOptions::LIN ) {
            TF s = 0;
            for ( SI i = 0; i < n; ++i ) s += ( nu[ i ] - A[ i ] ) * ( nu[ i ] - A[ i ] );
            return std::sqrt( s );
        }
        TF m = 0;
        for ( SI i = 0; i < n; ++i ) m += g( A[ i ] / nu[ i ] );
        m /= n;
        TF s = 0;
        for ( SI i = 0; i < n; ++i ) { const TF e = g( A[ i ] / nu[ i ] ) - m; s += e * e; }
        return std::sqrt( s );
    }

    /// LA BOUCLE, depuis `w_init` ( zero : Voronoi ). Rend `true` si le critere d'arret est atteint.
    /// `deja_mesure` : `a`, `fa` ( et `da` ) sont DEJA ceux de `w_init` ( qui porte la jauge ) --
    /// l'appelant les a calcules en choisissant son depart, on ne refait pas ce diagramme.
    bool resout( const std::vector<TF> &w_init, bool deja_mesure = false ) {
        const SI n = pd.n;
        TF nr_prec = 0;                                  // `|r|_2` de l'iteration precedente
        std::vector<TF> a2, b, w2, da2;
        std::vector<Facette> fa2;
        std::vector<LimiteCellule> lim;
        std::vector<TF> ucel, u2, b2, d2, sflux, dgard, del;  // la passe CIBLE
        std::vector<char> touche, dumm;
        std::vector<TF> d_sur;                       // LE FILET : la direction de Newton, avant
        std::vector<ModeleCellule> mods;             // ... son juge POLYNOME
        std::vector<SI> imod;
        std::vector<TF> ecor, dlt, alim, t1, t2;
        std::vector<char> coeurp;
        TF seuil_prec = 0;                           // la limite du polynome de l'iteration d'avant
        std::vector<char> dans;
        Laplacien L;
        std::vector<TF> *pda = rho && derivee ? &da : nullptr, *pda2 = pda ? &da2 : nullptr;

        w = w_init;
        const TF jauge = w[ 0 ];
        for ( SI i = 0; i < n; ++i )                     // la jauge, imposee ici et maintenue par
            w[ i ] -= jauge;                             // `d[ 0 ] = 0` ensuite
        if ( ! deja_mesure )
            mesures_et_facettes( w, a, fa, pda );
        if ( o.apres_pas ) o.apres_pas( -1, 0, 0 );

        TF eps = 0, t_prec = 0, beta = o.beta0, t_sur = -1;
        std::vector<char> protegee;                      // les cellules NON VIDES au depart : celles que `eps` defend
        for ( int it = 0; it < o.maxit; ++it ) {
            TF pire = 0;
            SI nvide = 0;
            b.assign( n, TF( 0 ) );
            for ( SI i = 0; i < n; ++i ) {
                nvide += ! ( a[ i ] > 0 );
                pire = std::max( pire, std::fabs( nu[ i ] - a[ i ] ) / nu[ i ] );
                b[ i ] = nu[ i ] - a[ i ];               // `-r`, le second membre de Newton
            }
            if ( o.residu != NewtonOptions::LIN ) {      // `J = diag( g' / nu ) L` : `L d = ( nu / g' ) ( c - g )`,
                TF su = 0, sug = 0;                      // `c` la moyenne ponderee qui fait sommer `b` a zero
                for ( SI i = 0; i < n; ++i ) {
                    const TF x = a[ i ] / nu[ i ], u = nu[ i ] / gp( x );
                    su += u; sug += u * g( x );
                }
                const TF c = sug / su;
                for ( SI i = 0; i < n; ++i ) {
                    const TF x = a[ i ] / nu[ i ];
                    b[ i ] = nu[ i ] / gp( x ) * ( c - g( x ) );
                }
            }
            if ( it == 0 ) {
                // LE PLANCHER D'AIRE DE L'AMORTISSEMENT, ET LES CELLULES QU'IL DEFEND.
                //
                // Il se lisait `0.5 min( min nu, min a )` sur TOUTES les cellules. Une seule cellule
                // vide au depart mettait donc `eps` a ZERO -- et alors le critere d'acceptation
                // `m2 >= eps` est satisfait par n'importe quel pas : le garde-fou d'aire disparaissait
                // en silence, et le premier pas pouvait vider des milliers de cellules ( mesure :
                // 1 vide au depart -> 7750 apres un pas, README § 8.6 ). Or un depart avec une poignee
                // de vides est exactement ce que rend une reparation incomplete, et il n'a rien de
                // fatal en soi : a nombre de vides nul, un residu de depart quatre fois pire ne coute
                // qu'une iteration ( § 8.6 ).
                //
                // Le plancher se lit donc sur les cellules VIVANTES, et il ne defend que celles-la :
                // une cellule deja vide ne peut pas etre remontee par l'amortissement, et l'exiger
                // au-dessus du plancher refuserait TOUT pas.
                TF am = INFINI, nm = nu[ 0 ];
                protegee.assign( n, 0 );
                for ( SI i = 0; i < n; ++i ) {
                    nm = std::min( nm, nu[ i ] );
                    if ( a[ i ] > 0 ) { am = std::min( am, a[ i ] ); protegee[ i ] = 1; }
                }
                eps = TF( 0.5 ) * std::min( nm, am < INFINI ? am : nm );
            }
            const TF nr = merite( a );
            if ( o.plancher > 0 && nr < o.plancher ) {
                st.fin = "PLANCHER DE BRUIT";            // la mesure ne sait plus rien dire
                st.reste = pire;
                return false;
            }
            if ( o.progres_min > 0 && it > 0 && nr > ( 1 - o.progres_min ) * nr_prec ) {
                st.fin = "PROGRES INSUFFISANT";          // le bruit de la mesure, pas un echec
                st.reste = pire;
                return false;
            }
            nr_prec = nr;
            st.reste = pire;
            if ( it == 0 ) st.reste0 = pire;

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
            if ( it == 0 ) {                             // ce que l'etape a reclame, pour qui enchaine
                TF m = 0;
                for ( SI i = 0; i < n; ++i ) m = std::max( m, std::fabs( d[ i ] ) );
                st.amp_d0 = m;
            }
            if ( it == o.extraire ) {                    // la direction est ce qu'on venait chercher
                st.fin = "DIRECTION EXTRAITE";
                return false;
            }

            // ---- LE PAS PAR LES LIMITES, s'il est demande
            TF t = 1, alpha_lim = -1;
            t_sur = -1;
            d_sur.clear();
            TF gain = 1;                                 // ce que `t = 1` vise : `nu` pour `d`, la cible
                                                         // partielle `theta` pour un pas tensoriel

            // ---- LA CIBLE MODIFIEE ( `Cible.h` ) : on ne touche QU'AU SECOND MEMBRE, et on
            // resout sur la meme factorisation -- aucun diagramme, aucun assemblage
            // ( SEULES se branche plus bas, dans la boucle de `essai-limites` )
            if ( o.cible.mode == OptionsCible::PLAFOND || o.cible.mode == OptionsCible::GEL ) {
                const double tc0 = now();
                EtatCible ec;
                ec.u0 = pas_par_flux( L, a, d, eps, ucel, ec.lie0, &sflux );
                ec.vise = std::min( TF( 1 ), o.cible.facteur * ec.u0 );
                if ( ec.lie0 >= 0 && ec.u0 < o.cible.seuil ) {
                    // `resoudre2` : une descente de plus, sur la factorisation deja faite
                    auto resoudre2 = [ & ]( TF lam ) {
                        melange_cible( b, nu, dans, lam, b2 );
                        ++ec.nb_res;
                        if ( lin.sait_encore() ) { lin.resout_encore( b2, d2 ); return true; }
                        return lin.resout( L, b2, d2 );
                    };
                    auto noyau = [ & ]( int anneaux ) {
                        dans.assign( n, 0 );
                        ec.nb_noyau = 0;
                        for ( SI i = 0; i < n; ++i )
                            if ( ucel[ i ] < ec.vise ) { dans[ i ] = 1; ++ec.nb_noyau; }
                        ec.nb_patch = dilate( L, dans, anneaux );
                    };

                    if ( o.cible.mode == OptionsCible::PLAFOND ) {
                        // LE PLAFOND, sans rien modifier : le patch gele rend `d` harmonique
                        // dedans, ce qui est le plus petit gradient qu'un second membre atteigne
                        if ( o.trace )
                            std::printf( "      cible : U* %.3e ( cellule %d ), vise %.3e -- plafond harmonique :\n",
                                         double( ec.u0 ), int( ec.lie0 ), double( ec.vise ) );
                        for ( int r : { 1, 2, 4, 8 } ) {
                            noyau( r );
                            resoudre2( 0 );
                            SI li = -1;
                            const TF u = pas_par_flux( L, a, d2, eps, u2, li );
                            if ( o.trace )
                                std::printf( "              %d anneaux : noyau %d, patch %d ( %.2f %% ) -> U* %.3e"
                                             "  ( x%.2f ), borne par %d %s\n",
                                             r, int( ec.nb_noyau ), int( ec.nb_patch ),
                                             100.0 * double( ec.nb_patch ) / double( n ), double( u ),
                                             double( u / ec.u0 ), int( li ),
                                             li >= 0 && dans[ li ] ? "DEDANS" : "dehors" );
                        }
                        // ... ET L'AUTRE SENS : nourrir le noyau au lieu de le taire
                        dans.assign( n, 0 );
                        for ( SI i = 0; i < n; ++i ) dans[ i ] = ucel[ i ] < ec.vise;
                        TF sn = 0;
                        for ( SI i = 0; i < n; ++i ) sn += nu[ i ];
                        for ( TF kap : { TF( 0.25 ), TF( 1 ), TF( 4 ), TF( 16 ), TF( 64 ) } ) {
                            const TF don = nourri_cible( b, nu, sflux, dans, kap, b2 );
                            ++ec.nb_res;
                            if ( lin.sait_encore() ) lin.resout_encore( b2, d2 ); else lin.resout( L, b2, d2 );
                            SI li = -1;
                            const TF u = pas_par_flux( L, a, d2, eps, u2, li );
                            TF s2 = 0, sb = 0;
                            for ( SI i = 0; i < n; ++i ) { const TF e = b2[ i ] - b[ i ]; s2 += e * e; sb += b[ i ] * b[ i ]; }
                            if ( o.trace )
                                std::printf( "              nourri kappa %5.2f : masse donnee %.2e = %.2f %% de nu,"
                                             " |db|/|b| %.2e -> U* %.3e  ( x%.2f ), borne par %d %s\n",
                                             double( kap ), double( don ), 100.0 * double( don / sn ),
                                             double( sb > 0 ? std::sqrt( s2 / sb ) : TF( 0 ) ), double( u ),
                                             double( u / ec.u0 ), int( li ), li >= 0 && dans[ li ] ? "le noyau" : "une autre" );
                        }
                    } else {
                        // LA DOSE AUTO-REGLEE : `delta_i = kappa ( S_i - ( a_i - eps ) / F )+`, nulle
                        // d'elle-meme sur les cellules qui tiennent deja. `U*( kappa )` n'est PAS
                        // monotone -- sauver les unes en tue d'autres -- donc on balaye et on garde
                        // le meilleur. Chaque essai est une descente de plus, jamais un diagramme.
                        // L'HORIZON : le pas qu'on veut vraiment atteindre. En ESSAI_LIMITES
                        // c'est `beta`, l'essai que l'iteration va faire -- pas un multiple du pas
                        // courant, qui n'est qu'une lecture.
                        // ... mais seulement pour le critere POPULATION : la DOSE, elle, se
                        // regle sur `vise`, sinon le deficit se calcule contre un horizon que
                        // presque aucune cellule n'atteint et tout le diagramme devient « malade ».
                        const TF horiz = o.cible.critere == OptionsCible::POPULATION
                                       && o.pas == NewtonOptions::ESSAI_LIMITES
                                       ? std::min( TF( 1 ), beta ) : ec.vise;
                        auto malades = [ & ]( const std::vector<TF> &U ) {
                            SI c = 0;
                            for ( SI i = 0; i < n; ++i ) c += U[ i ] < horiz;
                            return c;
                        };
                        // ================= LE JUGE POLYNOME ( § 7 ) =================
                        // `U` n'est pas un predicteur d'extinction, c'est une linearisation des
                        // flux en `t = 0`. Le predicteur, c'est l'aire a combinatoire figee. On la
                        // construit pour les seules candidates, puis on lit TOUT le balayage
                        // dessus -- zero calcul de cellule par essai, une seule resolution.
                        bool poly_fait = false;
                        if constexpr ( PD::dim == 2 ) {
                        if ( o.cible.juge == OptionsCible::POLYNOME ) {
                            const double tp0 = now();
                            // LE FILTRE. `U` est pessimiste, donc `U_i < X` contient toutes les
                            // cellules dont la limite du polynome est sous `X` : c'est un bon
                            // crible, a condition de le regler a l'echelle du POLYNOME et non a
                            // celle de `U`. On prend donc la limite trouvee au tour d'avant, et
                            // on compte apres coup ce qu'on a MANQUE.
                            const TF seuil_mod = std::max( o.cible.filtre * ec.vise, seuil_prec );
                            imod.assign( n, -1 );
                            SI nm = 0;
                            for ( SI i = 0; i < n; ++i )
                                if ( ucel[ i ] < seuil_mod ) imod[ i ] = nm++;
                            mods.assign( nm, ModeleCellule{} );
                            parallel_for( n, par, [ & ]( SI k, int ) {
                                const SI id = pd.ids[ k ];
                                if ( imod[ id ] < 0 ) return;
                                typename PD::Cell cel;
                                pd.cellule( k, cel );
                                mods[ imod[ id ] ].depuis( cel, id, P, w.data() );
                            } );
                            // `e` : la correction de direction par unite de `kappa`. Le coeur et
                            // les donneurs ne dependent pas de `kappa`, donc `delta` lui est
                            // proportionnel et UNE resolution sert tout le balayage.
                            SI nc = 0, na = 0, nd = 0;
                            // LA LIMITE LUE SUR LE POLYNOME, pour une dose donnee : trois
                            // evaluations d'aire par candidate suffisent ( elle est de degre 2 ).
                            // LA LIMITE LUE SUR LE POLYNOME, pour une dose donnee : trois
                            // evaluations d'aire par candidate suffisent ( elle est de degre 2 ),
                            // et la geometrie ne bouge pas -- zero calcul de cellule par essai.
                            t1.resize( n ); t2.resize( n );
                            alim.assign( n, INFINI );
                            auto limite_poly = [ & ]( TF kap, SI &lie, bool garde ) {
                                for ( SI i = 0; i < n; ++i ) {
                                    const TF v = d[ i ] + kap * ecor[ i ];
                                    t1[ i ] = v; t2[ i ] = 2 * v;
                                }
                                TF am = INFINI;
                                lie = -1;
                                for ( SI i = 0; i < n; ++i ) {
                                    if ( imod[ i ] < 0 ) continue;
                                    const ModeleCellule &m = mods[ imod[ i ] ];
                                    if ( m.nb < 3 ) continue;
                                    const TF A0 = m.aire( nullptr ), A1 = m.aire( t1.data() ), A2 = m.aire( t2.data() );
                                    PolyCellule q;
                                    q.a0 = A0;
                                    q.a1 = 2 * A1 - TF( 1.5 ) * A0 - TF( 0.5 ) * A2;
                                    q.a2 = TF( 0.5 ) * ( A2 - 2 * A1 + A0 );
                                    const TF r = q.premiere_racine( eps );
                                    if ( garde ) alim[ i ] = r;
                                    if ( r < am ) { am = r; lie = i; }
                                }
                                return am;
                            };
                            SI lp = -1;
                            ecor.assign( n, TF( 0 ) );   // pas encore de correction : `kappa = 0`
                            const TF a_ref = limite_poly( 0, lp, true );
                            seuil_prec = a_ref;
                            // CE QU'ON A MANQUE : une cellule non modelisee dont `U` tombe sous la
                            // limite trouvee aurait pu border. On le compte plutot que l'ignorer.
                            SI manques = 0;
                            for ( SI i = 0; i < n; ++i ) manques += imod[ i ] < 0 && ucel[ i ] < a_ref;
                            // LE COEUR, DESIGNE PAR LE BON PREDICTEUR : les cellules dont le
                            // POLYNOME dit qu'elles s'eteignent avant la cible.
                            const TF cible_poly = std::min( TF( 1 ), o.cible.facteur * a_ref );
                            coeurp.assign( n, 0 );
                            SI ncp = 0;
                            for ( SI i = 0; i < n; ++i )
                                if ( imod[ i ] >= 0 && alim[ i ] < cible_poly ) { coeurp[ i ] = 1; ++ncp; }
                            nourri_local( L, b, nu, sflux, a, d, eps, ec.vise, TF( 1 ), o.cible.repris,
                                          o.cible.ep_anneau, b2, dlt, touche, nc, na, nd, &coeurp );
                            ++ec.nb_res;
                            if ( lin.sait_encore() ) lin.resout_encore( dlt, ecor );
                            else if ( ! lin.resout( L, dlt, ecor ) ) ecor.assign( n, TF( 0 ) );
                            TF abest = a_ref, kp = 0;
                            SI lbest = lp;
                            for ( TF kap : { TF( 0.25 ), TF( 0.5 ), TF( 1 ), TF( 2 ), TF( 4 ) } ) {
                                SI li = -1;
                                const TF al = limite_poly( kap, li, false );
                                if ( al > abest ) { abest = al; kp = kap; lbest = li; }
                            }
                            if ( o.trace )
                                std::printf( "      cible/poly : %d modeles ( U < %.2e, %d manques ), limite"
                                             " %.3e  ( U* dit %.3e, soit x%.0f trop petit ), %d a nourrir"
                                             " -> kappa %.2f, limite %.3e  ( x%.2f ), %.3f s\n",
                                             int( nm ), double( seuil_mod ), int( manques ), double( a_ref ),
                                             double( ec.u0 ), double( ec.u0 > 0 ? a_ref / ec.u0 : 0 ),
                                             int( ncp ), double( kp ), double( abest ),
                                             double( a_ref > 0 ? abest / a_ref : 1 ), now() - tp0 );
                            if ( kp > 0 ) {
                                nourri_local( L, b, nu, sflux, a, d, eps, ec.vise, kp, o.cible.repris,
                                              o.cible.ep_anneau, b2, del, touche, nc, na, nd, &coeurp );
                                TF s2 = 0, sb = 0, sn = 0, don = 0;
                                for ( SI i = 0; i < n; ++i ) {
                                    const TF e = b2[ i ] - b[ i ];
                                    s2 += e * e; sb += b[ i ] * b[ i ]; sn += nu[ i ];
                                    if ( del[ i ] > 0 ) don += del[ i ];
                                }
                                for ( SI i = 0; i < n; ++i ) d[ i ] += kp * ecor[ i ];
                                d[ 0 ] = 0;
                                ec.u1 = abest; ec.u0 = a_ref; ec.lie1 = lbest;
                                ec.nb_noyau = nc; ec.lambda = kp;
                                ec.masse = sn > 0 ? don / sn : TF( 0 );
                                ec.ecart = sb > 0 ? std::sqrt( s2 / sb ) : TF( 0 );
                                gain = std::max( TF( 0 ), TF( 1 ) - ec.ecart );
                                ++st.nb_cible_pris;
                                st.cible_gain += std::log( abest / a_ref );
                            }
                            poly_fait = true;
                        }
                        }
                        if ( ! poly_fait ) {
                        ec.u1 = ec.u0;
                        ec.lie1 = ec.lie0;
                        const SI mal0 = malades( ucel );
                        SI mbest = mal0, nc = 0, na = 0, nd = 0;
                        TF kbest = 0, fuite = 0, emax = 0;
                        for ( TF kap : { TF( 0.25 ), TF( 0.5 ), TF( 1 ), TF( 2 ), TF( 4 ) } ) {
                            nourri_local( L, b, nu, sflux, a, d, eps, horiz, kap, o.cible.repris,
                                          o.cible.ep_anneau, b2, del, touche, nc, na, nd );
                            ++ec.nb_res;
                            if ( lin.sait_encore() ) lin.resout_encore( b2, d2 );
                            else if ( ! lin.resout( L, b2, d2 ) ) break;
                            SI li = -1;
                            const TF u = pas_par_flux( L, a, d2, eps, u2, li );
                            const SI m = malades( u2 );
                            const bool mieux = o.cible.critere == OptionsCible::POPULATION
                                             ? m < mbest : u > ec.u1;
                            if ( mieux ) {
                                mbest = m; ec.u1 = u; ec.lie1 = li; kbest = kap; dgard = d2;
                                fuite = fuite_direction( d, d2, touche, emax );
                            }
                        }
                        ec.lie0 = mal0;                      // la trace : malades au depart
                        ec.nb_patch = mbest;                 // ... et ce qu'il en reste
                        ec.lambda = kbest;
                        if ( kbest > 0 ) {
                            const TF don = nourri_local( L, b, nu, sflux, a, d, eps, horiz, kbest,
                                                         o.cible.repris, o.cible.ep_anneau,
                                                         b2, del, touche, nc, na, nd );
                            TF s2 = 0, sb = 0, sn = 0;
                            for ( SI i = 0; i < n; ++i ) {
                                const TF e = b2[ i ] - b[ i ];
                                s2 += e * e; sb += b[ i ] * b[ i ]; sn += nu[ i ];
                            }
                            ec.nb_noyau = nc;
                            ec.masse = sn > 0 ? don / sn : TF( 0 );
                            ec.ecart = sb > 0 ? std::sqrt( s2 / sb ) : TF( 0 );
                            d.swap( dgard );
                            ++st.nb_cible_pris;
                            // LE MERITE RESTE SUR `nu` : a l'ordre un le residu devient
                            // `( 1 - t ) r - t delta`, donc la decroissance garantie n'est plus
                            // `t / 2` mais `( 1 - |delta| / |r| ) t / 2`. On le dit au test.
                            gain = std::max( TF( 0 ), TF( 1 ) - ec.ecart );
                            st.cible_gain += std::log( ec.u1 / ec.u0 );
                        }
                        if ( o.trace )
                            std::printf( "      cible : U* %.3e -> %.3e ( x%.2f ), malades %d -> %d, horizon %.3e,"
                                         " kappa %.2f, %d nourries en %d amas, masse %.2f %%,"
                                         " |db|/|b| %.2e, FUITE %.1f %%, %d res%s\n",
                                         double( ec.u0 ), double( ec.u1 ), double( ec.u1 / ec.u0 ),
                                         int( ec.lie0 ), int( ec.nb_patch ), double( horiz ),
                                         double( ec.lambda ), int( ec.nb_noyau ), int( na ),
                                         100.0 * double( ec.masse ), double( ec.ecart ),
                                         100.0 * double( fuite ), ec.nb_res,
                                         kbest > 0 ? "" : "  ( RIEN NE FAIT MIEUX )" );
                        }
                    }
                }
                st.nb_cible_res += ec.nb_res;
                st.t_cible += now() - tc0;
            }

            if ( o.pas != NewtonOptions::ESSAIS && o.pas != NewtonOptions::ESSAI_LIMITES ) {
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
                        PasTensoriel pt;
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

            // ---- L'ESSAI PUIS LES LIMITES LOCALES : le diagramme du pas d'abord, et si des cellules
            // y passent sous `eps`, leurs limites ( a elles seules ), le pas ramene sous la plus
            // petite, et on recommence -- la non-monotonie peut en reveler d'autres
            bool deja = false;                           // ESSAI_LIMITES : le diagramme en `t` est deja fait
            if ( o.pas == NewtonOptions::ESSAI_LIMITES ) {
                if constexpr ( PD::dim == 2 ) {
                    t = beta;
                    OptionsLimites ol = o.lim;
                    ol.niveau = eps;
                    ol.global = true;
                    std::vector<SI> mauvaises;
                    w2.resize( n );
                    TF t_fait = -1;                      // le pas dont le diagramme est dans `a2`
                    const TF t_max_essai = beta;         // on ne depasse pas l'essai qu'on visait

                    // ---- LE DOSEUR BRANCHE SUR `seules` ( `Cible.h`, mode SEULES )
                    //
                    // Le bon predicteur -- le polynome d'aire a combinatoire figee -- n'est pas
                    // cher PAR CELLULE ; ce qui coute, c'est de savoir LESQUELLES modeliser
                    // ( § 20.6 : un crible en `U` en attrape 99 % ). Ici la liste est deja la,
                    // elle est courte, et elle ne soupconne pas : un diagramme a VU ces cellules
                    // passer sous `eps`. On les nourrit juste assez pour que le polynome les donne
                    // vivantes en `t`, et on re-essaye le meme pas au lieu de le diviser.
                    //
                    // `ModeleCellule::aire` est une aire de LEBESGUE : pas de densite ici.
                    int def_faites = 0;
                    // Rend la limite ATTEINTE ( 0 : rien a faire ). On ne vise pas `beta` -- aller
                    // de 4e-3 a 0.25 serait un facteur soixante sur six mille cellules, et le
                    // doseur a raison de refuser -- mais `facteur` fois la limite COURANTE.
                    auto deformer = [ & ]( const std::vector<SI> &cibles, TF t_max ) -> TF {
                        const double tc0 = now();
                        imod.assign( n, -1 );
                        SI nm = 0;
                        for ( SI i : cibles ) if ( imod[ i ] < 0 ) imod[ i ] = nm++;
                        mods.assign( nm, ModeleCellule{} );
                        pd.set_weights( w.data(), par );
                        parallel_for( n, par, [ & ]( SI k, int ) {
                            const SI id = pd.ids[ k ];
                            if ( imod[ id ] < 0 ) return;
                            typename PD::Cell cel;
                            pd.cellule( k, cel );
                            mods[ imod[ id ] ].depuis( cel, id, P, w.data() );
                        } );
                        // le debit sortant de la direction COURANTE : l'echelle de la dose
                        SI lx = -1;
                        pas_par_flux( L, a, d, eps, ucel, lx, &sflux );
                        coeurp.assign( n, 0 );
                        for ( SI i : cibles ) coeurp[ i ] = 1;
                        SI nc = 0, na = 0, nd = 0;
                        nourri_local( L, b, nu, sflux, a, d, eps, t_max, TF( 1 ), o.cible.repris,
                                      o.cible.ep_anneau, b2, dlt, touche, nc, na, nd, &coeurp );
                        ++st.nb_cible_res;
                        if ( lin.sait_encore() ) lin.resout_encore( dlt, ecor );
                        else if ( ! lin.resout( L, dlt, ecor ) ) { st.t_cible += now() - tc0; return 0; }
                        TF dose = 0, emx = 0, dmx = 0;
                        SI ndose = 0;
                        for ( SI i = 0; i < n; ++i ) {
                            if ( dlt[ i ] > 0 ) { dose += dlt[ i ]; ++ndose; }
                            emx = std::max( emx, std::fabs( ecor[ i ] ) );
                            dmx = std::max( dmx, std::fabs( d[ i ] ) );
                        }
                        t1.resize( n ); t2.resize( n );
                        auto limite = [ & ]( TF kap ) {
                            for ( SI i = 0; i < n; ++i ) {
                                const TF v = d[ i ] + kap * ecor[ i ];
                                t1[ i ] = v; t2[ i ] = 2 * v;
                            }
                            TF am = INFINI;
                            for ( SI i : cibles ) {
                                const ModeleCellule &m = mods[ imod[ i ] ];
                                if ( m.nb < 3 ) continue;
                                const TF A0 = m.aire( nullptr ), A1 = m.aire( t1.data() ), A2 = m.aire( t2.data() );
                                PolyCellule q;
                                q.a0 = A0;
                                q.a1 = 2 * A1 - TF( 1.5 ) * A0 - TF( 0.5 ) * A2;
                                q.a2 = TF( 0.5 ) * ( A2 - 2 * A1 + A0 );
                                am = std::min( am, q.premiere_racine( eps ) );
                            }
                            return am;
                        };
                        const TF a_av = limite( 0 );
                        // UNE CELLULE DEJA AU PLANCHER rend `a_av` nul, et alors tout `kappa`
                        // « atteint » une cible nulle : le doseur declarerait victoire en imposant
                        // un pas nul. On rend la main aux limites.
                        if ( ! ( a_av > 0 ) ) { st.t_cible += now() - tc0; return 0; }
                        // ... ET LE POLYNOME QUI MENT. Ces cellules-la, un DIAGRAMME les a vues
                        // sous `eps` en `t_max` ; si le polynome les donne vivantes au-dela, c'est
                        // qu'un voisin NOUVEAU les a mangees ( § 7 : invisible depuis la cellule
                        // seule ). On ne dose pas sur une prediction qu'on sait fausse.
                        if ( a_av >= t_max ) { st.t_cible += now() - tc0; return 0; }
                        // LE COMPROMIS, et c'est le seul reglage qui compte : on ne vise pas, on
                        // prend la MEILLEURE limite achetable dans un budget de deformation
                        // `| db | / | b |`. Le budget se traduit exactement en plafond sur
                        // `kappa`, `delta` lui etant proportionnel -- aucun essai perdu.
                        TF nb2 = 0, nd1 = 0;
                        for ( SI i = 0; i < n; ++i ) { nb2 += b[ i ] * b[ i ]; nd1 += dlt[ i ] * dlt[ i ]; }
                        nb2 = std::sqrt( nb2 ); nd1 = std::sqrt( nd1 );
                        const TF kmax = o.cible.budget > 0 && nd1 > 0
                                      ? o.cible.budget * nb2 / nd1 : INFINI;
                        TF kb = 0, ab = a_av;
                        for ( TF kap : { TF( 0.0625 ), TF( 0.125 ), TF( 0.25 ), TF( 0.5 ), TF( 1 ),
                                         TF( 2 ), TF( 4 ), TF( 8 ), TF( 16 ), TF( 32 ) } ) {
                            if ( kap > kmax ) break;
                            const TF al = limite( kap );
                            if ( al > ab * TF( 1.02 ) ) { ab = al; kb = kap; }
                        }
                        const bool pris = kb > 0;
                        TF ecart = 0;
                        if ( pris && d_sur.empty() ) d_sur = d;   // LE FILET, pose une seule fois
                        if ( pris ) {
                            nourri_local( L, b, nu, sflux, a, d, eps, t_max, kb, o.cible.repris,
                                          o.cible.ep_anneau, b2, del, touche, nc, na, nd, &coeurp );
                            TF s2 = 0, sb = 0;
                            for ( SI i = 0; i < n; ++i ) {
                                const TF e = b2[ i ] - b[ i ];
                                s2 += e * e; sb += b[ i ] * b[ i ];
                            }
                            ecart = sb > 0 ? std::sqrt( s2 / sb ) : TF( 0 );
                            for ( SI i = 0; i < n; ++i ) d[ i ] += kb * ecor[ i ];
                            d[ 0 ] = 0;
                            // le merite reste sur `nu` : la decroissance garantie tombe a
                            // `( 1 - |delta| / |r| ) t / 2` ( § 20.3 )
                            gain = std::max( TF( 0 ), TF( 1 ) - ecart );
                            ++st.nb_cible_pris;
                            st.cible_gain += std::log( std::min( ab / a_av, TF( 1e6 ) ) );
                        }
                        if ( o.trace )
                            std::printf( "      cible/seules : %d mauvaises modelisees, polynome %.3e -> %.3e"
                                         " ( x%.2f ), %d dosees, kappa %.3f ( plafond %.3f ),"
                                         " |db|/|b| %.2e, masse %.1f %% -- %s ( %.3f s )\n",
                                         int( nm ), double( a_av ), double( ab ), double( ab / a_av ),
                                         int( ndose ), double( kb ), double( std::min( kmax, TF( 99 ) ) ),
                                         double( ecart ), 100.0 * double( kb * dose ),
                                         pris ? "on REESSAYE plus loin" : "rien a acheter",
                                         now() - tc0 );
                        st.t_cible += now() - tc0;
                        return pris ? std::min( ab, t_max ) : TF( 0 );
                    };

                    for ( int tour = 0; tour < 8; ++tour ) {
                        for ( SI i = 0; i < n; ++i ) w2[ i ] = w[ i ] + t * d[ i ];
                        w2[ 0 ] = 0;
                        mesures_et_facettes( w2, a2, fa2, pda2 );
                        t_fait = t;
                        mauvaises.clear();
                        for ( SI i = 0; i < n; ++i ) if ( protegee[ i ] && a2[ i ] < eps ) mauvaises.push_back( i );
                        if ( mauvaises.empty() ) break;
                        st.nb_cell_mauvaises += SI( mauvaises.size() );
                        ++st.nb_tours_essai;
                        t0 = now();
                        ol.horizon = t;
                        pd.set_weights( w.data(), par );
                        if ( rho )                       // en masse : la bissection, pas le polynome
                            limites_masse( pd, P, w, d, par, ol, lim, Voisinage{ L.row.data(), L.col.data() }, mauvaises,
                                           [ & ]( const typename PD::Cell &cel ) { return rho->mesure( cel, []( int, TF ) {} ); } );
                        else
                            limites( pd, P, w, d, par, ol, lim, Voisinage{ L.row.data(), L.col.data() }, &mauvaises );
                        TF al = t;
                        for ( SI i : mauvaises ) { st.nb_cell_lim += lim[ i ].tours; al = std::min( al, lim[ i ].alpha ); }
                        st.t_lim += now() - t0;
                        if ( o.trace )
                            std::printf( "      essai t %.3e : %d cellules sous eps, limite locale %.3e ( %.3f s, %d cellules calculees )\n",
                                         double( t ), int( mauvaises.size() ), double( al ), now() - t0,
                                         int( [ & ]{ SI c = 0; for ( SI i : mauvaises ) c += lim[ i ].tours; return c; }() ) );
                        t = o.facteur * al;
                        if ( o.cible.mode == OptionsCible::SEULES && rho == nullptr
                             && def_faites < o.cible.essais
                             && ( o.cible.max_seules <= 0 || SI( mauvaises.size() ) <= o.cible.max_seules ) ) {
                            // LE PAS SUR, celui que les limites viennent de donner a la direction
                            // de Newton : on n'y touche plus, et on y revient si la deformation
                            // ne rend rien.
                            const TF t_garde = t;
                            const bool neuf = d_sur.empty();
                            const TF na = deformer( mauvaises, t_max_essai );
                            if ( na > 0 ) {
                                if ( neuf ) t_sur = t_garde;
                                ++def_faites;
                                t = std::min( t_max_essai, o.facteur * na );
                                continue;                // direction corrigee, pas plus long
                            }
                        }
                        if ( t < o.t_min ) break;
                    }
                    // une limite nulle ( la cellule est deja au plancher, ou la bissection n'a rien
                    // trouve ) n'est pas une raison de stagner : on rend la main aux essais, depuis la
                    // moitie du dernier pas calcule
                    if ( t < o.t_min ) t = t_fait / 2;
                    deja = t == t_fait;
                    alpha_lim = t;                       // pour la trace : le pas retenu
                    // le prochain essai : `mult_ok` fois celui-ci s'il est passe direct, et jamais moins
                    // que `confiance` fois le pas retenu
                    const bool direct = t >= beta;
                    beta = std::min( TF( 1 ), std::max( direct ? o.mult_ok * beta : beta, o.confiance * t ) );
                    // le diagramme en `t` est fait : on rejoint l'amortissement au test du residu
                }
            }

            // ---- L'AMORTISSEMENT, EN DEUX PASSES
            //
            // Une cible deformee peut allonger le pas ET ne plus faire descendre le residu : la
            // direction n'est plus celle de Newton. Sans filet, l'amortissement echoue et Newton
            // sort en STAGNATION -- a un residu qui n'a presque pas bouge, ce qui rend toute
            // comparaison de diagrammes MENSONGERE. On revient donc a la direction de Newton et
            // au pas que les limites lui avaient donne, et on recommence.
            bool pris = false;
            TF t_lim0 = t;
            w2.resize( n );
            for ( int passe = 0; passe < 2 && ! pris; ++passe ) {
                if ( passe == 1 ) {                      // la deformation n'a rien rendu
                    if ( d_sur.empty() || t_sur <= 0 ) break;
                    d.swap( d_sur );
                    d_sur.clear();
                    t = t_sur;
                    t_lim0 = t;
                    gain = 1;
                    deja = false;
                    ++st.nb_cible_refus;
                }
                for ( int essai = 0; essai < o.max_reculs; ++essai ) {
                    if ( ! ( essai == 0 && deja ) ) {    // sinon, deja fait en `t`
                        for ( SI i = 0; i < n; ++i ) w2[ i ] = w[ i ] + t * d[ i ];
                        w2[ 0 ] = 0;                     // la jauge, imposee et non esperee
                        mesures_et_facettes( w2, a2, fa2, pda2 );
                    }
                    TF m2 = INFINI;                      // le plancher `eps` est une aire ABSOLUE, et il ne
                    for ( SI i = 0; i < n; ++i )         // porte que sur les cellules VIVANTES AU DEPART
                        if ( protegee[ i ] ) m2 = std::min( m2, a2[ i ] );
                    const TF n2r = merite( a2 );
                    if ( m2 >= eps && n2r <= ( 1 - gain * t / 2 ) * nr && n2r < nr ) { pris = true; break; }
                    t /= 2;
                    ++st.nb_recul;
                    if ( t < o.t_min )
                        break;
                }
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
            t_prec = t;
            w.swap( w2 );
            a.swap( a2 );
            fa.swap( fa2 );
            if ( pda ) da.swap( da2 );
            if ( o.apres_pas ) o.apres_pas( it, t, st.nb_recul );
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

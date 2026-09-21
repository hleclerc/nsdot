#pragma once

// =====================================================================================
// LES METHODES DU PREMIER ORDRE sur le dual : L-BFGS et le gradient conjugue non lineaire, a
// comparer a Newton -- meme diagramme, meme critere d'arret, meme compteur de diagrammes.
//
// = Ce qu'on minimise
//
//     F( w ) = -Phi( w ),   grad F = a( w ) - nu,   F CONVEXE, invariante par les constantes.
//
// Le gradient est ce que le diagramme livre ( les mesures ), et rien de plus : pas de hessienne,
// pas de systeme lineaire. C'est tout l'interet en 3D, ou la factorisation pese.
//
// = La recherche lineaire SANS VALEUR DE FONCTION
//
// On n'a pas `F` ( il faudrait le second moment de chaque cellule, et avec une densite c'est un
// noyau de plus ) ; on a sa derivee le long de `d`, `phi'( alpha ) = ( a( w + alpha d ) - nu ) . d`,
// et elle est CROISSANTE en `alpha` ( `F` convexe ). Deux consequences, qui font la recherche :
//   * tout `alpha` avec `phi'( alpha ) <= 0` fait decroitre `F` -- `F( alpha ) <= F( 0 ) + alpha phi'( alpha )`
//     par convexite -- donc chaque point essaye du bon cote est un progres garanti, sans Armijo ;
//   * `phi'` est monotone : la SECANTE entre un point ou elle est negative et un ou elle est
//     positive converge vers le minimum sur la droite.
// On accepte des que `|phi'( alpha )| <= c2 |phi'( 0 )|` ( Wolfe forte ), et si la recherche
// s'epuise on garde le dernier point ou `phi' <= 0` ( un progres, quand meme ). La courbure
// `s . y > 0` de L-BFGS est alors automatique : `phi'( alpha ) > phi'( 0 )`.
//
// = Le plancher de masse, comme Newton
//
// Sans lui, ca s'effondre ( mesure : uniforme n = 20 000, 19 702 cellules vides a l'iteration 20 ).
// Une cellule vide a un gradient constant `-nu_i` et une courbure nulle ; la paire `( s, y )` de
// L-BFGS le lui dit ( `y_i = 0` ), et l'inverse de la hessienne devient enorme dans ces directions :
// le pas suivant pousse ces poids sans mesure, ces cellules avalent leurs voisines, qui se vident a
// leur tour. Le remede est celui de l'amortissement KMT : un pas ou une cellule passe sous
// `eps = min( min nu, min a_0 ) / 2` est REFUSE ( les cellules vides au depart sont exemptees tant
// qu'elles le restent : on n'a pas d'autre moyen de les remplir que de les pousser ). Les paires
// n'ont alors jamais de courbure nulle, et la recherche lineaire garde son sens.
//
// = Le preconditionnement « ad hoc »
//
// Le diagramme livre aussi les facettes, donc la DIAGONALE du laplacien pour rien : `H0 = D^-1`
// ( Jacobi, `precond = 1` ) donne l'echelle des poids des la premiere iteration -- sans elle un
// premier pas n'a aucune unite -- et, pour L-BFGS, `H0 = gamma D^-1` avec `gamma = s.y / y.D^-1 y`
// ensuite. Mais une direction Jacobi est RUGUEUSE : `d_i` suit le residu de la cellule `i` seule,
// deux voisines partent dans des sens opposes, et c'est `d_i - d_j` qui deplace leur bissectrice
// ( de `( d_i - d_j ) / 2 |p_i - p_j|` ). Mesure sur l'uniforme : `|d|max = 2 h^2`, une cellule
// de la moitie de la cible vidée par ses voisines des `alpha = 0.25`, puis le plancher ne laisse
// plus que `alpha ~ 5e-3` -- la ou la direction de Newton, LISSE, passe entiere.
//
// D'ou `precond = 2` : `H0 = L0^-1`, le laplacien FACTORISE UNE FOIS ( `resout_encore` du solveur
// lineaire : la descente de Cholesky, ou la hierarchie AMG gardee ), et une descente par iteration.
// La direction a la regularite de Newton, les paires de L-BFGS corrigent ce que la combinatoire a
// change depuis. C'est la methode de la corde avec une memoire, et c'est ce qui reutilise la
// factorisation -- le levier que § 7.4 designait. Elle est REFAITE quand la corde ne mord plus :
// toutes les `refacto` iterations, quand le plancher a borne le pas sous `refacto_borne`, quand
// `|r|_2` n'a pas ete divise par `1 / refacto_taux` au dernier pas, et quand la recherche lineaire
// echoue sur un laplacien perime. A chaque refonte la memoire est VIDEE : mesure, les paires
// apprises sur l'ancien laplacien, bornees par le plancher, tirent la direction fraiche vers la
// cellule qui bloque, et le pas tombe a 1e-15.
// Une cellule VIDE a une diagonale nulle et un gradient `-nu_i` : on lui prete la diagonale
// moyenne, et le pas la remplit peu a peu -- la ou Newton, lui, n'a plus d'equation.
//
// = La bascule vers Newton
//
// `bascule > 0` : des que `max_i |a_i - nu_i| / nu_i <= bascule` ( toute cellule a une masse
// raisonnable, donc le plancher KMT aussi ), on rend la main a Newton depuis la, avec le
// diagramme deja fait. C'est l'hybride : le premier ordre pour partir de loin, Newton pour finir.
// =====================================================================================

#include "solver/Newton.h"
#include <deque>
#include <string>

namespace sf {

struct PremierOrdreOptions {
    enum Methode : int { LBFGS = 0, CG };
    int  methode  = LBFGS;
    int  memoire  = 10;        ///< L-BFGS : paires `( s, y )` gardees
    int  precond  = 2;         ///< `H0` : 0 = `gamma I`, 1 = `D^-1` ( diagonale du laplacien ), 2 = `L0^-1` ( factorise au depart )
    int  refacto  = 0;         ///< precond 2 : refactoriser le laplacien toutes les K iterations ( 0 : jamais )
    TF   refacto_borne = 0.25; ///< precond 2 : refactoriser quand le plancher a borne le pas sous cette fraction ( 0 : jamais )
    TF   refacto_taux = 0.5;   ///< precond 2 : refactoriser quand `|r|_2` n'a pas ete divise par `1 / taux` au dernier pas ( 0 : jamais )
    TF   t_min    = 1e-10;     ///< sous ce pas, STAGNATION
    TF   c2       = 0.5;       ///< Wolfe forte : `|phi'( alpha )| <= c2 |phi'( 0 )|` ( CG : 0.1 conseille )
    int  max_ls   = 12;        ///< diagrammes par recherche lineaire, au plus
    bool plancher = true;      ///< refuser un pas qui met une cellule sous `eps` ( KMT )
    bool sauter_borne = true;  ///< L-BFGS : ne pas garder la paire d'un pas que le plancher a borne
    TF   tol      = 1e-6;      ///< arret : `max_i |a_i - nu_i| <= tol nu_i`
    int  maxit    = 3000;
    TF   bascule  = 0;         ///< > 0 : passer a Newton sous ce `max|a-nu|/nu`
    int  bascule_it = 0;       ///< > 0 : passer a Newton apres autant d'iterations
    bool trace    = true;

    /// l'etiquette des courbes et des tableaux : la methode et ce qui la distingue
    std::string nom() const {
        std::string r = methode == CG ? "cg" : "lbfgs";
        r += precond == 2 ? " L0" : precond == 1 ? " jacobi" : " nu";
        if ( refacto > 0 ) r += " refacto " + std::to_string( refacto );
        if ( refacto_taux > 0 ) { char b[ 32 ]; std::snprintf( b, sizeof( b ), " taux %g", double( refacto_taux ) ); r += b; }
        if ( ! plancher ) r += " sans plancher";
        if ( bascule > 0 ) r += " +newton";
        return r;
    }
};

struct PremierOrdreStats {
    const char *fin = "?";
    int nb_iter = 0, nb_ls = 0, nb_restart = 0, nb_diag = 0;   ///< `nb_diag` : les diagrammes AVANT la bascule
    int nb_plancher = 0;       ///< essais refuses par le plancher
    int nb_facto = 0;          ///< factorisations du laplacien ( precond 2 )
    int it_bascule = -1;       ///< l'iteration ou Newton a pris la main ( -1 : jamais )
    TF  reste = 0, reste0 = 0;
    double temps = 0;
};

/// Le solveur du premier ordre, sur le `Newton` qui porte le diagramme, la cible, la densite et les
/// compteurs ( `nw.st.nb_diag`, `nw.st.t_diag` continuent de compter ; `nw.w`, `nw.a`, `nw.fa` sont
/// l'etat courant, comme apres Newton ).
template<class PD, class Lin>
struct PremierOrdre {
    Newton<PD,Lin>     &nw;
    PremierOrdreOptions o;
    PremierOrdreStats   st;

    PremierOrdre( Newton<PD,Lin> &nw, PremierOrdreOptions o = {} ) : nw( nw ), o( o ) {}

    static TF dot( const std::vector<TF> &x, const std::vector<TF> &y ) {
        TF s = 0;
        for ( SI i = 0; i < SI( x.size() ); ++i ) s += x[ i ] * y[ i ];
        return s;
    }

    /// la diagonale du laplacien depuis les facettes ; une cellule sans facette recoit la moyenne
    static void diagonale( SI n, const std::vector<Facette> &fa, std::vector<TF> &dia ) {
        dia.assign( n, TF( 0 ) );
        for ( const Facette &f : fa ) dia[ f.i ] += f.c;
        TF m = 0; SI k = 0;
        for ( TF v : dia ) if ( v > 0 ) { m += v; ++k; }
        m = k ? m / k : 1;
        for ( TF &v : dia ) if ( ! ( v > TF( 1e-6 ) * m ) ) v = m;   // vide, ou une facette de masse 1e-300 ( densite )
    }

    /// `max_i |a_i - nu_i| / nu_i`, et le nombre de cellules vides
    TF pire( const std::vector<TF> &a, SI *nvide = nullptr ) const {
        TF p = 0; SI nv = 0;
        for ( SI i = 0; i < SI( a.size() ); ++i ) {
            p = std::max( p, std::fabs( nw.nu[ i ] - a[ i ] ) / nw.nu[ i ] );
            nv += ! ( a[ i ] > 0 );
        }
        if ( nvide ) *nvide = nv;
        return p;
    }

    bool resout( const std::vector<TF> &w_init, bool deja_mesure = false ) {
        const SI n = nw.pd.n;
        const double debut = now();
        std::vector<TF> &w = nw.w, &a = nw.a;
        std::vector<Facette> &fa = nw.fa;
        w = w_init;
        if ( ! deja_mesure ) nw.mesures_et_facettes( w, a, fa );
        if ( nw.o.apres_pas ) nw.o.apres_pas( -1, 0, 0 );

        std::vector<TF> g( n ), g_old, d( n ), dia, pg, y, s, w2( n ), a2, g2( n ), w_lo, a_lo, g_lo, q, r;
        std::vector<Facette> fa2, fa_lo;
        std::deque<std::vector<TF>> S, Y;               // L-BFGS
        std::deque<TF> RHO;
        std::vector<TF> d_old, pg_old;                   // CG
        Laplacien L;                                     // precond 2
        TF phi0_old = 0, alpha_old = 1, eps = 0;
        bool borne = false;                              // le dernier pas a-t-il ete borne par le plancher ?
        bool forcer = false, frais = false;              // refactoriser ( la recherche a echoue sur un laplacien perime ) ; l'a-t-on fait ?
        TF nr_ref = INFINI, nr_prec = 0; int it_ref = 0;              // la stagnation : `|r|_2` n'a pas baisse de 1 % en 20 iterations
        // `x = L0^-1 b` : la factorisation faite ( `refaire` ) ou reprise ( `resout_encore`, si le
        // solveur l'offre -- Cholesky ; sinon on resout a nouveau, hierarchie comprise )
        auto applique_L0 = [ & ]( bool refaire, const std::vector<TF> &b, std::vector<TF> &x ) {
            const double t0 = now();
            bool ok = true;
            if constexpr ( requires { nw.lin.resout_encore( b, x ); } ) {
                if ( refaire ) ok = nw.lin.resout( L, b, x );
                else nw.lin.resout_encore( b, x );
            } else
                ok = nw.lin.resout( L, b, x );
            nw.st.t_lin += now() - t0;
            return ok;
        };
        int it = 0;
        for ( ; it < o.maxit; ++it ) {
            SI nvide = 0;
            const TF p = pire( a, &nvide );
            st.reste = p;
            if ( it == 0 ) st.reste0 = p;
            for ( SI i = 0; i < n; ++i ) g[ i ] = a[ i ] - nw.nu[ i ];
            const TF nr = nw.merite( a );
            if ( p <= o.tol ) {
                if ( o.trace ) std::printf( "    it %3d  |r|_2 %.3e  max|a-nu|/nu %.3e  CONVERGE\n", it, double( nr ), double( p ) );
                st.fin = "CONVERGE";
                break;
            }
            if ( nr < TF( 0.99 ) * nr_ref ) { nr_ref = nr; it_ref = it; }
            else if ( it - it_ref >= 20 ) {
                if ( o.trace ) std::printf( "    it %3d  |r|_2 %.3e  max|a-nu|/nu %.3e  STAGNATION ( pas 1 %% en 20 iterations )\n", it, double( nr ), double( p ) );
                st.fin = "STAGNATION";
                break;
            }
            if ( ( o.bascule > 0 && p <= o.bascule ) || ( o.bascule_it > 0 && it >= o.bascule_it ) ) {
                if ( o.trace ) std::printf( "    it %3d  |r|_2 %.3e  max|a-nu|/nu %.3e  -> NEWTON\n", it, double( nr ), double( p ) );
                st.it_bascule = it;
                st.nb_diag = nw.st.nb_diag;
                st.temps = now() - debut;
                const bool ok = nw.resout( w, true );
                st.fin = nw.st.fin;
                st.reste = nw.st.reste;
                return ok;
            }
            if ( it == 0 ) {                             // le plancher, comme Newton ( sur les cellules non vides )
                TF am = INFINI, nm = nw.nu[ 0 ];
                for ( SI i = 0; i < n; ++i ) { if ( a[ i ] > 0 ) am = std::min( am, a[ i ] ); nm = std::min( nm, nw.nu[ i ] ); }
                eps = o.plancher ? TF( 0.5 ) * std::min( nm, am ) : 0;
            }
            ++st.nb_iter;
            const int g0 = nw.st.nb_diag;
            const double d0 = nw.st.t_diag, l0 = nw.st.t_lin;

            // ---- LA DIRECTION
            diagonale( n, fa, dia );
            pg.resize( n );
            if ( o.precond == 2 ) {                      // `L0^-1 g` : la factorisation du depart ( ou refaite )
                // la factorisation du depart, ou refaite : periodiquement, ou quand le plancher vient
                // de borner le pas -- la cellule qui bloque a change de combinatoire, et seul un
                // laplacien a jour lui rend une equation ( sans ca, mesure : alpha -> 1e-15 )
                const bool refaire = it == 0 || ( o.refacto > 0 && it % o.refacto == 0 )
                                  || ( o.refacto_borne > 0 && borne && alpha_old < o.refacto_borne )
                                  || ( o.refacto_taux > 0 && nr > o.refacto_taux * nr_prec ) || forcer;
                frais = refaire;
                if ( refaire ) {                         // la memoire repart avec le laplacien : les paires apprises
                    L.assemble( n, fa );                 // sur l'ancien ( et bornees par le plancher ) tirent la
                    ++st.nb_facto;                       // direction fraiche vers la cellule qui bloque ( mesure )
                    S.clear(); Y.clear(); RHO.clear();
                }
                if ( ! applique_L0( refaire, g, pg ) ) { st.fin = "SOLVEUR LINEAIRE EN ECHEC"; break; }
            } else
                for ( SI i = 0; i < n; ++i ) pg[ i ] = o.precond ? g[ i ] / dia[ i ] : g[ i ];
            bool restart = false;
            if ( o.methode == PremierOrdreOptions::LBFGS ) {
                // les deux boucles, `H0 = gamma D^-1`
                q = g;
                const int m = int( S.size() );
                std::vector<TF> al( m );
                for ( int k = m - 1; k >= 0; --k ) {
                    al[ k ] = RHO[ k ] * dot( S[ k ], q );
                    for ( SI i = 0; i < n; ++i ) q[ i ] -= al[ k ] * Y[ k ][ i ];
                }
                TF gamma = 1;
                r.resize( n );
                if ( o.precond == 2 ) {                  // `L0` a deja l'echelle : pas de gamma
                    if ( ! applique_L0( false, q, r ) ) { st.fin = "SOLVEUR LINEAIRE EN ECHEC"; break; }
                } else {
                    if ( m ) {
                        TF yy = 0;
                        for ( SI i = 0; i < n; ++i ) yy += Y[ m - 1 ][ i ] * ( o.precond ? Y[ m - 1 ][ i ] / dia[ i ] : Y[ m - 1 ][ i ] );
                        gamma = dot( S[ m - 1 ], Y[ m - 1 ] ) / yy;
                    }
                    for ( SI i = 0; i < n; ++i ) r[ i ] = gamma * ( o.precond ? q[ i ] / dia[ i ] : q[ i ] );
                }
                for ( int k = 0; k < m; ++k ) {
                    const TF be = RHO[ k ] * dot( Y[ k ], r );
                    for ( SI i = 0; i < n; ++i ) r[ i ] += ( al[ k ] - be ) * S[ k ][ i ];
                }
                for ( SI i = 0; i < n; ++i ) d[ i ] = -r[ i ];
            } else {
                // Polak-Ribiere+ preconditionne, redemarre si la direction n'est plus de descente
                TF beta = 0;
                if ( ! d_old.empty() ) {
                    TF num = 0, den = 0;
                    for ( SI i = 0; i < n; ++i ) { num += pg[ i ] * ( g[ i ] - g_old[ i ] ); den += pg_old[ i ] * g_old[ i ]; }
                    beta = std::max( TF( 0 ), num / den );
                }
                for ( SI i = 0; i < n; ++i ) d[ i ] = -pg[ i ] + ( d_old.empty() ? 0 : beta * d_old[ i ] );
                restart = beta == 0 && ! d_old.empty();
            }
            // la jauge : `sum d = 0` ( `F` ne bouge pas le long des constantes, autant ne pas deriver )
            TF md = 0;
            for ( TF v : d ) md += v;
            md /= n;
            for ( TF &v : d ) v -= md;
            TF phi0 = dot( g, d );
            if ( ! ( phi0 < 0 ) ) {                      // pas une descente : le gradient preconditionne
                for ( SI i = 0; i < n; ++i ) d[ i ] = -pg[ i ];
                md = 0; for ( TF v : d ) md += v; md /= n; for ( TF &v : d ) v -= md;
                phi0 = dot( g, d );
                S.clear(); Y.clear(); RHO.clear();
                restart = true;
            }
            if ( restart ) ++st.nb_restart;

            // ---- LA RECHERCHE LINEAIRE, sur `phi'` seule
            // le premier essai : 1, sauf si le plancher a borne le pas precedent -- alors le double de
            // celui-ci ( comme `beta` dans ESSAI_LIMITES ), et pour CG l'echelle de la derniere pente
            TF alpha = 1;
            if ( o.methode == PremierOrdreOptions::CG && it > 0 && ! restart && ! borne )
                alpha = std::min( TF( 4 ), std::max( TF( 0.25 ), alpha_old * phi0_old / phi0 ) );
            if ( borne ) alpha = std::min( TF( 1 ), 2 * alpha_old );
            borne = false;
            TF lo = 0, plo = phi0, hi = 0, phi = 0, a_prev = 0, p_prev = phi0;
            bool ok = false, a_lo_ok = false, hi_plancher = false;
            int ls = 0, nb_sous = 0;                     // `nb_sous` : les reculs du plancher, comptes a part
            for ( ; ls < o.max_ls && alpha >= o.t_min; ) {
                for ( SI i = 0; i < n; ++i ) w2[ i ] = w[ i ] + alpha * d[ i ];
                nw.mesures_et_facettes( w2, a2, fa2 );
                for ( SI i = 0; i < n; ++i ) g2[ i ] = a2[ i ] - nw.nu[ i ];
                phi = dot( g2, d );
                bool sous = false;                       // une cellule sous le plancher ?
                for ( SI i = 0; i < n && ! sous; ++i ) sous = a2[ i ] < eps && a[ i ] > 0;
                if ( sous ) {                            // refuse, quoi que dise `phi'` : on recule ( jusqu'a `t_min`, comme Newton )
                    ++st.nb_plancher; ++nb_sous;
                    borne = true;
                    hi = alpha; hi_plancher = true;
                    if ( a_lo_ok && hi - lo <= TF( 0.25 ) * hi ) break;   // le plancher est cerne : `lo` fera l'affaire
                    alpha = lo + TF( 0.5 ) * ( hi - lo );
                    continue;
                }
                ++ls;
                if ( std::fabs( phi ) <= o.c2 * std::fabs( phi0 ) ) { ok = true; break; }
                if ( phi < 0 ) {                         // encore en descente : le minimum est plus loin
                    lo = alpha; plo = phi;
                    w_lo.swap( w2 ); a_lo.swap( a2 ); g_lo.swap( g2 ); fa_lo.swap( fa2 ); a_lo_ok = true;
                    w2.resize( n ); g2.resize( n );
                    if ( hi > 0 ) {
                        const TF sec = hi_plancher ? lo + TF( 0.5 ) * ( hi - lo ) : lo - plo * ( hi - lo ) / ( p_prev - plo );
                        alpha = std::min( std::max( sec, lo + TF( 0.1 ) * ( hi - lo ) ), hi - TF( 0.1 ) * ( hi - lo ) );
                    } else {
                        const TF pente = ( phi - p_prev ) / ( alpha - a_prev );
                        const TF sec = pente > 0 ? alpha - phi / pente : 4 * alpha;
                        a_prev = alpha; p_prev = phi;
                        alpha = std::min( std::max( sec, 2 * alpha ), 10 * alpha );
                    }
                } else {                                 // depasse : le minimum est entre `lo` et `alpha`
                    hi = alpha; p_prev = phi; hi_plancher = false;
                    const TF sec = lo - plo * ( hi - lo ) / ( phi - plo );
                    alpha = std::min( std::max( sec, lo + TF( 0.1 ) * ( hi - lo ) ), hi - TF( 0.1 ) * ( hi - lo ) );
                }
            }
            if ( ! ok ) {
                if ( ! a_lo_ok || lo < o.t_min ) {       // tout depasse ou refuse : rien de garanti
                    if ( o.precond == 2 && ! frais ) {   // ... sur un laplacien perime : on le refait, et on reessaie
                        if ( o.trace ) std::printf( "    it %3d  recherche lineaire en echec ( rien sous %.2e ) : laplacien refait\n", it, double( hi ) );
                        forcer = true;
                        S.clear(); Y.clear(); RHO.clear();
                        continue;
                    }
                    if ( o.trace ) std::printf( "    it %3d  recherche lineaire en echec ( rien sous %.2e, phi'( 0 ) %.3e )\n", it, double( hi ), double( phi0 ) );
                    st.fin = "STAGNATION";
                    break;
                }
                alpha = lo; phi = plo;
                w2.swap( w_lo ); a2.swap( a_lo ); g2.swap( g_lo ); fa2.swap( fa_lo );
            }
            st.nb_ls += ls + nb_sous;
            forcer = false;

            // ---- LA MISE A JOUR
            s.resize( n ); y.resize( n );
            for ( SI i = 0; i < n; ++i ) { s[ i ] = w2[ i ] - w[ i ]; y[ i ] = g2[ i ] - g[ i ]; }
            if ( o.methode == PremierOrdreOptions::LBFGS ) {
                const TF sy = dot( s, y );
                if ( sy > 0 && ! ( o.sauter_borne && borne ) ) {
                    S.push_back( s ); Y.push_back( y ); RHO.push_back( 1 / sy );
                    if ( int( S.size() ) > o.memoire ) { S.pop_front(); Y.pop_front(); RHO.pop_front(); }
                }
            } else {
                d_old = d; pg_old = pg; g_old = g;
            }
            phi0_old = phi0; alpha_old = alpha; nr_prec = nr;
            w.swap( w2 ); a.swap( a2 ); fa.swap( fa2 );
            if ( o.trace ) {
                std::printf( "    it %3d  |r|_2 %.3e  max|a-nu|/nu %.3e  %d vides  alpha %.2e  phi' %.1e -> %.1e  %d diag  [diag %.2f  lin %.2f]%s\n",
                             it, double( nr ), double( p ), int( nvide ), double( alpha ), double( phi0 ), double( phi ),
                             nw.st.nb_diag - g0, nw.st.t_diag - d0, nw.st.t_lin - l0, restart ? "  redemarrage" : "" );
                std::fflush( stdout );
            }
            if ( nw.o.apres_pas ) nw.o.apres_pas( it, alpha, ls );
        }
        if ( it >= o.maxit ) st.fin = "MAX ITERATIONS";
        st.nb_diag = nw.st.nb_diag;
        st.temps = now() - debut;
        return st.fin == std::string( "CONVERGE" );
    }
};

} // namespace sf

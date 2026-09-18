#pragma once

// =====================================================================================
// L'ECRASEMENT DES CELLULES LE LONG D'UNE DIRECTION : jusqu'ou peut-on aller ?
//
// Newton propose `d` ; l'amortissement essaye `w + t d` pour `t = 1, 1/2, 1/4, ...` et refuse
// tant qu'une cellule passe sous le plancher. Chaque essai est un diagramme. La question posee
// ici : peut-on PREDIRE, depuis le diagramme en `w` seul, la valeur de `alpha` a partir de
// laquelle `w + alpha d` vide une cellule ?
//
// = Le polynome a combinatoire figee
//
// Le plan qui separe `i` de `j` est `( p_j - p_i ) . x <= c_ij + alpha delta_ij`, avec
// `delta_ij = ( d_i - d_j ) / 2` : la normale ne bouge pas, seul le decalage glisse, lineairement
// en `alpha`. Tant que la cellule garde les MEMES aretes, chaque sommet ( intersection de deux
// droites dont les decalages sont affines ) est AFFINE en `alpha`, et l'aire -- une somme de
// produits vectoriels de sommets -- est un POLYNOME DE DEGRE 2 en `alpha` ( 3 en 3D ). On le
// calcule exactement, sommet par sommet : `v( alpha ) = v0 + alpha v1`.
//
// Le polynome ment des que la combinatoire change : une arete s'annule ( deux sommets se
// rejoignent -- c'est visible depuis la cellule, la longueur signee d'une arete est AFFINE en
// `alpha` ), ou un germe qui n'etait pas voisin le devient ( invisible depuis la cellule seule ).
// `alpha_arete` est la premiere de ces annulations : avant, le polynome est exact sauf voisin
// nouveau ; apres, il est faux par construction. Ce que vaut le polynome au-dela, c'est
// justement ce qu'on veut mesurer.
//
// = Le temoin
//
// Le diagramme recalcule pour chaque `alpha` d'une grille. C'est ce qui dit la verite, et ce
// qu'on voudrait ne plus avoir a payer neuf fois par iteration.
//
// = Predire, verifier, corriger ( `limites` )
//
// Pour chaque cellule, dans le meme parcours : le polynome en `alpha = 0` donne une limite
// predite ; on calcule la cellule EXACTE a cette limite ( `FournisseurAlpha` : l'arbre n'est pas
// rafraichi, le depart est a chaud ) ; si elle a les MEMES aretes, le polynome etait exact et la
// limite est confirmee -- un test exact, pas un seuil. Sinon la cellule calculee porte la nouvelle
// combinatoire, donc un nouveau polynome, donc une limite corrigee, et on recommence depuis la.
// Une cellule vide ne porte rien : on resserre par bissection. Le cout nominal est UNE cellule
// par cellule -- au lieu d'un diagramme par pas essaye -- et on obtient une limite PAR CELLULE.
// =====================================================================================

#include "cell/FournisseurAlpha2D.h"
#include "diagram/PowerDiagram.h"
#include "util/parallel.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

namespace sf {

constexpr TF INFINI = std::numeric_limits<TF>::infinity();

/// LE POLYNOME D'UNE CELLULE : `q( alpha ) = a0 + a1 alpha + a2 alpha^2`, et ce qu'on en tire.
struct PolyCellule {
    enum Etat : int { OK = 0, VIDE_AU_DEPART, DEBORDE, DEGENERE };

    TF  a0 = 0, a1 = 0, a2 = 0;
    TF  alpha_arete = INFINI;   ///< premiere arete qui s'annule : la combinatoire change
    int nb_aretes = 0;
    int etat = OK;

    TF operator()( TF alpha ) const { return a0 + alpha * ( a1 + alpha * a2 ); }

    /// les racines reelles de `q( alpha ) == niveau`, triees ; rend leur nombre ( 0, 1 ou 2 ).
    int racines( TF niveau, TF &r1, TF &r2 ) const {
        const TF c = a0 - niveau;
        if ( std::fabs( a2 ) <= TF( 1e-300 ) ) {
            if ( a1 == 0 ) return 0;
            r1 = -c / a1;
            return 1;
        }
        const TF disc = a1 * a1 - 4 * a2 * c;
        if ( disc < 0 ) return 0;
        const TF s = std::sqrt( disc );
        // la forme stable : le signe qui evite la soustraction
        const TF qd = a1 >= 0 ? -a1 - s : -a1 + s;
        r1 = qd / ( 2 * a2 );
        r2 = qd != 0 ? 2 * c / qd : r1;
        if ( r1 > r2 ) std::swap( r1, r2 );
        return 2;
    }

    /// le plus petit `alpha > 0` tel que `q( alpha ) == niveau`, sachant `q( 0 ) > niveau` ;
    /// `INFINI` si le polynome ne redescend jamais jusque la.
    TF premiere_racine( TF niveau ) const {
        const TF c = a0 - niveau;
        if ( ! ( c > 0 ) ) return 0;
        if ( std::fabs( a2 ) <= TF( 1e-300 ) )
            return a1 < 0 ? -c / a1 : INFINI;
        const TF disc = a1 * a1 - 4 * a2 * c;
        if ( disc < 0 ) return INFINI;
        const TF s = std::sqrt( disc );
        // la forme stable : `q = 2 c / ( -a1 -+ s )`, le signe qui evite la soustraction
        const TF qd = a1 >= 0 ? -a1 - s : -a1 + s;
        TF r1 = qd / ( 2 * a2 ), r2 = qd != 0 ? 2 * c / qd : INFINI;
        if ( r1 > r2 ) std::swap( r1, r2 );
        if ( r1 > 0 ) return r1;
        if ( r2 > 0 ) return r2;
        return INFINI;
    }
};

/// Une droite `n . x <= c + alpha delta`, la normale fixe.
struct Droite2 {
    TF nx, ny, c, delta;
};

/// LE POLYNOME D'UNE CELLULE `cel` du germe `i`, calculee aux poids `w + alpha0 d`, le long de
/// `d` : `q( beta )` est l'aire en `w + ( alpha0 + beta ) d`, combinatoire figee.
template<class Cell>
PolyCellule polynome_cellule( const Cell &cel, SI i, const TF *const *P, const TF *w, const TF *d,
                              TF alpha0 = 0 ) {
    PolyCellule q;
    if ( cel.nb < 0 )  { q.etat = PolyCellule::DEBORDE; return q; }
    if ( cel.nb == 0 ) { q.etat = PolyCellule::VIDE_AU_DEPART; return q; }
    const int nb = cel.nb;
    q.nb_aretes = nb;
    {
        // ---- les droites : l'arete `j` va de `v_j` a `v_j+1`, portee par la coupe `cid[ j ]`
        Droite2 dr[ Cell::max_nb ];
        const TF xi = P[ 0 ][ i ], yi = P[ 1 ][ i ], wi = w[ i ] + alpha0 * d[ i ];
        for ( int j = 0; j < nb; ++j ) {
            const auto id = cel.cid[ j ];
            if ( id >= 0 ) {
                const TF xj = P[ 0 ][ id ], yj = P[ 1 ][ id ], wj = w[ id ] + alpha0 * d[ id ];
                const TF nx = xj - xi, ny = yj - yi;
                dr[ j ] = { nx, ny, TF( 0.5 ) * ( nx * ( xj + xi ) + ny * ( yj + yi ) + wi - wj ),
                            TF( 0.5 ) * ( d[ i ] - d[ id ] ) };
            } else {                                     // le carre unite, cotes -1 .. -4
                switch ( id ) {
                    case -1: dr[ j ] = {  0, -1, 0, 0 }; break;
                    case -2: dr[ j ] = {  1,  0, 1, 0 }; break;
                    case -3: dr[ j ] = {  0,  1, 1, 0 }; break;
                    default: dr[ j ] = { -1,  0, 0, 0 }; break;
                }
            }
        }

        // ---- les sommets, affines : `v_j` est l'intersection des aretes `j-1` et `j`
        TF v0x[ Cell::max_nb ], v0y[ Cell::max_nb ], v1x[ Cell::max_nb ], v1y[ Cell::max_nb ];
        for ( int j = 0; j < nb; ++j ) {
            const Droite2 &a = dr[ j ? j - 1 : nb - 1 ], &b = dr[ j ];
            const TF det = a.nx * b.ny - a.ny * b.nx;
            if ( ! ( std::fabs( det ) > 0 ) ) { q.etat = PolyCellule::DEGENERE; return q; }
            v0x[ j ] = ( a.c * b.ny - b.c * a.ny ) / det;
            v0y[ j ] = ( a.nx * b.c - b.nx * a.c ) / det;
            v1x[ j ] = ( a.delta * b.ny - b.delta * a.ny ) / det;
            v1y[ j ] = ( a.nx * b.delta - b.nx * a.delta ) / det;
        }

        // ---- l'aire signee, et la longueur signee de chaque arete
        TF a0 = 0, a1 = 0, a2 = 0;
        for ( int j = 0; j < nb; ++j ) {
            const int l = j + 1 < nb ? j + 1 : 0;
            a0 += v0x[ j ] * v0y[ l ] - v0x[ l ] * v0y[ j ];
            a1 += v0x[ j ] * v1y[ l ] - v0x[ l ] * v1y[ j ] + v1x[ j ] * v0y[ l ] - v1x[ l ] * v0y[ j ];
            a2 += v1x[ j ] * v1y[ l ] - v1x[ l ] * v1y[ j ];
        }
        const TF sg = a0 < 0 ? TF( -0.5 ) : TF( 0.5 );
        q.a0 = sg * a0; q.a1 = sg * a1; q.a2 = sg * a2;

        for ( int j = 0; j < nb; ++j ) {
            const int l = j + 1 < nb ? j + 1 : 0;
            const TF tx = -dr[ j ].ny, ty = dr[ j ].nx; // le long de l'arete `j`
            TF l0 = ( v0x[ l ] - v0x[ j ] ) * tx + ( v0y[ l ] - v0y[ j ] ) * ty;
            TF l1 = ( v1x[ l ] - v1x[ j ] ) * tx + ( v1y[ l ] - v1y[ j ] ) * ty;
            if ( l0 < 0 ) { l0 = -l0; l1 = -l1; }
            if ( l1 < 0 )
                q.alpha_arete = std::min( q.alpha_arete, -l0 / l1 );
        }
    }
    return q;
}

/// LES POLYNOMES DE TOUTES LES CELLULES du diagramme `pd` ( aux poids `w` ), le long de `d`.
/// `P` et `w` dans l'ordre des identifiants. Rend `poly[ id ]`.
template<class PD>
void polynomes( const PD &pd, const TF *const *P, const std::vector<TF> &w, const std::vector<TF> &d,
                const Parallel &par, std::vector<PolyCellule> &poly ) {
    static_assert( PD::dim == 2, "les polynomes ne sont ecrits qu'en 2D pour l'instant" );
    using Cell = typename PD::Cell;
    const SI n = pd.n;
    poly.assign( n, PolyCellule{} );
    parallel_for( n, par, [ & ]( SI k, int ) {
        Cell cel;
        pd.cellule( k, cel );
        poly[ pd.ids[ k ] ] = polynome_cellule( cel, pd.ids[ k ], P, w.data(), d.data() );
    } );
}

/// LE MINIMUM d'une prediction sur les cellules, et qui le porte.
struct Premier {
    TF alpha = INFINI;
    SI cellule = -1;
    void propose( TF a, SI i ) { if ( a < alpha ) { alpha = a; cellule = i; } }
};

/// LES MESURES EXACTES en `w + alpha d` : un diagramme.
template<class PD>
void mesures_en( PD &pd, const std::vector<TF> &w, const std::vector<TF> &d, TF alpha,
                 const Parallel &par, std::vector<TF> &w2, std::vector<TF> &res ) {
    const SI n = pd.n;
    w2.resize( n );
    for ( SI i = 0; i < n; ++i ) w2[ i ] = w[ i ] + alpha * d[ i ];
    pd.set_weights( w2.data(), par );
    pd.measures( res, par );
}

/// LE PREMIER `alpha` de `[ lo, hi ]` ou `critere( mesures )` devient vrai, sachant qu'il est
/// faux en `lo` et vrai en `hi` : une bissection, un diagramme par pas. Rend aussi la cellule
/// designee par `critere` a l'arrivee.
template<class PD, class Crit>
TF bissection( PD &pd, const std::vector<TF> &w, const std::vector<TF> &d, TF lo, TF hi,
               const Parallel &par, int pas, Crit &&critere, SI &cellule ) {
    std::vector<TF> w2, res;
    cellule = -1;
    for ( int k = 0; k < pas; ++k ) {
        const TF m = TF( 0.5 ) * ( lo + hi );
        mesures_en( pd, w, d, m, par, w2, res );
        SI c = -1;
        if ( critere( res, c ) ) { hi = m; cellule = c; }
        else                       lo = m;
    }
    return hi;
}

// ------------------------------------------------------------------------------------ limites
/// CE QU'ON SAIT D'UNE CELLULE a la fin : sa limite, et comment on l'a obtenue.
struct LimiteCellule {
    enum Etat : int { CONFIRMEE = 0, CORRIGEE, HORIZON, VIDE_AU_DEPART, ECHEC };
    TF  alpha      = INFINI;    ///< le premier `alpha` ou la cellule passe sous `niveau`
    TF  alpha_poly = INFINI;    ///< ce que le polynome en 0 predisait
    int tours      = 0;         ///< cellules calculees en plus de celle en 0
    int etat       = ECHEC;
};

struct OptionsLimites {
    TF  niveau    = 0;          ///< le plancher ( `eps` de l'amortissement )
    TF  coeff     = 0.99;       ///< on verifie en `a_ok + coeff * ( predit - a_ok )` : la
                                ///< prediction est confirmee si la combinatoire y est la meme
    TF  horizon   = 1;          ///< au-dela, on ne verifie qu'a l'horizon ( le pas plein )
    TF  tol       = 1e-2;       ///< precision relative demandee sur la limite
    int max_tours = 12;
    SI  trace     = -1;         ///< une cellule dont on imprime chaque tour
};

/// LA PASSE « predire, verifier, corriger », sur toutes les cellules. `pd` doit porter les poids
/// `w` ( `set_weights( w )` ). Rend `lim[ id ]`.
template<class PD>
void limites( const PD &pd, const TF *const *P, const std::vector<TF> &w, const std::vector<TF> &d,
              const Parallel &par, const OptionsLimites &o, std::vector<LimiteCellule> &lim ) {
    static_assert( PD::dim == 2, "2D seulement pour l'instant" );
    using Cell = typename PD::Cell;
    using TK   = typename PD::TKernel;
    const SI n = pd.n;
    const auto &arbre = pd.arbre;

    // ---- `d` dans l'ordre de l'arbre, et son majorant par noeud : une fois par direction
    std::vector<TF> dt( n );
    for ( SI k = 0; k < n; ++k ) dt[ k ] = d[ arbre.order[ k ] ];
    std::vector<WMajT<2>> dm( arbre.nodes.size() );
    parallel_for( SI( arbre.nodes.size() ), par, [ & ]( SI m, int ) {
        const auto &nd = arbre.nodes[ m ];
        dm[ m ] = weight_majorant<2>( nd.beg, nd.end, [ & ]( SI k, Vec<2> &q, TF &v ) {
            q[ 0 ] = arbre.p[ 0 ][ k ]; q[ 1 ] = arbre.p[ 1 ][ k ]; v = dt[ k ];
        } );
    } );

    lim.assign( n, LimiteCellule{} );
    parallel_for( n, par, [ & ]( SI k, int ) {
        const SI i = pd.ids[ k ];
        LimiteCellule &L = lim[ i ];
        Cell cel;
        pd.cellule( k, cel );
        PolyCellule q = polynome_cellule( cel, i, P, w.data(), d.data() );
        if ( q.etat != PolyCellule::OK ) { L.etat = LimiteCellule::VIDE_AU_DEPART; return; }
        L.alpha_poly = q.premiere_racine( o.niveau );

        // L'ETAT DE LA RECHERCHE. `q_ok` est le polynome de la cellule en `a_ok`, exact tant que
        // la combinatoire ne bouge pas ; `pred` est sa racine, en absolu. `a_bad` est le premier
        // `alpha` connu sous le niveau. `cible` est ou l'on va verifier : la prediction, ou un
        // milieu quand la prediction est au-dela de `a_bad` ( la combinatoire a forcement bouge
        // entre les deux ), ou ce que dit le polynome d'une cellule trouvee trop petite.
        TF a_ok = 0, a_bad = INFINI, pred = L.alpha_poly, cible = pred;
        d2::SI32 cids_ok[ Cell::max_nb ];
        int nb_ok = cel.nb;
        for ( int j = 0; j < nb_ok; ++j ) cids_ok[ j ] = cel.cid[ j ];
        std::sort( cids_ok, cids_ok + nb_ok );

        for ( ; L.tours < o.max_tours; ) {
            // ---- ou verifier
            bool sur_pred = cible == pred;                // on vise la prediction du polynome
            TF a_test = std::min( cible, o.horizon );
            if ( ! ( a_test < a_bad ) ) { a_test = TF( 0.5 ) * ( a_ok + a_bad ); sur_pred = false; }
            if ( sur_pred && a_test < o.horizon ) a_test = a_ok + o.coeff * ( a_test - a_ok );
            if ( ! ( a_test > a_ok ) ) { L.alpha = a_ok; L.etat = LimiteCellule::CORRIGEE; return; }

            // ---- la cellule exacte en `a_test`, a chaud depuis la derniere bonne
            d2::FournisseurAlpha<TK> f( &arbre, dm.data(), dt.data(), P, w.data(), d.data(), a_test,
                                        d2::SI32( i ), cids_ok, nb_ok );
            d2::moteur<TK>( &f, &cel );
            ++L.tours;
            if ( i == o.trace )
                std::printf( "    cellule %d tour %d : a_ok %.6e a_bad %.6e pred %.6e cible %.6e -> a_test %.6e :"
                             " nb %d ( avant %d ), aire %.4e / niveau %.4e\n",
                             int( i ), L.tours, double( a_ok ), double( a_bad ), double( pred ), double( cible ),
                             double( a_test ), cel.nb, nb_ok, double( cel.nb > 0 ? PD::mesure( cel ) : TF( 0 ) ),
                             double( o.niveau ) );

            // ---- memes aretes : `q_ok` etait exact de `a_ok` a `a_test`
            bool memes = cel.nb == nb_ok;
            if ( memes ) {
                d2::SI32 c2[ Cell::max_nb ];
                for ( int j = 0; j < cel.nb; ++j ) c2[ j ] = cel.cid[ j ];
                std::sort( c2, c2 + cel.nb );
                for ( int j = 0; j < cel.nb && memes; ++j ) memes = c2[ j ] == cids_ok[ j ];
            }
            if ( memes ) {
                if ( a_test >= o.horizon ) { L.alpha = pred; L.etat = LimiteCellule::HORIZON; return; }
                if ( sur_pred ) {                        // la prediction est confirmee, a `coeff` pres
                    L.alpha = pred;
                    L.etat = L.tours == 1 ? LimiteCellule::CONFIRMEE : LimiteCellule::CORRIGEE;
                    return;
                }
                a_ok = a_test;                           // le polynome tient encore ici ; `pred` reste
                cible = pred;
                if ( a_bad < INFINI && a_bad - a_ok <= o.tol * a_bad ) { L.alpha = a_ok; L.etat = LimiteCellule::CORRIGEE; return; }
                continue;
            }

            // ---- la combinatoire a change : la cellule calculee porte le nouveau polynome
            const PolyCellule q2 = polynome_cellule( cel, i, P, w.data(), d.data(), a_test );
            const TF m = q2.etat == PolyCellule::OK ? q2.a0 : TF( 0 );
            if ( i == o.trace )
                std::printf( "      q2 : etat %d a0 %.4e a1 %.4e a2 %.4e, racine %.4e\n", q2.etat, double( q2.a0 ),
                             double( q2.a1 ), double( q2.a2 ), double( q2.premiere_racine( o.niveau ) ) );
            if ( m >= o.niveau ) {                       // bonne : on repart d'ici
                a_ok = a_test;
                nb_ok = cel.nb;
                for ( int j = 0; j < nb_ok; ++j ) cids_ok[ j ] = cel.cid[ j ];
                std::sort( cids_ok, cids_ok + nb_ok );
                const TF beta = q2.premiere_racine( o.niveau );
                pred = cible = a_test + beta;
                if ( a_test >= o.horizon ) { L.alpha = pred; L.etat = LimiteCellule::HORIZON; return; }
                if ( beta <= o.tol * a_test ) { L.alpha = a_test; L.etat = LimiteCellule::CORRIGEE; return; }
            } else {                                     // mauvaise : la limite est avant
                a_bad = a_test;
                TF r1, r2, beta = -INFINI;
                if ( q2.etat == PolyCellule::OK ) {      // la racine negative la plus proche
                    const int nr = q2.racines( o.niveau, r1, r2 );
                    if ( nr >= 1 && r1 < 0 ) beta = r1;
                    if ( nr >= 2 && r2 < 0 ) beta = r2;
                }
                cible = a_test + beta;                   // un indice, pas une prediction de `q_ok`
                if ( ! ( cible > a_ok && cible < a_bad ) ) cible = TF( 0.5 ) * ( a_ok + a_bad );
                if ( cible == pred ) cible = std::nextafter( cible, a_ok );
            }
            if ( a_bad < INFINI && a_bad - a_ok <= o.tol * a_bad ) { L.alpha = a_ok; L.etat = LimiteCellule::CORRIGEE; return; }
        }
        L.alpha = a_ok;                                  // le conservatif, faute de mieux
        L.etat = LimiteCellule::ECHEC;
    } );
}

} // namespace sf

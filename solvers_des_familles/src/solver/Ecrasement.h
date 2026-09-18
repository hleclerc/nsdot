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
#include "solver/Laplacien.h"
#include "util/parallel.h"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <limits>
#include <type_traits>
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

/// LE MODELE QUADRATIQUE D'UNE CELLULE dans TOUT l'espace des poids : ses droites a combinatoire
/// figee, `n_k . x <= c_k + ( delta_i - delta_j_k ) / 2`, et l'aire signee qui en sort pour un
/// `delta` quelconque -- une forme quadratique en `delta`, exacte tant que la combinatoire tient.
struct ModeleCellule {
    SI  i = -1;
    int nb = 0;
    TF  signe = 1;                  ///< l'orientation en `delta = 0`, pour rendre l'aire positive
    std::vector<d2::SI32> vois;     ///< le germe de chaque arete ( `< 0` : un cote du domaine )
    std::vector<TF> nx, ny, c0;

    template<class Cell>
    void depuis( const Cell &cel, SI id, const TF *const *P, const TF *w ) {
        i = id; nb = std::max( cel.nb, 0 );
        vois.resize( nb ); nx.resize( nb ); ny.resize( nb ); c0.resize( nb );
        const TF xi = P[ 0 ][ i ], yi = P[ 1 ][ i ];
        for ( int j = 0; j < nb; ++j ) {
            const auto id2 = cel.cid[ j ];
            vois[ j ] = id2;
            if ( id2 >= 0 ) {
                const TF xj = P[ 0 ][ id2 ], yj = P[ 1 ][ id2 ];
                nx[ j ] = xj - xi; ny[ j ] = yj - yi;
                c0[ j ] = TF( 0.5 ) * ( nx[ j ] * ( xj + xi ) + ny[ j ] * ( yj + yi ) + w[ i ] - w[ id2 ] );
            } else switch ( id2 ) {
                case -1: nx[ j ] =  0; ny[ j ] = -1; c0[ j ] = 0; break;
                case -2: nx[ j ] =  1; ny[ j ] =  0; c0[ j ] = 1; break;
                case -3: nx[ j ] =  0; ny[ j ] =  1; c0[ j ] = 1; break;
                default: nx[ j ] = -1; ny[ j ] =  0; c0[ j ] = 0; break;
            }
        }
        signe = 1;
        const TF a = aire( nullptr );
        if ( a < 0 ) signe = -1;
    }

    /// l'aire signee pour le deplacement `delta` des poids ( `nullptr` : zero ), et pour qui les
    /// veut les longueurs SIGNEES des aretes : `facette( j, l / ( 2 |p_j - p_i| ) )` -- la derivee
    /// de l'aire par rapport au decalage de l'arete est sa longueur, signe compris.
    template<class F>
    TF aire( const TF *delta, F &&facette ) const {
        if ( nb < 3 ) return 0;
        TF c[ 128 ], vx[ 128 ], vy[ 128 ];
        const TF di = delta ? delta[ i ] : TF( 0 );
        for ( int j = 0; j < nb; ++j )
            c[ j ] = c0[ j ] + ( vois[ j ] >= 0 && delta ? TF( 0.5 ) * ( di - delta[ vois[ j ] ] ) : TF( 0 ) );
        for ( int j = 0; j < nb; ++j ) {
            const int a = j ? j - 1 : nb - 1;
            const TF det = nx[ a ] * ny[ j ] - ny[ a ] * nx[ j ];
            if ( ! ( std::fabs( det ) > 0 ) ) return 0;
            vx[ j ] = ( c[ a ] * ny[ j ] - c[ j ] * ny[ a ] ) / det;
            vy[ j ] = ( nx[ a ] * c[ j ] - nx[ j ] * c[ a ] ) / det;
        }
        TF s = 0;
        for ( int j = 0; j < nb; ++j ) {
            const int l = j + 1 < nb ? j + 1 : 0;
            s += vx[ j ] * vy[ l ] - vx[ l ] * vy[ j ];
            if ( vois[ j ] >= 0 ) {                      // l'arete `j` va de `v_j` a `v_l`, le long de `t = ( -ny, nx )`
                const TF n2 = nx[ j ] * nx[ j ] + ny[ j ] * ny[ j ];
                const TF lg = signe * ( ( vx[ l ] - vx[ j ] ) * ( -ny[ j ] ) + ( vy[ l ] - vy[ j ] ) * nx[ j ] ) / std::sqrt( n2 );
                facette( vois[ j ], lg / ( 2 * std::sqrt( n2 ) ) );
            }
        }
        return signe * TF( 0.5 ) * s;
    }
    TF aire( const TF *delta ) const { return aire( delta, []( d2::SI32, TF ) {} ); }
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

/// LA DECOMPOSITION EN TRIANGLES `( p_i, v_j, v_j+1 )`, un polynome par triangle, pour tester
/// deux estimateurs de l'aire a combinatoire figee : la somme des PARTIES POSITIVES, et la somme
/// ou chaque triangle est mis a ZERO passe son premier changement de signe. `offs[ i ] ..
/// offs[ i + 1 ]` sont les triangles de la cellule `i`.
struct Triangle { TF a0, a1, a2, alpha_signe; TF operator()( TF al ) const { return a0 + al * ( a1 + al * a2 ); } };

template<class Cell>
void triangles_cellule( const Cell &cel, SI i, const TF *const *P, const TF *w, const TF *d,
                        std::vector<Triangle> &out ) {
    if ( cel.nb <= 0 ) return;
    const int nb = cel.nb;
    Droite2 dr[ Cell::max_nb ];
    const TF xi = P[ 0 ][ i ], yi = P[ 1 ][ i ];
    for ( int j = 0; j < nb; ++j ) {
        const auto id = cel.cid[ j ];
        if ( id >= 0 ) {
            const TF xj = P[ 0 ][ id ], yj = P[ 1 ][ id ], nx = xj - xi, ny = yj - yi;
            dr[ j ] = { nx, ny, TF( 0.5 ) * ( nx * ( xj + xi ) + ny * ( yj + yi ) + w[ i ] - w[ id ] ),
                        TF( 0.5 ) * ( d[ i ] - d[ id ] ) };
        } else switch ( id ) {
            case -1: dr[ j ] = {  0, -1, 0, 0 }; break;
            case -2: dr[ j ] = {  1,  0, 1, 0 }; break;
            case -3: dr[ j ] = {  0,  1, 1, 0 }; break;
            default: dr[ j ] = { -1,  0, 0, 0 }; break;
        }
    }
    TF v0x[ Cell::max_nb ], v0y[ Cell::max_nb ], v1x[ Cell::max_nb ], v1y[ Cell::max_nb ];
    for ( int j = 0; j < nb; ++j ) {
        const Droite2 &a = dr[ j ? j - 1 : nb - 1 ], &b = dr[ j ];
        const TF det = a.nx * b.ny - a.ny * b.nx;
        if ( ! ( std::fabs( det ) > 0 ) ) return;
        v0x[ j ] = ( a.c * b.ny - b.c * a.ny ) / det - xi;   // relatif au germe
        v0y[ j ] = ( a.nx * b.c - b.nx * a.c ) / det - yi;
        v1x[ j ] = ( a.delta * b.ny - b.delta * a.ny ) / det;
        v1y[ j ] = ( a.nx * b.delta - b.nx * a.delta ) / det;
    }
    TF sg = 0;
    for ( int j = 0; j < nb; ++j ) { const int l = j + 1 < nb ? j + 1 : 0; sg += v0x[ j ] * v0y[ l ] - v0x[ l ] * v0y[ j ]; }
    sg = sg < 0 ? TF( -0.5 ) : TF( 0.5 );
    for ( int j = 0; j < nb; ++j ) {
        const int l = j + 1 < nb ? j + 1 : 0;
        Triangle t;
        t.a0 = sg * ( v0x[ j ] * v0y[ l ] - v0x[ l ] * v0y[ j ] );
        t.a1 = sg * ( v0x[ j ] * v1y[ l ] - v0x[ l ] * v1y[ j ] + v1x[ j ] * v0y[ l ] - v1x[ l ] * v0y[ j ] );
        t.a2 = sg * ( v1x[ j ] * v1y[ l ] - v1x[ l ] * v1y[ j ] );
        PolyCellule q; q.a0 = t.a0; q.a1 = t.a1; q.a2 = t.a2;
        t.alpha_signe = t.a0 > 0 ? q.premiere_racine( 0 ) : TF( 0 );
        out.push_back( t );
    }
}

template<class PD>
void decomposition( const PD &pd, const TF *const *P, const std::vector<TF> &w, const std::vector<TF> &d,
                    const Parallel &par, std::vector<SI> &offs, std::vector<Triangle> &tris ) {
    using Cell = typename PD::Cell;
    const SI n = pd.n;
    std::vector<std::vector<Triangle>> par_cell( n );
    parallel_for( n, par, [ & ]( SI k, int ) {
        Cell cel;
        pd.cellule( k, cel );
        triangles_cellule( cel, pd.ids[ k ], P, w.data(), d.data(), par_cell[ pd.ids[ k ] ] );
    } );
    offs.assign( n + 1, 0 );
    for ( SI i = 0; i < n; ++i ) offs[ i + 1 ] = offs[ i ] + SI( par_cell[ i ].size() );
    tris.clear(); tris.reserve( offs[ n ] );
    for ( SI i = 0; i < n; ++i ) tris.insert( tris.end(), par_cell[ i ].begin(), par_cell[ i ].end() );
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

/// LE VOISINAGE DEJA CONNU : les voisins de chaque cellule en `alpha = 0`, en CSR par identifiant
/// ( ce que le laplacien de Newton porte deja ). Avec lui, la cellule en 0 se refait en coupant
/// le carre par ses seuls voisins, sans parcours -- au lieu d'un diagramme complet.
struct Voisinage {
    const SI *row = nullptr, *col = nullptr;
    bool connu() const { return row && col; }
};

struct OptionsLimites {
    TF  niveau    = 0;          ///< le plancher ( `eps` de l'amortissement )
    TF  coeff     = 0.99;       ///< on verifie en `a_ok + coeff * ( predit - a_ok )` : la
                                ///< prediction est confirmee si la combinatoire y est la meme
    TF  horizon   = 1;          ///< au-dela, on ne verifie qu'a l'horizon ( le pas plein )
    TF  tol       = 1e-2;       ///< precision relative demandee sur la limite
    int max_tours = 12;
    SI  trace     = -1;         ///< une cellule dont on imprime chaque tour
    bool global   = false;      ///< seul `min_i alpha_i` compte : voir `limites`
};

/// LA PASSE « predire, verifier, corriger », sur toutes les cellules. `pd` doit porter les poids
/// `w` ( `set_weights( w )` ). Rend `lim[ id ]`.
///
/// `o.global` : on ne veut que `min_i alpha_i`. Une cellule dont la prediction depasse le minimum
/// COURANT ( atomique, partage entre les threads ) n'est verifiee qu'a `1.1 x` ce minimum : si
/// elle y est bonne, sa limite est au-dela et c'est tout ce qu'on a besoin de savoir ( etat
/// HORIZON, `alpha` = ce point ). C'est une cellule a PETIT `alpha`, presque gratuite a chaud,
/// la ou l'horizon `alpha = 1` est un diagramme monstrueux. Si le minimum trouve a la fin depasse
/// la borne de certaines cellules, on les repasse -- rare, une prediction conservative de plus
/// de 10 % sur la premiere cellule.
template<class PD>
void limites( const PD &pd, const TF *const *P, const std::vector<TF> &w, const std::vector<TF> &d,
              const Parallel &par, const OptionsLimites &o, std::vector<LimiteCellule> &lim,
              Voisinage vois = {} ) {
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

    // le minimum courant des limites trouvees, partage
    std::atomic<TF> courant{ INFINI };
    auto abaisse = [ & ]( TF a ) {
        TF c = courant.load();
        while ( a < c && ! courant.compare_exchange_weak( c, a ) ) {}
    };

    // ---- UNE CELLULE : depuis son rang `k`, avec l'horizon `hz`
    auto une = [ & ]( SI k, TF hz ) {
        const SI i = pd.ids[ k ];
        LimiteCellule &L = lim[ i ];
        L.tours = 0;
        Cell cel;
        if ( vois.connu() ) {                            // la cellule en 0 par ses seuls voisins
            static_assert( std::is_same_v<SI, d2::SI32>, "le CSR se lit tel quel" );
            d2::FournisseurAlpha<TK> f0( &arbre, dm.data(), dt.data(), P, w.data(), d.data(), TF( 0 ),
                                         d2::SI32( i ), vois.col + vois.row[ i ], int( vois.row[ i + 1 ] - vois.row[ i ] ) );
            f0.parcours = false;
            d2::moteur<TK>( &f0, &cel );
        } else
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

        auto fini = [ & ]( TF a, int etat ) { L.alpha = a; L.etat = etat; if ( etat != LimiteCellule::HORIZON ) abaisse( a ); };

        for ( ; L.tours < o.max_tours; ) {
            // ---- ou verifier
            bool sur_pred = cible == pred;                // on vise la prediction du polynome
            TF a_test = std::min( cible, hz );
            if ( ! ( a_test < a_bad ) ) { a_test = TF( 0.5 ) * ( a_ok + a_bad ); sur_pred = false; }
            if ( sur_pred && a_test < hz ) a_test = a_ok + o.coeff * ( a_test - a_ok );
            if ( ! ( a_test > a_ok ) ) return fini( a_ok, LimiteCellule::CORRIGEE );

            // ---- la cellule exacte en `a_test`, a chaud depuis la derniere bonne
            d2::FournisseurAlpha<TK> f( &arbre, dm.data(), dt.data(), P, w.data(), d.data(), a_test,
                                        d2::SI32( i ), cids_ok, nb_ok );
            d2::moteur<TK>( &f, &cel );
            ++L.tours;
            if ( i == o.trace )
                std::printf( "    cellule %d tour %d : a_ok %.6e a_bad %.6e pred %.6e cible %.6e hz %.6e -> a_test %.6e :"
                             " nb %d ( avant %d ), aire %.4e / niveau %.4e\n",
                             int( i ), L.tours, double( a_ok ), double( a_bad ), double( pred ), double( cible ),
                             double( hz ), double( a_test ), cel.nb, nb_ok,
                             double( cel.nb > 0 ? PD::mesure( cel ) : TF( 0 ) ), double( o.niveau ) );

            // ---- memes aretes : `q_ok` etait exact de `a_ok` a `a_test`
            bool memes = cel.nb == nb_ok;
            if ( memes ) {
                d2::SI32 c2[ Cell::max_nb ];
                for ( int j = 0; j < cel.nb; ++j ) c2[ j ] = cel.cid[ j ];
                std::sort( c2, c2 + cel.nb );
                for ( int j = 0; j < cel.nb && memes; ++j ) memes = c2[ j ] == cids_ok[ j ];
            }
            if ( memes ) {
                if ( a_test >= hz ) return fini( pred < INFINI && pred < hz ? pred : hz, LimiteCellule::HORIZON );
                if ( sur_pred )                          // la prediction est confirmee, a `coeff` pres
                    return fini( pred, L.tours == 1 ? LimiteCellule::CONFIRMEE : LimiteCellule::CORRIGEE );
                a_ok = a_test;                           // le polynome tient encore ici ; `pred` reste
                cible = pred;
                if ( a_bad < INFINI && a_bad - a_ok <= o.tol * a_bad ) return fini( a_ok, LimiteCellule::CORRIGEE );
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
                if ( a_test >= hz ) return fini( hz, LimiteCellule::HORIZON );
                if ( ! ( pred < a_bad ) ) cible = TF( 0.5 ) * ( a_ok + a_bad );
                if ( beta <= o.tol * a_test ) return fini( a_test, LimiteCellule::CORRIGEE );
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
            if ( a_bad < INFINI && a_bad - a_ok <= o.tol * a_bad ) return fini( a_ok, LimiteCellule::CORRIGEE );
        }
        fini( a_ok, LimiteCellule::ECHEC );              // le conservatif, faute de mieux
    };

    lim.assign( n, LimiteCellule{} );
    if ( ! o.global ) {
        parallel_for( n, par, [ & ]( SI k, int ) { une( k, o.horizon ); } );
        return;
    }

    // ---- le mode global : l'horizon est le minimum courant, avec une marge
    parallel_for( n, par, [ & ]( SI k, int ) {
        une( k, std::min( o.horizon, TF( 1.1 ) * courant.load() ) );
    } );
    for ( int passe = 0; passe < 8; ++passe ) {          // les bornes passees sous le minimum trouve
        const TF c = courant.load();
        std::vector<SI> encore;
        for ( SI k = 0; k < n; ++k ) {
            const LimiteCellule &L = lim[ pd.ids[ k ] ];
            if ( L.etat == LimiteCellule::HORIZON && L.alpha < c && L.alpha < o.horizon ) encore.push_back( k );
        }
        if ( encore.empty() ) break;
        parallel_for( SI( encore.size() ), par, [ & ]( SI j, int ) {
            une( encore[ j ], std::min( o.horizon, TF( 1.1 ) * courant.load() ) );
        } );
    }
}

// ------------------------------------------------------------------------------------ le pas tensoriel
/// LE PAS TENSORIEL : `delta` tel que le modele quadratique de chaque cellule atteigne la cible
/// partielle `a + theta ( nu - a )`, par Newton SUR LE MODELE ( jacobien = laplacien aux longueurs
/// signees, meme motif que `L`, refactorise ), amorti sur le residu du modele. Rend le residu
/// relatif atteint ; `delta` part de `theta d`.
template<class Lin>
struct PasTensoriel {
    int    max_it = 8;
    TF     tol    = 1e-6;
    int    nb_it  = 0;
    double t      = 0;

    TF resout( const std::vector<ModeleCellule> &mod, const std::vector<TF> &a, const std::vector<TF> &nu,
               const std::vector<TF> &d, TF theta, const Parallel &par, Lin &lin, std::vector<TF> &delta ) {
        const SI n = SI( mod.size() );
        const double t0 = now();
        std::vector<TF> b( n ), am( n ), r( n ), corr, essai( n ), am2( n );
        std::vector<std::vector<Facette>> par_th( std::max( par.threads, 1 ) );
        std::vector<Facette> fa;
        Laplacien J;
        TF rb = 0;
        for ( SI i = 0; i < n; ++i ) { b[ i ] = a[ i ] + theta * ( nu[ i ] - a[ i ] ); rb += ( b[ i ] - a[ i ] ) * ( b[ i ] - a[ i ] ); }
        rb = std::sqrt( rb );
        delta.resize( n );
        for ( SI i = 0; i < n; ++i ) delta[ i ] = theta * d[ i ];
        auto modele = [ & ]( const std::vector<TF> &dl, std::vector<TF> &out ) {
            parallel_for( n, par, [ & ]( SI i, int ) { out[ i ] = mod[ i ].aire( dl.data() ); } );
        };
        TF rn = INFINI;
        for ( nb_it = 0; nb_it < max_it; ++nb_it ) {
            for ( auto &v : par_th ) v.clear();
            parallel_for( n, par, [ & ]( SI i, int t ) {
                am[ i ] = mod[ i ].aire( delta.data(), [ & ]( d2::SI32 j, TF c ) { par_th[ t ].push_back( Facette{ i, j, c } ); } );
            } );
            rn = 0;
            for ( SI i = 0; i < n; ++i ) { r[ i ] = b[ i ] - am[ i ]; rn += r[ i ] * r[ i ]; }
            rn = std::sqrt( rn );
            if ( rn <= tol * rb ) break;
            fa.clear();
            for ( auto &v : par_th ) fa.insert( fa.end(), v.begin(), v.end() );
            J.assemble( n, fa );
            if ( ! lin.resout( J, r, corr ) ) break;
            TF tm = 1;
            bool mieux = false;
            for ( int rec = 0; rec < 20; ++rec, tm /= 2 ) {
                for ( SI i = 0; i < n; ++i ) essai[ i ] = delta[ i ] + tm * corr[ i ];
                essai[ 0 ] = 0;
                modele( essai, am2 );
                TF rn2 = 0;
                for ( SI i = 0; i < n; ++i ) rn2 += ( b[ i ] - am2[ i ] ) * ( b[ i ] - am2[ i ] );
                if ( std::sqrt( rn2 ) < rn ) { mieux = true; break; }
            }
            if ( ! mieux ) break;
            delta.swap( essai );
        }
        t += now() - t0;
        return rn / rb;
    }
};

} // namespace sf

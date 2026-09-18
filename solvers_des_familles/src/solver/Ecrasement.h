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
// =====================================================================================

#include "diagram/PowerDiagram.h"
#include "util/parallel.h"
#include <algorithm>
#include <cmath>
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
        const SI i = pd.ids[ k ];
        PolyCellule &q = poly[ i ];
        Cell cel;
        if ( ! pd.cellule( k, cel ) ) { q.etat = PolyCellule::DEBORDE; return; }
        if ( cel.nb <= 0 )            { q.etat = PolyCellule::VIDE_AU_DEPART; return; }
        const int nb = cel.nb;
        q.nb_aretes = nb;

        // ---- les droites : l'arete `j` va de `v_j` a `v_j+1`, portee par la coupe `cid[ j ]`
        Droite2 dr[ Cell::max_nb ];
        const TF xi = P[ 0 ][ i ], yi = P[ 1 ][ i ];
        for ( int j = 0; j < nb; ++j ) {
            const auto id = cel.cid[ j ];
            if ( id >= 0 ) {
                const TF xj = P[ 0 ][ id ], yj = P[ 1 ][ id ];
                const TF nx = xj - xi, ny = yj - yi;
                dr[ j ] = { nx, ny, TF( 0.5 ) * ( nx * ( xj + xi ) + ny * ( yj + yi ) + w[ i ] - w[ id ] ),
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
            if ( ! ( std::fabs( det ) > 0 ) ) { q.etat = PolyCellule::DEGENERE; return; }
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

} // namespace sf

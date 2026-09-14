#pragma once

// =====================================================================================
// LE TEST D'ELAGAGE, UNE FOIS POUR TOUTES.
//
//     un germe `q` de la boite `B`, de poids majore par `w( q ) <= a . q + b`, peut-il encore
//     retrancher quelque chose a la cellule de `( p0, w0 )` ?
//
// Si `q` coupe la cellule, il en retranche au moins un SOMMET, et ce sommet verifie
// `|v - q|^2 - w( q ) < |v - p0|^2 - w0`. On rejette donc `B` des que TOUS les sommets ont
//
//     min_{ q in B } ( |v - q|^2 - a . q - b )  >=  |v - p0|^2 - w0
//
// et c'est EXACT au sens ou le minimum est calcule, pas majore : `|v - q|^2 - a . q` est separable
// par axe, son minimum libre est en `q = v + a / 2`, et un `clamp` par axe le donne. Le majorant
// constant est le cas `a = 0` et ne coute pas moins cher. C'est ce qui separe ce test d'une
// premiere version qui comparait la BOITE de la cellule a celle du noeud : 280 candidats par
// cellule la ou 26 suffisent, une boite majorant tres mal un polygone convexe.
//
// `<= 0` et non `< 0` : un plan qui passe exactement par un sommet n'enleve rien, donc l'admettre
// coute une coupe inutile la ou le refuser sur un arrondi perdrait une coupe VRAIE.
//
// LE SIMD EST ICI ET PAS DANS LE NOYAU : les sommets lui appartiennent et arrivent en registres
// en 2D ( `EtatReg` ), le fournisseur les lit tels quels -- deux `max`, deux `fma`, une
// comparaison, quel que soit leur nombre. En memoire ( `EtatMem*` ) c'est une boucle.
// =====================================================================================

#include <asimd/asimd.h>
#include "Etat.h"

namespace sdot {

/// une boite de germes et le majorant affine de leurs poids, dans le flottant du noyau
template<class TK,int D>
struct Boite {
    TK lo[ D ], hi[ D ];
    TK a[ D ], b;                                        ///< `w( q ) <= a . q + b`
};

/// le test, pour un etat en MEMOIRE ( `nb` sommets, `D` tableaux )
template<bool POIDS,class TK,int D>
inline bool peut_couper_boite( int nb, const TK *const *v, const TK *p0, TK w0, const Boite<TK,D> &B ) {
    const TK cb = POIDS ? w0 - B.b : TK( 0 );
    for ( int i = 0; i < nb; ++i ) {
        TK s = cb;
        for ( int d = 0; d < D; ++d ) {
            const TK x = v[ d ][ i ];
            TK y = x + ( POIDS ? B.a[ d ] / 2 : TK( 0 ) );
            y = y < B.lo[ d ] ? B.lo[ d ] : ( y > B.hi[ d ] ? B.hi[ d ] : y );
            const TK u = y - x, f = x - p0[ d ];
            s += u * u - f * f;
            if constexpr ( POIDS ) s -= B.a[ d ] * y;
        }
        if ( s <= 0 )
            return true;
    }
    return false;
}

/// le meme test, sur les huit voies d'un `EtatReg` ( 2D, registres )
template<bool POIDS,class TK,class Etat>
inline bool peut_couper_boite_reg( const Etat &e, const TK *p0, TK w0, const Boite<TK,2> &B ) {
    using V = asimd::SimdVec<TK,8>;
    // le point de la boite le plus proche du sommet, DECALE d'une demi-pente
    V y0 = e.vx, y1 = e.vy;
    if constexpr ( POIDS ) {
        y0 = y0 + V( B.a[ 0 ] / 2 );
        y1 = y1 + V( B.a[ 1 ] / 2 );
    }
    y0 = asimd::min( asimd::max( y0, V( B.lo[ 0 ] ) ), V( B.hi[ 0 ] ) );
    y1 = asimd::min( asimd::max( y1, V( B.lo[ 1 ] ) ), V( B.hi[ 1 ] ) );

    const V g0 = y0 - e.vx, f0 = e.vx - V( p0[ 0 ] );
    const V g1 = y1 - e.vy, f1 = e.vy - V( p0[ 1 ] );
    V s = asimd::fma( g0, g0, g1 * g1 ) - asimd::fma( f0, f0, f1 * f1 );
    if constexpr ( POIDS )
        s = s + V( w0 - B.b ) - asimd::fma( V( B.a[ 0 ] ), y0, V( B.a[ 1 ] ) * y1 );
    const unsigned m = unsigned( asimd::to_bits( asimd::ge( V( TK( 0 ) ), s ) ) );
    return ( m & ( ( 1u << Etat::nb ) - 1 ) ) != 0;
}

/// LA PORTE UNIQUE : quel que soit l'etat, la meme question.
template<bool POIDS,class TK,int D,class Etat>
inline bool peut_couper( const Etat &e, const TK *p0, TK w0, const Boite<TK,D> &B ) {
    // une cellule NON BORNEE est un simplexe de remplacement dont les coins sont inventes : rien
    // a elaguer contre, et la reponse honnete est « peut-etre ». L'accelerateur degenere alors en
    // balayage complet, ce qui est la bonne reponse et non un chemin lent que quelqu'un a choisi.
    if constexpr ( requires { e.bounded; } )
        if ( ! e.bounded )
            return true;
    if constexpr ( requires { e.vx + e.vx; } ) {
        static_assert( D == 2 );
        return peut_couper_boite_reg<POIDS,TK>( e, p0, w0, B );
    } else if constexpr ( D == 2 ) {
        const TK *v[ 2 ] = { e.vx, e.vy };
        return peut_couper_boite<POIDS,TK,2>( e.nb, v, p0, w0, B );
    } else {
        return peut_couper_boite<POIDS,TK,D>( e.nb, e.v, p0, w0, B );
    }
}

} // namespace sdot

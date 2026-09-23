#pragma once

// =====================================================================================
// LE TEST D'ELAGAGE 2D. La question, posee a un noeud de l'arbre :
//
//     un germe `q` de la boite `B`, de poids majore par `w( q ) <= a . q + b`, peut-il encore
//     retrancher quelque chose a la cellule de `p0` ?
//
// Si `q` coupe la cellule, il en retranche au moins un SOMMET `v`, qui verifie alors
// `|v - q|^2 - w( q ) < |v - p0|^2 - w0`. On rejette donc `B` des que TOUS les sommets ont
//
//     min_{ q in B } ( |v - q|^2 - a . q - b )  >=  |v - p0|^2 - w0
//
// et c'est EXACT : le minimum est separable par axe, libre en `q = v + a / 2`, un `clamp` par
// axe le donne. Le test ne rejette jamais a tort ; il n'est pas fait pour accepter le moins
// possible. ( Une premiere version comparait la BOITE de la cellule a celle du noeud : 280
// candidats par cellule la ou 26 suffisent. )
//
// `<= 0` et non `< 0` : un plan qui passe exactement par un sommet n'enleve rien, l'admettre coute
// une coupe inutile la ou le refuser sur un arrondi perdrait une coupe VRAIE.
//
// Les sommets sont en REGISTRES quand la cellule y tient -- le test coute une operation SIMD par
// boite, quel que soit leur nombre -- et en memoire pendant une excursion.
// =====================================================================================

#include <asimd/asimd.h>
#include <type_traits>

namespace sf::d2 {

/// une boite de germes et le majorant affine de leurs poids, dans le flottant du noyau.
template<class TK>
struct Boite2 {
    TK lo[ 2 ], hi[ 2 ];
    TK a[ 2 ] = { 0, 0 }, b = 0;                         ///< `w( q ) <= a . q + b`
};

/// `POIDS` : Laguerre ( les termes en `a` et `b` ) ou Voronoi ( ils disparaissent a la compilation ).
template<bool POIDS, class TK, class Etat>
inline bool peut_couper_boite( const Etat &e, TK x0, TK y0, TK w0, const Boite2<TK> &B ) {
    const TK a0 = POIDS ? B.a[ 0 ] : TK( 0 );
    const TK a1 = POIDS ? B.a[ 1 ] : TK( 0 );
    const TK cb = POIDS ? w0 - B.b : TK( 0 );

    if constexpr ( requires { e.vx + e.vx; } ) {         // registres
        using V = std::decay_t<decltype( e.vx )>;
        // le point de la boite le plus proche du sommet, DECALE d'une demi-pente
        V y0v = e.vx, y1v = e.vy;
        if constexpr ( POIDS ) {
            y0v = y0v + V( TK( 0.5 ) * a0 );
            y1v = y1v + V( TK( 0.5 ) * a1 );
        }
        y0v = asimd::min( asimd::max( y0v, V( B.lo[ 0 ] ) ), V( B.hi[ 0 ] ) );
        y1v = asimd::min( asimd::max( y1v, V( B.lo[ 1 ] ) ), V( B.hi[ 1 ] ) );

        const V g0 = y0v - e.vx, f0 = e.vx - V( x0 );
        const V g1 = y1v - e.vy, f1 = e.vy - V( y0 );
        V s = asimd::fma( g0, g0, g1 * g1 ) - asimd::fma( f0, f0, f1 * f1 );
        if constexpr ( POIDS )
            s = V( cb ) - asimd::fma( V( a0 ), y0v, V( a1 ) * y1v ) + s;
        return ( unsigned( asimd::to_bits( asimd::ge( V( TK( 0 ) ), s ) ) )
                 & ( ( 1u << Etat::nb ) - 1 ) ) != 0;
    } else {                                             // excursion : la cellule est en memoire
        const TK a[ 2 ] = { a0, a1 };
        const TK p0[ 2 ] = { x0, y0 };
        for ( int i = 0; i < e.nb; ++i ) {
            const TK v[ 2 ] = { e.vx[ i ], e.vy[ i ] };
            TK s = POIDS ? cb : TK( 0 );
            for ( int d = 0; d < 2; ++d ) {
                TK y = v[ d ] + ( POIDS ? TK( 0.5 ) * a[ d ] : TK( 0 ) );
                y = y < B.lo[ d ] ? B.lo[ d ] : ( y > B.hi[ d ] ? B.hi[ d ] : y );
                const TK u = y - v[ d ], f = v[ d ] - p0[ d ];
                s += u * u - f * f;
                if constexpr ( POIDS ) s -= a[ d ] * y;
            }
            if ( s <= 0 ) return true;
        }
        return false;
    }
}

} // namespace sf::d2

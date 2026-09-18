#pragma once

// =====================================================================================
// LE TEST D'ELAGAGE 3D : le critere de `Elagage2D.h`, un axe de plus. Exact pour la meme raison.
//
// CE QUE LA 3D CHANGE : les sommets sont en MEMOIRE, vingt a vingt-cinq par cellule, donc le test
// coute une boucle -- en asimd par blocs de `W` voies, chargements pleins puis une queue partielle
// dont les voies hors queue sont MASQUEES ( sinon un `s` de hasard ferait renoncer a un elagage
// legitime ). Sortie des qu'UN sommet peut etre coupe. Mesure : 2558 sommets lus par cellule a
// `leaf = 1`, ce qui reste petit ( `2d_des_familles`, en-tete de `FournisseurBsp3.h` ).
// =====================================================================================

#include <asimd/asimd.h>

namespace sf::d3 {

template<class TK>
struct Boite3 {
    TK lo[ 3 ], hi[ 3 ];
    TK a[ 3 ] = { 0, 0, 0 }, b = 0;                      ///< `w( q ) <= a . q + b`
};

/// `e` : `nb` sommets dans trois tableaux ALIGNES. Rend `true` s'il faut descendre dans la boite.
template<bool POIDS, int W = 8, class TK, class Etat>
inline bool peut_couper_boite3( const Etat &e, TK x0, TK y0, TK z0, TK w0, const Boite3<TK> &B ) {
    using V = asimd::SimdVec<TK,W>;

    const V l0( B.lo[0] ), h0( B.hi[0] ), l1( B.lo[1] ), h1( B.hi[1] ), l2( B.lo[2] ), h2( B.hi[2] );
    const V q0( x0 ), q1( y0 ), q2( z0 ), zero( TK( 0 ) );
    const V a0( POIDS ? B.a[0] : TK( 0 ) ), a1( POIDS ? B.a[1] : TK( 0 ) ), a2( POIDS ? B.a[2] : TK( 0 ) );
    const V m0( POIDS ? TK( 0.5 ) * B.a[0] : TK( 0 ) ), m1( POIDS ? TK( 0.5 ) * B.a[1] : TK( 0 ) ),
            m2( POIDS ? TK( 0.5 ) * B.a[2] : TK( 0 ) );
    const V cb( POIDS ? w0 - B.b : TK( 0 ) );

    // `s = min_q ( |v - q|^2 - a . q ) - |v - p0|^2 - b + w0`, par axe, pour `W` sommets.
    auto bilan = [ & ]( const V &vx, const V &vy, const V &vz ) {
        V y0v = vx, y1v = vy, y2v = vz;
        if constexpr ( POIDS ) { y0v = y0v + m0; y1v = y1v + m1; y2v = y2v + m2; }
        y0v = asimd::min( asimd::max( y0v, l0 ), h0 );   // le point de la boite le plus proche,
        y1v = asimd::min( asimd::max( y1v, l1 ), h1 );   // DECALE d'une demi-pente
        y2v = asimd::min( asimd::max( y2v, l2 ), h2 );

        const V g0 = y0v - vx, g1 = y1v - vy, g2 = y2v - vz;
        const V f0 = vx - q0,  f1 = vy - q1,  f2 = vz - q2;
        V s = asimd::fma( g0, g0, asimd::fma( g1, g1, g2 * g2 ) )
            - asimd::fma( f0, f0, asimd::fma( f1, f1, f2 * f2 ) );
        if constexpr ( POIDS )
            s = cb - asimd::fma( a0, y0v, asimd::fma( a1, y1v, a2 * y2v ) ) + s;
        return s;
    };

    const int plein = e.nb & ~( W - 1 );
    int i = 0;
    for ( ; i < plein; i += W ) {
        const V s = bilan( V::load_aligned( e.vx + i ),
                           V::load_aligned( e.vy + i ),
                           V::load_aligned( e.vz + i ) );
        if ( asimd::to_bits( asimd::ge( zero, s ) ) )
            return true;                                 // ce sommet-la peut etre coupe
    }
    if ( i < e.nb ) {
        const auto r = asimd::LaneRange<0>( e.nb - i );
        const V s = bilan( V::load_partial( e.vx + i, r ),
                           V::load_partial( e.vy + i, r ),
                           V::load_partial( e.vz + i, r ) );
        if ( asimd::to_bits( asimd::ge( zero, s ) ) & ( ( 1ull << ( e.nb - i ) ) - 1 ) )
            return true;
    }
    return false;
}

} // namespace sf::d3

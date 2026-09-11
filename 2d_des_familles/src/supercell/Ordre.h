// L'ORDRE SPATIAL DES GERMES -- la seule chose que l'echantillonnage demande au nuage.
//
// Echantillonner « un germe tous les `rho` » n'a de sens que dans un ordre ou deux voisins de
// liste sont voisins dans l'espace. Deux ordres sont fournis, et ce n'est pas de l'indecision :
// ils donnent des agregats INDISCERNABLES (meme rayon, meme degre, meme distribution des tailles)
// et Morton coute 2 a 3x moins cher, sans construire d'arbre sur les `n` germes. Le BSP reste
// parce qu'il est le temoin qui a permis de le dire.

#pragma once

#include "spatial_accel/AaBsp.h"
#include "bench/Bench.h"
#include <algorithm>
#include <cstdint>
#include <numeric>
#include <vector>

namespace pd::supercell {

using namespace pd::bench;

// ------------------------------------------------------------------ L'ORDRE SPATIAL

/// La cle de Morton, `64 / D` bits par coordonnee -- 32 en 2D, 21 en 3D. Les coordonnees sont
/// ramenees dans `[ 0, 1 )` sur la boite du nuage : quantifier sur la boite REELLE et non sur le
/// cube unite evite qu'un nuage etroit (les cinq plans, par exemple) ne perde toute sa resolution
/// dans la direction fine.
template<int D>
uint64_t morton( Vec<D> x, Vec<D> lo, Vec<D> inv ) {
    constexpr int B = 64 / D;
    constexpr uint64_t M = ( uint64_t( 1 ) << B ) - 1;
    uint64_t k = 0;
    for ( int d = 0; d < D; ++d ) {
        const TF t = ( x[ d ] - lo[ d ] ) * inv[ d ];
        const double q = double( t ) * double( M + 1 );
        uint64_t u = q <= 0 ? 0 : ( q >= double( M ) ? M : uint64_t( q ) );
        for ( int b = 0; b < B; ++b )
            k |= ( ( u >> b ) & uint64_t( 1 ) ) << ( b * D + d );
    }
    return k;
}

template<int D>
void ordre_morton( const Cloud<D> &cl, std::vector<SI> &ord ) {
    Vec<D> lo, hi;
    for ( int d = 0; d < D; ++d ) { lo[ d ] = cl.P[ d ][ 0 ]; hi[ d ] = cl.P[ d ][ 0 ]; }
    for ( SI i = 1; i < cl.n; ++i )
        for ( int d = 0; d < D; ++d ) {
            lo[ d ] = std::min( lo[ d ], cl.P[ d ][ i ] );
            hi[ d ] = std::max( hi[ d ], cl.P[ d ][ i ] );
        }
    Vec<D> inv;
    for ( int d = 0; d < D; ++d ) inv[ d ] = hi[ d ] > lo[ d ] ? TF( 1 ) / ( hi[ d ] - lo[ d ] ) : TF( 0 );

    std::vector<uint64_t> key( cl.n );
    for ( SI i = 0; i < cl.n; ++i ) {
        Vec<D> x;
        for ( int d = 0; d < D; ++d ) x[ d ] = cl.P[ d ][ i ];
        key[ i ] = morton<D>( x, lo, inv );
    }
    ord.resize( cl.n );
    std::iota( ord.begin(), ord.end(), SI( 0 ) );
    std::sort( ord.begin(), ord.end(), [ & ]( SI a, SI b ) { return key[ a ] < key[ b ]; } );
}

/// L'ordre du BSP, tel quel. `AaBsp::build` produit deja `order` -- des medianes recursives -- et
/// le relire ne coute rien de plus que l'arbre, qu'on aurait construit de toute facon.
template<int D>
void ordre_bsp( const Cloud<D> &cl, std::vector<SI> &ord, SI leaf ) {
    AaBspT<D> t;
    t.build( cl.P, cl.W, cl.n, leaf );
    ord = t.order;
}

} // namespace pd::supercell

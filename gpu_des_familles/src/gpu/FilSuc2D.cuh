#pragma once

// =====================================================================================
// TOUT EN MASQUES : LA CELLULE EST UNE RELATION DE SUCCESSION ( suite de `FilOrd2D.cuh` ).
//
// `filord` gardait l'ordre cyclique dans un registre `O` ( position -> slot ), ce qui obligeait a
// faire l'aller-retour slot -> position -> index -> slot a chaque coupe : une CHAINE sequentielle,
// 9.0 cycles par instruction emise contre 7.1 pour `filnrm` ( README § 4 ). Ici il n'y a plus de
// positions du tout : deux registres de 64 bits portent la SUCCESSION et la PRECEDENCE --
// l'octet `i` de `SU` est le masque one-hot du successeur du slot `i`, celui de `PR` son
// predecesseur -- et tout se calcule en masques de slots.
//
// LES QUATRE SOMMETS DE LA FRONTIERE, directement depuis `M` ( les slots dehors ) :
//     j0 = dedans & pred( M )      j3 = dedans & succ( M )        <- INDEPENDANTS
//     i1 = succ( j0 )              j2 = pred( j3 )                <- independants
// ( l'exterieur etant un arc contigu, chacun de ces masques n'a qu'un bit ). `succ` et `pred`
// d'un ensemble se lisent en `hor_or( SU & spread( X ) )` ; d'un singleton dont on a l'indice,
// en un simple decalage.
//
// LA MISE A JOUR est purement locale : le cycle devient `... j0 -> A -> B -> j3 ...`, soit trois
// octets a ecrire dans `SU` ( en `j0`, `A`, `B` ) et trois dans `PR` ( en `A`, `B`, `j3` ).
// Aucune rotation, aucun decalage global, aucune notion d'ordre a maintenir.
// =====================================================================================

#include "gpu/FilOrd2D.cuh"

namespace sf::gpu {

/// un masque de huit bits en huit OCTETS : `0xff` la ou le bit est a un.
__device__ __forceinline__ unsigned long long spread( unsigned X ) {
    unsigned long long t = ( unsigned long long ) X * 0x0101010101010101ull;
    t &= 0x8040201008040201ull;                          // octet `j` : `2^j` ou zero
    t |= t >> 4; t |= t >> 2; t |= t >> 1;
    t &= 0x0101010101010101ull;
    return t * 0xffull;
}

/// le OU de tous les octets.
__device__ __forceinline__ unsigned hor_or( unsigned long long t ) {
    t |= t >> 32; t |= t >> 16; t |= t >> 8;
    return unsigned( t ) & 0xffu;
}

/// l'octet `i` d'un registre de succession : le voisin du slot `i`.
__device__ __forceinline__ unsigned voisin( unsigned long long V, int i ) {
    return unsigned( V >> ( 8 * i ) ) & 0xffu;
}

/// y poser `val` : `V[ i ] = val`.
__device__ __forceinline__ void pose_voisin( unsigned long long &V, int i, unsigned val ) {
    const int d = 8 * i;
    V = ( V & ~( 0xffull << d ) ) | ( ( unsigned long long ) val << d );
}

template<bool POIDS, class TK>
__global__ void __launch_bounds__( 128 ) noyau2_filsuc( Arbre<TK,2> ar, double *res, int *deborde, int *liste_deb ) {
    constexpr int R = 8;
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if ( k >= ar.n ) return;
    const TK p0[ 2 ] = { ar.c[ 0 ][ k ], ar.c[ 1 ][ k ] };
    const TK w0 = POIDS ? ar.w[ k ] : TK( 0 );
    const int i0 = ar.ids[ k ];

    TK  x[ R ], y[ R ];
    int c[ R ];
#pragma unroll
    for ( int i = 0; i < R; ++i ) { x[ i ] = TK( i == 1 || i == 2 ); y[ i ] = TK( i == 2 || i == 3 ); c[ i ] = i < 4 ? -1 - i : 0; }
    unsigned vivant = 0x0fu;
    unsigned long long SU = 0x01080402ull;               // 0 -> 1 -> 2 -> 3 -> 0
    unsigned long long PR = 0x04020108ull;
    bool vide = false;

    int pile[ PILE ];
    int haut = 0;
    pile[ haut++ ] = 0;
    while ( haut ) {
        const int h = pile[ --haut ];
        const Noeud<TK,2> nd = ar.nodes[ h ];

        bool peut = false;
#pragma unroll
        for ( int i = 0; i < R; ++i ) {
            const TK v[ 2 ] = { x[ i ], y[ i ] };
            peut |= ( vivant >> i ) & 1u && bilan_sommet<POIDS>( nd, v, p0, w0 ) <= TK( 0 );
        }
        if ( ! peut )
            continue;

        if ( nd.right >= 0 ) {
            const int g = h + 1, dr = nd.right;
            const bool gp = proximite( ar.nodes[ g ], p0 ) <= proximite( ar.nodes[ dr ], p0 );
            pile[ haut++ ] = gp ? dr : g;
            pile[ haut++ ] = gp ? g : dr;
            continue;
        }

        for ( int q = nd.beg; q < nd.end; ++q ) {
            const Plan2<TK> p = bissect2<POIDS>( ar, q, p0[ 0 ], p0[ 1 ], w0 );

            unsigned M = 0;
#pragma unroll
            for ( int i = 0; i < R; ++i )
                M |= unsigned( p.dx * x[ i ] + p.dy * y[ i ] - p.off > TK( 0 ) ) << i;
            M &= vivant;
            if ( PROBABLE( ! M ) )
                continue;
            const unsigned In = vivant & ~M;
            if ( IMPROBABLE( ! In ) ) { vide = true; goto fin; }

            // ---- LA FRONTIERE, DEUX BRANCHES INDEPENDANTES
            const unsigned long long sp = spread( M );
            const unsigned mj0 = In & hor_or( PR & sp );
            const unsigned mj3 = In & hor_or( SU & sp );
            const int j0 = __ffs( int( mj0 ) ) - 1;
            const int j3 = __ffs( int( mj3 ) ) - 1;
            const int i1 = __ffs( int( voisin( SU, j0 ) ) ) - 1;
            const int j2 = __ffs( int( voisin( PR, j3 ) ) ) - 1;

            const int nb_in = __popc( In );
            if ( IMPROBABLE( nb_in + 2 > R ) ) { vivant = 0; goto fin; }

            const TK x0v = selR( x, j0 ), y0v = selR( y, j0 );
            const TK x1 = selR( x, i1 ), y1 = selR( y, i1 );
            const TK x2 = selR( x, j2 ), y2 = selR( y, j2 );
            const TK x3 = selR( x, j3 ), y3 = selR( y, j3 );
            const int bid = selR( c, j2 );
            const TK s0 = p.dx * x0v + p.dy * y0v - p.off, s1 = p.dx * x1 + p.dy * y1 - p.off;
            const TK s2 = p.dx * x2 + p.dy * y2 - p.off,   s3 = p.dx * x3 + p.dy * y3 - p.off;
            const TK ta = s0 / ( s0 - s1 ), tb = s3 / ( s3 - s2 );
            const TK pax = x0v + ( x1 - x0v ) * ta, pay = y0v + ( y1 - y0v ) * ta;
            const TK pbx = x3 + ( x2 - x3 ) * tb,   pby = y3 + ( y2 - y3 ) * tb;

            // ---- DEUX SLOTS LIBRES, puis le recollement `j0 -> A -> B -> j3`
            unsigned libre = ( ~vivant | M ) & 0xffu;
            const int sA = __ffs( int( libre ) ) - 1; libre &= ~( 1u << sA );
            const int sB = __ffs( int( libre ) ) - 1; libre &= ~( 1u << sB );
            const unsigned mA = 1u << sA, mB = 1u << sB;
#pragma unroll
            for ( int i = 0; i < R; ++i ) {
                x[ i ] = i == sA ? pax : ( i == sB ? pbx : x[ i ] );
                y[ i ] = i == sA ? pay : ( i == sB ? pby : y[ i ] );
                c[ i ] = i == sA ? p.id : ( i == sB ? bid : c[ i ] );
            }
            pose_voisin( SU, j0, mA ); pose_voisin( SU, sA, mB ); pose_voisin( SU, sB, mj3 );
            pose_voisin( PR, sA, mj0 ); pose_voisin( PR, sB, mA ); pose_voisin( PR, j3, mB );
            vivant = In | mA | mB;
        }
    }

fin:
    double a = 0;
    if ( ! vide && vivant ) {
        int t = __ffs( int( vivant ) ) - 1;
        const int t0 = t;
        TK xp = selR( x, t ), yp = selR( y, t );
        const TK x00 = xp, y00 = yp;
#pragma unroll
        for ( int i = 1; i < R; ++i ) {
            t = __ffs( int( voisin( SU, t ) ) ) - 1;
            if ( t == t0 ) break;
            const TK xq = selR( x, t ), yq = selR( y, t );
            a += double( xp ) * double( yq ) - double( xq ) * double( yp );
            xp = xq; yp = yq;
        }
        a += double( xp ) * double( y00 ) - double( x00 ) * double( yp );
    }
    if ( ! vide && ! vivant ) liste_deb[ atomicAdd( deborde, 1 ) ] = k;
    res[ i0 ] = 0.5 * fabs( a );
}

} // namespace sf::gpu

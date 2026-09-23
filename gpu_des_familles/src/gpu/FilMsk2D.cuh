#pragma once

// =====================================================================================
// LES REGISTRES RESTENT TRIES ( `filnrm` / `filrot` ), MAIS LA FRONTIERE EST CUEILLIE PAR DES
// MASQUES PARTAGES ENTRE x, y ET c ( idee de H. L. ).
//
// `filord` et `filsuc` laissaient les sommets sur place et payaient le desordre ; ici on garde
// l'ordre -- les sommets sont toujours en `0 .. nb - 1` -- et on ne change QUE deux choses :
//
//   1. LA CUEILLETTE DES QUATRE SOMMETS DE LA FRONTIERE. Au lieu de quatre indices ( `j0`, `i1`,
//      `j2`, `j3` ) suivis de neuf lectures a indice dynamique ( `selR` : un compare et un
//      `select` par case ), on fabrique QUATRE MASQUES DE ROLE de huit bits, chacun a un seul
//      bit -- ils sortent de `m`, `prev` et `next` en quatre instructions :
//          r0 = ~m & next   ( dedans, le suivant dehors  -> `v_j0` )
//          r1 =  m & ~prev  ( dehors, le precedent dedans -> `v_i1` )
//          r2 =  m & ~next  ( dehors, le suivant dedans   -> `v_j2` )
//          r3 = ~m & prev   ( dedans, le precedent dehors -> `v_j3` )
//      puis, case par case, UN MASQUE PLEIN par role ( `0` ou `-1` ), PARTAGE par `x`, `y` et
//      `c` : `ax0 |= bits( x[ i ] ) & k0` -- sur le GPU c'est un seul `LOP3` ( `et` puis `ou` en
//      une instruction ). Plus un indice dynamique, plus un compare par case et par valeur.
//
//   2. LE REMONTAGE EN UN SEUL BARILLET. `filrot` traite le cas `d == -1` ( une seule coupe
//      sortante, le cas le plus frequent ) par une copie decalee a droite, soit un `select` de
//      plus par case et par tableau. Ici on decale A PRIORI d'UNE CASE, ce qui est GRATUIT -- un
//      simple renommage de registres a la compilation -- en travaillant sur `u[ k ] = v[ k - 1 ]`
//      de neuf cases, et le barillet part de `e = d + 1 >= 0`. Un etage de moins a ecrire, trois
//      tableaux au lieu de six chez `filnrm`.
//
// `MASQ = false` garde le meme remontage mais cueille par indices ( `__ffs` + `selR` ) : c'est
// la mesure temoin qui isole ce que les masques valent.
// =====================================================================================

#include "gpu/FilNrm2D.cuh"

namespace sf::gpu {

/// le flottant vu comme un mot d'entiers, pour les `et` / `ou` masques
template<class TK> struct Mot;
template<> struct Mot<float> {
    using U = unsigned;
    static __device__ __forceinline__ U     de( float v )  { return __float_as_uint( v ); }
    static __device__ __forceinline__ float vers( U b )    { return __uint_as_float( b ); }
};
template<> struct Mot<double> {
    using U = unsigned long long;
    static __device__ __forceinline__ U      de( double v ) { return ( U ) __double_as_longlong( v ); }
    static __device__ __forceinline__ double vers( U b )    { return __longlong_as_double( ( long long ) b ); }
};

template<bool POIDS, bool MASQ, class TK>
__global__ void __launch_bounds__( 128 ) noyau2_filmsk( Arbre<TK,2> ar, double *res, int *deborde, int *liste_deb ) {
    using U = typename Mot<TK>::U;
    constexpr int R = 8, SUR = 3;
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if ( k >= ar.n ) return;
    const TK p0[ 2 ] = { ar.c[ 0 ][ k ], ar.c[ 1 ][ k ] };
    const TK w0 = POIDS ? ar.w[ k ] : TK( 0 );
    const int i0 = ar.ids[ k ];

    TK  x[ R ], y[ R ];
    int c[ R ];
#pragma unroll
    for ( int i = 0; i < R; ++i ) { x[ i ] = TK( i == 1 || i == 2 ); y[ i ] = TK( i == 2 || i == 3 ); c[ i ] = i < 4 ? -1 - i : 0; }
    int nb = 4;

    int pile[ PILE ];
    int haut = 0;
    pile[ haut++ ] = 0;
    while ( haut ) {
        const int h = pile[ --haut ];
        const Noeud<TK,2> nd = ar.nodes[ h ];

        bool peut = false;
#pragma unroll
        for ( int i = 0; i < R; ++i ) {
            if ( i >= SUR && i >= nb ) break;
            const TK v[ 2 ] = { x[ i ], y[ i ] };
            peut |= bilan_sommet<POIDS>( nd, v, p0, w0 ) <= TK( 0 );
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

            // ---- LA PREMIERE PASSE : le masque `m` des sommets dehors
            unsigned m = 0;
#pragma unroll
            for ( int i = 0; i < R; ++i ) {
                if ( i >= SUR && i >= nb ) break;
                m |= unsigned( p.dx * x[ i ] + p.dy * y[ i ] - p.off > TK( 0 ) ) << i;
            }
            if ( PROBABLE( ! m ) )
                continue;
            const unsigned valid = ( 1u << nb ) - 1;
            if ( IMPROBABLE( m == valid ) ) { nb = 0; goto fin; }

            // ---- LES QUATRE MASQUES DE ROLE, un seul bit chacun ( la plage dehors est contigue )
            const unsigned prev = ( ( m << 1 ) | ( m >> ( nb - 1 ) ) ) & valid;
            const unsigned next = ( ( m >> 1 ) | ( m << ( nb - 1 ) ) ) & valid;
            const unsigned r0 = ~m & next & valid;      // `v_j0` : dedans, le suivant dehors
            const unsigned r1 =  m & ~prev;             // `v_i1` : dehors, le precedent dedans
            const unsigned r2 =  m & ~next;             // `v_j2` : dehors, le suivant dedans
            const unsigned r3 = ~m & prev & valid;      // `v_j3` : dedans, le precedent dehors
            const int nb_out = __popc( m );
            const int nn = nb - nb_out + 2;
            if ( IMPROBABLE( nn > R ) ) { nb = -1; goto fin; }   // pour la seconde passe

            TK x0v, y0v, x1v, y1v, x2v, y2v, x3v, y3v;
            int bid;
            if constexpr ( MASQ ) {
                // ---- LA CUEILLETTE PAR MASQUES, DEROULEE A LA MAIN, CASE PAR CASE.
                //      `b?` : le bit du role pour cette case ; `- U( b? )` : le masque plein.
                //      Les quatre masques d'une case servent a `x`, a `y` et ( pour `r2` ) a `c`.
                U ax0 = 0, ay0 = 0, ax1 = 0, ay1 = 0, ax2 = 0, ay2 = 0, ax3 = 0, ay3 = 0;
                unsigned ac2 = 0;

                { const unsigned b0 = r0 & 1u, b1 = r1 & 1u, b2 = r2 & 1u, b3 = r3 & 1u;
                  const U bx = Mot<TK>::de( x[ 0 ] ), by = Mot<TK>::de( y[ 0 ] );
                  ax0 |= bx & -U( b0 ); ay0 |= by & -U( b0 );
                  ax1 |= bx & -U( b1 ); ay1 |= by & -U( b1 );
                  ax2 |= bx & -U( b2 ); ay2 |= by & -U( b2 ); ac2 |= unsigned( c[ 0 ] ) & -b2;
                  ax3 |= bx & -U( b3 ); ay3 |= by & -U( b3 ); }

                { const unsigned b0 = ( r0 >> 1 ) & 1u, b1 = ( r1 >> 1 ) & 1u, b2 = ( r2 >> 1 ) & 1u, b3 = ( r3 >> 1 ) & 1u;
                  const U bx = Mot<TK>::de( x[ 1 ] ), by = Mot<TK>::de( y[ 1 ] );
                  ax0 |= bx & -U( b0 ); ay0 |= by & -U( b0 );
                  ax1 |= bx & -U( b1 ); ay1 |= by & -U( b1 );
                  ax2 |= bx & -U( b2 ); ay2 |= by & -U( b2 ); ac2 |= unsigned( c[ 1 ] ) & -b2;
                  ax3 |= bx & -U( b3 ); ay3 |= by & -U( b3 ); }

                { const unsigned b0 = ( r0 >> 2 ) & 1u, b1 = ( r1 >> 2 ) & 1u, b2 = ( r2 >> 2 ) & 1u, b3 = ( r3 >> 2 ) & 1u;
                  const U bx = Mot<TK>::de( x[ 2 ] ), by = Mot<TK>::de( y[ 2 ] );
                  ax0 |= bx & -U( b0 ); ay0 |= by & -U( b0 );
                  ax1 |= bx & -U( b1 ); ay1 |= by & -U( b1 );
                  ax2 |= bx & -U( b2 ); ay2 |= by & -U( b2 ); ac2 |= unsigned( c[ 2 ] ) & -b2;
                  ax3 |= bx & -U( b3 ); ay3 |= by & -U( b3 ); }

                { const unsigned b0 = ( r0 >> 3 ) & 1u, b1 = ( r1 >> 3 ) & 1u, b2 = ( r2 >> 3 ) & 1u, b3 = ( r3 >> 3 ) & 1u;
                  const U bx = Mot<TK>::de( x[ 3 ] ), by = Mot<TK>::de( y[ 3 ] );
                  ax0 |= bx & -U( b0 ); ay0 |= by & -U( b0 );
                  ax1 |= bx & -U( b1 ); ay1 |= by & -U( b1 );
                  ax2 |= bx & -U( b2 ); ay2 |= by & -U( b2 ); ac2 |= unsigned( c[ 3 ] ) & -b2;
                  ax3 |= bx & -U( b3 ); ay3 |= by & -U( b3 ); }

                { const unsigned b0 = ( r0 >> 4 ) & 1u, b1 = ( r1 >> 4 ) & 1u, b2 = ( r2 >> 4 ) & 1u, b3 = ( r3 >> 4 ) & 1u;
                  const U bx = Mot<TK>::de( x[ 4 ] ), by = Mot<TK>::de( y[ 4 ] );
                  ax0 |= bx & -U( b0 ); ay0 |= by & -U( b0 );
                  ax1 |= bx & -U( b1 ); ay1 |= by & -U( b1 );
                  ax2 |= bx & -U( b2 ); ay2 |= by & -U( b2 ); ac2 |= unsigned( c[ 4 ] ) & -b2;
                  ax3 |= bx & -U( b3 ); ay3 |= by & -U( b3 ); }

                { const unsigned b0 = ( r0 >> 5 ) & 1u, b1 = ( r1 >> 5 ) & 1u, b2 = ( r2 >> 5 ) & 1u, b3 = ( r3 >> 5 ) & 1u;
                  const U bx = Mot<TK>::de( x[ 5 ] ), by = Mot<TK>::de( y[ 5 ] );
                  ax0 |= bx & -U( b0 ); ay0 |= by & -U( b0 );
                  ax1 |= bx & -U( b1 ); ay1 |= by & -U( b1 );
                  ax2 |= bx & -U( b2 ); ay2 |= by & -U( b2 ); ac2 |= unsigned( c[ 5 ] ) & -b2;
                  ax3 |= bx & -U( b3 ); ay3 |= by & -U( b3 ); }

                { const unsigned b0 = ( r0 >> 6 ) & 1u, b1 = ( r1 >> 6 ) & 1u, b2 = ( r2 >> 6 ) & 1u, b3 = ( r3 >> 6 ) & 1u;
                  const U bx = Mot<TK>::de( x[ 6 ] ), by = Mot<TK>::de( y[ 6 ] );
                  ax0 |= bx & -U( b0 ); ay0 |= by & -U( b0 );
                  ax1 |= bx & -U( b1 ); ay1 |= by & -U( b1 );
                  ax2 |= bx & -U( b2 ); ay2 |= by & -U( b2 ); ac2 |= unsigned( c[ 6 ] ) & -b2;
                  ax3 |= bx & -U( b3 ); ay3 |= by & -U( b3 ); }

                { const unsigned b0 = ( r0 >> 7 ) & 1u, b1 = ( r1 >> 7 ) & 1u, b2 = ( r2 >> 7 ) & 1u, b3 = ( r3 >> 7 ) & 1u;
                  const U bx = Mot<TK>::de( x[ 7 ] ), by = Mot<TK>::de( y[ 7 ] );
                  ax0 |= bx & -U( b0 ); ay0 |= by & -U( b0 );
                  ax1 |= bx & -U( b1 ); ay1 |= by & -U( b1 );
                  ax2 |= bx & -U( b2 ); ay2 |= by & -U( b2 ); ac2 |= unsigned( c[ 7 ] ) & -b2;
                  ax3 |= bx & -U( b3 ); ay3 |= by & -U( b3 ); }

                x0v = Mot<TK>::vers( ax0 ); y0v = Mot<TK>::vers( ay0 );
                x1v = Mot<TK>::vers( ax1 ); y1v = Mot<TK>::vers( ay1 );
                x2v = Mot<TK>::vers( ax2 ); y2v = Mot<TK>::vers( ay2 );
                x3v = Mot<TK>::vers( ax3 ); y3v = Mot<TK>::vers( ay3 );
                bid = int( ac2 );
            } else {
                // ---- LE TEMOIN : par indices, comme `filrot`
                const int j0 = __ffs( int( r0 ) ) - 1, i1 = __ffs( int( r1 ) ) - 1;
                const int j2 = __ffs( int( r2 ) ) - 1, j3 = __ffs( int( r3 ) ) - 1;
                x0v = selR( x, j0 ); y0v = selR( y, j0 );
                x1v = selR( x, i1 ); y1v = selR( y, i1 );
                x2v = selR( x, j2 ); y2v = selR( y, j2 );
                x3v = selR( x, j3 ); y3v = selR( y, j3 );
                bid = selR( c, j2 );
            }

            // ---- LES DEUX POINTS CREES ; `s` recalcule pour les quatre ( un `fma` chacun )
            const TK s0 = p.dx * x0v + p.dy * y0v - p.off, s1 = p.dx * x1v + p.dy * y1v - p.off;
            const TK s2 = p.dx * x2v + p.dy * y2v - p.off, s3 = p.dx * x3v + p.dy * y3v - p.off;
            const TK ta = s0 / ( s0 - s1 ), tb = s3 / ( s3 - s2 );
            const TK pax = x0v + ( x1v - x0v ) * ta, pay = y0v + ( y1v - y0v ) * ta;
            const TK pbx = x3v + ( x2v - x3v ) * tb, pby = y3v + ( y2v - y3v ) * tb;

            // ---- LE REMONTAGE : `new[ o ] = o < a ? old[ o ] : o == a ? A : o == a + 1 ? B :
            //      old[ o + d ]`, avec `d >= -1`. On pose `u[ k ] = old[ k - 1 ]` ( GRATUIT : un
            //      renommage ) et `e = d + 1 >= 0`, d'ou `old[ o + d ] = u[ o + e ]` : UN SEUL
            //      barillet, trois etages, la meme condition pour les trois tableaux.
            const bool boucle = r1 > r2;                        // `i1 > j2` : la plage dehors boucle
            const int  a = boucle ? 0 : __ffs( int( r1 ) ) - 1;
            const int  e = boucle ? __ffs( int( r3 ) ) - 2 : nb_out - 1;
            TK  ux[ R + 1 ], uy[ R + 1 ];
            int uc[ R + 1 ];
            ux[ 0 ] = x[ 0 ]; uy[ 0 ] = y[ 0 ]; uc[ 0 ] = c[ 0 ];   // jamais lu : `o + e >= 1`
#pragma unroll
            for ( int o = 1; o < R + 1; ++o ) { ux[ o ] = x[ o - 1 ]; uy[ o ] = y[ o - 1 ]; uc[ o ] = c[ o - 1 ]; }
            const bool e1 = e & 1, e2 = e & 2, e4 = e & 4;
#pragma unroll
            for ( int o = 0; o + 1 < R + 1; ++o ) { ux[ o ] = e1 ? ux[ o + 1 ] : ux[ o ]; uy[ o ] = e1 ? uy[ o + 1 ] : uy[ o ]; uc[ o ] = e1 ? uc[ o + 1 ] : uc[ o ]; }
#pragma unroll
            for ( int o = 0; o + 2 < R + 1; ++o ) { ux[ o ] = e2 ? ux[ o + 2 ] : ux[ o ]; uy[ o ] = e2 ? uy[ o + 2 ] : uy[ o ]; uc[ o ] = e2 ? uc[ o + 2 ] : uc[ o ]; }
#pragma unroll
            for ( int o = 0; o + 4 < R + 1; ++o ) { ux[ o ] = e4 ? ux[ o + 4 ] : ux[ o ]; uy[ o ] = e4 ? uy[ o + 4 ] : uy[ o ]; uc[ o ] = e4 ? uc[ o + 4 ] : uc[ o ]; }
#pragma unroll
            for ( int o = 0; o < R; ++o ) {
                if ( o >= SUR && o >= nn ) break;
                x[ o ] = o < a ? x[ o ] : ( o == a ? pax  : ( o == a + 1 ? pbx : ux[ o ] ) );
                y[ o ] = o < a ? y[ o ] : ( o == a ? pay  : ( o == a + 1 ? pby : uy[ o ] ) );
                c[ o ] = o < a ? c[ o ] : ( o == a ? p.id : ( o == a + 1 ? bid : uc[ o ] ) );
            }
            nb = nn;
        }
    }

fin:
    double area = 0;
    if ( nb > 0 ) {
        double a = 0;
#pragma unroll
        for ( int i = 0; i < R; ++i ) {
            if ( i >= SUR && i >= nb ) break;
            const int j = i + 1 < nb ? i + 1 : 0;
            a += double( x[ i ] ) * double( selR( y, j ) ) - double( selR( x, j ) ) * double( y[ i ] );
        }
        area = 0.5 * fabs( a );
    }
    if ( nb < 0 ) liste_deb[ atomicAdd( deborde, 1 ) ] = k;
    res[ i0 ] = area;
}

} // namespace sf::gpu

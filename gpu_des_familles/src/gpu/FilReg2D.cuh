#pragma once

// =====================================================================================
// UNE CELLULE PAR THREAD, 2D, LES POSITIONS EN REGISTRES. Le noyau a registres du CPU
// ( `cell/Noyau2D.h` ) execute par UN thread scalaire : huit `x`, huit `y` dans des registres
// ( tout est deroule, une lecture a indice dynamique est une chaine de `select` ), `nb` un
// entier, la coupe sans boucle -- masque de signe, ses deux bouts par `ffs`, deux intersections,
// et le remontage par huit `select` a huit entrees. Les `cid` restent en memoire locale : on ne
// les lit qu'au remontage et a la fin, huit registres de plus ne se justifient pas.
//
// Ce que ca teste : `fil` ( Fil2D.cuh ) a ses sommets en memoire locale, L1 a 49 % de succes, 51
// cycles par instruction emise. Ici plus une lecture de sommet en memoire ; la divergence entre
// les 32 cellules d'un warp, elle, ne change pas.
//
// Le debordement est une excursion : au-dela de huit sommets la cellule se pose en memoire
// locale et `coupe2` continue en scalaire, jusqu'a ce qu'elle redescende a huit.
// =====================================================================================

#include "gpu/Fil2D.cuh"

namespace sf::gpu {

template<class T>
__device__ __forceinline__ T sel8( const T ( &a )[ 8 ], int i ) {
    T r = a[ 0 ];
#pragma unroll
    for ( int q = 1; q < 8; ++q ) r = i == q ? a[ q ] : r;
    return r;
}

template<bool POIDS, int MaxNb, bool CIDREG, class TK>
__global__ void __launch_bounds__( 128 ) noyau2_filreg( Arbre<TK,2> ar, double *res, int *deborde ) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if ( k >= ar.n ) return;
    const TK p0[ 2 ] = { ar.c[ 0 ][ k ], ar.c[ 1 ][ k ] };
    const TK w0 = POIDS ? ar.w[ k ] : TK( 0 );
    const int i0 = ar.ids[ k ];

    TK x[ 8 ], y[ 8 ];
    int cid[ MaxNb ];                                    // memoire locale ; sert aussi a l'excursion
    int cr[ 8 ];                                         // `CIDREG` : les huit cid en registres
#pragma unroll
    for ( int i = 0; i < 8; ++i ) { x[ i ] = TK( i == 1 || i == 2 ); y[ i ] = TK( i == 2 || i == 3 ); cr[ i ] = i < 4 ? -1 - i : 0; }
    cid[ 0 ] = -1; cid[ 1 ] = -2; cid[ 2 ] = -3; cid[ 3 ] = -4;
    int nb = 4;
    TK  lvx[ MaxNb ], lvy[ MaxNb ], ls[ MaxNb ];
    bool large = false;

    int pile[ PILE ];
    int haut = 0;
    pile[ haut++ ] = 0;
    while ( haut ) {
        const int h = pile[ --haut ];
        const Noeud<TK,2> nd = ar.nodes[ h ];

        // ---- le test d'elagage : huit bilans, masques par `nb`
        bool peut;
        if ( ! large ) {
            peut = false;
#pragma unroll
            for ( int i = 0; i < 8; ++i ) {
                const TK v[ 2 ] = { x[ i ], y[ i ] };
                peut |= i < nb && bilan_sommet<POIDS>( nd, v, p0, w0 ) <= TK( 0 );
            }
        } else
            peut = peut_couper2<POIDS>( nd, p0, w0, nb, lvx, lvy );
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
            if ( ar.ids[ q ] == i0 ) continue;
            const Plan2<TK> p = bissect2<POIDS>( ar, q, p0[ 0 ], p0[ 1 ], w0 );

            if ( ! large ) {
                // LE TEST, QUI EST DEJA LA COUPE
                TK s[ 8 ];
                unsigned m = 0;
#pragma unroll
                for ( int i = 0; i < 8; ++i ) {
                    s[ i ] = p.dx * x[ i ] + p.dy * y[ i ] - p.off;
                    m |= unsigned( i < nb && s[ i ] > TK( 0 ) ) << i;
                }
                if ( ! m )
                    continue;
                const unsigned valid = ( 1u << nb ) - 1;
                if ( m == valid ) { nb = 0; goto fin; }

                const unsigned prev = ( ( m << 1 ) | ( m >> ( nb - 1 ) ) ) & valid;
                const unsigned next = ( ( m >> 1 ) | ( m << ( nb - 1 ) ) ) & valid;
                const int i1 = __ffs( m & ~prev ) - 1;
                const int j2 = __ffs( m & ~next ) - 1;
                const int j0 = i1 ? i1 - 1 : nb - 1;
                const int j3 = j2 + 1 < nb ? j2 + 1 : 0;
                const int nb_in = nb - __popc( m );
                const int nn = nb_in + 2;

                // les deux intersections, ancrees sur le sommet dedans
                const TK s0 = sel8( s, j0 ), s1 = sel8( s, i1 ), s2 = sel8( s, j2 ), s3 = sel8( s, j3 );
                const TK x0v = sel8( x, j0 ), y0v = sel8( y, j0 ), x1v = sel8( x, i1 ), y1v = sel8( y, i1 );
                const TK x2v = sel8( x, j2 ), y2v = sel8( y, j2 ), x3v = sel8( x, j3 ), y3v = sel8( y, j3 );
                const TK ta = s0 / ( s0 - s1 ), tb = s3 / ( s3 - s2 );
                const TK pax = x0v + ( x1v - x0v ) * ta, pay = y0v + ( y1v - y0v ) * ta;
                const TK pbx = x3v + ( x2v - x3v ) * tb, pby = y3v + ( y2v - y3v ) * tb;
                const int bid = CIDREG ? sel8( cr, j2 ) : cid[ j2 ];

                if ( nn > 8 ) {                          // L'EXCURSION PART DE LA
#pragma unroll
                    for ( int i = 0; i < 8; ++i ) { lvx[ i ] = x[ i ]; lvy[ i ] = y[ i ]; if ( CIDREG ) cid[ i ] = cr[ i ]; }
                    large = true;
                } else {
                    // LE REMONTAGE : `[ v_j3, ..., v_j0, A, B ]`
                    TK  nx[ 8 ], ny[ 8 ];
                    int nc[ 8 ];
#pragma unroll
                    for ( int o = 0; o < 8; ++o ) {
                        int og = j3 + o; og = og >= nb ? og - nb : og;
                        nx[ o ] = o == nb_in ? pax : ( o == nb_in + 1 ? pbx : sel8( x, og ) );
                        ny[ o ] = o == nb_in ? pay : ( o == nb_in + 1 ? pby : sel8( y, og ) );
                        nc[ o ] = o == nb_in ? p.id : ( o == nb_in + 1 ? bid : ( CIDREG ? sel8( cr, og ) : cid[ og ] ) );
                    }
#pragma unroll
                    for ( int o = 0; o < 8; ++o ) { x[ o ] = nx[ o ]; y[ o ] = ny[ o ]; if ( CIDREG ) cr[ o ] = nc[ o ]; else cid[ o ] = nc[ o ]; }
                    nb = nn;
                    continue;
                }
            }

            // EN EXCURSION : en scalaire, en memoire, jusqu'au retour a huit
            nb = coupe2<MaxNb>( lvx, lvy, cid, nb, p, ls );
            if ( nb <= 0 )
                goto fin;
            if ( nb <= 8 ) {
#pragma unroll
                for ( int i = 0; i < 8; ++i ) { x[ i ] = i < nb ? lvx[ i ] : TK( 0 ); y[ i ] = i < nb ? lvy[ i ] : TK( 0 ); if ( CIDREG ) cr[ i ] = cid[ i ]; }
                large = false;
            }
        }
    }

fin:
    double area = 0;
    if ( nb > 0 ) {
        double a = 0;
        if ( ! large ) {
#pragma unroll
            for ( int i = 0; i < 8; ++i ) {
                const int j = i + 1 < nb ? i + 1 : 0;
                if ( i < nb ) a += double( x[ i ] ) * double( sel8( y, j ) ) - double( sel8( x, j ) ) * double( y[ i ] );
            }
        } else
            for ( int i = 0; i < nb; ++i ) {
                const int j = i + 1 < nb ? i + 1 : 0;
                a += double( lvx[ i ] ) * double( lvy[ j ] ) - double( lvx[ j ] ) * double( lvy[ i ] );
            }
        area = 0.5 * fabs( a );
    }
    if ( nb < 0 ) atomicAdd( deborde, 1 );
    res[ i0 ] = area;
}

} // namespace sf::gpu

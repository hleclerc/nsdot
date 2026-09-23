#pragma once

// =====================================================================================
// UNE CELLULE PAR THREAD, 2D, `R` SOMMETS EN REGISTRES ET LA QUEUE EN MEMOIRE. La generalisation
// de `FilReg2D.cuh` : les sommets `0 .. R-1` sont dans des registres et leur code est deroule,
// les sommets `R .. nb-1` sont dans des tableaux locaux et parcourus par une boucle ordinaire --
// qui diverge entre les threads d'un warp, mais qui ne fait pas plus de travail, et qui ne fait
// RIEN pour une cellule qui tient dans ses registres. Il n'y a plus d'excursion : `nb` va
// jusqu'a `MaxNb` sans changer de mode, la coupe est une seule et meme suite d'instructions.
//
// Ce que `R` regle : le code deroule fait toujours `R` cases, qu'elles soient occupees ou non
// ( une cellule finie a six sommets ) ; la queue coute une boucle divergente. `R = 8` est
// `filreg` sans excursion, `R = 4` parie sur le peu de travail perdu, `R = 16` sur l'absence de
// queue ( Laguerre, ou les cellules ont plus de sommets ).
//
// Les masques sont sur 64 bits ( `MaxNb <= 64` ), `ffsll` / `popcll` les lisent.
// =====================================================================================

#include "gpu/Fil2D.cuh"

namespace sf::gpu {

template<class T, int R>
__device__ __forceinline__ T selR( const T ( &a )[ R ], int i ) {
    T r = a[ 0 ];
#pragma unroll
    for ( int q = 1; q < R; ++q ) r = i == q ? a[ q ] : r;
    return r;
}

/// `liste` : les rangs a faire ( `nullptr` : tous ) -- la seconde passe de `filbrk`.
template<bool POIDS, int MaxNb, int R, bool CIDREG, class TK>
__global__ void __launch_bounds__( 128 ) noyau2_filmix( Arbre<TK,2> ar, double *res, int *deborde, const int *liste = nullptr, int nl = 0 ) {
    static_assert( MaxNb <= 64 && R <= MaxNb && R >= 4, "les masques sont sur 64 bits, le carre tient dans les registres" );
    const int ti = blockIdx.x * blockDim.x + threadIdx.x;
    const int k = liste ? ( ti < nl ? liste[ ti ] : ar.n ) : ti;
    if ( k >= ar.n ) return;
    const TK p0[ 2 ] = { ar.c[ 0 ][ k ], ar.c[ 1 ][ k ] };
    const TK w0 = POIDS ? ar.w[ k ] : TK( 0 );
    const int i0 = ar.ids[ k ];

    // ---- les registres, puis la queue
    TK  x[ R ], y[ R ];
    int cr[ R ];                                         // `CIDREG`
    TK  lx[ MaxNb ], ly[ MaxNb ], ls[ MaxNb ];          // la queue ( entrees `>= R` ), et `s` de la queue
    int lc[ MaxNb ];                                     // les cid : tous si `! CIDREG`, la queue sinon
#pragma unroll
    for ( int i = 0; i < R; ++i ) { x[ i ] = TK( i == 1 || i == 2 ); y[ i ] = TK( i == 2 || i == 3 ); cr[ i ] = i < 4 ? -1 - i : 0; }
    lc[ 0 ] = -1; lc[ 1 ] = -2; lc[ 2 ] = -3; lc[ 3 ] = -4;
    int nb = 4;

    auto get_x = [ & ]( int j ) { return j < R ? selR( x, j ) : lx[ j ]; };
    auto get_y = [ & ]( int j ) { return j < R ? selR( y, j ) : ly[ j ]; };
    auto get_c = [ & ]( int j ) { return CIDREG && j < R ? selR( cr, j ) : lc[ j ]; };

    int pile[ PILE ];
    int haut = 0;
    pile[ haut++ ] = 0;
    while ( haut ) {
        const int h = pile[ --haut ];
        const Noeud<TK,2> nd = ar.nodes[ h ];

        // ---- le test d'elagage : `R` bilans deroules, la queue en boucle
        bool peut = false;
#pragma unroll
        for ( int i = 0; i < R; ++i ) {
            const TK v[ 2 ] = { x[ i ], y[ i ] };
            peut |= i < nb && bilan_sommet<POIDS>( nd, v, p0, w0 ) <= TK( 0 );
        }
        for ( int i = R; i < nb && ! peut; ++i ) {
            const TK v[ 2 ] = { lx[ i ], ly[ i ] };
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
            if ( ar.ids[ q ] == i0 ) continue;
            const Plan2<TK> p = bissect2<POIDS>( ar, q, p0[ 0 ], p0[ 1 ], w0 );

            // LE TEST, QUI EST DEJA LA COUPE : le masque de signe sur 64 bits
            TK s[ R ];
            unsigned long long m = 0;
#pragma unroll
            for ( int i = 0; i < R; ++i ) {
                s[ i ] = p.dx * x[ i ] + p.dy * y[ i ] - p.off;
                m |= ( unsigned long long ) ( i < nb && s[ i ] > TK( 0 ) ) << i;
            }
            for ( int i = R; i < nb; ++i ) {
                ls[ i ] = p.dx * lx[ i ] + p.dy * ly[ i ] - p.off;
                m |= ( unsigned long long ) ( ls[ i ] > TK( 0 ) ) << i;
            }
            if ( ! m )
                continue;
            const unsigned long long valid = nb >= 64 ? ~0ull : ( 1ull << nb ) - 1;
            if ( m == valid ) { nb = 0; goto fin; }

            const unsigned long long prev = ( ( m << 1 ) | ( m >> ( nb - 1 ) ) ) & valid;
            const unsigned long long next = ( ( m >> 1 ) | ( m << ( nb - 1 ) ) ) & valid;
            const int i1 = __ffsll( ( long long ) ( m & ~prev ) ) - 1;
            const int j2 = __ffsll( ( long long ) ( m & ~next ) ) - 1;
            const int j0 = i1 ? i1 - 1 : nb - 1;
            const int j3 = j2 + 1 < nb ? j2 + 1 : 0;
            const int nb_in = nb - __popcll( m );
            const int nn = nb_in + 2;
            if ( nn > MaxNb ) { nb = -1; goto fin; }

            // les deux intersections, ancrees sur le sommet dedans
            auto get_s = [ & ]( int j ) { return j < R ? selR( s, j ) : ls[ j ]; };
            const TK s0 = get_s( j0 ), s1 = get_s( i1 ), s2 = get_s( j2 ), s3 = get_s( j3 );
            const TK x0v = get_x( j0 ), y0v = get_y( j0 ), x1v = get_x( i1 ), y1v = get_y( i1 );
            const TK x2v = get_x( j2 ), y2v = get_y( j2 ), x3v = get_x( j3 ), y3v = get_y( j3 );
            const TK ta = s0 / ( s0 - s1 ), tb = s3 / ( s3 - s2 );
            const TK pax = x0v + ( x1v - x0v ) * ta, pay = y0v + ( y1v - y0v ) * ta;
            const TK pbx = x3v + ( x2v - x3v ) * tb, pby = y3v + ( y2v - y3v ) * tb;
            const int bid = get_c( j2 );

            // LE REMONTAGE `[ v_j3, ..., v_j0, A, B ]` : les registres deroules, la queue en boucle
            // dans un tampon ( elle lit derriere elle ), rien n'est ecrit avant que tout soit lu
            TK  nx[ R ], ny[ R ];
            int nc[ R ];
#pragma unroll
            for ( int o = 0; o < R; ++o ) {
                int og = j3 + o; og = og >= nb ? og - nb : og;
                nx[ o ] = o == nb_in ? pax : ( o == nb_in + 1 ? pbx : get_x( og ) );
                ny[ o ] = o == nb_in ? pay : ( o == nb_in + 1 ? pby : get_y( og ) );
                nc[ o ] = o == nb_in ? p.id : ( o == nb_in + 1 ? bid : get_c( og ) );
            }
            TK  tx[ MaxNb ], ty[ MaxNb ];
            int tc[ MaxNb ];
            for ( int o = R; o < nn; ++o ) {
                int og = j3 + o; og = og >= nb ? og - nb : og;
                tx[ o ] = o == nb_in ? pax : ( o == nb_in + 1 ? pbx : get_x( og ) );
                ty[ o ] = o == nb_in ? pay : ( o == nb_in + 1 ? pby : get_y( og ) );
                tc[ o ] = o == nb_in ? p.id : ( o == nb_in + 1 ? bid : get_c( og ) );
            }
#pragma unroll
            for ( int o = 0; o < R; ++o ) { x[ o ] = nx[ o ]; y[ o ] = ny[ o ]; if ( CIDREG ) cr[ o ] = nc[ o ]; else lc[ o ] = nc[ o ]; }
            for ( int o = R; o < nn; ++o ) { lx[ o ] = tx[ o ]; ly[ o ] = ty[ o ]; lc[ o ] = tc[ o ]; }
            nb = nn;
        }
    }

fin:
    double area = 0;
    if ( nb > 0 ) {
        double a = 0;
#pragma unroll
        for ( int i = 0; i < R; ++i ) {
            const int j = i + 1 < nb ? i + 1 : 0;
            if ( i < nb ) a += double( x[ i ] ) * double( get_y( j ) ) - double( get_x( j ) ) * double( y[ i ] );
        }
        for ( int i = R; i < nb; ++i ) {
            const int j = i + 1 < nb ? i + 1 : 0;
            a += double( lx[ i ] ) * double( get_y( j ) ) - double( get_x( j ) ) * double( ly[ i ] );
        }
        area = 0.5 * fabs( a );
    }
    if ( nb < 0 ) atomicAdd( deborde, 1 );
    res[ i0 ] = area;
}

} // namespace sf::gpu

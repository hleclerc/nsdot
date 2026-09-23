#pragma once

// =====================================================================================
// `filbrk` AVEC UN AUTRE REMONTAGE : LE DECALAGE EN BARILLET. Dans `filbrk` chaque case de sortie
// va CHERCHER son sommet par une lecture a indice dynamique ( `selR` : sept compares, sept
// `select` ), trois tableaux, huit cases -- la moitie des instructions du noyau ( mesure ). Ici
// les sommets gardes sont DECALES en bloc d'un pas `d` dynamique, en barillet : trois etages
// ( decale de 1, de 2, de 4 selon les bits de `d` ), un `select` par case et par etage, sans un
// compare par case. Les points crees sont a des positions FIXES quand c'est possible.
//
// La plage exterieure `[ i1 .. j2 ]` est cyclique contigue. Deux cas :
//   * elle BOUCLE ( `i1 > j2` ) : les gardes `[ j3 .. j0 ]` sont contigus ; la sortie est
//     `[ A, B, v_j3, ..., v_j0 ]` -- A et B en 0 et 1, le reste decale a gauche de `j3 - 2` ;
//   * sinon : `[ v_0, ..., v_i1-1, A, B, v_j3, ..., v_nb-1 ]` -- la tete ne bouge pas, A et B en
//     `i1` et `i1 + 1`, la queue decalee de `nb_out - 2` ( a droite de 1 si `nb_out == 1` ).
// Les deux se lisent d'une seule formule : `new[ o ] = o < a ? old[ o ] : o == a ? A : o == a + 1
// ? B : old[ o + d ]`, avec `( a, d ) = ( 0, j3 - 2 )` ou `( i1, nb_out - 2 )`.
// =====================================================================================

#include "gpu/FilBrk2D.cuh"

namespace sf::gpu {

/// `a` decale a gauche de `d >= 0` en barillet ( `R` puissance de deux ou pas, les bits de `d`
/// jusqu'a `R - 1` ) : `r[ o ] = a[ o + d ]`, garbage au-dela de `R`.
template<class T, int R>
__device__ __forceinline__ void barillet( T ( &a )[ R ], int d ) {
#pragma unroll
    for ( int b = 1; b < R; b *= 2 ) {
        const bool on = d & b;
#pragma unroll
        for ( int o = 0; o < R; ++o ) a[ o ] = on && o + b < R ? a[ o + b ] : a[ o ];
    }
}

template<bool POIDS, int R, class TK>
__global__ void __launch_bounds__( 128 ) noyau2_filrot( Arbre<TK,2> ar, double *res, int *deborde, int *liste_deb ) {
    constexpr int SUR = 3;
    static_assert( R >= 4 && R <= 32, "le carre tient dans les registres, le masque dans 32 bits" );
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
            if ( IMPROBABLE( ar.ids[ q ] == i0 ) ) continue;
            const Plan2<TK> p = bissect2<POIDS>( ar, q, p0[ 0 ], p0[ 1 ], w0 );

            TK s[ R ];
            unsigned m = 0;
#pragma unroll
            for ( int i = 0; i < R; ++i ) {
                if ( i >= SUR && i >= nb ) break;
                s[ i ] = p.dx * x[ i ] + p.dy * y[ i ] - p.off;
                m |= unsigned( s[ i ] > TK( 0 ) ) << i;
            }
            if ( PROBABLE( ! m ) )
                continue;
            const unsigned valid = nb >= 32 ? 0xffffffffu : ( 1u << nb ) - 1;
            if ( IMPROBABLE( m == valid ) ) { nb = 0; goto fin; }

            const unsigned prev = ( ( m << 1 ) | ( m >> ( nb - 1 ) ) ) & valid;
            const unsigned next = ( ( m >> 1 ) | ( m << ( nb - 1 ) ) ) & valid;
            const int i1 = __ffs( m & ~prev ) - 1;      // l'ENTREE : le premier dehors
            const int j2 = __ffs( m & ~next ) - 1;      // le dernier dehors
            const int j0 = i1 ? i1 - 1 : nb - 1;
            const int j3 = j2 + 1 < nb ? j2 + 1 : 0;
            const int nb_out = __popc( m );
            const int nn = nb - nb_out + 2;
            if ( IMPROBABLE( nn > R ) ) { nb = -1; goto fin; }   // pour la seconde passe

            const TK s0 = selR( s, j0 ), s1 = selR( s, i1 ), s2 = selR( s, j2 ), s3 = selR( s, j3 );
            const TK x0v = selR( x, j0 ), y0v = selR( y, j0 ), x1v = selR( x, i1 ), y1v = selR( y, i1 );
            const TK x2v = selR( x, j2 ), y2v = selR( y, j2 ), x3v = selR( x, j3 ), y3v = selR( y, j3 );
            const TK ta = s0 / ( s0 - s1 ), tb = s3 / ( s3 - s2 );
            const TK pax = x0v + ( x1v - x0v ) * ta, pay = y0v + ( y1v - y0v ) * ta;
            const TK pbx = x3v + ( x2v - x3v ) * tb, pby = y3v + ( y2v - y3v ) * tb;
            const int bid = selR( c, j2 );

            // LE REMONTAGE PAR DECALAGE : `( a, d )`, la queue decalee en barillet, A et B poses
            const bool boucle = i1 > j2;
            const int  a = boucle ? 0 : i1;
            const int  d = boucle ? j3 - 2 : nb_out - 2;
            TK  sx[ R ], sy[ R ];
            int sc[ R ];
#pragma unroll
            for ( int o = 0; o < R; ++o ) {              // `d == -1` : a droite de un
                sx[ o ] = d < 0 ? ( o ? x[ o - 1 ] : x[ 0 ] ) : x[ o ];
                sy[ o ] = d < 0 ? ( o ? y[ o - 1 ] : y[ 0 ] ) : y[ o ];
                sc[ o ] = d < 0 ? ( o ? c[ o - 1 ] : c[ 0 ] ) : c[ o ];
            }
            const int dd = d < 0 ? 0 : d;
            barillet( sx, dd ); barillet( sy, dd ); barillet( sc, dd );
#pragma unroll
            for ( int o = 0; o < R; ++o ) {
                if ( o >= SUR && o >= nn ) break;
                x[ o ] = o < a ? x[ o ] : ( o == a ? pax  : ( o == a + 1 ? pbx : sx[ o ] ) );
                y[ o ] = o < a ? y[ o ] : ( o == a ? pay  : ( o == a + 1 ? pby : sy[ o ] ) );
                c[ o ] = o < a ? c[ o ] : ( o == a ? p.id : ( o == a + 1 ? bid : sc[ o ] ) );
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

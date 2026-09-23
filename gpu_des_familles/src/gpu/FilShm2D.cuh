#pragma once

// =====================================================================================
// `filnrm` AVEC LA ROTATION EN MEMOIRE PARTAGEE. La cellule vit en registres ( le test d'elagage,
// la premiere passe ), mais la ROTATION de la coupe -- ce que les registres ne savent pas faire
// sans barillet, une lecture a indice dynamique -- passe par la memoire partagee : huit stores a
// indices fixes, puis les lectures a indices calcules ( les quatre sommets des intersections,
// et `new[ o ] = old[ ( o + j3 - 2 ) mod nb ]` pour la sortie ). Une cinquantaine d'operations
// memoire contre deux cents `select`.
//
// LE LAYOUT est `[ case ][ thread ]` : une ligne par thread mettrait les 32 threads d'un warp
// sur quatre bancs ( conflits x8 ) ; ainsi chaque thread reste sur son banc quel que soit
// l'indice qu'il lit -- pas un conflit, meme pour des indices differents par lane.
//
// L'EMPREINTE est ce qui perd : trois tableaux ( `x`, `y`, `cid` ), 96 octets par thread, 12 Ko
// par bloc, cinq blocs par SM au lieu de huit -- 62 % d'occupation au lieu de 87 -- et le gain
// d'instructions ( 1816 -> 1343 par cellule, -26 % ) s'y perd : 8.0 contre 7.7 ns/germe. Reutiliser
// le meme tampon pour `x` puis `y` ( 8 Ko, la dependance store -> load est dans le thread ) remonte
// l'occupation a 75 % mais met deux allers-retours en serie dans le chemin critique : 8.3. Mesure.
// =====================================================================================

#include "gpu/FilNrm2D.cuh"

namespace sf::gpu {

template<bool POIDS, int R, class TK>
__global__ void __launch_bounds__( 128 ) noyau2_filshm( Arbre<TK,2> ar, double *res, int *deborde, int *liste_deb ) {
    constexpr int SUR = 3, BL = 128;
    static_assert( R >= 4 && R <= 32, "le carre tient dans les registres, le masque dans 32 bits" );
    __shared__ TK  shx[ R ][ BL ], shy[ R ][ BL ];
    __shared__ int shc[ R ][ BL ];
    const int t = threadIdx.x;
    const int k = blockIdx.x * blockDim.x + t;
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

            unsigned m = 0;
#pragma unroll
            for ( int i = 0; i < R; ++i ) {
                if ( i >= SUR && i >= nb ) break;
                m |= unsigned( p.dx * x[ i ] + p.dy * y[ i ] - p.off > TK( 0 ) ) << i;
            }
            if ( PROBABLE( ! m ) )
                continue;
            const unsigned valid = nb >= 32 ? 0xffffffffu : ( 1u << nb ) - 1;
            if ( IMPROBABLE( m == valid ) ) { nb = 0; goto fin; }

            const unsigned prev = ( ( m << 1 ) | ( m >> ( nb - 1 ) ) ) & valid;
            const unsigned next = ( ( m >> 1 ) | ( m << ( nb - 1 ) ) ) & valid;
            const int i1 = __ffs( m & ~prev ) - 1;
            const int j2 = __ffs( m & ~next ) - 1;
            const int j0 = i1 ? i1 - 1 : nb - 1;
            const int j3 = j2 + 1 < nb ? j2 + 1 : 0;
            const int nb_out = __popc( m );
            const int nb_in = nb - nb_out;
            const int nn = nb_in + 2;
            if ( IMPROBABLE( nn > R ) ) { nb = -1; goto fin; }   // pour la seconde passe

            // ---- LA CELLULE EN MEMOIRE PARTAGEE, a indices fixes, et les lectures a indices
            // calcules : les quatre sommets des intersections, puis la rotation `( o - 2 + j3 ) mod nb`
            TK nx[ R ], ny[ R ];
            int nc[ R ];
#pragma unroll
            for ( int i = 0; i < R; ++i ) { if ( i >= SUR && i >= nb ) break; shx[ i ][ t ] = x[ i ]; shy[ i ][ t ] = y[ i ]; shc[ i ][ t ] = c[ i ]; }
            const TK  x0v = shx[ j0 ][ t ], x1 = shx[ i1 ][ t ], x2 = shx[ j2 ][ t ], x3 = shx[ j3 ][ t ];
            const TK  y0v = shy[ j0 ][ t ], y1 = shy[ i1 ][ t ], y2 = shy[ j2 ][ t ], y3 = shy[ j3 ][ t ];
            const int bid = shc[ j2 ][ t ];
#pragma unroll
            for ( int o = 2; o < R; ++o ) {
                if ( o >= SUR && o >= nn ) break;
                int idx = o - 2 + j3; idx = idx >= nb ? idx - nb : idx;
                nx[ o ] = shx[ idx ][ t ]; ny[ o ] = shy[ idx ][ t ]; nc[ o ] = shc[ idx ][ t ];
            }
            const TK  s0 = p.dx * x0v + p.dy * y0v - p.off, s1 = p.dx * x1 + p.dy * y1 - p.off;
            const TK  s2 = p.dx * x2 + p.dy * y2 - p.off,   s3 = p.dx * x3 + p.dy * y3 - p.off;
            const TK  ta = s0 / ( s0 - s1 ), tb = s3 / ( s3 - s2 );
            const TK  pax = x0v + ( x1 - x0v ) * ta, pay = y0v + ( y1 - y0v ) * ta;
            const TK  pbx = x3 + ( x2 - x3 ) * tb,   pby = y3 + ( y2 - y3 ) * tb;

            // ---- LA SORTIE `[ A, B, v_j3, ..., v_j0 ]`
            x[ 0 ] = pax; y[ 0 ] = pay; c[ 0 ] = p.id;
            x[ 1 ] = pbx; y[ 1 ] = pby; c[ 1 ] = bid;
#pragma unroll
            for ( int o = 2; o < R; ++o ) {
                if ( o >= SUR && o >= nn ) break;
                x[ o ] = nx[ o ]; y[ o ] = ny[ o ]; c[ o ] = nc[ o ];
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

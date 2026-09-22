#pragma once

// =====================================================================================
// `filbrk` AVEC LA CELLULE NORMALISEE AVANT LA COUPE. L'idee : mettre les sommets dans un ordre
// canonique AVANT de calculer les intersections, pour que tout se lise a des positions FIXES et
// que rien ne soit a recomposer apres. Plus une lecture a indice dynamique ( `selR` ) dans la coupe.
//
// La sortie est toujours `[ A, B, v_j3, ..., v_j0 ]` -- les gardes dans l'ordre cyclique depuis
// `j3`, A et B en 0 et 1. Les gardes sont amenes la par une ROTATION MODULO `nb` de `j3 - 2`,
// faite en deux decalages en barillet ( a gauche de `j3 - 2` pour ce qui vient de `[ j3, nb )`,
// a droite de `nb - j3 + 2` pour ce qui vient de `[ 0, i1 )` ) et un `select` par case ; quand
// la plage exterieure boucle la seconde moitie est vide, le `select` ne choisit rien.
// Les quatre sommets des intersections :
//   * `v_j2, v_j3` sont en 1 et 2 de la moitie gauche ( `v_j2` est LE DERNIER sommet si
//     `j3 == 0` : il vit dans un registre a part, tenu a jour a chaque coupe -- c'est `v_j0` ) ;
//   * `v_j0, v_i1` sont adjacents : un decalage a droite de `R - 1 - i1` les met en `R - 2` et
//     `R - 1` ( `v_j0` est le dernier sommet si `i1 == 0` ).
// `s` n'est pas decale : on le recalcule pour ces quatre sommets, un `fma` chacun.
//
// Le compte : trois barillets ( `x, y, cid` a gauche et a droite, `x, y` pour `j0 / i1` ) et un
// `select` par case, contre un barillet, douze `selR` et une recomposition dans `filrot`.
// =====================================================================================

#include "gpu/FilRot2D.cuh"

namespace sf::gpu {

/// `a` decale a DROITE de `e >= 0` en barillet : `r[ o ] = a[ o - e ]`, garbage en dessous.
template<class T, int R>
__device__ __forceinline__ void barillet_d( T ( &a )[ R ], int e ) {
#pragma unroll
    for ( int b = 1; b < R; b *= 2 ) {
        const bool on = e & b;
#pragma unroll
        for ( int o = R - 1; o >= 0; --o ) a[ o ] = on && o - b >= 0 ? a[ o - b ] : a[ o ];
    }
}

template<bool POIDS, int R, class TK>
__global__ void __launch_bounds__( 128 ) noyau2_filnrm( Arbre<TK,2> ar, double *res, int *deborde, int *liste_deb ) {
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
    TK  xl = 0, yl = 1; int cl = -4;                    // LE DERNIER SOMMET, `v_nb-1`

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
            // pas de test « c'est mon germe » : son plan a `d = 0` et `off = 0` EXACTEMENT, donc
            // `s = 0` partout, jamais `> 0` -- le test coutait une lecture par germe ( 7 % )
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
            const int i1 = __ffs( m & ~prev ) - 1;      // l'ENTREE : le premier dehors
            const int j2 = __ffs( m & ~next ) - 1;      // le dernier dehors
            const int j3 = j2 + 1 < nb ? j2 + 1 : 0;
            const int nb_out = __popc( m );
            const int nb_in = nb - nb_out;
            const int nn = nb_in + 2;
            if ( IMPROBABLE( nn > R ) ) { nb = -1; goto fin; }   // pour la seconde passe

            // ---- LA NORMALISATION : la rotation modulo `nb` de `j3 - 2`, en deux moities
            TK  gx[ R ], gy[ R ]; int gc[ R ];           // ce qui vient de `[ j3, nb )`, a gauche de `j3 - 2`
            TK  dx[ R ], dy[ R ]; int dc[ R ];           // ce qui vient de `[ 0, i1 )`, a droite de `nb - j3 + 2`
#pragma unroll
            for ( int o = 0; o < R; ++o ) { gx[ o ] = x[ o ]; gy[ o ] = y[ o ]; gc[ o ] = c[ o ]; dx[ o ] = x[ o ]; dy[ o ] = y[ o ]; dc[ o ] = c[ o ]; }
            const int d1 = j3 - 2;                       // dans `[ -2, R - 3 ]`
            if ( d1 < 0 ) { barillet_d( gx, -d1 ); barillet_d( gy, -d1 ); barillet_d( gc, -d1 ); }
            else          { barillet( gx, d1 );    barillet( gy, d1 );    barillet( gc, d1 ); }
            const int e = nb - j3 + 2;                   // la frontiere : `[ 2, e )` a gauche, `[ e, nn )` a droite
            const int ee = e < R ? e : R - 1;
            barillet_d( dx, ee ); barillet_d( dy, ee ); barillet_d( dc, ee );

            // ---- LES QUATRE SOMMETS DES INTERSECTIONS, a des positions fixes
            const TK  x3 = gx[ 2 ], y3 = gy[ 2 ];                                   // `v_j3`
            const TK  x2 = j3 ? gx[ 1 ] : xl, y2 = j3 ? gy[ 1 ] : yl;               // `v_j2`
            const int bid = j3 ? gc[ 1 ] : cl;
            TK ax[ R ], ay[ R ];                                                     // `v_j0, v_i1` en `R - 2, R - 1`
#pragma unroll
            for ( int o = 0; o < R; ++o ) { ax[ o ] = x[ o ]; ay[ o ] = y[ o ]; }
            barillet_d( ax, R - 1 - i1 ); barillet_d( ay, R - 1 - i1 );
            const TK  x1 = ax[ R - 1 ], y1 = ay[ R - 1 ];                           // `v_i1`
            const TK  x0v = i1 ? ax[ R - 2 ] : xl, y0v = i1 ? ay[ R - 2 ] : yl;     // `v_j0`
            const TK  s0 = p.dx * x0v + p.dy * y0v - p.off, s1 = p.dx * x1 + p.dy * y1 - p.off;
            const TK  s2 = p.dx * x2 + p.dy * y2 - p.off,   s3 = p.dx * x3 + p.dy * y3 - p.off;
            const TK  ta = s0 / ( s0 - s1 ), tb = s3 / ( s3 - s2 );
            const TK  pax = x0v + ( x1 - x0v ) * ta, pay = y0v + ( y1 - y0v ) * ta;
            const TK  pbx = x3 + ( x2 - x3 ) * tb,   pby = y3 + ( y2 - y3 ) * tb;

            // ---- LA SORTIE : A, B, puis les deux moities ; rien a recomposer
            x[ 0 ] = pax; y[ 0 ] = pay; c[ 0 ] = p.id;
            x[ 1 ] = pbx; y[ 1 ] = pby; c[ 1 ] = bid;
#pragma unroll
            for ( int o = 2; o < R; ++o ) {
                if ( o >= SUR && o >= nn ) break;
                x[ o ] = o < e ? gx[ o ] : dx[ o ];
                y[ o ] = o < e ? gy[ o ] : dy[ o ];
                c[ o ] = o < e ? gc[ o ] : dc[ o ];
            }
            xl = x0v; yl = y0v; cl = i1 ? selR( c, nn - 1 ) : cl;   // le dernier est `v_j0` ( son cid ne change pas )
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

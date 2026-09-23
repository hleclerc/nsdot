#pragma once

// =====================================================================================
// `filnrm` / `filrot` AVEC DES MASQUES DE ROLE ET UN SEUL BARILLET ( idee de H. L. ).
// Les registres restent TRIES -- les sommets sont toujours en `0 .. nb - 1` -- et deux choses
// changent par rapport a `filrot` :
//
//   1. LES QUATRE SOMMETS DE LA FRONTIERE sortent de QUATRE MASQUES DE ROLE de huit bits,
//      fabriques en quatre instructions a partir de `m`, `prev` et `next`. Chacun n'a qu'UN SEUL
//      BIT, puisque la plage exterieure est un arc cyclique contigu :
//          r0 = ~m & next    ( dedans, le suivant dehors    -> `v_j0` )
//          r1 =  m & ~prev   ( dehors, le precedent dedans  -> `v_i1`, l'entree )
//          r2 =  m & ~next   ( dehors, le suivant dedans    -> `v_j2` )
//          r3 = ~m & prev    ( dedans, le precedent dehors  -> `v_j3` )
//      Un `__ffs` donne l'indice, et « la plage boucle » se lit `r1 > r2` ( les deux masques sont
//      one-hot : les comparer, c'est comparer `i1` et `j2` ). `filrot` faisait le meme travail en
//      deux `__ffs` puis quatre corrections cycliques ( `i1 ? i1 - 1 : nb - 1` ... ).
//
//      ( La variante ou les quatre masques servaient DIRECTEMENT a cueillir `x`, `y` et `c` par
//        des `et` / `ou` pleine largeur a ete ecrite et mesuree : +11 % d'instructions. Le partage
//        entre les trois tableaux existait deja -- ptxas met un seul `ISETP.EQ` par case en
//        facteur de trois `SEL` -- et un masque plein coute deux a trois instructions la ou un
//        predicat en coute une. Voir README § 4. )
//
//   2. LE REMONTAGE TIENT EN UN SEUL BARILLET. La sortie est
//          new[ o ] = o < a ? old[ o ] : o == a ? A : o == a + 1 ? B : old[ o + d ]
//      avec `( a, d ) = ( 0, j3 - 2 )` si la plage boucle, `( i1, nb_out - 2 )` sinon. Le `d`
//      peut valoir -1 ( une seule coupe sortante : le cas le plus frequent ), et `filrot` payait
//      ce cas par une copie decalee a droite -- un `select` de plus par case et par tableau.
//      Ici on decale A PRIORI d'une case, ce qui est GRATUIT ( un renommage de registres a la
//      compilation ) : on travaille sur `u[ k ] = old[ k - 1 ]`, neuf cases, et le barillet part
//      de `e = d + 1 >= 0`. Mesure : -8 % d'instructions et -8 % de temps contre `filrot8`.
//
// Trois tableaux temporaires de neuf cases, contre les huit de huit de `filnrm` : 96 registres en
// `double` et 58 en `float`, contre 128 et 74 -- LE PLUS PETIT NOYAU DE LA FAMILLE, a +5 % de
// temps. `BSM` force l'occupation ( `__launch_bounds__( 128, BSM )` : ptxas rabote les registres
// pour loger `BSM` blocs par SM ) ; `BSM = 1` ne contraint rien.
// =====================================================================================

#include "gpu/FilNrm2D.cuh"

namespace sf::gpu {

template<bool POIDS, int BSM, class TK>
__global__ void __launch_bounds__( 128, BSM ) noyau2_filmsk( Arbre<TK,2> ar, double *res, int *deborde, int *liste_deb ) {
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

            // ---- LA PREMIERE PASSE : `m`, le masque des sommets dehors
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

            // ---- LES QUATRE MASQUES DE ROLE, un seul bit chacun
            const unsigned prev = ( ( m << 1 ) | ( m >> ( nb - 1 ) ) ) & valid;
            const unsigned next = ( ( m >> 1 ) | ( m << ( nb - 1 ) ) ) & valid;
            const unsigned r0 = ~m & next & valid;      // `v_j0` : dedans, le suivant dehors
            const unsigned r1 =  m & ~prev;             // `v_i1` : dehors, le precedent dedans
            const unsigned r2 =  m & ~next;             // `v_j2` : dehors, le suivant dedans
            const unsigned r3 = ~m & prev & valid;      // `v_j3` : dedans, le precedent dehors
            const int j0 = __ffs( int( r0 ) ) - 1, i1 = __ffs( int( r1 ) ) - 1;
            const int j2 = __ffs( int( r2 ) ) - 1, j3 = __ffs( int( r3 ) ) - 1;
            const int nb_out = __popc( m );
            const int nn = nb - nb_out + 2;
            if ( IMPROBABLE( nn > R ) ) { nb = -1; goto fin; }   // pour la seconde passe

            // ---- LES DEUX POINTS CREES ; `s` recalcule pour les quatre ( un `fma` chacun )
            const TK x0v = selR( x, j0 ), y0v = selR( y, j0 ), x1v = selR( x, i1 ), y1v = selR( y, i1 );
            const TK x2v = selR( x, j2 ), y2v = selR( y, j2 ), x3v = selR( x, j3 ), y3v = selR( y, j3 );
            const int bid = selR( c, j2 );
            const TK s0 = p.dx * x0v + p.dy * y0v - p.off, s1 = p.dx * x1v + p.dy * y1v - p.off;
            const TK s2 = p.dx * x2v + p.dy * y2v - p.off, s3 = p.dx * x3v + p.dy * y3v - p.off;
            const TK ta = s0 / ( s0 - s1 ), tb = s3 / ( s3 - s2 );
            const TK pax = x0v + ( x1v - x0v ) * ta, pay = y0v + ( y1v - y0v ) * ta;
            const TK pbx = x3v + ( x2v - x3v ) * tb, pby = y3v + ( y2v - y3v ) * tb;

            // ---- LE REMONTAGE : `u[ k ] = old[ k - 1 ]` ( gratuit ), puis UN barillet de `e`
            const bool boucle = r1 > r2;                        // `i1 > j2` : la plage dehors boucle
            const int  a = boucle ? 0 : i1;
            const int  e = boucle ? j3 - 1 : nb_out - 1;        // `= d + 1 >= 0`
            TK  ux[ R + 1 ], uy[ R + 1 ];
            int uc[ R + 1 ];
            ux[ 0 ] = x[ 0 ]; uy[ 0 ] = y[ 0 ]; uc[ 0 ] = c[ 0 ];   // jamais lu : `o + e >= 1`
#pragma unroll
            for ( int o = 1; o < R + 1; ++o ) { ux[ o ] = x[ o - 1 ]; uy[ o ] = y[ o - 1 ]; uc[ o ] = c[ o - 1 ]; }
            // trois etages, la meme condition pour les trois tableaux ; les cases au-dela de
            // `o + e = R` ne servent jamais ( `o < nn` et `nn - 1 + e <= R` ), d'ou les bornes
#pragma unroll
            for ( int b = 1; b < R + 1; b *= 2 ) {
                const bool on = e & b;
#pragma unroll
                for ( int o = 0; o + b < R + 1; ++o ) { ux[ o ] = on ? ux[ o + b ] : ux[ o ]; uy[ o ] = on ? uy[ o + b ] : uy[ o ]; uc[ o ] = on ? uc[ o + b ] : uc[ o ]; }
            }
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

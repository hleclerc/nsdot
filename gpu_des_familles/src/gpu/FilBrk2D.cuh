#pragma once

// =====================================================================================
// UNE CELLULE PAR THREAD, 2D, TOUT EN REGISTRES, TOUT DEROULE, AVEC DES SORTIES. `R` sommets
// ( `x`, `y`, `cid` ) en registres, chaque boucle deroulee SORT des que `i >= nb` : une cellule a
// six sommets n'execute pas le code des cases 6 .. R-1. Des branches, donc de la divergence entre
// les threads d'un warp -- mais pas de travail sur des cases vides, et le warp n'execute que
// jusqu'au plus grand `nb` de ses voies.
//
// Une cellule qui depasserait `R` sommets n'est pas geree ici : son rang est pousse dans une liste
// ( atomique ) et une SECONDE PASSE, un autre noyau a plus de largeur ( `filmix` a 64 sommets ),
// la refait. Les grandes cellules sont rares ( 0.1 % des etats au-dela de 8 en uniforme ), le
// noyau commun ne porte rien pour elles.
//
// `OPT`, les micro-optimisations : ( 1 ) une cellule non vide a TROIS sommets au moins, et une
// coupe en laisse `nb_in + 2 >= 3` : les trois premieres cases n'ont pas a tester `i < nb` ;
// ( 2 ) `__builtin_expect` d'apres les compteurs -- 58 % des plans testes ne coupent pas, une
// cellule vide, un debordement ou son propre germe sont rares.
// =====================================================================================

#define PROBABLE( c )   ( __builtin_expect( !! ( c ), 1 ) )
#define IMPROBABLE( c ) ( __builtin_expect( !! ( c ), 0 ) )

#include "gpu/FilMix2D.cuh"

namespace sf::gpu {

template<bool POIDS, int R, bool OPT, class TK>
__global__ void __launch_bounds__( 128 ) noyau2_filbrk( Arbre<TK,2> ar, double *res, int *deborde, int *liste_deb ) {
    constexpr int SUR = OPT ? 3 : 0;                     // les cases toujours occupees
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
            if ( OPT ? IMPROBABLE( ar.ids[ q ] == i0 ) : ar.ids[ q ] == i0 ) continue;
            const Plan2<TK> p = bissect2<POIDS>( ar, q, p0[ 0 ], p0[ 1 ], w0 );

            TK s[ R ];
            unsigned m = 0;
#pragma unroll
            for ( int i = 0; i < R; ++i ) {
                if ( i >= SUR && i >= nb ) break;
                s[ i ] = p.dx * x[ i ] + p.dy * y[ i ] - p.off;
                m |= unsigned( s[ i ] > TK( 0 ) ) << i;
            }
            if ( OPT ? PROBABLE( ! m ) : ! m )
                continue;
            const unsigned valid = nb >= 32 ? 0xffffffffu : ( 1u << nb ) - 1;
            if ( OPT ? IMPROBABLE( m == valid ) : m == valid ) { nb = 0; goto fin; }

            const unsigned prev = ( ( m << 1 ) | ( m >> ( nb - 1 ) ) ) & valid;
            const unsigned next = ( ( m >> 1 ) | ( m << ( nb - 1 ) ) ) & valid;
            const int i1 = __ffs( m & ~prev ) - 1;
            const int j2 = __ffs( m & ~next ) - 1;
            const int j0 = i1 ? i1 - 1 : nb - 1;
            const int j3 = j2 + 1 < nb ? j2 + 1 : 0;
            const int nb_in = nb - __popc( m );
            const int nn = nb_in + 2;
            if ( OPT ? IMPROBABLE( nn > R ) : nn > R ) { nb = -1; goto fin; }   // pour la seconde passe

            const TK s0 = selR( s, j0 ), s1 = selR( s, i1 ), s2 = selR( s, j2 ), s3 = selR( s, j3 );
            const TK x0v = selR( x, j0 ), y0v = selR( y, j0 ), x1v = selR( x, i1 ), y1v = selR( y, i1 );
            const TK x2v = selR( x, j2 ), y2v = selR( y, j2 ), x3v = selR( x, j3 ), y3v = selR( y, j3 );
            const TK ta = s0 / ( s0 - s1 ), tb = s3 / ( s3 - s2 );
            const TK pax = x0v + ( x1v - x0v ) * ta, pay = y0v + ( y1v - y0v ) * ta;
            const TK pbx = x3v + ( x2v - x3v ) * tb, pby = y3v + ( y2v - y3v ) * tb;
            const int bid = selR( c, j2 );

            TK  nx[ R ], ny[ R ];
            int nc[ R ];
#pragma unroll
            for ( int o = 0; o < R; ++o ) {
                if ( o >= SUR && o >= nn ) break;
                int og = j3 + o; og = og >= nb ? og - nb : og;
                nx[ o ] = o == nb_in ? pax : ( o == nb_in + 1 ? pbx : selR( x, og ) );
                ny[ o ] = o == nb_in ? pay : ( o == nb_in + 1 ? pby : selR( y, og ) );
                nc[ o ] = o == nb_in ? p.id : ( o == nb_in + 1 ? bid : selR( c, og ) );
            }
#pragma unroll
            for ( int o = 0; o < R; ++o ) {
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

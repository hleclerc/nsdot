#pragma once

// =====================================================================================
// LES SOMMETS NE BOUGENT PLUS : L'ORDRE EST DANS UN REGISTRE ( idee de H. L. ).
//
// `filnrm` normalise la cellule a chaque coupe -- trois barillets sur `x`, `y`, `cid`, huit
// tableaux temporaires, 166 registres. Ici les valeurs NE SONT JAMAIS DEPLACEES : le sommet vit
// dans son slot tant qu'il vit, et un registre `O` de 64 bits porte l'ORDRE CYCLIQUE -- son octet
// `i` est le masque ONE-HOT du slot ou se trouve le sommet de position `i`. Un masque `vivant`
// de huit bits dit quels slots sont occupes ; son complement est la liste des slots libres.
//
// CE QUE CA CHANGE, coupe par coupe :
//   * la premiere passe calcule `s` pour LES HUIT SLOTS, occupes ou non ( du gachis, mais pas une
//     branche ) et rend un masque `M` PAR SLOT ;
//   * le masque PAR POSITION se recompose en sept instructions entieres : `t = O & ( M x
//     0x0101010101010101 )` a un octet non nul la ou le sommet est dehors, et « octet non nul ->
//     bit » se fait par trois `or` decales puis une multiplication magique ;
//   * les quatre sommets de la frontiere se lisent par leur slot ( `ffs` de l'octet, puis un
//     `select` ) et leur `s` se RECALCULE ( un fma, moins cher qu'une lecture indexee ) ;
//   * les deux sommets neufs prennent DEUX SLOTS LIBRES ( les sortants viennent de se liberer ) :
//     deux ecritures masquees, rien d'autre ne bouge ;
//   * `O` se met a jour PRESQUE GRATUITEMENT : comme la sortie est normalisee depuis `j3`, les
//     gardes sont contigus dans l'ordre cyclique -- une rotation de `j3` octets, une troncature a
//     `nb_in` octets, et les deux octets neufs a la suite. Six instructions.
//
// Plus un barillet, plus un tableau temporaire.
// =====================================================================================

#include "gpu/FilNrm2D.cuh"

namespace sf::gpu {

/// « quels octets de `t` sont non nuls », en huit bits.
__device__ __forceinline__ unsigned octets_non_nuls( unsigned long long t ) {
    t |= t >> 4; t |= t >> 2; t |= t >> 1;
    t &= 0x0101010101010101ull;
    return unsigned( ( t * 0x0102040810204080ull ) >> 56 );
}

/// le slot du sommet de position `pos` : l'octet `pos` de `O` est un masque one-hot.
__device__ __forceinline__ int slot_de( unsigned long long O, int pos ) {
    return __ffs( int( ( O >> ( 8 * pos ) ) & 0xffu ) ) - 1;
}

template<bool POIDS, class TK>
__global__ void __launch_bounds__( 128 ) noyau2_filord( Arbre<TK,2> ar, double *res, int *deborde, int *liste_deb ) {
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
    int nb = 4;
    unsigned vivant = 0x0fu;
    unsigned long long O = 0x08040201ull;                // positions 0..3 -> slots 0..3

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

            // ---- LA PREMIERE PASSE, par SLOT : les huit, occupes ou non
            unsigned M = 0;
#pragma unroll
            for ( int i = 0; i < R; ++i )
                M |= unsigned( p.dx * x[ i ] + p.dy * y[ i ] - p.off > TK( 0 ) ) << i;
            M &= vivant;
            if ( PROBABLE( ! M ) )
                continue;

            // ---- LE MASQUE PAR POSITION, en sept instructions
            const unsigned m = octets_non_nuls( O & ( 0x0101010101010101ull * ( unsigned long long ) M ) );
            const unsigned valid = ( 1u << nb ) - 1;
            if ( IMPROBABLE( m == valid ) ) { nb = 0; goto fin; }

            const unsigned prev = ( ( m << 1 ) | ( m >> ( nb - 1 ) ) ) & valid;
            const unsigned next = ( ( m >> 1 ) | ( m << ( nb - 1 ) ) ) & valid;
            const int i1 = __ffs( m & ~prev ) - 1;
            const int j2 = __ffs( m & ~next ) - 1;
            const int j0 = i1 ? i1 - 1 : nb - 1;
            const int j3 = j2 + 1 < nb ? j2 + 1 : 0;
            const int nb_in = nb - __popc( m );
            const int nn = nb_in + 2;
            if ( IMPROBABLE( nn > R ) ) { nb = -1; goto fin; }

            // ---- LES QUATRE SOMMETS DE LA FRONTIERE, par leur slot ; leur `s` RECALCULE
            const int t0 = slot_de( O, j0 ), t1 = slot_de( O, i1 ), t2 = slot_de( O, j2 ), t3 = slot_de( O, j3 );
            const TK x0v = selR( x, t0 ), y0v = selR( y, t0 );
            const TK x1 = selR( x, t1 ), y1 = selR( y, t1 );
            const TK x2 = selR( x, t2 ), y2 = selR( y, t2 );
            const TK x3 = selR( x, t3 ), y3 = selR( y, t3 );
            const int bid = selR( c, t2 );
            const TK s0 = p.dx * x0v + p.dy * y0v - p.off, s1 = p.dx * x1 + p.dy * y1 - p.off;
            const TK s2 = p.dx * x2 + p.dy * y2 - p.off,   s3 = p.dx * x3 + p.dy * y3 - p.off;
            const TK ta = s0 / ( s0 - s1 ), tb = s3 / ( s3 - s2 );
            const TK pax = x0v + ( x1 - x0v ) * ta, pay = y0v + ( y1 - y0v ) * ta;
            const TK pbx = x3 + ( x2 - x3 ) * tb,   pby = y3 + ( y2 - y3 ) * tb;

            // ---- DEUX SLOTS LIBRES : les sortants viennent de se liberer
            unsigned libre = ( ~vivant | M ) & 0xffu;
            const int sA = __ffs( int( libre ) ) - 1; libre &= ~( 1u << sA );
            const int sB = __ffs( int( libre ) ) - 1; libre &= ~( 1u << sB );
#pragma unroll
            for ( int i = 0; i < R; ++i ) {
                x[ i ] = i == sA ? pax : ( i == sB ? pbx : x[ i ] );
                y[ i ] = i == sA ? pay : ( i == sB ? pby : y[ i ] );
                c[ i ] = i == sA ? p.id : ( i == sB ? bid : c[ i ] );
            }
            vivant = ( ~libre ) & 0xffu;

            // ---- L'ORDRE : une rotation de `j3` octets, tronquee a `nb_in`, puis les deux neufs
            const int rot = 8 * j3;
            const unsigned long long Or = j3 ? ( ( O >> rot ) | ( O << ( 8 * nb - rot ) ) ) : O;
            const unsigned long long garde = nb_in >= 8 ? ~0ull : ( ( 1ull << ( 8 * nb_in ) ) - 1 );
            O = ( Or & garde )
              | ( ( unsigned long long ) ( 1u << sA ) << ( 8 * nb_in ) )
              | ( ( unsigned long long ) ( 1u << sB ) << ( 8 * ( nb_in + 1 ) ) );
            nb = nn;
        }
    }

fin:
    double a = 0;
    if ( nb > 0 ) {
        int tp = slot_de( O, 0 );
        TK xp = selR( x, tp ), yp = selR( y, tp );
        const TK x00 = xp, y00 = yp;
#pragma unroll
        for ( int i = 1; i < R; ++i ) {
            if ( i >= nb ) break;
            const int tq = slot_de( O, i );
            const TK xq = selR( x, tq ), yq = selR( y, tq );
            a += double( xp ) * double( yq ) - double( xq ) * double( yp );
            xp = xq; yp = yq;
        }
        a += double( xp ) * double( y00 ) - double( x00 ) * double( yp );
    }
    if ( nb < 0 ) liste_deb[ atomicAdd( deborde, 1 ) ] = k;
    res[ i0 ] = 0.5 * fabs( a );
}

} // namespace sf::gpu

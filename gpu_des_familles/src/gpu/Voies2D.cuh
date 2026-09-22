#pragma once

// =====================================================================================
// LA CELLULE SUR HUIT VOIES, 2D. Le pendant CUDA du noyau a registres du CPU ( `cell/Noyau2D.h` ) :
// la voie `l` porte le sommet `l` ( `vx`, `vy`, `cid` ), le nombre de sommets `nb` est un scalaire
// UNIFORME dans le groupe, la coupe est le meme calcul sans boucle -- un masque de signe par
// `ballot`, ses deux bouts par `ffs`, deux intersections en une division, deux permutations par
// `shfl` -- et le test d'elagage coute UNE operation par voie et un `ballot`.
//
// Un warp porte QUATRE cellules ( `shfl` et `ballot` de largeur 8 ). Les quatre parcourent des
// arbres differents : le warp execute l'union de leurs chemins, et pendant qu'un groupe travaille
// les trois autres attendent -- c'est le prix de ce mappage. En 2D une cellule finie a six
// sommets en moyenne et 98 % des etats intermediaires en ont huit ou moins : sur huit voies,
// deux dorment ; sur trente-deux, vingt-six dormiraient. D'ou huit, et pas le warp.
//
// LE DEBORDEMENT EST UNE EXCURSION, comme au CPU : au-dela de huit sommets la cellule se pose dans
// des tableaux locaux de la voie 0, qui coupe en scalaire ( `coupe2` ) pendant que les sept
// autres attendent, et DES QU'ELLE REDESCEND a huit la cellule revient sur les voies.
//
// La pile du parcours est en memoire PARTAGEE, une par groupe, ecrite par la voie 0 et lue par
// toutes ( un `syncwarp` du groupe entre les deux ) ; les scalaires du parcours sont repliques
// dans les registres de chaque voie, ce qui ne coute rien.
// =====================================================================================

#include "gpu/Fil2D.cuh"

namespace sf::gpu {

constexpr int BLOC2 = 128;              ///< threads par bloc

/// `V` voies par cellule : 8 ( quatre cellules par warp ), 16 ou 32 ( une seule ).
template<bool POIDS, int MaxNb, int V, class TK>
__global__ void __launch_bounds__( BLOC2 ) noyau2_voies( Arbre<TK,2> ar, double *res, int *deborde ) {
    static_assert( V == 8 || V == 16 || V == 32, "un groupe est une fraction du warp" );
    constexpr int VOIES2 = V;
    const int lane = threadIdx.x & 31;
    const int l    = lane & ( V - 1 );                   // ma voie dans le groupe
    const int grp  = lane / V;                           // mon groupe dans le warp
    const unsigned gm = ( V == 32 ? 0xffffffffu : ( 1u << V ) - 1 ) << ( V * grp );   // le masque de mon groupe
    const int k = ( blockIdx.x * blockDim.x + threadIdx.x ) / V;

    __shared__ int piles[ BLOC2 / V ][ PILE ];
    int *pile = piles[ threadIdx.x / V ];
    if ( k >= ar.n ) return;                             // uniforme dans le groupe

    const TK p0[ 2 ] = { ar.c[ 0 ][ k ], ar.c[ 1 ][ k ] };
    const TK w0 = POIDS ? ar.w[ k ] : TK( 0 );
    const int i0 = ar.ids[ k ];

    // le carre unite sur les voies 0..3, cotes `-1 .. -4`
    TK  vx  = TK( l == 1 || l == 2 ), vy = TK( l == 2 || l == 3 );
    int cid = l < 4 ? -1 - l : 0;                       // ( `valid` masque les voies au-dela de `nb` )
    int nb  = 4;

    // l'excursion : la voie 0 seule y touche
    TK  lvx[ MaxNb ], lvy[ MaxNb ], ls[ MaxNb ];
    int lcid[ MaxNb ];
    bool large = false;

    int haut = 1;
    if ( l == 0 ) pile[ 0 ] = 0;
    __syncwarp( gm );

    while ( haut ) {
        const int h = pile[ --haut ];
        const Noeud<TK,2> nd = ar.nodes[ h ];

        // ---- LE TEST D'ELAGAGE : une operation par voie, ou la boucle de la voie 0
        bool peut;
        if ( ! large ) {
            const TK v[ 2 ] = { vx, vy };
            const bool mienne = l < nb && bilan_sommet<POIDS>( nd, v, p0, w0 ) <= 0;
            peut = ( __ballot_sync( gm, mienne ) & gm ) != 0;
        } else {
            int r = l == 0 ? int( peut_couper2<POIDS>( nd, p0, w0, nb, lvx, lvy ) ) : 0;
            peut = __shfl_sync( gm, r, 0, VOIES2 ) != 0;
        }
        if ( ! peut )
            continue;

        if ( nd.right >= 0 ) {                           // un noeud : ses deux fils, le proche dessus
            const int g = h + 1, dr = nd.right;
            const bool gp = proximite( ar.nodes[ g ], p0 ) <= proximite( ar.nodes[ dr ], p0 );
            if ( l == 0 ) { pile[ haut ] = gp ? dr : g; pile[ haut + 1 ] = gp ? g : dr; }
            haut += 2;
            __syncwarp( gm );
            continue;
        }

        // ---- UNE FEUILLE : ses germes, dans l'ordre
        for ( int q = nd.beg; q < nd.end; ++q ) {
            if ( ar.ids[ q ] == i0 ) continue;
            const Plan2<TK> p = bissect2<POIDS>( ar, q, p0[ 0 ], p0[ 1 ], w0 );

            if ( ! large ) {
                // LE TEST, QUI EST DEJA LA COUPE : `s > 0` dehors
                const TK s = p.dx * vx + p.dy * vy - p.off;
                const unsigned valid = ( 1u << nb ) - 1;
                const unsigned m = ( __ballot_sync( gm, l < nb && s > 0 ) >> ( V * grp ) ) & valid;
                if ( ! m )
                    continue;
                if ( m == valid ) { nb = 0; goto fin; }

                // LES DEUX BOUTS DE LA PLAGE EXTERIEURE
                const unsigned prev = ( ( m << 1 ) | ( m >> ( nb - 1 ) ) ) & valid;
                const unsigned next = ( ( m >> 1 ) | ( m << ( nb - 1 ) ) ) & valid;
                const int i1 = __ffs( m & ~prev ) - 1;   // premier DEHORS
                const int j2 = __ffs( m & ~next ) - 1;   // dernier DEHORS
                const int j0 = i1 ? i1 - 1 : nb - 1;     // dernier DEDANS avant
                const int j3 = j2 + 1 < nb ? j2 + 1 : 0; // premier DEDANS apres
                const int nb_in = nb - __popc( m );
                const int nn = nb_in + 2;

                // LES DEUX INTERSECTIONS : A sur la voie 0 ( ancre `j0`, l'autre `i1` ), B sur la
                // voie 1 ( ancre `j3`, l'autre `j2` ), ancrees sur le sommet dedans
                const int anc = l == 1 ? j3 : j0, oth = l == 1 ? j2 : i1;
                const TK vax = __shfl_sync( gm, vx, anc, VOIES2 ), vox = __shfl_sync( gm, vx, oth, VOIES2 );
                const TK vay = __shfl_sync( gm, vy, anc, VOIES2 ), voy = __shfl_sync( gm, vy, oth, VOIES2 );
                const TK sa  = __shfl_sync( gm, s,  anc, VOIES2 ), so  = __shfl_sync( gm, s,  oth, VOIES2 );
                const TK t   = sa / ( sa - so );
                const TK pcx = vax + ( vox - vax ) * t;
                const TK pcy = vay + ( voy - vay ) * t;

                if ( nn <= VOIES2 ) {
                    // LE REMONTAGE : `[ v_j3, ..., v_j0, A, B ]`, `A` sur la coupe neuve, `B` sur ce
                    // qui reste de la coupe `j2`
                    int og = j3 + l; og = og >= nb ? og - nb : og;
                    TK  nvx = __shfl_sync( gm, vx,  og & ( V - 1 ), VOIES2 );
                    TK  nvy = __shfl_sync( gm, vy,  og & ( V - 1 ), VOIES2 );
                    int nid = __shfl_sync( gm, cid, og & ( V - 1 ), VOIES2 );
                    const TK  ax  = __shfl_sync( gm, pcx, 0,  VOIES2 ), ay = __shfl_sync( gm, pcy, 0, VOIES2 );
                    const TK  bx  = __shfl_sync( gm, pcx, 1,  VOIES2 ), by = __shfl_sync( gm, pcy, 1, VOIES2 );
                    const int bid = __shfl_sync( gm, cid, j2, VOIES2 );
                    if ( l == nb_in )          { nvx = ax; nvy = ay; nid = p.id; }
                    else if ( l == nb_in + 1 ) { nvx = bx; nvy = by; nid = bid;  }
                    vx = nvx; vy = nvy; cid = nid; nb = nn;
                    continue;
                }

                // L'EXCURSION PART DE LA : les huit sommets sur la voie 0, et cette coupe rejouee
#pragma unroll
                for ( int i = 0; i < VOIES2; ++i ) {
                    const TK  x = __shfl_sync( gm, vx,  i, VOIES2 ), y = __shfl_sync( gm, vy, i, VOIES2 );
                    const int c = __shfl_sync( gm, cid, i, VOIES2 );
                    if ( l == 0 ) { lvx[ i ] = x; lvy[ i ] = y; lcid[ i ] = c; }
                }
                large = true;
            }

            // EN EXCURSION : la voie 0 coupe en scalaire, `nb` est rediffuse
            int r = 0;
            if ( l == 0 ) r = coupe2<MaxNb>( lvx, lvy, lcid, nb, p, ls );
            nb = __shfl_sync( gm, r, 0, VOIES2 );
            if ( nb <= 0 )
                goto fin;
            if ( nb <= VOIES2 ) {                        // la cellule REVIENT sur les voies
#pragma unroll
                for ( int i = 0; i < VOIES2; ++i ) {
                    const TK  x = __shfl_sync( gm, l == 0 && i < nb ? lvx[ i ]  : TK( 0 ), 0, VOIES2 );
                    const TK  y = __shfl_sync( gm, l == 0 && i < nb ? lvy[ i ]  : TK( 0 ), 0, VOIES2 );
                    const int c = __shfl_sync( gm, l == 0 && i < nb ? lcid[ i ] : 0,       0, VOIES2 );
                    if ( l == i ) { vx = x; vy = y; cid = c; }
                }
                large = false;
            }
        }
    }

fin:
    double area = 0;
    if ( nb > 0 ) {
        if ( ! large ) {                                 // le lacet, une arete par voie
            const int j = l + 1 < nb ? l + 1 : 0;
            const TK xj = __shfl_sync( gm, vx, j, VOIES2 ), yj = __shfl_sync( gm, vy, j, VOIES2 );
            double a = l < nb ? double( vx ) * double( yj ) - double( xj ) * double( vy ) : 0.0;
#pragma unroll
            for ( int d = V / 2; d >= 1; d /= 2 )
                a += __shfl_xor_sync( gm, a, d, VOIES2 );
            area = 0.5 * fabs( a );
        } else if ( l == 0 ) {
            double a = 0;
            for ( int i = 0; i < nb; ++i ) {
                const int j = i + 1 < nb ? i + 1 : 0;
                a += double( lvx[ i ] ) * double( lvy[ j ] ) - double( lvx[ j ] ) * double( lvy[ i ] );
            }
            area = 0.5 * fabs( a );
        }
    }
    if ( l == 0 ) {
        if ( nb < 0 ) atomicAdd( deborde, 1 );
        res[ i0 ] = area;
    }
}

} // namespace sf::gpu

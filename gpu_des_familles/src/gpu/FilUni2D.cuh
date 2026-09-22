#pragma once

// =====================================================================================
// `filnrm` EN UNE SEULE BOUCLE, AVEC DES LANES PERSISTANTES. Deux sources de divergence dans
// `filnrm` ( 6.8 threads actifs par warp sur 32, mesure ) :
//   * les PHASES : les deux boucles imbriquees ( les noeuds, puis les germes d'une feuille ) mettent
//     les lanes d'un warp dans des chemins differents -- l'un depile, l'autre teste un germe, un
//     troisieme coupe -- et le warp les serialise ;
//   * la QUEUE : les 32 cellules d'un warp finissent a des moments differents, et le warp vit
//     jusqu'a la plus lente ( le max de 32 tirages vaut ~1.6 fois la moyenne ).
// Ici UNE boucle : chaque iteration fait UN pas -- le germe suivant de la feuille ouverte s'il y
// en a un, sinon le noeud suivant de la pile, sinon la cellule est finie -- et un lane qui finit
// PREND UNE AUTRE CELLULE ( un compteur global, `atomicAdd` ) au lieu d'attendre le warp. L'etat
// du parcours ( la pile, le curseur de feuille ) vit avec la cellule, dans le lane. Le nombre de
// threads est celui qui remplit la carte, pas `n`.
//
// La coupe est celle de `filnrm` ( normalisation en barillet, positions fixes ), mot pour mot.
// =====================================================================================

#include "gpu/FilNrm2D.cuh"

namespace sf::gpu {

/// `PERS` : les lanes persistantes ( sinon un thread par cellule, la boucle unique seule ).
template<bool POIDS, int R, bool PERS, class TK>
__global__ void __launch_bounds__( 128 ) noyau2_filuni( Arbre<TK,2> ar, double *res, int *deborde, int *liste_deb, int *compteur ) {
    constexpr int SUR = 3;
    static_assert( R >= 4 && R <= 32, "le carre tient dans les registres, le masque dans 32 bits" );

    // ---- l'etat d'une cellule, dans le lane
    TK  x[ R ], y[ R ];
    int c[ R ];
    int nb = 0;                                          // 0 : pas de cellule en cours
    TK  xl = 0, yl = 0; int cl = 0;
    TK  p0[ 2 ] = { 0, 0 }, w0 = 0;
    int i0 = 0, k = -1;
    int pile[ PILE ];
    int haut = 0;
    int q = 0, fin = 0;                                  // la feuille ouverte

    for ( ;; ) {
        // ---- PAS DE CELLULE : en prendre une, ou s'arreter
        if ( nb <= 0 ) {
            if ( nb < 0 ) { liste_deb[ atomicAdd( deborde, 1 ) ] = k; res[ i0 ] = 0; }   // debordee : la seconde passe
            else if ( haut < 0 ) res[ i0 ] = 0;                                          // vide
            if ( PERS ) k = atomicAdd( compteur, 1 );
            else { if ( haut != 0 || nb != 0 || k >= 0 ) break; k = blockIdx.x * blockDim.x + threadIdx.x; }   // une seule cellule : `k < 0` au depart
            if ( k >= ar.n ) break;
            p0[ 0 ] = ar.c[ 0 ][ k ]; p0[ 1 ] = ar.c[ 1 ][ k ];
            w0 = POIDS ? ar.w[ k ] : TK( 0 );
            i0 = ar.ids[ k ];
#pragma unroll
            for ( int i = 0; i < R; ++i ) { x[ i ] = TK( i == 1 || i == 2 ); y[ i ] = TK( i == 2 || i == 3 ); c[ i ] = i < 4 ? -1 - i : 0; }
            nb = 4; xl = 0; yl = 1; cl = -4;
            haut = 0; pile[ haut++ ] = 0;
            q = fin = 0;
        }

        // ---- UN GERME DE LA FEUILLE OUVERTE
        if ( q < fin ) {
            const Plan2<TK> p = bissect2<POIDS>( ar, q++, p0[ 0 ], p0[ 1 ], w0 );
            unsigned m = 0;
#pragma unroll
            for ( int i = 0; i < R; ++i ) {
                if ( i >= SUR && i >= nb ) break;
                m |= unsigned( p.dx * x[ i ] + p.dy * y[ i ] - p.off > TK( 0 ) ) << i;
            }
            if ( PROBABLE( ! m ) )
                continue;
            const unsigned valid = nb >= 32 ? 0xffffffffu : ( 1u << nb ) - 1;
            if ( IMPROBABLE( m == valid ) ) { nb = 0; haut = -1; continue; }   // vide : finie

            const unsigned prev = ( ( m << 1 ) | ( m >> ( nb - 1 ) ) ) & valid;
            const unsigned next = ( ( m >> 1 ) | ( m << ( nb - 1 ) ) ) & valid;
            const int i1 = __ffs( m & ~prev ) - 1;
            const int j2 = __ffs( m & ~next ) - 1;
            const int j3 = j2 + 1 < nb ? j2 + 1 : 0;
            const int nb_out = __popc( m );
            const int nb_in = nb - nb_out;
            const int nn = nb_in + 2;
            if ( IMPROBABLE( nn > R ) ) { nb = -1; continue; }   // pour la seconde passe

            TK  gx[ R ], gy[ R ]; int gc[ R ];
            TK  dx[ R ], dy[ R ]; int dc[ R ];
#pragma unroll
            for ( int o = 0; o < R; ++o ) { gx[ o ] = x[ o ]; gy[ o ] = y[ o ]; gc[ o ] = c[ o ]; dx[ o ] = x[ o ]; dy[ o ] = y[ o ]; dc[ o ] = c[ o ]; }
            const int d1 = j3 - 2;
            if ( d1 < 0 ) { barillet_d( gx, -d1 ); barillet_d( gy, -d1 ); barillet_d( gc, -d1 ); }
            else          { barillet( gx, d1 );    barillet( gy, d1 );    barillet( gc, d1 ); }
            const int e = nb - j3 + 2;
            const int ee = e < R ? e : R - 1;
            barillet_d( dx, ee ); barillet_d( dy, ee ); barillet_d( dc, ee );

            const TK  x3 = gx[ 2 ], y3 = gy[ 2 ];
            const TK  x2 = j3 ? gx[ 1 ] : xl, y2 = j3 ? gy[ 1 ] : yl;
            const int bid = j3 ? gc[ 1 ] : cl;
            TK ax[ R ], ay[ R ];
#pragma unroll
            for ( int o = 0; o < R; ++o ) { ax[ o ] = x[ o ]; ay[ o ] = y[ o ]; }
            barillet_d( ax, R - 1 - i1 ); barillet_d( ay, R - 1 - i1 );
            const TK  x1 = ax[ R - 1 ], y1 = ay[ R - 1 ];
            const TK  x0v = i1 ? ax[ R - 2 ] : xl, y0v = i1 ? ay[ R - 2 ] : yl;
            const TK  s0 = p.dx * x0v + p.dy * y0v - p.off, s1 = p.dx * x1 + p.dy * y1 - p.off;
            const TK  s2 = p.dx * x2 + p.dy * y2 - p.off,   s3 = p.dx * x3 + p.dy * y3 - p.off;
            const TK  ta = s0 / ( s0 - s1 ), tb = s3 / ( s3 - s2 );
            const TK  pax = x0v + ( x1 - x0v ) * ta, pay = y0v + ( y1 - y0v ) * ta;
            const TK  pbx = x3 + ( x2 - x3 ) * tb,   pby = y3 + ( y2 - y3 ) * tb;

            x[ 0 ] = pax; y[ 0 ] = pay; c[ 0 ] = p.id;
            x[ 1 ] = pbx; y[ 1 ] = pby; c[ 1 ] = bid;
#pragma unroll
            for ( int o = 2; o < R; ++o ) {
                if ( o >= SUR && o >= nn ) break;
                x[ o ] = o < e ? gx[ o ] : dx[ o ];
                y[ o ] = o < e ? gy[ o ] : dy[ o ];
                c[ o ] = o < e ? gc[ o ] : dc[ o ];
            }
            xl = x0v; yl = y0v; cl = i1 ? selR( c, nn - 1 ) : cl;
            nb = nn;
            continue;
        }

        // ---- UN NOEUD DE LA PILE
        if ( haut > 0 ) {
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
            if ( nd.right < 0 ) { q = nd.beg; fin = nd.end; continue; }
            const int g = h + 1, dr = nd.right;
            const bool gp = proximite( ar.nodes[ g ], p0 ) <= proximite( ar.nodes[ dr ], p0 );
            pile[ haut++ ] = gp ? dr : g;
            pile[ haut++ ] = gp ? g : dr;
            continue;
        }

        // ---- LA CELLULE EST FINIE : son aire, et la prochaine au tour suivant
        double a = 0;
#pragma unroll
        for ( int i = 0; i < R; ++i ) {
            if ( i >= SUR && i >= nb ) break;
            const int j = i + 1 < nb ? i + 1 : 0;
            a += double( x[ i ] ) * double( selR( y, j ) ) - double( selR( x, j ) ) * double( y[ i ] );
        }
        res[ i0 ] = 0.5 * fabs( a );
        nb = 0; haut = 0;
    }
}

} // namespace sf::gpu

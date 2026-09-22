#pragma once

// =====================================================================================
// LES PHASES, EN VRAI : UN NOYAU PERSISTANT PAR SM, DES FILES PAR BLOC, L'ETAT EN RAM.
//
// Pas de noyau global par phase -- les phases sont INTERNES AU BLOC, synchronisees par
// `__syncthreads()` ( une vingtaine de cycles ) et non par des lancements ( cinq microsecondes ).
// Chaque bloc prend des cellules CONSECUTIVES dans l'ordre de l'arbre, donc voisines dans
// l'espace : la localite par warp -- ce qui avait tue le tri par cout ( README § 4 ) -- est
// preservee, et la redistribution ne se fait qu'entre les `CAP` cellules du bloc.
//
// TROIS CATEGORIES, comme demande :
//   A. les cellules qui attendent une BOITE ( phase « parcours » : depiler jusqu'a une feuille )
//   B. les cellules qui ont une feuille a traiter ( phase « coupe » : les germes de la feuille )
//   C. les cellules FINIES ( phase « mesure » : l'aire, l'ecriture, et le slot rendu )
// Files DOUBLES : une phase lit `f[ cur ]`, ecrit `f[ 1 - cur ]` -- rien n'est lu et ecrit dans le
// meme tour. `libres` est l'ensemble des slots disponibles.
//
// LES FILES SONT DES BITMAPS, pas des listes : un bit par slot, l'ecriture est un `atomicOr` et la
// LECTURE EST ORDONNEE ( les bits parcourus du plus petit au plus grand, par un prefixe sur les
// `popc` des seize mots ). C'est ce qui rend la COALESCENCE : les lanes consecutives d'un warp
// prennent des slots consecutifs, donc `st.x[ i * S + slot ]` se lit en une transaction au lieu
// de trente-deux. Avec des listes remplies par `atomicAdd` ( l'ordre d'arrivee ), le meme noyau
// lisait 8.8 Ko par cellule au lieu de ~1 et saturait la DRAM ( 365 Go/s, mesure ).
//
// L'ETAT D'UNE CELLULE, dans la RAM du GPU, en SoA `[ champ * S + slot ]` ( `S` = tous les slots
// de la grille ) : `x[ R ]`, `y[ R ]`, `cid[ R ]`, la pile du parcours, et quatre entiers
// ( `nb`, `haut`, le rang `k`, le noeud courant ). Le germe et son poids ne sont pas stockes :
// ils se relisent dans l'arbre depuis `k`. ~200 octets par cellule en vol.
//
// LA COUPE est celle de `filnrm` ( normalisation en barillet, positions fixes ), en registres :
// une cellule descend en RAM entre deux phases, pas pendant.
// =====================================================================================

#include "gpu/FilNrm2D.cuh"

namespace sf::gpu {

/// l'etat des cellules en vol, en RAM. `S` slots en tout ( `gridDim.x * CAP` ).
template<class TK>
struct EtatPh {
    TK  *x, *y;                 ///< `[ i * S + slot ]`
    int *c;                     ///< idem
    int *pile;                  ///< `[ d * S + slot ]`
    int *meta;                  ///< `[ m * S + slot ]` : 0 = nb, 1 = haut, 2 = rang, 3 = noeud
    int  S;
};

constexpr int PILE_PH = 24;     ///< la profondeur du BSP median : `log2( n / feuille )` + marge
constexpr int META_PH = 4;

/// `CAP` cellules en vol par bloc, `BL` threads par bloc.
template<bool POIDS, int R, int CAP, int BL, class TK>
__global__ void __launch_bounds__( BL ) noyau2_filph( Arbre<TK,2> ar, double *res, int *deborde, int *liste_deb,
                                                       int *compteur, EtatPh<TK> st ) {
    constexpr int SUR = 3;
    const int tid = threadIdx.x;
    const int base = blockIdx.x * CAP;
    const int S = st.S;

    constexpr int MOTS = CAP / 32;
    __shared__ unsigned mA[ 2 ][ MOTS ], mB[ 2 ][ MOTS ], mC[ MOTS ], mlibre[ MOTS ];
    __shared__ unsigned short file[ CAP ];               // la file ordonnee de la phase courante
    __shared__ int off[ MOTS ], nfile, cur, reste;

    for ( int w = tid; w < MOTS; w += BL ) { mA[ 0 ][ w ] = mA[ 1 ][ w ] = mB[ 0 ][ w ] = mB[ 1 ][ w ] = mC[ w ] = 0; mlibre[ w ] = 0xffffffffu; }
    if ( tid == 0 ) { cur = 0; reste = 1; }
    __syncthreads();

    // ---- LE BITMAP EN FILE ORDONNEE : un prefixe sur les `popc`, puis chaque mot ecrit ses bits
    auto deplie = [ & ]( unsigned *m ) {
        if ( tid < MOTS ) off[ tid ] = __popc( m[ tid ] );
        __syncthreads();
        if ( tid == 0 ) {
            int s = 0;
            for ( int w = 0; w < MOTS; ++w ) { const int v = off[ w ]; off[ w ] = s; s += v; }
            nfile = s;
        }
        __syncthreads();
        if ( tid < MOTS ) {
            unsigned w = m[ tid ];
            int p = off[ tid ];
            while ( w ) { const int b = __ffs( w ) - 1; file[ p++ ] = ( unsigned short ) ( tid * 32 + b ); w &= w - 1; }
        }
        __syncthreads();
    };
    auto pose = [ & ]( unsigned *m, int sl ) { atomicOr( &m[ sl >> 5 ], 1u << ( sl & 31 ) ); };

    // ---- les accesseurs de l'etat
    auto charge = [ & ]( int slot, TK *x, TK *y, int *c ) {
#pragma unroll
        for ( int i = 0; i < R; ++i ) { x[ i ] = st.x[ i * S + slot ]; y[ i ] = st.y[ i * S + slot ]; c[ i ] = st.c[ i * S + slot ]; }
    };
    auto range = [ & ]( int slot, const TK *x, const TK *y, const int *c ) {
#pragma unroll
        for ( int i = 0; i < R; ++i ) { st.x[ i * S + slot ] = x[ i ]; st.y[ i * S + slot ] = y[ i ]; st.c[ i * S + slot ] = c[ i ]; }
    };

    for ( ;; ) {
        // ================================================================ PHASE MESURE
        deplie( mC );
        const int nmes = nfile;
        for ( int e = tid; e < nmes; e += BL ) {
            const int slot = base + file[ e ];
            const int nb = st.meta[ 0 * S + slot ], k = st.meta[ 2 * S + slot ];
            const int i0 = ar.ids[ k ];
            double a = 0;
            if ( nb > 0 ) {
                TK x[ R ], y[ R ]; int c[ R ];
                charge( slot, x, y, c );
#pragma unroll
                for ( int i = 0; i < R; ++i ) {
                    if ( i >= SUR && i >= nb ) break;
                    const int j = i + 1 < nb ? i + 1 : 0;
                    a += double( x[ i ] ) * double( selR( y, j ) ) - double( selR( x, j ) ) * double( y[ i ] );
                }
            } else if ( nb < 0 )
                liste_deb[ atomicAdd( deborde, 1 ) ] = k;
            res[ i0 ] = 0.5 * fabs( a );
            pose( mlibre, slot - base );
        }
        __syncthreads();
        for ( int w = tid; w < MOTS; w += BL ) mC[ w ] = 0;
        __syncthreads();

        // ================================================================ RECHARGE
        // chaque slot libre prend une cellule neuve ; elle entre en A avec la racine sur la pile
        deplie( mlibre );
        const int nlib = nfile;
        for ( int w = tid; w < MOTS; w += BL ) mlibre[ w ] = 0;   // les non repris sont perdus :
        __syncthreads();                                          // il n'y a plus de cellules
        for ( int e = tid; e < nlib; e += BL ) {
            const int k = atomicAdd( compteur, 1 );
            if ( k >= ar.n ) continue;
            const int slot = base + file[ e ];
            TK x[ R ], y[ R ]; int c[ R ];
#pragma unroll
            for ( int i = 0; i < R; ++i ) { x[ i ] = TK( i == 1 || i == 2 ); y[ i ] = TK( i == 2 || i == 3 ); c[ i ] = i < 4 ? -1 - i : 0; }
            range( slot, x, y, c );
            st.pile[ 0 * S + slot ] = 0;
            st.meta[ 0 * S + slot ] = 4;                 // nb
            st.meta[ 1 * S + slot ] = 1;                 // haut
            st.meta[ 2 * S + slot ] = k;
            pose( mA[ 1 - cur ], slot - base );
        }
        __syncthreads();

        // ================================================================ PHASE PARCOURS
        // depiler jusqu'a une feuille ( -> B ) ou jusqu'a la pile vide ( -> C )
        deplie( mA[ cur ] );
        const int npar = nfile;
        for ( int e = tid; e < npar; e += BL ) {
            const int slot = base + file[ e ];
            const int nb = st.meta[ 0 * S + slot ], k = st.meta[ 2 * S + slot ];
            int haut = st.meta[ 1 * S + slot ];
            const TK p0[ 2 ] = { ar.c[ 0 ][ k ], ar.c[ 1 ][ k ] };
            const TK w0 = POIDS ? ar.w[ k ] : TK( 0 );
            TK x[ R ], y[ R ]; int c[ R ];
            charge( slot, x, y, c );

            int feuille = -1;
            while ( haut > 0 ) {
                const int h = st.pile[ ( --haut ) * S + slot ];
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
                if ( nd.right < 0 ) { feuille = h; break; }
                const int g = h + 1, dr = nd.right;
                const bool gp = proximite( ar.nodes[ g ], p0 ) <= proximite( ar.nodes[ dr ], p0 );
                st.pile[ haut * S + slot ] = gp ? dr : g;
                st.pile[ ( haut + 1 ) * S + slot ] = gp ? g : dr;
                haut += 2;
            }
            st.meta[ 1 * S + slot ] = haut;
            if ( feuille >= 0 ) {
                st.meta[ 3 * S + slot ] = feuille;
                pose( mB[ 1 - cur ], slot - base );
            } else
                pose( mC, slot - base );
        }
        __syncthreads();
        for ( int w = tid; w < MOTS; w += BL ) mA[ cur ][ w ] = 0;
        __syncthreads();

        // ================================================================ PHASE COUPE
        // les germes de la feuille, tous, puis retour en A ( ou en C si la cellule est finie )
        deplie( mB[ cur ] );
        const int ncou = nfile;
        for ( int e = tid; e < ncou; e += BL ) {
            const int slot = base + file[ e ];
            int nb = st.meta[ 0 * S + slot ];
            const int k = st.meta[ 2 * S + slot ], h = st.meta[ 3 * S + slot ];
            const Noeud<TK,2> nd = ar.nodes[ h ];
            const TK p0[ 2 ] = { ar.c[ 0 ][ k ], ar.c[ 1 ][ k ] };
            const TK w0 = POIDS ? ar.w[ k ] : TK( 0 );
            TK x[ R ], y[ R ]; int c[ R ];
            charge( slot, x, y, c );
            TK xl = 0, yl = 0; int cl = 0;
            bool vu_dernier = false;                     // `v_nb-1` : relu au besoin

            for ( int q = nd.beg; q < nd.end && nb > 0; ++q ) {
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
                if ( IMPROBABLE( m == valid ) ) { nb = 0; break; }

                const unsigned prev = ( ( m << 1 ) | ( m >> ( nb - 1 ) ) ) & valid;
                const unsigned next = ( ( m >> 1 ) | ( m << ( nb - 1 ) ) ) & valid;
                const int i1 = __ffs( m & ~prev ) - 1;
                const int j2 = __ffs( m & ~next ) - 1;
                const int j3 = j2 + 1 < nb ? j2 + 1 : 0;
                const int nb_in = nb - __popc( m );
                const int nn = nb_in + 2;
                if ( IMPROBABLE( nn > R ) ) { nb = -1; break; }

                if ( ! vu_dernier ) { xl = selR( x, nb - 1 ); yl = selR( y, nb - 1 ); cl = selR( c, nb - 1 ); vu_dernier = true; }

                TK  gx[ R ], gy[ R ]; int gc[ R ];
                TK  dx[ R ], dy[ R ]; int dc[ R ];
#pragma unroll
                for ( int o = 0; o < R; ++o ) { gx[ o ] = x[ o ]; gy[ o ] = y[ o ]; gc[ o ] = c[ o ]; dx[ o ] = x[ o ]; dy[ o ] = y[ o ]; dc[ o ] = c[ o ]; }
                const int d1 = j3 - 2;
                if ( d1 < 0 ) { barillet_d( gx, -d1 ); barillet_d( gy, -d1 ); barillet_d( gc, -d1 ); }
                else          { barillet( gx, d1 );    barillet( gy, d1 );    barillet( gc, d1 ); }
                const int e2 = nb - j3 + 2;
                const int ee = e2 < R ? e2 : R - 1;
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
                    x[ o ] = o < e2 ? gx[ o ] : dx[ o ];
                    y[ o ] = o < e2 ? gy[ o ] : dy[ o ];
                    c[ o ] = o < e2 ? gc[ o ] : dc[ o ];
                }
                xl = x0v; yl = y0v; cl = i1 ? selR( c, nn - 1 ) : cl;
                nb = nn;
            }

            st.meta[ 0 * S + slot ] = nb;
            if ( nb > 0 ) {
                range( slot, x, y, c );
                pose( mA[ 1 - cur ], slot - base );
            } else
                pose( mC, slot - base );
        }
        __syncthreads();
        for ( int w = tid; w < MOTS; w += BL ) mB[ cur ][ w ] = 0;
        __syncthreads();
        if ( tid == 0 ) {
            cur = 1 - cur;
            int r = 0;
            for ( int w = 0; w < MOTS; ++w ) r |= mA[ cur ][ w ] | mB[ cur ][ w ] | mC[ w ];
            reste = r;
        }
        __syncthreads();
        if ( reste == 0 && *compteur >= ar.n )
            break;
    }
}

} // namespace sf::gpu

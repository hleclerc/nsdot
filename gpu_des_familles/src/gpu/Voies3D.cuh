#pragma once

// =====================================================================================
// LA CELLULE SUR LES TRENTE-DEUX VOIES D'UN WARP, 3D. Le polyedre de `cell/Cellule3D.h` -- trois
// coupes et trois voisins par sommet, le voisin `j` EN FACE de la coupe `j` -- porte par les
// voies : la voie `l` tient les sommets `l, l + 32, l + 64, ...` dans `S` CASES de registres, les
// trois coupes d'un sommet dans un mot ( un octet chacune, triees ), ses trois voisins dans un
// autre. `nv` et `nc` sont uniformes. Une cellule par warp, donc pas de divergence entre
// groupes ; en 3D une cellule a vingt a quarante sommets, les voies sont pleines.
//
// CE QUE LE WARP FAIT ENSEMBLE, a chaque coupe :
//   * la premiere passe : un `fma` par case et un `ballot` par case -- les masques des sommets
//     dehors ( `out[ s ]` ) SONT l'etat de la coupe, tout le reste s'en deduit ;
//   * les sommets neufs : un CANDIDAT par voie ( un sommet dehors, une de ses trois aretes ),
//     garde si l'autre bout est dedans, compacte par `ballot` / `popc` -- le sommet neuf `j` vit
//     dans la voie `j` le temps de la coupe ;
//   * la RENUMEROTATION : pas de trous a boucher ( l'astuce du CPU pour ne pas toucher les
//     survivants n'a pas de sens ici, toutes les voies travaillent de toute facon ), les
//     survivants sont compactes dans l'ordre et les neufs mis a la suite ; le nouveau numero d'un
//     sommet est `popc` du masque des survivants sous lui -- les `ballot` sont la table, aucune
//     memoire partagee ;
//   * chaque voie rassemble ses nouveaux sommets par `shfl` ( une par case source et par
//     champ ), et recolle les voisins : un octet qui nommait un sommet dehors nomme maintenant le
//     sommet neuf ne sur cette arete ( retrouve par une boucle sur les neufs ).
// La cellule n'est ecrite qu'a la fin : sur debordement elle reste INTACTE.
//
// LE VOLUME : la somme des tetraedres ( g, o_f, a, b ) pour chaque arete `( a, b )` et chacune de
// ses deux faces, `o_f` un sommet fixe de la face ( le plus petit, par `atomicMin` en memoire
// partagee ) -- equivalent a l'accumulation par face du CPU sur un polygone convexe, sans
// accumulateur.
//
// LA MEMOIRE PARTAGEE ne sert qu'au parcours ( la pile ), au compactage des coupes mortes ( un
// masque de vivantes ) et au volume ( `o_f` ) : trois petits tableaux par warp.
// =====================================================================================

#include "gpu/Arbre.cuh"

namespace sf::gpu {

constexpr unsigned PLEIN = 0xffffffffu;
constexpr int      BLOC3 = 128;         ///< threads par bloc : quatre cellules

/// `a[ s ]` avec `s` dynamique, SANS passer par la memoire locale : une chaine de `select`.
template<class T, int S>
__device__ __forceinline__ T sel( const T ( &a )[ S ], int s ) {
    T r = a[ 0 ];
#pragma unroll
    for ( int q = 1; q < S; ++q ) r = s == q ? a[ q ] : r;
    return r;
}

/// LE RASSEMBLEMENT : la valeur du champ `a` pour le sommet `idx` ( propre a chaque voie ), quel
/// que soit la voie et la case ou il vit. TOUTES les cases sont diffusees quand il y en a deux :
/// une `shfl` de trop coute moins qu'une branche et sa barriere de reconvergence ( mesure : les
/// `if ( s < su )` faisaient 4 % des instructions ) ; au-dela, les cases hors d'usage sont sautees.
template<class T, int S>
__device__ __forceinline__ T rassemble( const T ( &a )[ S ], int idx, int su ) {
    T r = T( 0 );
#pragma unroll
    for ( int s = 0; s < S; ++s )
        if ( S <= 2 || s < su ) {
            const T v = __shfl_sync( PLEIN, a[ s ], idx & 31 );
            r = ( idx >> 5 ) == s ? v : r;
        }
    return r;
}

/// LE `n`-IEME BIT A 1 d'un mot ( `n` depuis 0 ), SANS BOUCLE : `__fns` n'est pas une instruction
/// sur sm_75 mais une boucle logicielle ( 7 % des instructions, mesure ) ; ici une dichotomie sur
/// `popc`, cinq etages de `select`.
__device__ __forceinline__ int bit_nieme( unsigned m, int n ) {
    int pos = 0;
    int c = __popc( m & 0xffffu );  bool h = n >= c; n -= h ? c : 0; pos += h ? 16 : 0; m = h ? m >> 16 : m;
    c = __popc( m & 0xffu );        h = n >= c; n -= h ? c : 0; pos += h ? 8 : 0;  m = h ? m >> 8 : m;
    c = __popc( m & 0xfu );         h = n >= c; n -= h ? c : 0; pos += h ? 4 : 0;  m = h ? m >> 4 : m;
    c = __popc( m & 0x3u );         h = n >= c; n -= h ? c : 0; pos += h ? 2 : 0;  m = h ? m >> 2 : m;
    c = __popc( m & 0x1u );         h = n >= c; pos += h ? 1 : 0;
    return pos;
}

/// le `n`-ieme bit a 1 ( `n` depuis 0 ) des masques, en numero de sommet.
template<int S>
__device__ __forceinline__ int nieme( const unsigned ( &m )[ S ], int n ) {
    int slot = 0, pre = 0, acc = 0;
#pragma unroll
    for ( int s = 0; s < S - 1; ++s ) {                  // la case : combien de prefixes sont sous `n`
        acc += __popc( m[ s ] );
        const bool h = n >= acc;
        slot = h ? s + 1 : slot;
        pre  = h ? acc : pre;
    }
    return slot * 32 + bit_nieme( sel( m, slot ), n - pre );
}

/// le rang du sommet `w` parmi les bits a 1 : combien en ont un numero plus petit.
template<int S>
__device__ __forceinline__ int rang( const unsigned ( &m )[ S ], int w ) {
    int r = 0;
#pragma unroll
    for ( int s = 0; s < S; ++s ) if ( s < ( w >> 5 ) ) r += __popc( m[ s ] );
    return r + __popc( sel( m, w >> 5 ) & ( ( 1u << ( w & 31 ) ) - 1 ) );
}

template<int S>
__device__ __forceinline__ int total( const unsigned ( &m )[ S ] ) {
    int r = 0;
#pragma unroll
    for ( int s = 0; s < S; ++s ) r += __popc( m[ s ] );
    return r;
}

__device__ __forceinline__ int octet( unsigned w, int j ) { return int( ( w >> ( 8 * j ) ) & 255u ); }
__device__ __forceinline__ unsigned mot3( int a, int b, int c ) { return ( unsigned( a ) & 255u ) | ( ( unsigned( b ) & 255u ) << 8 ) | ( ( unsigned( c ) & 255u ) << 16 ); }

/// LES DEUX FACES QUI PORTENT L'ARETE `j` DU SOMMET : ses coupes, privees de la `j`-ieme.
__device__ __forceinline__ void faces_de( unsigned k, int j, int &f0, int &f1 ) {
    f0 = octet( k, j == 0 ? 1 : 0 );
    f1 = octet( k, j == 2 ? 1 : 2 );
}

/// LA CELLULE D'UN WARP : `S` cases par voie, `32 S` sommets et coupes au plus.
template<class TK, int S>
struct Cel3V {
    static constexpr int MaxNv = 32 * S;
    TK       x[ S ], y[ S ], z[ S ];
    unsigned k[ S ];            ///< les trois coupes du sommet, un octet chacune, TRIEES
    unsigned nn[ S ];           ///< ses trois voisins, `nn_j` EN FACE de `k_j`
    int      cid[ S ];          ///< l'identifiant global de la coupe `s * 32 + l`
    int      nv, nc;            ///< uniformes
    int      lane;

    __device__ int su_v() const { return ( nv + 31 ) >> 5; }    ///< cases en usage, sommets
    __device__ int su_c() const { return ( nc + 31 ) >> 5; }    ///< cases en usage, coupes

    /// LE CUBE UNITE : sommet `b` dans la voie `b`, coupes `0:x=0 1:x=1 2:y=0 3:y=1 4:z=0 5:z=1`.
    __device__ void init_cube( int l ) {
        lane = l;
        nv = 8; nc = 6;
        const int b = l & 7, i = b & 1, j = ( b >> 1 ) & 1, kk = ( b >> 2 ) & 1;
        x[ 0 ] = TK( i ); y[ 0 ] = TK( j ); z[ 0 ] = TK( kk );
        k[ 0 ]  = mot3( i, 2 + j, 4 + kk );
        nn[ 0 ] = mot3( b ^ 1, b ^ 2, b ^ 4 );
        cid[ 0 ] = -1 - l;
#pragma unroll
        for ( int s = 1; s < S; ++s ) { x[ s ] = y[ s ] = z[ s ] = TK( 0 ); k[ s ] = nn[ s ] = 0; cid[ s ] = 0; }
    }

    /// LE TEST D'ELAGAGE : une operation par case, un `ballot`.
    template<bool POIDS>
    __device__ bool peut_couper( const Noeud<TK,3> &B, const TK *p0, TK w0 ) const {
        const int su = su_v();
        unsigned any = 0;
#pragma unroll
        for ( int s = 0; s < S; ++s )
            if ( s < su ) {
                const TK v[ 3 ] = { x[ s ], y[ s ], z[ s ] };
                const bool m = s * 32 + lane < nv && bilan_sommet<POIDS>( B, v, p0, w0 ) <= TK( 0 );
                any |= __ballot_sync( PLEIN, m );
            }
        return any != 0;
    }

    /// LA COUPE. `viv` : quatre mots partages du warp, pour le compactage des coupes mortes.
    __device__ int coupe( const Plan3<TK> &p, unsigned *viv ) {
        const int su = su_v();

        // ---- LA PREMIERE PASSE : `s` par case, les masques des sommets DEHORS
        TK       sv[ S ];
        unsigned out[ S ], viva[ S ];
#pragma unroll
        for ( int s = 0; s < S; ++s ) {
            sv[ s ] = p.dx * x[ s ] + p.dy * y[ s ] + p.dz * z[ s ] - p.off;
            const bool valide = s * 32 + lane < nv;
            out[ s ] = viva[ s ] = 0u;
            if ( s < su ) {                              // uniforme
                out[ s ]  = __ballot_sync( PLEIN, valide && sv[ s ] > TK( 0 ) );
                viva[ s ] = __ballot_sync( PLEIN, valide && ! ( sv[ s ] > TK( 0 ) ) );
            }
        }
        const int nb_out = total( out );
        if ( nb_out == 0 )
            return INCHANGEE;
        const int nb_in = total( viva );
        if ( nb_in == 0 ) { nv = 0; nc = 0; return VIDE; }

        if ( nc >= 32 * S ) {
            compacte( viv );
            if ( nc >= 32 * S ) return DEBORDE;
        }
        const int knew = nc;

        // ---- LES SOMMETS NEUFS : un candidat par voie -- le sommet dehors `io`, son arete `j` --
        // garde si l'autre bout est dedans ; le neuf `m` vit dans la voie `m`
        TK       NX = 0, NY = 0, NZ = 0;
        unsigned N01 = 0;                                // les deux coupes heritees, n0 | n1 << 8
        int      RV = 0, RF = 0;                         // le sommet dedans a recoller, et sa fente
        int      nm = 0;
        for ( int base = 0; base < 3 * nb_out; base += 32 ) {
            const int  c = base + lane;
            const bool cand = c < 3 * nb_out;
            const int  io = c / 3, j = c - 3 * io;
            const int  o  = cand ? nieme( out, io ) : 0;
            const unsigned ko  = rassemble( k,  o, su );
            const unsigned nno = rassemble( nn, o, su );
            const int  u  = octet( nno, j );
            const TK   suu = rassemble( sv, u, su ), so = rassemble( sv, o, su );
            const bool dedans = cand && ! ( suu > TK( 0 ) );
            const TK   xo = rassemble( x, o, su ), yo = rassemble( y, o, su ), zo = rassemble( z, o, su );
            const TK   xu = rassemble( x, u, su ), yu = rassemble( y, u, su ), zu = rassemble( z, u, su );
            const unsigned nnu = rassemble( nn, u, su );
            const TK   t  = suu / ( suu - so );          // ANCRE SUR LE SOMMET DEDANS
            const TK   nx = xu + ( xo - xu ) * t, ny = yu + ( yo - yu ) * t, nz = zu + ( zo - zu ) * t;
            int f0, f1;
            faces_de( ko, j, f0, f1 );
            const int rf = octet( nnu, 0 ) == o ? 0 : ( octet( nnu, 1 ) == o ? 1 : 2 );

            const unsigned bal = __ballot_sync( PLEIN, dedans );
            const int cnt = __popc( bal );
            if ( nm + cnt > 32 )
                return DEBORDE;                          // plus de trente-deux sommets sur une face
            // le compactage : la voie `nm + r` prend le `r`-ieme candidat garde
            const int  r   = lane - nm;
            const int  src = bit_nieme( bal, ( r >= 0 && r < cnt ) ? r : 0 );
            const TK   gx = __shfl_sync( PLEIN, nx, src ), gy = __shfl_sync( PLEIN, ny, src ), gz = __shfl_sync( PLEIN, nz, src );
            const unsigned g01 = __shfl_sync( PLEIN, unsigned( f0 ) | ( unsigned( f1 ) << 8 ), src );
            const int  gv = __shfl_sync( PLEIN, u, src ), gf = __shfl_sync( PLEIN, rf, src );
            if ( r >= 0 && r < cnt ) { NX = gx; NY = gy; NZ = gz; N01 = g01; RV = gv; RF = gf; }
            nm += cnt;
        }
        const int new_nv = nb_in + nm;
        if ( new_nv > 32 * S )
            return DEBORDE;

        // ---- LES VOISINS DES NEUFS ENTRE EUX : deux neufs sont voisins exactement quand ils
        // partagent une ANCIENNE coupe ; le neuf `m` devient le sommet `nb_in + m`
        int M0 = -1, M1 = -1;
        {
            const int n0 = octet( N01, 0 ), n1 = octet( N01, 1 );
            for ( int j = 0; j < nm; ++j ) {
                const unsigned nj = __shfl_sync( PLEIN, N01, j );
                if ( j == lane || lane >= nm ) continue;
                const int a = octet( nj, 0 ), b = octet( nj, 1 );
                if      ( n0 == a || n0 == b ) M1 = j;
                else if ( n1 == a || n1 == b ) M0 = j;
            }
        }

        // ---- LE NOUVEL ETAT : les survivants compactes dans l'ordre, puis les neufs. Chaque voie
        // rassemble LES DEUX ( un `shfl` ne peut pas vivre sous une branche divergente ) et garde
        // ce qui la concerne.
        const int su2 = ( new_nv + 31 ) >> 5;
        TK       x2[ S ], y2[ S ], z2[ S ];
        unsigned k2[ S ], nn2[ S ];
#pragma unroll
        for ( int s = 0; s < S; ++s ) {
            x2[ s ] = y2[ s ] = z2[ s ] = TK( 0 ); k2[ s ] = nn2[ s ] = 0;
            if ( s >= su2 ) continue;
            const int  t    = s * 32 + lane;
            const bool surv = t < nb_in, neuf = ! surv && t < new_nv;

            // un SURVIVANT : le `t`-ieme sommet dedans, ses voisins renumerotes ou recolles
            const int i = surv ? nieme( viva, t ) : 0;
            const TK  sx = rassemble( x, i, su ), sy = rassemble( y, i, su ), sz = rassemble( z, i, su );
            const unsigned sk = rassemble( k, i, su ), nni = rassemble( nn, i, su );
            int w0 = 0, w1 = 0, w2 = 0;
#pragma unroll
            for ( int f = 0; f < 3; ++f ) {
                const int  old    = octet( nni, f );
                const bool dehors = ( sel( out, old >> 5 ) >> ( old & 31 ) ) & 1u;
                const int  wf     = dehors ? -1 : rang( viva, old );
                if ( f == 0 ) w0 = wf; else if ( f == 1 ) w1 = wf; else w2 = wf;
            }
            // le recollage : l'octet qui nommait un sommet dehors nomme le neuf ne sur l'arete
            for ( int j = 0; j < nm; ++j ) {
                const int rv = __shfl_sync( PLEIN, RV, j ), rf = __shfl_sync( PLEIN, RF, j );
                if ( surv && rv == i ) {
                    if ( rf == 0 ) w0 = nb_in + j; else if ( rf == 1 ) w1 = nb_in + j; else w2 = nb_in + j;
                }
            }

            // un NEUF : `( n0, n1, knew )` et `( m0, m1, u )`
            const int j = neuf ? t - nb_in : 0;
            const TK  gx = __shfl_sync( PLEIN, NX, j ), gy = __shfl_sync( PLEIN, NY, j ), gz = __shfl_sync( PLEIN, NZ, j );
            const unsigned n01 = __shfl_sync( PLEIN, N01, j );
            const int m0 = __shfl_sync( PLEIN, M0, j ), m1 = __shfl_sync( PLEIN, M1, j ), rv = __shfl_sync( PLEIN, RV, j );

            if ( surv ) { x2[ s ] = sx; y2[ s ] = sy; z2[ s ] = sz; k2[ s ] = sk; nn2[ s ] = mot3( w0, w1, w2 ); }
            if ( neuf ) {
                x2[ s ] = gx; y2[ s ] = gy; z2[ s ] = gz;
                k2[ s ]  = n01 | ( unsigned( knew ) << 16 );
                nn2[ s ] = mot3( nb_in + m0, nb_in + m1, rang( viva, rv ) );
            }
        }
#pragma unroll
        for ( int s = 0; s < S; ++s ) { x[ s ] = x2[ s ]; y[ s ] = y2[ s ]; z[ s ] = z2[ s ]; k[ s ] = k2[ s ]; nn[ s ] = nn2[ s ]; }
        if ( ( knew & 31 ) == lane ) {
#pragma unroll
            for ( int s = 0; s < S; ++s ) if ( ( knew >> 5 ) == s ) cid[ s ] = p.id;
        }
        nc = knew + 1;
        nv = new_nv;
        return COUPEE;
    }

    /// ENLEVER LES COUPES MORTES : un masque de vivantes en memoire partagee, la renumerotation par
    /// `popc` en dessous -- MONOTONE, donc les triplets restent tries.
    __device__ void compacte( unsigned *viv ) {
        const int su = su_v(), suc = su_c();
        if ( lane < S ) viv[ lane ] = 0;
        __syncwarp();
#pragma unroll
        for ( int s = 0; s < S; ++s )
            if ( s < su && s * 32 + lane < nv ) {
#pragma unroll
                for ( int f = 0; f < 3; ++f ) {
                    const int c = octet( k[ s ], f );
                    atomicOr( &viv[ c >> 5 ], 1u << ( c & 31 ) );
                }
            }
        __syncwarp();
        unsigned m[ S ];
#pragma unroll
        for ( int s = 0; s < S; ++s ) m[ s ] = viv[ s ];
        __syncwarp();
        const int ncn = total( m );
#pragma unroll
        for ( int s = 0; s < S; ++s )
            if ( s < su && s * 32 + lane < nv )
                k[ s ] = mot3( rang( m, octet( k[ s ], 0 ) ), rang( m, octet( k[ s ], 1 ) ), rang( m, octet( k[ s ], 2 ) ) );
        int c2[ S ];
#pragma unroll
        for ( int s = 0; s < S; ++s ) {
            const int t = s * 32 + lane;
            const int v = rassemble( cid, t < ncn ? nieme( m, t ) : 0, suc );
            c2[ s ] = t < ncn ? v : 0;
        }
#pragma unroll
        for ( int s = 0; s < S; ++s ) cid[ s ] = c2[ s ];
        nc = ncn;
    }

    /// LE VOLUME : les tetraedres `( g, o_f, a, b )`, `o_f` le plus petit sommet de la face ( `v0`,
    /// `32 S` entiers partages du warp ).
    __device__ double volume( int *v0 ) const {
        if ( nv < 4 ) return 0;
        const int su = su_v();
#pragma unroll
        for ( int s = 0; s < S; ++s ) v0[ s * 32 + lane ] = 0x7fffffff;
        __syncwarp();
        double gx = 0, gy = 0, gz = 0;
#pragma unroll
        for ( int s = 0; s < S; ++s ) {
            const int i = s * 32 + lane;
            if ( s < su && i < nv ) {
                gx += x[ s ]; gy += y[ s ]; gz += z[ s ];
#pragma unroll
                for ( int f = 0; f < 3; ++f ) atomicMin( &v0[ octet( k[ s ], f ) ], i );
            }
        }
#pragma unroll
        for ( int d = 16; d >= 1; d /= 2 ) {
            gx += __shfl_xor_sync( PLEIN, gx, d ); gy += __shfl_xor_sync( PLEIN, gy, d ); gz += __shfl_xor_sync( PLEIN, gz, d );
        }
        gx /= nv; gy /= nv; gz /= nv;
        __syncwarp();

        double v = 0;
#pragma unroll
        for ( int s = 0; s < S; ++s ) {
            if ( s >= su ) continue;
            const int a = s * 32 + lane;
            const bool ok = a < nv;
            const double ax = double( x[ s ] ) - gx, ay = double( y[ s ] ) - gy, az = double( z[ s ] ) - gz;
#pragma unroll
            for ( int j = 0; j < 3; ++j ) {
                const int b = ok ? octet( nn[ s ], j ) : 0;
                const double bx = double( rassemble( x, b, su ) ) - gx, by = double( rassemble( y, b, su ) ) - gy, bz = double( rassemble( z, b, su ) ) - gz;
                int f0, f1;
                faces_de( ok ? k[ s ] : 0u, j, f0, f1 );
#pragma unroll
                for ( int r = 0; r < 2; ++r ) {
                    const int o = ok ? v0[ r ? f1 : f0 ] : 0;
                    const double ox = double( rassemble( x, o, su ) ) - gx, oy = double( rassemble( y, o, su ) ) - gy, oz = double( rassemble( z, o, su ) ) - gz;
                    // ( a - o ) x ( b - o ) . ( o - g )
                    const double ex = ax - ox, ey = ay - oy, ez = az - oz;
                    const double fx = bx - ox, fy = by - oy, fz = bz - oz;
                    const double cx = ey * fz - ez * fy, cy = ez * fx - ex * fz, cz = ex * fy - ey * fx;
                    if ( ok && b > a ) v += fabs( cx * ox + cy * oy + cz * oz );
                }
            }
        }
#pragma unroll
        for ( int d = 16; d >= 1; d /= 2 ) v += __shfl_xor_sync( PLEIN, v, d );
        return v / 6;
    }
};

/// `liste` : les rangs a faire ( `nullptr` : tous ) ; une cellule qui deborde est comptee dans
/// `deborde` et son rang pousse dans `liste_deb` s'il est la -- la passe suivante, avec plus de
/// cases, les rejoue. `__launch_bounds__( 128, 4 )` plafonne a 128 registres ( quatre blocs par
/// SM ) : 137 registres libres donnaient trois blocs et 247 ns / germe, quatre en donnent 216 ;
/// a 102 ( cinq blocs ) ca deborde en memoire locale et remonte a 251, a 85 ( six ) a 343.
template<bool POIDS, int S, class TK>
__global__ void __launch_bounds__( BLOC3, 4 ) noyau3_voies( Arbre<TK,3> ar, double *res, int *deborde,
                                                          const int *liste, int nl, int *liste_deb ) {
    constexpr int W = BLOC3 / 32;
    const int lane = threadIdx.x & 31, wp = threadIdx.x >> 5;
    const int wi = ( blockIdx.x * blockDim.x + threadIdx.x ) >> 5;
    const int k = liste ? ( wi < nl ? liste[ wi ] : ar.n ) : wi;
    __shared__ int      piles[ W ][ PILE ];
    __shared__ unsigned viv[ W ][ S ];
    __shared__ int      v0[ W ][ 32 * S ];
    if ( k >= ar.n ) return;
    int *pile = piles[ wp ];

    const TK p0[ 3 ] = { ar.c[ 0 ][ k ], ar.c[ 1 ][ k ], ar.c[ 2 ][ k ] };
    const TK w0 = POIDS ? ar.w[ k ] : TK( 0 );
    const int i0 = ar.ids[ k ];

    Cel3V<TK,S> c;
    c.init_cube( lane );
    int haut = 1;
    if ( lane == 0 ) pile[ 0 ] = 0;
    __syncwarp();
    int r = 0;
    while ( haut ) {
        const int h = pile[ --haut ];
        const Noeud<TK,3> nd = ar.nodes[ h ];
        if ( ! c.template peut_couper<POIDS>( nd, p0, w0 ) )
            continue;
        if ( nd.right < 0 ) {
            for ( int q = nd.beg; q < nd.end; ++q ) {
                if ( ar.ids[ q ] == i0 ) continue;
                const Plan3<TK> p = bissect3<POIDS>( ar, q, p0[ 0 ], p0[ 1 ], p0[ 2 ], w0 );
                r = c.coupe( p, viv[ wp ] );
                if ( r == VIDE || r == DEBORDE ) { haut = 0; break; }
            }
            continue;
        }
        const int g = h + 1, dr = nd.right;
        const bool gp = proximite( ar.nodes[ g ], p0 ) <= proximite( ar.nodes[ dr ], p0 );
        if ( lane == 0 ) { pile[ haut ] = gp ? dr : g; pile[ haut + 1 ] = gp ? g : dr; }
        haut += 2;
        __syncwarp();
    }
    const double vol = r == DEBORDE ? 0.0 : c.volume( v0[ wp ] );
    if ( lane == 0 ) {
        if ( r == DEBORDE ) {
            const int q = atomicAdd( deborde, 1 );
            if ( liste_deb ) liste_deb[ q ] = k;
        }
        res[ i0 ] = vol;
    }
}

} // namespace sf::gpu

#pragma once

// =====================================================================================
// L'ARBRE CONSTRUIT SUR LE GPU, 2D, DE BOUT EN BOUT -- rien ne redescend. La mesure disait que
// c'etait devenu le goulot : 306 ms de construction CPU pour 96 ms de mesure GPU a n = 1e6, et
// 5.5 s contre 1.0 a n = 1e7.
//
// LE CPU est recursif ( `accel/AaBsp.h` ) : boite englobante serree, coupe MEDIANE PAR RANG sur
// l'axe le plus long ( `nth_element` ), feuille des que la tranche tient dans `leaf`. Le GPU fait
// la meme chose NIVEAU PAR NIVEAU :
//
//   1. LES BOITES, par atomiques. Chaque germe connait son noeud et pousse sa position dans son
//      min / max. Les flottants passent par un CODAGE ENTIER ORDONNE ( bit de signe retourne,
//      negatifs complementes ) qui rend `atomicMin` / `atomicMax` exacts -- et qui sert AUSSI de
//      clef de tri, gratuitement.
//   2. L'AXE le plus long, un thread par noeud, et la decision feuille / coupe.
//   3. LE TRI. Une coupe mediane par rang, c'est « trier la tranche et prendre le milieu ». Les
//      tranches etant contigues, UN SEUL tri radix global par niveau sur la clef
//      `( beg << 32 ) | coordonnee` les trie toutes a la fois. `beg` et non le numero du noeud :
//      il est constant sur la tranche ET CROISSANT AVEC LA POSITION, donc les tranches ne se
//      melangent pas -- y compris celles des feuilles finies, que leur clef basse nulle et la
//      stabilite du radix laissent en place.
//   4. LES FILS, deux par noeud coupe, sur un compteur atomique.
//
// LE PREORDRE vient apres : tailles de sous-arbre remontees niveau par niveau a l'envers, indices
// redescendus ( gauche = moi + 1, droit = moi + 1 + taille du gauche ).
//
// LE MAJORANT AFFINE ( `accel/WeightMajorant.h` ) suit, en QUATRE PASSES sur les germes, chacune
// une reduction par noeud faite en atomiques :
//   A. les sommes et les extremes ( somme des `w` et des `y`, `w` min / max, boite ) ;
//   B. la matrice normale centree `M` et le second membre `rhs` ;
//      -> un thread par noeud resout le 2x2 et applique les garde-fous de pente ;
//   C. l'etalement des residus `w - a . y`, qui decide d'accepter la pente ou de la jeter ;
//   D. `b = max( w - a . y )` avec les pentes ARRONDIES, plus la marge de quelques ulp.
// Les atomiques flottantes ne somment pas dans le meme ordre que le CPU, donc `a` peut differer
// d'un arrondi : sans importance, `a` est un CHOIX et `b` est calcule APRES lui.
// =====================================================================================

#include "gpu/Arbre.cuh"
#include <cub/cub.cuh>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <algorithm>

namespace sf::gpu {
namespace {                                              // interne : ce fichier est inclus deux fois

/// un flottant en ENTIER ORDONNE : `a < b` equivaut a `ord( a ) < ord( b )`, negatifs compris
__device__ __forceinline__ unsigned ordf( float f ) {
    const unsigned u = __float_as_uint( f );
    return ( u & 0x80000000u ) ? ~u : ( u | 0x80000000u );
}
__device__ __forceinline__ float deordf( unsigned u ) {
    return __uint_as_float( ( u & 0x80000000u ) ? ( u & 0x7fffffffu ) : ~u );
}
__device__ __forceinline__ unsigned long long ordd( double d ) {
    const unsigned long long u = ( unsigned long long ) __double_as_longlong( d );
    return ( u >> 63 ) ? ~u : ( u | 0x8000000000000000ull );
}
__device__ __forceinline__ double deordd( unsigned long long u ) {
    return __longlong_as_double( ( long long ) ( ( u >> 63 ) ? ( u & 0x7fffffffffffffffull ) : ~u ) );
}

/// un noeud en ORDRE DE NIVEAU, avant renumerotation
struct Cru {
    unsigned lo[ 2 ], hi[ 2 ];      ///< la boite, en entiers ordonnes
    int beg, end;
    int gauche, droit;              ///< `-1` : feuille
    int axe;                        ///< `-1` : feuille finie ( clef basse nulle )
};

/// les accumulateurs du majorant, un jeu par noeud
struct Maj {
    double sw, sy[ 2 ];             ///< sommes
    unsigned long long wlo, whi;    ///< `w` min / max, en entiers ordonnes
    double M00, M01, M11, r0, r1;   ///< matrice normale centree et second membre
    unsigned long long rlo, rhi;    ///< etalement des residus
    unsigned long long b, ampl;     ///< `max( w - a . y )` et l'amplitude, pour la marge
    float a[ 2 ];
};

__global__ void k_init_boites( Cru *cru, int deb, int nb ) {
    const int i = deb + blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= deb + nb ) return;
    for ( int d = 0; d < 2; ++d ) { cru[ i ].lo[ d ] = 0xffffffffu; cru[ i ].hi[ d ] = 0u; }
}

__global__ void k_boites( const unsigned *cx, const unsigned *cy, const int *noeud_de, Cru *cru, int n ) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if ( k >= n ) return;
    const int nd = noeud_de[ k ];
    atomicMin( &cru[ nd ].lo[ 0 ], cx[ k ] ); atomicMax( &cru[ nd ].hi[ 0 ], cx[ k ] );
    atomicMin( &cru[ nd ].lo[ 1 ], cy[ k ] ); atomicMax( &cru[ nd ].hi[ 1 ], cy[ k ] );
}

__global__ void k_fend( Cru *cru, int deb, int nb, int leaf, int *compteur ) {
    const int i = deb + blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= deb + nb ) return;
    Cru &c = cru[ i ];
    const float lo0 = deordf( c.lo[ 0 ] ), hi0 = deordf( c.hi[ 0 ] );
    const float lo1 = deordf( c.lo[ 1 ] ), hi1 = deordf( c.hi[ 1 ] );
    const int ax = ( hi1 - lo1 ) > ( hi0 - lo0 ) ? 1 : 0;
    const float etendue = ax ? hi1 - lo1 : hi0 - lo0;
    c.axe = ax;
    if ( c.end - c.beg <= leaf || ! ( etendue > 0 ) ) { c.gauche = c.droit = -1; c.axe = -1; return; }
    const int g = atomicAdd( compteur, 2 );
    c.gauche = g; c.droit = g + 1;
    const int mid = c.beg + ( c.end - c.beg ) / 2;
    cru[ g ].beg = c.beg;     cru[ g ].end = mid;
    cru[ g + 1 ].beg = mid;   cru[ g + 1 ].end = c.end;
}

__global__ void k_clefs( const unsigned *cx, const unsigned *cy, const int *noeud_de, const Cru *cru,
                         unsigned long long *clef, int *val, int n ) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if ( k >= n ) return;
    const Cru &c = cru[ noeud_de[ k ] ];
    const unsigned bas = c.axe < 0 ? 0u : ( c.axe ? cy[ k ] : cx[ k ] );
    clef[ k ] = ( ( unsigned long long ) unsigned( c.beg ) << 32 ) | bas;
    val[ k ] = k;
}

__global__ void k_range( const int *perm, const int *ord0, const unsigned *cx0, const unsigned *cy0,
                         const int *noeud_de0, const Cru *cru,
                         int *ord, unsigned *cx, unsigned *cy, int *noeud_de, int n ) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if ( k >= n ) return;
    const int s = perm[ k ];
    ord[ k ] = ord0[ s ];
    cx[ k ] = cx0[ s ];
    cy[ k ] = cy0[ s ];
    const Cru &c = cru[ noeud_de0[ s ] ];
    if ( c.gauche < 0 ) { noeud_de[ k ] = noeud_de0[ s ]; return; }
    const int mid = c.beg + ( c.end - c.beg ) / 2;
    noeud_de[ k ] = k < mid ? c.gauche : c.droit;
}

__global__ void k_tailles( const Cru *cru, int *taille, int deb, int nb ) {
    const int i = deb + blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= deb + nb ) return;
    taille[ i ] = cru[ i ].gauche < 0 ? 1 : 1 + taille[ cru[ i ].gauche ] + taille[ cru[ i ].droit ];
}

/// le preordre, un niveau a la fois EN DESCENDANT ; `place[ i ]` rend aussi l'indice final
__global__ void k_preordre( const Cru *cru, const int *taille, int *pre, int *place, int deb, int nb ) {
    const int i = deb + blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= deb + nb ) return;
    const Cru &c = cru[ i ];
    const int me = pre[ i ];
    place[ i ] = me;
    if ( c.gauche < 0 ) return;
    pre[ c.gauche ] = me + 1;
    pre[ c.droit ]  = me + 1 + taille[ c.gauche ];
}

/// l'ecriture du noeud final, une fois les places connues et les majorants calcules
template<class TK>
__global__ void k_ecrit( const Cru *cru, const Maj *maj, const int *place, bool poids,
                         Noeud<TK,2> *out, int total ) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= total ) return;
    const Cru &c = cru[ i ];
    Noeud<TK,2> &o = out[ place[ i ] ];
    for ( int d = 0; d < 2; ++d ) {
        // les boites sont calculees en `float` : on les ELARGIT d'un ulp pour qu'elles contiennent
        // a coup sur les positions `double`, sinon l'elagage retrancherait un germe legitime
        o.lo[ d ] = TK( nextafterf( deordf( c.lo[ d ] ), -3e38f ) );
        o.hi[ d ] = TK( nextafterf( deordf( c.hi[ d ] ),  3e38f ) );
        o.a[ d ] = poids ? TK( maj[ i ].a[ d ] ) : TK( 0 );
    }
    o.b = poids ? TK( deordd( maj[ i ].b ) + 8 * 2.220446049250313e-16 * deordd( maj[ i ].ampl ) ) : TK( 0 );
    o.beg = c.beg;
    o.end = c.end;
    o.right = c.gauche < 0 ? -1 : place[ c.droit ];
}

__global__ void k_code( const double *px, const double *py, unsigned *cx, unsigned *cy, int *ord, int n ) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if ( k >= n ) return;
    cx[ k ] = ordf( float( px[ k ] ) );
    cy[ k ] = ordf( float( py[ k ] ) );
    ord[ k ] = k;
}

/// les positions et les poids RELUS EN `double` par la permutation : le tri n'a manipule que des
/// `float`, mais seule la permutation compte. Les codes en virgule fixe suivent ( `Arbre.cuh` ).
template<class TK>
__global__ void k_cueille( const double *px, const double *py, const double *w, const int *ord,
                           TK *cx, TK *cy, TK *cw, int *ux, int *uy, long long *gx, long long *gy, int n ) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if ( k >= n ) return;
    const int i = ord[ k ];
    const double x = px[ i ], y = py[ i ];
    cx[ k ] = TK( x ); cy[ k ] = TK( y );
    if ( cw ) cw[ k ] = TK( w[ i ] );
    ux[ k ] = int( min( max( round( x * double( ECH_FIXE ) ), 0.0 ), double( ECH_FIXE ) ) );
    uy[ k ] = int( min( max( round( y * double( ECH_FIXE ) ), 0.0 ), double( ECH_FIXE ) ) );
    gx[ k ] = ( long long ) llround( min( max( x, 0.0 ), 1.0 ) * double( ECH_F64 ) );
    gy[ k ] = ( long long ) llround( min( max( y, 0.0 ), 1.0 ) * double( ECH_F64 ) );
}

// ---------------------------------------------------------------- LE MAJORANT AFFINE

__global__ void k_maj_init( Maj *maj, int total ) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= total ) return;
    Maj &m = maj[ i ];
    m.sw = m.sy[ 0 ] = m.sy[ 1 ] = 0;
    m.M00 = m.M01 = m.M11 = m.r0 = m.r1 = 0;
    m.wlo = m.rlo = 0xffffffffffffffffull;
    m.whi = m.rhi = 0ull;
    m.b = m.ampl = 0ull;
    m.a[ 0 ] = m.a[ 1 ] = 0;
}

/// PASSE A : sommes et extremes. Un germe appartient a TOUS les noeuds de sa branche ; on remonte
/// donc la branche depuis sa feuille -- `parent` la donne en un saut par niveau.
__global__ void k_maj_a( const double *cw, const double *cx, const double *cy, const int *feuille_de,
                         const int *parent, Maj *maj, int n ) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if ( k >= n ) return;
    const double w = cw[ k ], x = cx[ k ], y = cy[ k ];
    const unsigned long long ow = ordd( w );
    for ( int i = feuille_de[ k ]; i >= 0; i = parent[ i ] ) {
        atomicAdd( &maj[ i ].sw, w );
        atomicAdd( &maj[ i ].sy[ 0 ], x );
        atomicAdd( &maj[ i ].sy[ 1 ], y );
        atomicMin( &maj[ i ].wlo, ow );
        atomicMax( &maj[ i ].whi, ow );
    }
}

/// PASSE B : la matrice normale centree
__global__ void k_maj_b( const double *cw, const double *cx, const double *cy, const int *feuille_de,
                         const int *parent, const Cru *cru, Maj *maj, int n ) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if ( k >= n ) return;
    const double w = cw[ k ], x = cx[ k ], y = cy[ k ];
    for ( int i = feuille_de[ k ]; i >= 0; i = parent[ i ] ) {
        const int m = cru[ i ].end - cru[ i ].beg;
        const double q0 = x - maj[ i ].sy[ 0 ] / m, q1 = y - maj[ i ].sy[ 1 ] / m, qw = w - maj[ i ].sw / m;
        atomicAdd( &maj[ i ].M00, q0 * q0 );
        atomicAdd( &maj[ i ].M01, q0 * q1 );
        atomicAdd( &maj[ i ].M11, q1 * q1 );
        atomicAdd( &maj[ i ].r0, q0 * qw );
        atomicAdd( &maj[ i ].r1, q1 * qw );
    }
}

/// un thread par noeud : le 2x2 et les garde-fous de pente ( `WeightMajorant.h` )
__global__ void k_maj_pente( const Cru *cru, Maj *maj, int total ) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= total ) return;
    Maj &m = maj[ i ];
    const int nb = cru[ i ].end - cru[ i ].beg;
    const double wlo = deordd( m.wlo ), whi = deordd( m.whi );
    const double spread = whi - wlo;
    m.a[ 0 ] = m.a[ 1 ] = 0;
    if ( nb < 6 || ! ( spread > 0 ) ) return;            // `m >= 3 D` avec `D = 2`
    const double det = m.M00 * m.M11 - m.M01 * m.M01;
    if ( ! ( fabs( det ) > 0 ) ) return;
    const double a0 = ( m.r0 * m.M11 - m.r1 * m.M01 ) / det;
    const double a1 = ( m.r1 * m.M00 - m.r0 * m.M01 ) / det;
    const double lo0 = deordf( cru[ i ].lo[ 0 ] ), hi0 = deordf( cru[ i ].hi[ 0 ] );
    const double lo1 = deordf( cru[ i ].lo[ 1 ] ), hi1 = deordf( cru[ i ].hi[ 1 ] );
    const double a[ 2 ] = { a0, a1 }, l[ 2 ] = { lo0, lo1 }, h[ 2 ] = { hi0, hi1 };
    for ( int d = 0; d < 2; ++d ) {
        const double reach = fmax( fabs( l[ d ] ), fabs( h[ d ] ) );
        if ( fabs( a[ d ] ) * ( h[ d ] - l[ d ] ) > 8 * spread || fabs( a[ d ] ) * reach > 100 * spread )
            return;                                      // une pente qui n'explique rien : on la jette
    }
    m.a[ 0 ] = float( a0 );
    m.a[ 1 ] = float( a1 );
}

/// PASSE C : l'etalement des residus avec la pente CANDIDATE
__global__ void k_maj_c( const double *cw, const double *cx, const double *cy, const int *feuille_de,
                         const int *parent, Maj *maj, int n ) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if ( k >= n ) return;
    const double w = cw[ k ], x = cx[ k ], y = cy[ k ];
    for ( int i = feuille_de[ k ]; i >= 0; i = parent[ i ] ) {
        const unsigned long long v = ordd( w - double( maj[ i ].a[ 0 ] ) * x - double( maj[ i ].a[ 1 ] ) * y );
        atomicMin( &maj[ i ].rlo, v );
        atomicMax( &maj[ i ].rhi, v );
    }
}

/// la pente n'est gardee que si elle resserre NETTEMENT plus que le hasard
__global__ void k_maj_garde( const Cru *cru, Maj *maj, int total ) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= total ) return;
    Maj &m = maj[ i ];
    if ( m.a[ 0 ] == 0 && m.a[ 1 ] == 0 ) return;
    const int nb = cru[ i ].end - cru[ i ].beg;
    const double spread = deordd( m.whi ) - deordd( m.wlo );
    const double par_hasard = sqrt( fmax( 0.0, 1.0 - 2.0 / ( nb - 1 ) ) );
    if ( ! ( deordd( m.rhi ) - deordd( m.rlo ) < 0.85 * par_hasard * spread ) )
        m.a[ 0 ] = m.a[ 1 ] = 0;
}

/// PASSE D : `b` avec les pentes ARRONDIES, et l'amplitude pour la marge
__global__ void k_maj_d( const double *cw, const double *cx, const double *cy, const int *feuille_de,
                         const int *parent, Maj *maj, int n ) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if ( k >= n ) return;
    const double w = cw[ k ], x = cx[ k ], y = cy[ k ];
    for ( int i = feuille_de[ k ]; i >= 0; i = parent[ i ] ) {
        const double t0 = double( maj[ i ].a[ 0 ] ) * x, t1 = double( maj[ i ].a[ 1 ] ) * y;
        atomicMax( &maj[ i ].b, ordd( w - t0 - t1 ) );
        atomicMax( &maj[ i ].ampl, ordd( fabs( w ) + fabs( t0 ) + fabs( t1 ) ) );
    }
}

/// le parent de chaque noeud ( `-1` pour la racine ), et la feuille de chaque germe
__global__ void k_parents( const Cru *cru, int *parent, int *feuille_de, int total ) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= total ) return;
    if ( i == 0 ) parent[ 0 ] = -1;
    const Cru &c = cru[ i ];
    if ( c.gauche < 0 ) { for ( int k = c.beg; k < c.end; ++k ) feuille_de[ k ] = i; return; }
    parent[ c.gauche ] = i;
    parent[ c.droit ] = i;
}

/// ce que la construction laisse SUR LE GPU
template<class TK>
struct SortieBsp {
    Noeud<TK,2> *nodes = nullptr;
    int         *ids = nullptr;                          ///< rang -> identifiant
    TK          *c[ 2 ] = {};                            ///< positions permutees
    TK          *w = nullptr;                            ///< poids permutes ( `nullptr` : Voronoi )
    int         *u[ 2 ] = {};                            ///< virgule fixe 32 bits
    long long   *u64[ 2 ] = {};                          ///< virgule fixe 64 bits
    int          n = 0, nn = 0;
};

#define BSP_OK( x ) do { cudaError_t e_ = ( x ); if ( e_ != cudaSuccess ) { \
    std::printf( "CUDA : %s ( %s:%d )\n", cudaGetErrorString( e_ ), __FILE__, __LINE__ ); std::exit( 2 ); } } while ( 0 )

/// LA CONSTRUCTION, de bout en bout sur le GPU. `ms` rend le temps GPU seul.
template<class TK>
void construit2_dev( const double *px_h, const double *py_h, const double *w_h, int n, int leaf,
                     SortieBsp<TK> &s, double *ms ) {
    const int BL = 256;
    auto gr = [ & ]( int m ) { return ( m + BL - 1 ) / BL; };
    const int cap = 4 * ( n / ( leaf > 1 ? leaf : 1 ) + 8 );

    double *dpx, *dpy, *dpw = nullptr, *tx, *ty, *tw = nullptr;
    unsigned *cx, *cy, *cx2, *cy2;
    int *ord, *ord2, *nde, *nde2, *val, *perm, *compteur, *taille, *pre, *place, *parent, *feuille;
    unsigned long long *clef, *clef2;
    Cru *cru;
    Maj *maj = nullptr;
    BSP_OK( cudaMalloc( &dpx, n * 8 ) ); BSP_OK( cudaMalloc( &dpy, n * 8 ) );
    BSP_OK( cudaMalloc( &tx, n * 8 ) );  BSP_OK( cudaMalloc( &ty, n * 8 ) );
    BSP_OK( cudaMalloc( &cx, n * 4 ) );  BSP_OK( cudaMalloc( &cy, n * 4 ) );
    BSP_OK( cudaMalloc( &cx2, n * 4 ) ); BSP_OK( cudaMalloc( &cy2, n * 4 ) );
    BSP_OK( cudaMalloc( &ord, n * 4 ) ); BSP_OK( cudaMalloc( &ord2, n * 4 ) );
    BSP_OK( cudaMalloc( &nde, n * 4 ) ); BSP_OK( cudaMalloc( &nde2, n * 4 ) );
    BSP_OK( cudaMalloc( &val, n * 4 ) ); BSP_OK( cudaMalloc( &perm, n * 4 ) );
    BSP_OK( cudaMalloc( &feuille, n * 4 ) );
    BSP_OK( cudaMalloc( &clef, size_t( n ) * 8 ) ); BSP_OK( cudaMalloc( &clef2, size_t( n ) * 8 ) );
    BSP_OK( cudaMalloc( &cru, size_t( cap ) * sizeof( Cru ) ) );
    BSP_OK( cudaMalloc( &taille, cap * 4 ) ); BSP_OK( cudaMalloc( &pre, cap * 4 ) );
    BSP_OK( cudaMalloc( &place, cap * 4 ) ); BSP_OK( cudaMalloc( &parent, cap * 4 ) );
    BSP_OK( cudaMalloc( &compteur, 4 ) );
    BSP_OK( cudaMemcpy( dpx, px_h, n * 8, cudaMemcpyHostToDevice ) );
    BSP_OK( cudaMemcpy( dpy, py_h, n * 8, cudaMemcpyHostToDevice ) );
    if ( w_h ) {
        BSP_OK( cudaMalloc( &dpw, n * 8 ) ); BSP_OK( cudaMalloc( &tw, n * 8 ) );
        BSP_OK( cudaMemcpy( dpw, w_h, n * 8, cudaMemcpyHostToDevice ) );
        BSP_OK( cudaMalloc( &maj, size_t( cap ) * sizeof( Maj ) ) );
    }

    void *tmp = nullptr; size_t tmp_o = 0;
    cub::DeviceRadixSort::SortPairs( tmp, tmp_o, clef, clef2, val, perm, n );
    BSP_OK( cudaMalloc( &tmp, tmp_o ) );

    cudaEvent_t e0, e1;
    BSP_OK( cudaEventCreate( &e0 ) ); BSP_OK( cudaEventCreate( &e1 ) );
    BSP_OK( cudaEventRecord( e0 ) );

    k_code<<<gr( n ), BL>>>( dpx, dpy, cx, cy, ord, n );
    BSP_OK( cudaMemset( nde, 0, n * 4 ) );
    const int un = 1;
    BSP_OK( cudaMemcpy( compteur, &un, 4, cudaMemcpyHostToDevice ) );
    { const Cru racine{ { 0xffffffffu, 0xffffffffu }, { 0u, 0u }, 0, n, -1, -1, 0 };
      BSP_OK( cudaMemcpy( cru, &racine, sizeof( Cru ), cudaMemcpyHostToDevice ) ); }

    std::vector<int> niv_deb{ 0 }, niv_nb{ 1 };
    int deb = 0, nb = 1, total = 1;
    while ( nb ) {
        k_init_boites<<<gr( nb ), BL>>>( cru, deb, nb );
        k_boites<<<gr( n ), BL>>>( cx, cy, nde, cru, n );
        k_fend<<<gr( nb ), BL>>>( cru, deb, nb, leaf, compteur );
        int apres = 0;
        BSP_OK( cudaMemcpy( &apres, compteur, 4, cudaMemcpyDeviceToHost ) );
        const int nb2 = apres - total;
        if ( nb2 == 0 ) break;
        k_clefs<<<gr( n ), BL>>>( cx, cy, nde, cru, clef, val, n );
        size_t tb = tmp_o;
        cub::DeviceRadixSort::SortPairs( tmp, tb, clef, clef2, val, perm, n );
        k_range<<<gr( n ), BL>>>( perm, ord, cx, cy, nde, cru, ord2, cx2, cy2, nde2, n );
        std::swap( ord, ord2 ); std::swap( cx, cx2 ); std::swap( cy, cy2 ); std::swap( nde, nde2 );
        deb = total; nb = nb2; total = apres;
        niv_deb.push_back( deb ); niv_nb.push_back( nb );
    }

    for ( int l = int( niv_deb.size() ) - 1; l >= 0; --l )
        k_tailles<<<gr( niv_nb[ l ] ), BL>>>( cru, taille, niv_deb[ l ], niv_nb[ l ] );
    BSP_OK( cudaMemset( pre, 0, 4 ) );
    for ( size_t l = 0; l < niv_deb.size(); ++l )
        k_preordre<<<gr( niv_nb[ l ] ), BL>>>( cru, taille, pre, place, niv_deb[ l ], niv_nb[ l ] );

    // ---- ce qui sort, et les codes en virgule fixe
    s.n = n; s.nn = total;
    BSP_OK( cudaMalloc( &s.nodes, size_t( total ) * sizeof( Noeud<TK,2> ) ) );
    BSP_OK( cudaMalloc( &s.ids, n * 4 ) );
    for ( int d = 0; d < 2; ++d ) {
        BSP_OK( cudaMalloc( &s.c[ d ], size_t( n ) * sizeof( TK ) ) );
        BSP_OK( cudaMalloc( &s.u[ d ], n * 4 ) );
        BSP_OK( cudaMalloc( &s.u64[ d ], size_t( n ) * 8 ) );
    }
    if ( w_h ) BSP_OK( cudaMalloc( &s.w, size_t( n ) * sizeof( TK ) ) );
    BSP_OK( cudaMemcpy( s.ids, ord, n * 4, cudaMemcpyDeviceToDevice ) );
    k_cueille<TK><<<gr( n ), BL>>>( dpx, dpy, dpw, ord, s.c[ 0 ], s.c[ 1 ], s.w,
                                    s.u[ 0 ], s.u[ 1 ], s.u64[ 0 ], s.u64[ 1 ], n );

    // ---- LE MAJORANT AFFINE ( Laguerre seulement )
    if ( w_h ) {
        k_cueille<double><<<gr( n ), BL>>>( dpx, dpy, dpw, ord, tx, ty, tw,
                                            s.u[ 0 ], s.u[ 1 ], s.u64[ 0 ], s.u64[ 1 ], n );
        k_parents<<<gr( total ), BL>>>( cru, parent, feuille, total );
        k_maj_init<<<gr( total ), BL>>>( maj, total );
        k_maj_a<<<gr( n ), BL>>>( tw, tx, ty, feuille, parent, maj, n );
        k_maj_b<<<gr( n ), BL>>>( tw, tx, ty, feuille, parent, cru, maj, n );
        k_maj_pente<<<gr( total ), BL>>>( cru, maj, total );
        k_maj_c<<<gr( n ), BL>>>( tw, tx, ty, feuille, parent, maj, n );
        k_maj_garde<<<gr( total ), BL>>>( cru, maj, total );
        k_maj_d<<<gr( n ), BL>>>( tw, tx, ty, feuille, parent, maj, n );
    }
    k_ecrit<TK><<<gr( total ), BL>>>( cru, maj, place, w_h != nullptr, s.nodes, total );

    BSP_OK( cudaEventRecord( e1 ) );
    BSP_OK( cudaEventSynchronize( e1 ) );
    float t = 0;
    BSP_OK( cudaEventElapsedTime( &t, e0, e1 ) );
    if ( ms ) *ms = t;
    BSP_OK( cudaGetLastError() );

    cudaFree( dpx ); cudaFree( dpy ); cudaFree( dpw ); cudaFree( tx ); cudaFree( ty ); cudaFree( tw );
    cudaFree( cx ); cudaFree( cy ); cudaFree( cx2 ); cudaFree( cy2 );
    cudaFree( ord ); cudaFree( ord2 ); cudaFree( nde ); cudaFree( nde2 );
    cudaFree( val ); cudaFree( perm ); cudaFree( feuille );
    cudaFree( clef ); cudaFree( clef2 ); cudaFree( cru ); cudaFree( maj );
    cudaFree( taille ); cudaFree( pre ); cudaFree( place ); cudaFree( parent );
    cudaFree( compteur ); cudaFree( tmp );
    cudaEventDestroy( e0 ); cudaEventDestroy( e1 );
}

} // namespace
} // namespace sf::gpu

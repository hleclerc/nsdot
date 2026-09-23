// =====================================================================================
// L'ARBRE CONSTRUIT SUR LE GPU ( 2D, Voronoi ). La mesure disait que c'etait devenu le goulot :
// 300 ms de construction CPU pour 95 ms de mesure GPU a n = 1e6, et 22 s contre 3 a n = 3e7.
//
// LE CPU est recursif ( `accel/AaBsp.h` ) : boite englobante serree du sous-arbre, coupe MEDIANE
// PAR RANG sur l'axe le plus long ( `nth_element` ), feuille des que la tranche tient dans
// `leaf` ou que la boite est plate. Le GPU ne peut pas recurser sur des tranches irregulieres ;
// il fait la MEME chose NIVEAU PAR NIVEAU :
//
//   1. LES BOITES, par atomiques. Chaque germe connait son noeud ( `noeud_de` ) et pousse sa
//      position dans le min / max de ce noeud. Les flottants passent par un codage ENTIER
//      ORDONNE ( le bit de signe retourne, les negatifs complementes ) qui rend `atomicMin` et
//      `atomicMax` entiers exacts -- et qui sert AUSSI de clef de tri, gratuitement.
//   2. L'AXE le plus long, un thread par noeud, et la decision feuille / coupe.
//   3. LE TRI. Une coupe mediane par rang, c'est « trier la tranche et prendre le milieu ». Les
//      tranches etant deja contigues et rangees par numero de noeud, UN SEUL tri radix global sur
//      la clef `( noeud << 32 ) | coordonnee_ordonnee` trie toutes les tranches a la fois, sans
//      segmentation. C'est le seul poste couteux : `log2( n / leaf )` tris de `n` clefs.
//   4. LES FILS : deux par noeud coupe, pris sur un compteur atomique, et `noeud_de` mis a jour.
//
// LE PREORDRE vient apres : les noeuds sont bâtis en ORDRE DE NIVEAU, puis on remonte les tailles
// de sous-arbre ( niveaux a l'envers ) et on redescend les indices ( fils gauche = moi + 1, fils
// droit = moi + 1 + taille du gauche ). Deux passes triviales, et la sortie a exactement la forme
// que `Arbre.cuh` attend.
//
// CE QUI N'EST PAS FAIT : le majorant affine des poids ( Laguerre ). En Voronoi il vaut zero.
// =====================================================================================

#include "gpu/Arbre.cuh"
#include "gpu/Bsp2D.h"
#include <cub/cub.cuh>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace sf::gpu {

#define CU_OK( x ) do { cudaError_t e_ = ( x ); if ( e_ != cudaSuccess ) { \
    std::fprintf( stderr, "CUDA : %s ( %s:%d )\n", cudaGetErrorString( e_ ), __FILE__, __LINE__ ); std::exit( 2 ); } } while ( 0 )

namespace {

/// un flottant en ENTIER ORDONNE : `a < b` equivaut a `ord( a ) < ord( b )`, y compris les
/// negatifs. Sert aux atomiques de boite ET de clef de tri.
__device__ __forceinline__ unsigned ordf( float f ) {
    unsigned u = __float_as_uint( f );
    return ( u & 0x80000000u ) ? ~u : ( u | 0x80000000u );
}
__device__ __forceinline__ float deordf( unsigned u ) {
    return __uint_as_float( ( u & 0x80000000u ) ? ( u & 0x7fffffffu ) : ~u );
}

/// un noeud en ORDRE DE NIVEAU, avant renumerotation
struct Cru {
    unsigned lo[ 2 ], hi[ 2 ];      ///< la boite, en entiers ordonnes
    int beg, end;
    int gauche, droit;              ///< `-1` : feuille
    int axe;
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

/// l'axe le plus long et la decision ; les noeuds coupes prennent deux fils sur `compteur`
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

/// la clef de tri : le noeud en haut, la coordonnee de SON axe en bas
__global__ void k_clefs( const unsigned *cx, const unsigned *cy, const int *noeud_de, const Cru *cru,
                         unsigned long long *clef, int *val, int n ) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if ( k >= n ) return;
    const int nd = noeud_de[ k ];
    const Cru &c = cru[ nd ];
    // `beg` et non le numero du noeud : il est le meme pour toute la tranche ET CROISSANT AVEC LA
    // POSITION, donc le tri global ne melange pas les tranches -- y compris celles des feuilles
    // deja finies, que leur coordonnee nulle et la stabilite du tri laissent en place.
    const unsigned bas = c.gauche < 0 && c.axe < 0 ? 0u : ( c.axe ? cy[ k ] : cx[ k ] );
    clef[ k ] = ( ( unsigned long long ) unsigned( c.beg ) << 32 ) | bas;
    val[ k ] = k;
}

/// apres le tri : la permutation et les positions suivent, et chaque germe descend chez son fils
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
    if ( c.gauche < 0 ) { noeud_de[ k ] = noeud_de0[ s ]; return; }        // feuille : elle ne bouge plus
    const int mid = c.beg + ( c.end - c.beg ) / 2;
    noeud_de[ k ] = k < mid ? c.gauche : c.droit;
}

/// les tailles de sous-arbre, un niveau a la fois EN REMONTANT
__global__ void k_tailles( const Cru *cru, int *taille, int deb, int nb ) {
    const int i = deb + blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= deb + nb ) return;
    taille[ i ] = cru[ i ].gauche < 0 ? 1 : 1 + taille[ cru[ i ].gauche ] + taille[ cru[ i ].droit ];
}

/// les indices de preordre, un niveau a la fois EN DESCENDANT, puis l'ecriture du noeud final
__global__ void k_preordre( const Cru *cru, const int *taille, int *pre, Noeud<float,2> *out,
                            int deb, int nb ) {
    const int i = deb + blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= deb + nb ) return;
    const Cru &c = cru[ i ];
    const int me = pre[ i ];
    Noeud<float,2> &o = out[ me ];
    for ( int d = 0; d < 2; ++d ) {
        o.lo[ d ] = deordf( c.lo[ d ] );
        o.hi[ d ] = deordf( c.hi[ d ] );
        o.a[ d ] = 0;
    }
    o.b = 0;
    o.beg = c.beg;
    o.end = c.end;
    if ( c.gauche < 0 ) { o.right = -1; return; }
    pre[ c.gauche ] = me + 1;
    pre[ c.droit ]  = me + 1 + taille[ c.gauche ];
    o.right = me + 1 + taille[ c.gauche ];
}

__global__ void k_code( const double *px, const double *py, unsigned *cx, unsigned *cy, int *ord, int n ) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if ( k >= n ) return;
    cx[ k ] = ordf( float( px[ k ] ) );
    cy[ k ] = ordf( float( py[ k ] ) );
    ord[ k ] = k;
}

__global__ void k_decode( const unsigned *cx, const unsigned *cy, float *px, float *py, int n ) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if ( k >= n ) return;
    px[ k ] = deordf( cx[ k ] );
    py[ k ] = deordf( cy[ k ] );
}

} // namespace

void construit2( const double *px_h, const double *py_h, int n, int leaf, ArbreHote &res, double *ms ) {
    const int BL = 256;
    auto grille = [ & ]( int m ) { return ( m + BL - 1 ) / BL; };
    const int cap = 4 * ( n / ( leaf > 1 ? leaf : 1 ) + 8 );   // large : 2 noeuds par feuille

    double *dpx, *dpy;
    unsigned *cx, *cy, *cx2, *cy2;
    int *ord, *ord2, *noeud_de, *noeud_de2, *val, *perm, *compteur, *taille, *pre;
    unsigned long long *clef, *clef2;
    Cru *cru;
    Noeud<float,2> *out;
    CU_OK( cudaMalloc( &dpx, n * sizeof( double ) ) );  CU_OK( cudaMalloc( &dpy, n * sizeof( double ) ) );
    CU_OK( cudaMalloc( &cx, n * 4 ) );  CU_OK( cudaMalloc( &cy, n * 4 ) );
    CU_OK( cudaMalloc( &cx2, n * 4 ) ); CU_OK( cudaMalloc( &cy2, n * 4 ) );
    CU_OK( cudaMalloc( &ord, n * 4 ) ); CU_OK( cudaMalloc( &ord2, n * 4 ) );
    CU_OK( cudaMalloc( &noeud_de, n * 4 ) ); CU_OK( cudaMalloc( &noeud_de2, n * 4 ) );
    CU_OK( cudaMalloc( &val, n * 4 ) ); CU_OK( cudaMalloc( &perm, n * 4 ) );
    CU_OK( cudaMalloc( &clef, size_t( n ) * 8 ) ); CU_OK( cudaMalloc( &clef2, size_t( n ) * 8 ) );
    CU_OK( cudaMalloc( &cru, size_t( cap ) * sizeof( Cru ) ) );
    CU_OK( cudaMalloc( &out, size_t( cap ) * sizeof( Noeud<float,2> ) ) );
    CU_OK( cudaMalloc( &taille, cap * 4 ) ); CU_OK( cudaMalloc( &pre, cap * 4 ) );
    CU_OK( cudaMalloc( &compteur, 4 ) );

    CU_OK( cudaMemcpy( dpx, px_h, n * sizeof( double ), cudaMemcpyHostToDevice ) );
    CU_OK( cudaMemcpy( dpy, py_h, n * sizeof( double ), cudaMemcpyHostToDevice ) );

    void *tmp = nullptr;
    size_t tmp_o = 0;
    cub::DeviceRadixSort::SortPairs( tmp, tmp_o, clef, clef2, val, perm, n );
    CU_OK( cudaMalloc( &tmp, tmp_o ) );

    cudaEvent_t e0, e1;
    CU_OK( cudaEventCreate( &e0 ) ); CU_OK( cudaEventCreate( &e1 ) );
    CU_OK( cudaEventRecord( e0 ) );

    k_code<<<grille( n ), BL>>>( dpx, dpy, cx, cy, ord, n );
    CU_OK( cudaMemset( noeud_de, 0, n * 4 ) );
    const int un = 1;
    CU_OK( cudaMemcpy( compteur, &un, 4, cudaMemcpyHostToDevice ) );
    { const Cru racine{ { 0xffffffffu, 0xffffffffu }, { 0u, 0u }, 0, n, -1, -1, 0 };
      CU_OK( cudaMemcpy( cru, &racine, sizeof( Cru ), cudaMemcpyHostToDevice ) ); }

    // ---- LES NIVEAUX
    std::vector<int> niv_deb{ 0 }, niv_nb{ 1 };
    int deb = 0, nb = 1, total = 1;
    while ( nb ) {
        k_init_boites<<<grille( nb ), BL>>>( cru, deb, nb );
        k_boites<<<grille( n ), BL>>>( cx, cy, noeud_de, cru, n );
        k_fend<<<grille( nb ), BL>>>( cru, deb, nb, leaf, compteur );

        int apres = 0;
        CU_OK( cudaMemcpy( &apres, compteur, 4, cudaMemcpyDeviceToHost ) );
        const int nb2 = apres - total;
        if ( nb2 == 0 ) break;

        k_clefs<<<grille( n ), BL>>>( cx, cy, noeud_de, cru, clef, val, n );
        size_t tmp_b = tmp_o;
        cub::DeviceRadixSort::SortPairs( tmp, tmp_b, clef, clef2, val, perm, n );
        k_range<<<grille( n ), BL>>>( perm, ord, cx, cy, noeud_de, cru, ord2, cx2, cy2, noeud_de2, n );
        std::swap( ord, ord2 ); std::swap( cx, cx2 ); std::swap( cy, cy2 ); std::swap( noeud_de, noeud_de2 );

        deb = total; nb = nb2; total = apres;
        niv_deb.push_back( deb ); niv_nb.push_back( nb );
    }

    // ---- LES TAILLES DE SOUS-ARBRE, en remontant
    for ( int l = int( niv_deb.size() ) - 1; l >= 0; --l )
        k_tailles<<<grille( niv_nb[ l ] ), BL>>>( cru, taille, niv_deb[ l ], niv_nb[ l ] );
    // ---- LE PREORDRE, en descendant
    CU_OK( cudaMemset( pre, 0, 4 ) );
    for ( size_t l = 0; l < niv_deb.size(); ++l )
        k_preordre<<<grille( niv_nb[ l ] ), BL>>>( cru, taille, pre, out, niv_deb[ l ], niv_nb[ l ] );

    float *fx, *fy;
    CU_OK( cudaMalloc( &fx, n * 4 ) ); CU_OK( cudaMalloc( &fy, n * 4 ) );
    k_decode<<<grille( n ), BL>>>( cx, cy, fx, fy, n );

    CU_OK( cudaEventRecord( e1 ) );
    CU_OK( cudaEventSynchronize( e1 ) );
    float t = 0;
    CU_OK( cudaEventElapsedTime( &t, e0, e1 ) );
    if ( ms ) *ms = t;
    CU_OK( cudaGetLastError() );

    std::vector<Noeud<float,2>> nds( total );
    CU_OK( cudaMemcpy( nds.data(), out, size_t( total ) * sizeof( Noeud<float,2> ), cudaMemcpyDeviceToHost ) );
    res.nn = total; res.n = n;
    res.lo.resize( 2 * total ); res.hi.resize( 2 * total );
    res.beg.resize( total ); res.end.resize( total ); res.right.resize( total );
    for ( int i = 0; i < total; ++i ) {
        for ( int d = 0; d < 2; ++d ) { res.lo[ 2 * i + d ] = nds[ i ].lo[ d ]; res.hi[ 2 * i + d ] = nds[ i ].hi[ d ]; }
        res.beg[ i ] = nds[ i ].beg; res.end[ i ] = nds[ i ].end; res.right[ i ] = nds[ i ].right;
    }
    res.order.resize( n ); res.px.resize( n ); res.py.resize( n );
    CU_OK( cudaMemcpy( res.order.data(), ord, n * 4, cudaMemcpyDeviceToHost ) );
    CU_OK( cudaMemcpy( res.px.data(), fx, n * 4, cudaMemcpyDeviceToHost ) );
    CU_OK( cudaMemcpy( res.py.data(), fy, n * 4, cudaMemcpyDeviceToHost ) );

    cudaFree( dpx ); cudaFree( dpy ); cudaFree( cx ); cudaFree( cy ); cudaFree( cx2 ); cudaFree( cy2 );
    cudaFree( ord ); cudaFree( ord2 ); cudaFree( noeud_de ); cudaFree( noeud_de2 );
    cudaFree( val ); cudaFree( perm ); cudaFree( clef ); cudaFree( clef2 );
    cudaFree( cru ); cudaFree( out ); cudaFree( taille ); cudaFree( pre ); cudaFree( compteur );
    cudaFree( tmp ); cudaFree( fx ); cudaFree( fy );
    cudaEventDestroy( e0 ); cudaEventDestroy( e1 );
}

} // namespace sf::gpu

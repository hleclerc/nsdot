#pragma once

// =====================================================================================
// UN MULTIGRILLE ALGEBRIQUE MAISON, sur la carte. Le CG a preconditionneur diagonal converge en
// `sqrt( n )` iterations ( 8167 a n = 1e6 ) et pese 99 % de l'iteration de Newton ; il lui faut
// une hierarchie.
//
// L'AGREGATION EST GRATUITE, et c'est le point. Les germes sont ranges DANS L'ORDRE DE L'ARBRE,
// qui est une courbe remplissante : des RANGS CONSECUTIFS sont voisins dans le plan. Agreger,
// c'est donc `rang >> 2` -- quatre germes par paquet, sans un noyau d'appariement, sans matching,
// sans compaction. Et comme les indices d'agregat restent ordonnes par rang, le niveau suivant
// s'agrege pareil : `a >> 2`. La hierarchie entiere tient dans un decalage.
// ( Le niveau fin est indexe par IDENTIFIANT ; on passe par `rang_de` une seule fois. )
//
// LE GROSSIER est le produit de Galerkin `A_c = P^T A P` avec `P` constant par morceaux. Sur un
// laplacien de graphe c'est encore un laplacien : il suffit de sommer les poids d'aretes entre
// paquets, `c_ab = somme des c_ij pour i dans a, j dans b`, et la diagonale est la somme de la
// ligne. On emet donc un triplet par arete, on TRIE ( CUB ), on REDUIT PAR CLEF, et le CSR sort
// de la.
//
// LE CYCLE EN V : un lissage de Jacobi amorti avant, un apres ( meme `omega`, donc l'operateur
// est SYMETRIQUE et le CG l'accepte comme preconditionneur ), restriction par somme sur le
// paquet, prolongation par diffusion. Le niveau le plus grossier -- moins de mille inconnues --
// est lisse cent fois.
//
// LA JAUGE passe de `x[ 0 ] = 0` a MOYENNE NULLE, qui est la bonne pour un multigrille : le
// laplacien a les constantes pour noyau, `b = mesures - cible` est deja de somme nulle, et
// projeter est symetrique la ou rayer une ligne ne l'est pas. Les deux jauges donnent la meme
// direction de Newton a une constante pres.
// =====================================================================================

#include <cub/cub.cuh>

namespace sf::gpu {
namespace {

/// un niveau de la hierarchie ( laplacien : hors-diagonaux positifs, diagonale = somme de ligne )
struct Niveau {
    int    *row = nullptr, *col = nullptr;
    double *val = nullptr, *dia = nullptr;
    double *x = nullptr, *b = nullptr, *r = nullptr;     ///< les vecteurs de travail du cycle
    double *v1 = nullptr, *v2 = nullptr, *t = nullptr, *rc = nullptr;   ///< ceux du K-cycle
    double *sc = nullptr;                                ///< huit scalaires, sur la carte
    int     n = 0, nnz = 0;
};

/// `m[ i ]` : le paquet du niveau suivant. Au niveau fin on passe par le rang, ensuite non.
__global__ void k_amg_map_fin( const int *rang_de, int *m, int n ) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i < n ) m[ i ] = rang_de[ i ] >> 2;
}
__global__ void k_amg_map( int *m, int n ) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i < n ) m[ i ] = i >> 2;
}

/// un triplet par arete inter-paquets : la clef porte `( a, b )`, la valeur le poids
__global__ void k_amg_triples( const int *row, const int *col, const double *val, const int *m,
                               int n, int nc, unsigned long long *clef, double *poids, int *cpt ) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= n ) return;
    const int a = m[ i ];
    for ( int p = row[ i ]; p < row[ i + 1 ]; ++p ) {
        const int b = m[ col[ p ] ];
        if ( a == b ) continue;                          // l'interieur du paquet disparait
        const int q = atomicAdd( cpt, 1 );
        clef[ q ] = ( unsigned long long ) a * nc + b;
        poids[ q ] = val[ p ];
    }
}

/// du tableau de clefs uniques ( trie ) au CSR : compter, puis placer
__global__ void k_amg_compte( const unsigned long long *clef, int nu, int nc, int *cnt ) {
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if ( q < nu ) atomicAdd( &cnt[ int( clef[ q ] / nc ) ], 1 );
}
__global__ void k_amg_place( const unsigned long long *clef, const double *poids, int nu, int nc,
                             const int *row, int *at, int *col, double *val ) {
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if ( q >= nu ) return;
    const int a = int( clef[ q ] / nc );
    const int p = row[ a ] + atomicAdd( &at[ a ], 1 );
    col[ p ] = int( clef[ q ] % nc );
    val[ p ] = poids[ q ];
}
__global__ void k_amg_dia( const int *row, const double *val, double *dia, int n ) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= n ) return;
    double s = 0;
    for ( int p = row[ i ]; p < row[ i + 1 ]; ++p ) s += val[ p ];
    dia[ i ] = s > 0 ? s : 1.0;
}

/// `r = b - A x` ( laplacien : diagonale moins les hors-diagonaux )
__global__ void k_amg_residu( const int *row, const int *col, const double *val, const double *dia,
                              const double *x, const double *b, double *r, int n ) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= n ) return;
    double s = dia[ i ] * x[ i ];
    for ( int p = row[ i ]; p < row[ i + 1 ]; ++p ) s -= val[ p ] * x[ col[ p ] ];
    r[ i ] = b[ i ] - s;
}
/// Jacobi amorti : `x += omega D^-1 ( b - A x )`
__global__ void k_amg_jacobi( const int *row, const int *col, const double *val, const double *dia,
                              double *x, const double *b, double omega, int n ) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= n ) return;
    double s = dia[ i ] * x[ i ];
    for ( int p = row[ i ]; p < row[ i + 1 ]; ++p ) s -= val[ p ] * x[ col[ p ] ];
    x[ i ] += omega * ( b[ i ] - s ) / dia[ i ];
}

__global__ void k_amg_restreint( const double *r, const int *m, double *bc, int n ) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i < n ) atomicAdd( &bc[ m[ i ] ], r[ i ] );
}
__global__ void k_amg_prolonge( double *x, const double *xc, const int *m, int n ) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i < n ) x[ i ] += xc[ m[ i ] ];
}

/// la somme d'un vecteur ( warp, bloc, un atomique ) -- pour la projection
__global__ void k_amg_somme( const double *u, double *acc, int n ) {
    __shared__ double part[ 32 ];
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    double s = i < n ? u[ i ] : 0.0;
#pragma unroll
    for ( int d = 16; d; d >>= 1 ) s += __shfl_down_sync( 0xffffffffu, s, d );
    const int voie = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if ( voie == 0 ) part[ warp ] = s;
    __syncthreads();
    if ( warp == 0 ) {
        s = voie < ( blockDim.x >> 5 ) ? part[ voie ] : 0.0;
#pragma unroll
        for ( int d = 16; d; d >>= 1 ) s += __shfl_down_sync( 0xffffffffu, s, d );
        if ( voie == 0 ) atomicAdd( acc, s );
    }
}

/// `rang_de` : identifiant -> rang dans l'arbre
__global__ void k_amg_rang( const int *ids, int *rang_de, int n ) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if ( k < n ) rang_de[ ids[ k ] ] = k;
}

// ---------------------------------------------------------------- LE K-CYCLE
//
// Le defaut mesure de l'agregation non lissee est que LA CORRECTION GROSSIERE EST TROP FAIBLE et
// que l'erreur s'accumule d'un niveau a l'autre ( descendre a 16 inconnues au lieu de 1000 faisait
// passer de 168 a 411 iterations ). Le K-cycle y repond non pas en changeant `P` -- ce que fait la
// prolongation lissee, au prix d'un vrai produit triple creux -- mais en ACCELERANT CHAQUE NIVEAU
// PAR KRYLOV : au lieu d'un appel recursif, DEUX pas d'un gradient conjugue sur le systeme
// grossier, dont le preconditionneur est le niveau d'en dessous.
//
// Les coefficients restent sur la carte ( un thread les calcule ) : sans ca, chaque niveau de
// chaque cycle couterait une synchronisation.

__global__ void k_kc_c1( const double *rho1, const double *a1, double *c1 ) {
    *c1 = *rho1 != 0 ? *a1 / *rho1 : 0.0;
}
/// `rc = b - c1 t`, le residu apres le premier pas
__global__ void k_kc_rc( double *rc, const double *b, const double *t, const double *c1, int n ) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i < n ) rc[ i ] = b[ i ] - *c1 * t[ i ];
}
/// les coefficients du second pas : `rho2 = b2 - g2^2 / rho1`, et la correction de `c1`
__global__ void k_kc_c2( const double *rho1, const double *a1, const double *g2, const double *b2,
                         const double *a2, double *c1, double *c2 ) {
    const double r1 = *rho1, r2 = *b2 - ( *g2 ) * ( *g2 ) / ( r1 != 0 ? r1 : 1.0 );
    *c2 = r2 != 0 ? *a2 / r2 : 0.0;
    *c1 = ( r1 != 0 ? *a1 / r1 : 0.0 ) - ( r1 != 0 && r2 != 0 ? ( *g2 ) * ( *a2 ) / ( r1 * r2 ) : 0.0 );
}
__global__ void k_kc_comb( double *x, const double *v1, const double *v2,
                           const double *c1, const double *c2, int n ) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i < n ) x[ i ] = *c1 * v1[ i ] + *c2 * v2[ i ];
}
__global__ void k_amg_matvec( const int *row, const int *col, const double *val, const double *dia,
                              const double *x, double *y, int n ) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= n ) return;
    double s = dia[ i ] * x[ i ];
    for ( int p = row[ i ]; p < row[ i + 1 ]; ++p ) s -= val[ p ] * x[ col[ p ] ];
    y[ i ] = s;
}
__global__ void k_amg_copie( double *dst, const double *src, int n ) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i < n ) dst[ i ] = src[ i ];
}

/// la projection sur la moyenne nulle : le noyau du laplacien
__global__ void k_amg_centre( double *x, const double *somme, int n ) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i < n ) x[ i ] -= *somme / n;
}

} // namespace
} // namespace sf::gpu

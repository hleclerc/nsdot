#pragma once

// =====================================================================================
// LE GRADIENT CONJUGUE PRECONDITIONNE, sur la carte.
//
// LA JAUGE. Le laplacien a les CONSTANTES pour noyau -- ajouter la meme chose a tous les poids ne
// change aucune cellule -- donc il est singulier. Le CPU raye la ligne et la colonne zero
// ( `solver/Laplacien.h` : `crs_reduit` ) ; ici on fait la meme chose SANS RIEN RECOPIER : le
// produit saute la colonne zero et rend zero sur la ligne zero. L'operateur est alors defini
// positif sur le sous-espace `{ x : x[ 0 ] = 0 }`, et comme `b[ 0 ] = 0` et `x = 0` au depart,
// TOUS les vecteurs du CG y restent -- residu, direction et preconditionne compris.
//
// LE PRECONDITIONNEUR est Jacobi : la diagonale est deja assemblee, et pour un laplacien de graphe
// presque planaire elle capture l'essentiel de l'echelle. ( Le CPU fait mieux avec un multigrille
// algebrique -- c'est le bon outil pour ce systeme -- mais il n'a pas sa place dans un premier
// jet GPU : voir `doc/06-ce-qui-reste.md`. )
//
// LES REDUCTIONS se font en un seul noyau : reduction dans le warp par `__shfl_down`, puis une
// case de memoire partagee par warp, puis UN atomique par bloc. Le produit scalaire coute donc
// une lecture de chaque vecteur et rien d'autre.
// =====================================================================================

namespace sf::gpu {
namespace {

/// `y = A x` avec la jauge : la colonne zero est sautee, la ligne zero rend zero
__global__ void k_cg_mul( const int *row, const int *col, const double *val, const double *dia,
                          const double *x, double *y, int n ) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= n ) return;
    if ( i == 0 ) { y[ 0 ] = 0; return; }
    double s = dia[ i ] * x[ i ];
    for ( int p = row[ i ]; p < row[ i + 1 ]; ++p ) {
        const int j = col[ p ];
        if ( j ) s -= val[ p ] * x[ j ];                 // la colonne zero est rayee
    }
    y[ i ] = s;
}

/// `<u, v>` : warp, puis bloc, puis un seul atomique
__global__ void k_cg_dot( const double *u, const double *v, double *acc, int n ) {
    __shared__ double part[ 32 ];
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    double s = i < n ? u[ i ] * v[ i ] : 0.0;
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

/// `x += a u` et `r -= a q`, en un noyau ( les deux marchent du meme pas )
__global__ void k_cg_avance( double *x, double *r, const double *p, const double *q, const double *a, int n ) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= n ) return;
    const double al = *a;
    x[ i ] += al * p[ i ];
    r[ i ] -= al * q[ i ];
}

/// `z = r / dia` ( Jacobi ), et `p = z + beta p`
__global__ void k_cg_prec( const double *r, const double *dia, double *z, int n ) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= n ) return;
    z[ i ] = i ? r[ i ] / dia[ i ] : 0.0;
}
__global__ void k_cg_dir( double *p, const double *z, const double *beta, int n ) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= n ) return;
    p[ i ] = z[ i ] + *beta * p[ i ];
}

/// les scalaires restent SUR LA CARTE : un thread fait la division et la remise a zero, ce qui
/// evite une synchronisation par iteration ( seul le test d'arret en demande une )
__global__ void k_cg_alpha( const double *rz, const double *pq, double *alpha ) {
    *alpha = *pq != 0 ? *rz / *pq : 0.0;
}
__global__ void k_cg_beta( const double *rz2, const double *rz, double *beta ) {
    *beta = *rz != 0 ? *rz2 / *rz : 0.0;
}
__global__ void k_cg_copie( double *dst, const double *src ) { *dst = *src; }
__global__ void k_cg_zero( double *a ) { *a = 0; }

} // namespace
} // namespace sf::gpu

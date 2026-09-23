#pragma once

// =====================================================================================
// L'AGREGATION LISSEE, sur la carte, avec `cusparseSpGEMM`.
//
// POURQUOI. Notre prolongation est une MARCHE D'ESCALIER : tous les germes d'un paquet recoivent
// la meme correction. Or c'est l'erreur LISSE qu'un multigrille doit corriger, et on la corrige
// par quelque chose de discontinu -- d'ou le besoin de lisser beaucoup ( deux passes avant, deux
// apres, la ou AMGCL en fait une ) et un cycle deux fois plus cher que le sien.
//
// LE REMEDE : `P^ = P - omega D^-1 A P`. Un pas de Jacobi applique a l'OPERATEUR D'INTERPOLATION
// lui-meme, qui arrondit les marches. `P` a une entree par ligne ( le paquet du germe ), `P^` en a
// trois a cinq ( son paquet et les paquets voisins ), et le grossier devient `P^t A P^` : un
// produit de trois matrices creuses.
//
// POURQUOI CA TIENT SUR GPU. Fait a la main -- emettre des triplets, trier, reduire -- ce produit
// donnerait ~16 triplets par non-zero de `A`, soit 1.8 Go a trier a n = 1e6. `cusparseSpGEMM` le
// fait a notre place, et `cusparseSpGEMMreuse` sait meme garder le MOTIF entre deux appels : dans
// un Newton les positions ne bougent pas, donc le motif non plus -- seules les valeurs changent.
// ( AmgX fait la meme chose depuis des annees : ce n'est pas le GPU qui s'y prete mal, c'est
// AMGCL qui monte sa hierarchie sur le CPU. )
//
// LE DETAIL QUI EPARGNE UNE ADDITION CREUSE : le motif de `P` est INCLUS dans celui de `A P`
// ( la ligne `i` de `A P` touche le paquet de chaque voisin de `i`, dont `i` lui-meme ). On calcule
// donc `A P`, on multiplie tout par `-omega / d_i`, et on ajoute un a la seule entree qui tombe
// sur le paquet de `i`. Pas de somme de matrices.
// =====================================================================================

#include <cusparse.h>

namespace sf::gpu {
namespace {

#define CUS_OK( x ) do { cusparseStatus_t s_ = ( x ); if ( s_ != CUSPARSE_STATUS_SUCCESS ) { \
    std::printf( "cuSPARSE %d ( %s:%d )\n", int( s_ ), __FILE__, __LINE__ ); std::exit( 2 ); } } while ( 0 )

/// une matrice creuse en CSR, possedee
struct Csr {
    int  lignes = 0, colonnes = 0, nnz = 0;
    int    *row = nullptr, *col = nullptr;
    double *val = nullptr;
};

void csr_libere( Csr &a ) {
    cudaFree( a.row ); cudaFree( a.col ); cudaFree( a.val );
    a = Csr{};
}

cusparseSpMatDescr_t descr( const Csr &a ) {
    cusparseSpMatDescr_t d;
    CUS_OK( cusparseCreateCsr( &d, a.lignes, a.colonnes, a.nnz,
                               ( void * ) a.row, ( void * ) a.col, ( void * ) a.val,
                               CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F ) );
    return d;
}

/// `c = a b`, motif et valeurs
void spgemm( cusparseHandle_t h, const Csr &a, const Csr &b, Csr &c ) {
    const double un = 1.0, zero = 0.0;
    cusparseSpMatDescr_t da = descr( a ), db = descr( b ), dc;
    c.lignes = a.lignes; c.colonnes = b.colonnes;
    cudaMalloc( &c.row, size_t( c.lignes + 1 ) * sizeof( int ) );
    CUS_OK( cusparseCreateCsr( &dc, c.lignes, c.colonnes, 0, c.row, nullptr, nullptr,
                               CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F ) );
    cusparseSpGEMMDescr_t sd;
    CUS_OK( cusparseSpGEMM_createDescr( &sd ) );
    const auto NT = CUSPARSE_OPERATION_NON_TRANSPOSE;

    size_t t1 = 0, t2 = 0;
    void *b1 = nullptr, *b2 = nullptr;
    CUS_OK( cusparseSpGEMM_workEstimation( h, NT, NT, &un, da, db, &zero, dc, CUDA_R_64F,
                                           CUSPARSE_SPGEMM_DEFAULT, sd, &t1, nullptr ) );
    cudaMalloc( &b1, t1 );
    CUS_OK( cusparseSpGEMM_workEstimation( h, NT, NT, &un, da, db, &zero, dc, CUDA_R_64F,
                                           CUSPARSE_SPGEMM_DEFAULT, sd, &t1, b1 ) );
    CUS_OK( cusparseSpGEMM_compute( h, NT, NT, &un, da, db, &zero, dc, CUDA_R_64F,
                                    CUSPARSE_SPGEMM_DEFAULT, sd, &t2, nullptr ) );
    cudaMalloc( &b2, t2 );
    CUS_OK( cusparseSpGEMM_compute( h, NT, NT, &un, da, db, &zero, dc, CUDA_R_64F,
                                    CUSPARSE_SPGEMM_DEFAULT, sd, &t2, b2 ) );

    int64_t nl = 0, nc = 0, nz = 0;
    CUS_OK( cusparseSpMatGetSize( dc, &nl, &nc, &nz ) );
    c.nnz = int( nz );
    cudaMalloc( &c.col, size_t( c.nnz ) * sizeof( int ) );
    cudaMalloc( &c.val, size_t( c.nnz ) * sizeof( double ) );
    CUS_OK( cusparseCsrSetPointers( dc, c.row, c.col, c.val ) );
    CUS_OK( cusparseSpGEMM_copy( h, NT, NT, &un, da, db, &zero, dc, CUDA_R_64F,
                                 CUSPARSE_SPGEMM_DEFAULT, sd ) );

    CUS_OK( cusparseSpGEMM_destroyDescr( sd ) );
    cusparseDestroySpMat( da ); cusparseDestroySpMat( db ); cusparseDestroySpMat( dc );
    cudaFree( b1 ); cudaFree( b2 );
}

__global__ void k_lis_gthr( const double *src, double *dst, const int *perm, int nnz );

/// LES COLONNES TRIEES, en place. `cusparseSpGEMM` les veut ainsi, et le niveau fin sort
/// d'atomiques, donc en desordre.
void trie( cusparseHandle_t h, Csr &a ) {
    cusparseMatDescr_t md;
    CUS_OK( cusparseCreateMatDescr( &md ) );
    size_t tb = 0;
    CUS_OK( cusparseXcsrsort_bufferSizeExt( h, a.lignes, a.colonnes, a.nnz, a.row, a.col, &tb ) );
    void *buf = nullptr;
    int *perm = nullptr;
    double *v2 = nullptr;
    cudaMalloc( &buf, tb );
    cudaMalloc( &perm, size_t( a.nnz ) * sizeof( int ) );
    cudaMalloc( &v2, size_t( a.nnz ) * sizeof( double ) );
    CUS_OK( cusparseCreateIdentityPermutation( h, a.nnz, perm ) );
    CUS_OK( cusparseXcsrsort( h, a.lignes, a.colonnes, a.nnz, md, a.row, a.col, perm, buf ) );
    k_lis_gthr<<<( a.nnz + 255 ) / 256, 256>>>( a.val, v2, perm, a.nnz );
    cudaFree( a.val );
    a.val = v2;
    cudaFree( buf ); cudaFree( perm );
    cusparseDestroyMatDescr( md );
}

/// `t = a^T`, par la conversion CSR -> CSC ( qui EST la transposee )
void transpose( cusparseHandle_t h, const Csr &a, Csr &t ) {
    t.lignes = a.colonnes; t.colonnes = a.lignes; t.nnz = a.nnz;
    cudaMalloc( &t.row, size_t( t.lignes + 1 ) * sizeof( int ) );
    cudaMalloc( &t.col, size_t( t.nnz ) * sizeof( int ) );
    cudaMalloc( &t.val, size_t( t.nnz ) * sizeof( double ) );
    size_t tb = 0;
    CUS_OK( cusparseCsr2cscEx2_bufferSize( h, a.lignes, a.colonnes, a.nnz, a.val, a.row, a.col,
                                           t.val, t.row, t.col, CUDA_R_64F, CUSPARSE_ACTION_NUMERIC,
                                           CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG1, &tb ) );
    void *buf = nullptr;
    cudaMalloc( &buf, tb );
    CUS_OK( cusparseCsr2cscEx2( h, a.lignes, a.colonnes, a.nnz, a.val, a.row, a.col,
                                t.val, t.row, t.col, CUDA_R_64F, CUSPARSE_ACTION_NUMERIC,
                                CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG1, buf ) );
    cudaFree( buf );
}

// ---------------------------------------------------------------- les noyaux qui vont avec

/// notre laplacien ( `dia` + hors-diagonaux POSITIFS ) vers un CSR ordinaire.
/// La diagonale est posee EN TETE de ligne et le reste suit dans son ordre : on ne suppose donc
/// rien sur l'ordre des colonnes en entree ( celles du niveau fin sortent d'atomiques ). Le tri
/// vient apres, une fois, par `trie`.
__global__ void k_lis_plein( const int *row, const int *col, const double *val, const double *dia,
                             int *prow, int *pcol, double *pval, int n ) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i > n ) return;
    prow[ i ] = row[ i ] + i;
    if ( i == n ) return;
    int p = prow[ i ];
    pcol[ p ] = i; pval[ p ] = dia[ i ]; ++p;
    for ( int q = row[ i ]; q < row[ i + 1 ]; ++q ) { pcol[ p ] = col[ q ]; pval[ p ] = -val[ q ]; ++p; }
}

/// la permutation appliquee aux valeurs ( `cusparseDgthr` a disparu )
__global__ void k_lis_gthr( const double *src, double *dst, const int *perm, int nnz ) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if ( k < nnz ) dst[ k ] = src[ perm[ k ] ];
}

/// la prolongation TENTATIVE : une entree par ligne, au paquet du germe
__global__ void k_lis_p0( const int *m, int *row, int *col, double *val, int n ) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i > n ) return;
    row[ i ] = i;
    if ( i == n ) return;
    col[ i ] = m[ i ];
    val[ i ] = 1.0;
}

/// `P^ = P - omega D^-1 ( A P )`, en place sur `A P` : le motif de `P` y est deja inclus
__global__ void k_lis_lisse( const int *row, int *col, double *val, const double *dia,
                             const int *m, double omega, int n ) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= n ) return;
    const double f = -omega / dia[ i ];
    const int a = m[ i ];
    for ( int p = row[ i ]; p < row[ i + 1 ]; ++p ) {
        val[ p ] *= f;
        if ( col[ p ] == a ) val[ p ] += 1.0;
    }
}

/// d'un CSR ordinaire vers notre laplacien : la diagonale a part, les hors-diagonaux positifs
__global__ void k_lis_compte( const int *row, const int *col, int n, int *cnt ) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= n ) return;
    int c = 0;
    for ( int p = row[ i ]; p < row[ i + 1 ]; ++p ) c += col[ p ] != i;
    cnt[ i ] = c;
}
__global__ void k_lis_verse( const int *prow, const int *pcol, const double *pval,
                             const int *row, int *col, double *val, double *dia, int n ) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= n ) return;
    int p = row[ i ];
    double d = 0;
    for ( int q = prow[ i ]; q < prow[ i + 1 ]; ++q ) {
        if ( pcol[ q ] == i ) { d = pval[ q ]; continue; }
        col[ p ] = pcol[ q ];
        val[ p ] = -pval[ q ];                           // notre convention : hors-diagonaux positifs
        ++p;
    }
    dia[ i ] = d > 0 ? d : 1.0;
}

/// `y = M x` et `y += M x`, CSR ordinaire -- la restriction et la prolongation deviennent ca
__global__ void k_lis_spmv( const int *row, const int *col, const double *val,
                            const double *x, double *y, int n, bool ajoute ) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= n ) return;
    double s = 0;
    for ( int p = row[ i ]; p < row[ i + 1 ]; ++p ) s += val[ p ] * x[ col[ p ] ];
    y[ i ] = ajoute ? y[ i ] + s : s;
}

} // namespace
} // namespace sf::gpu

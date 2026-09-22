// =====================================================================================
// LE DEBIT EN STREAMING SoA -- ce que couterait une PHASE d'un schema ou les cellules en cours
// vivent dans la RAM du GPU ( README § 4, « les phases et la RAM » ) : l'etat de `n` cellules
// ( `R` sommets x { x, y, cid } ) lu et reecrit, dans le layout structure-of-arrays qui coalesce
// ( `a[ case * n + cellule ]` ). C'est une borne SUPERIEURE du debit utile : rien d'autre ne
// tourne, les acces sont parfaits.
//
//   xmake run bande 1000000 8        -> 0.36 ns par cellule et par passe ( 532 Go/s )
// =====================================================================================
#include <cstdio>
#include <algorithm>
__global__ void passe( const float *xi, const float *yi, const int *ci, float *xo, float *yo, int *co, int n, int R ) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if ( k >= n ) return;
    for ( int i = 0; i < R; ++i ) {
        xo[ i * n + k ] = xi[ i * n + k ] + 1.0f;
        yo[ i * n + k ] = yi[ i * n + k ] + 1.0f;
        co[ i * n + k ] = ci[ i * n + k ] + 1;
    }
}
int main( int argc, char **argv ) {
    const int n = argc > 1 ? atoi( argv[ 1 ] ) : 1000000, R = argc > 2 ? atoi( argv[ 2 ] ) : 8;
    float *xi, *yi, *xo, *yo; int *ci, *co;
    cudaMalloc( &xi, size_t( n ) * R * 4 ); cudaMalloc( &yi, size_t( n ) * R * 4 ); cudaMalloc( &ci, size_t( n ) * R * 4 );
    cudaMalloc( &xo, size_t( n ) * R * 4 ); cudaMalloc( &yo, size_t( n ) * R * 4 ); cudaMalloc( &co, size_t( n ) * R * 4 );
    cudaMemset( xi, 0, size_t( n ) * R * 4 ); cudaMemset( yi, 0, size_t( n ) * R * 4 ); cudaMemset( ci, 0, size_t( n ) * R * 4 );
    cudaEvent_t e0, e1; cudaEventCreate( &e0 ); cudaEventCreate( &e1 );
    const int bloc = 256, grid = ( n + bloc - 1 ) / bloc;
    float best = 1e30f;
    for ( int r = -3; r < 10; ++r ) {
        cudaEventRecord( e0 );
        passe<<<grid, bloc>>>( xi, yi, ci, xo, yo, co, n, R );
        cudaEventRecord( e1 ); cudaEventSynchronize( e1 );
        float ms = 0; cudaEventElapsedTime( &ms, e0, e1 );
        if ( r >= 0 ) best = std::min( best, ms );
    }
    const double octets = 2.0 * size_t( n ) * R * 12;    // lus + ecrits
    printf( "n=%d R=%d : %.3f ms par passe, %.0f Go/s, %.2f ns par cellule et par passe\n",
            n, R, best, octets / ( best * 1e-3 ) / 1e9, best * 1e6 / n );
    return 0;
}

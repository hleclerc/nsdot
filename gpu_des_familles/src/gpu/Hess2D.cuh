#pragma once

// =====================================================================================
// LA HESSIENNE, ASSEMBLEE SUR LA CARTE. C'est le laplacien du graphe de Laguerre
// ( `solver/Laplacien.h` ) :
//
//     c_ij = |facette ij| / ( 2 |p_i - p_j| ),   L_ii = somme_j c_ij,   L_ij = -c_ij
//
// Les facettes sortent deja du noyau de mesure, `NF` cases par cellule en SoA
// ( `[ arete * n + identifiant ]` ) : l'assemblage est donc COMPTER, SCANNER, REMPLIR -- les trois
// passes d'un CSR, sans un tri.
//
//   * COMPTER : une ligne par thread, `NF` lectures separees de `n` -- des voies consecutives
//     lisent des adresses consecutives, donc tout est coalesce.
//   * SCANNER : une somme prefixe exclusive ( CUB ) donne `row`.
//   * REMPLIR : la meme boucle ecrit `col`, `val`, et accumule la diagonale au passage.
//
// LES COLONNES NE SONT PAS TRIEES : un gradient conjugue n'en a pas besoin. Le CPU les trie pour
// `crs_reduit`, qui sert a un solveur direct.
//
// LE NOYAU DU LAPLACIEN est les constantes : chaque ligne somme EXACTEMENT a zero ( la diagonale
// est la somme des hors-diagonaux, calculee dans la meme passe ). Une ligne sans voisin -- qui ne
// peut arriver que si une cellule est vide -- rendrait le systeme singulier SANS LE DIRE : on la
// neutralise a un, comme le CPU.
// =====================================================================================

#include <cub/cub.cuh>

namespace sf::gpu {
namespace {

/// COMPTER : combien de voisins reels pour chaque ligne
__global__ void k_hess_compte( const int *fj, int n, int nf, int *cnt ) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= n ) return;
    int c = 0;
    for ( int s = 0; s < nf; ++s )
        c += fj[ size_t( s ) * n + i ] >= 0;
    cnt[ i ] = c;
}

/// REMPLIR : `col`, `val`, et la diagonale dans la meme passe
template<class TK>
__global__ void k_hess_remplit( const int *fj, const TK *fl, const double *px, const double *py,
                                const int *row, int n, int nf, int *col, double *val, double *dia ) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= n ) return;
    const double xi = px[ i ], yi = py[ i ];
    int p = row[ i ];
    double d = 0;
    for ( int s = 0; s < nf; ++s ) {
        const int j = fj[ size_t( s ) * n + i ];
        if ( j < 0 ) continue;
        const double ex = px[ j ] - xi, ey = py[ j ] - yi;
        const double d2 = ex * ex + ey * ey;
        const double c = d2 > 0 ? double( fl[ size_t( s ) * n + i ] ) / ( 2 * sqrt( d2 ) ) : 0.0;
        col[ p ] = j;
        val[ p ] = c;
        ++p;
        d += c;
    }
    dia[ i ] = d > 0 ? d : 1.0;                          // une ligne nulle rendrait le systeme singulier
}

/// la scatter des positions dans l'ordre DE L'APPELANT ( les tableaux de l'arbre sont permutes )
__global__ void k_hess_pos( const double *tx, const double *ty, const int *ids, double *px, double *py, int n ) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if ( k >= n ) return;
    px[ ids[ k ] ] = tx[ k ];
    py[ ids[ k ] ] = ty[ k ];
}

/// `y = L x` ( la diagonale plus les hors-diagonaux, signes ) -- pour verifier, et pour le CG
__global__ void k_hess_mul( const int *row, const int *col, const double *val, const double *dia,
                            const double *x, double *y, int n ) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= n ) return;
    double s = dia[ i ] * x[ i ];
    for ( int p = row[ i ]; p < row[ i + 1 ]; ++p )
        s -= val[ p ] * x[ col[ p ] ];
    y[ i ] = s;
}

} // namespace
} // namespace sf::gpu

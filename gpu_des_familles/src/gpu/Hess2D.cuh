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
// = ELLE EST SYMETRIQUE AU BIT PRES, ET CE N'EST PAS UN LUXE
//
// Chaque facette est vue DEUX FOIS, une par cellule, et les deux cellules n'en mesurent pas
// exactement la meme longueur. On gardait la mesure DE LA LIGNE : `L_ij` venait de la cellule `i`
// et `L_ji` de la cellule `j`. En `double` les deux vues different de 1e-16 et le gradient
// conjugue ne s'en apercoit pas. EN `float` ELLES DIFFERENT DE 100 % sur les facettes presque
// degenerees -- et le CG, qui suppose un operateur symetrique, cesse de converger. Mesure sur
// cette carte, uniforme 2D `n = 2e5` en `float` : 20 000 iterations sans converger, 56 s de
// solveur. Le meme bug avait ete trouve et repare sur CPU
// ( `solvers_des_familles`, README § 19.10 ) : le diagramme en `float` etait juste, c'est
// l'assemblage qui ne l'etait pas.
//
// La reparation est celle du CPU, mot pour mot : ON NE GARDE QUE LA VUE `i < j` ET ON LA MIROITE
// dans les deux lignes. `col[ p ]` et `col[ q ]` recoivent LE MEME `double`, donc `L = L^T`
// exactement. Ce que ca coute sur une carte, c'est que la ligne `j` recoit une entree qu'elle
// n'a pas produite : le compte et le placement passent par des ATOMIQUES, une par arete et par
// cote. Mesure : l'assemblage passe de 0.73 a 0.9 ms a `n = 2e5` -- rien, en regard des 56 s.
//
// Une facette vue d'un SEUL cote ( `i > j` dans l'unique vue ) est ecartee, comme au CPU : sur
// les millions d'aretes d'un diagramme, `fp32` en compte une dizaine, et une arete que l'une des
// deux cellules ne voit meme pas est microscopique.
//
//   * COMPTER : une ligne par thread, `NF` lectures separees de `n` -- des voies consecutives
//     lisent des adresses consecutives, donc tout est coalesce. Deux atomiques par arete gardee.
//   * SCANNER : une somme prefixe exclusive ( CUB ) donne `row`.
//   * REMPLIR : la meme boucle, le curseur de chaque ligne en atomique.
//   * LA DIAGONALE est sommee A PART, depuis le CSR fini et dans SON ordre : `L . 1` vaut alors
//     zero AU BIT PRES, et pas seulement a 1e-16 pres ( `k_hess_mul` somme dans le meme ordre ).
//
// LES COLONNES NE SONT PAS TRIEES : un gradient conjugue n'en a pas besoin. Le CPU les trie pour
// `crs_reduit`, qui sert a un solveur direct. L'ordre des entrees d'une ligne depend ici de
// l'ordonnancement des atomiques, donc d'un lancement a l'autre ; la MATRICE, elle, ne change
// pas, et sa symetrie est exacte dans tous les cas.
//
// LE NOYAU DU LAPLACIEN est les constantes. Une ligne sans voisin -- qui ne peut arriver que si
// une cellule est vide -- rendrait le systeme singulier SANS LE DIRE : on la neutralise a un,
// comme le CPU.
// =====================================================================================

#include <cub/cub.cuh>

namespace sf::gpu {
namespace {

/// COMPTER, LES DEUX COTES. Seule la vue `i < j` compte, et elle compte DANS LES DEUX LIGNES :
/// une entree pour `i`, une pour `j`. `cnt` doit etre remis a zero avant.
__global__ void k_hess_compte( const int *fj, int n, int nf, int *cnt ) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= n ) return;
    int c = 0;
    for ( int s = 0; s < nf; ++s ) {
        const int j = fj[ size_t( s ) * n + i ];
        if ( j <= i ) continue;                          // une case vide, un cote de boite, ou l'autre vue
        ++c;
        atomicAdd( cnt + j, 1 );                         // la ligne d'en face, qui ne la produit pas
    }
    if ( c ) atomicAdd( cnt + i, c );                    // la notre, en une fois
}

/// REMPLIR : `col` et `val`, la meme valeur des deux cotes. `at` part d'une copie de `row`.
template<class TK>
__global__ void k_hess_remplit( const int *fj, const TK *fl, const double *px, const double *py,
                                int *at, int n, int nf, int *col, double *val ) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= n ) return;
    const double xi = px[ i ], yi = py[ i ];
    for ( int s = 0; s < nf; ++s ) {
        const int j = fj[ size_t( s ) * n + i ];
        if ( j <= i ) continue;
        const double ex = px[ j ] - xi, ey = py[ j ] - yi;
        const double d2 = ex * ex + ey * ey;
        const double c = d2 > 0 ? double( fl[ size_t( s ) * n + i ] ) / ( 2 * sqrt( d2 ) ) : 0.0;
        const int p = atomicAdd( at + i, 1 ), q = atomicAdd( at + j, 1 );
        col[ p ] = j; val[ p ] = c;                      // LE MEME `double` des deux cotes
        col[ q ] = i; val[ q ] = c;
    }
}

/// LA DIAGONALE, sommee depuis le CSR fini et dans son ordre -- donc `L . 1 = 0` exactement.
__global__ void k_hess_dia( const int *row, const double *val, double *dia, int n ) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= n ) return;
    double d = 0;
    for ( int p = row[ i ]; p < row[ i + 1 ]; ++p )
        d += val[ p ];
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

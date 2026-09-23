#pragma once

// =====================================================================================
// L'ARBRE CONSTRUIT SUR LE GPU ( 2D, Voronoi ) -- voir `Bsp2D.cu`. L'interface est en types nus
// pour que l'hote ( compile par g++ ) l'appelle sans voir de CUDA.
// =====================================================================================

#include <vector>

namespace sf::gpu {

/// l'arbre rendu a l'hote : noeuds en PREORDRE, comme `accel/AaBsp.h`
struct ArbreHote {
    std::vector<float> lo, hi;      ///< `2 * nn`, entrelaces ( x, y )
    std::vector<int>   beg, end;    ///< la tranche du noeud
    std::vector<int>   right;       ///< le fils droit ; `< 0` dit FEUILLE ( le gauche est `i + 1` )
    std::vector<int>   order;       ///< rang -> identifiant
    std::vector<float> px, py;      ///< positions permutees
    int nn = 0, n = 0;
};

/// construit l'arbre sur le GPU ; `ms` rend le temps GPU seul
void construit2( const double *px, const double *py, int n, int leaf, ArbreHote &out, double *ms );

} // namespace sf::gpu

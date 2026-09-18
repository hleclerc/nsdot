#pragma once

// =====================================================================================
// LES TYPES DE BASE, et rien d'autre : le flottant de la geometrie, l'indice, le point, l'horloge.
// =====================================================================================

#include <chrono>
#include <cstdint>

namespace sf {

using TF = double;   ///< le flottant des POSITIONS, des poids et des mesures. Le noyau de coupe a le
                     ///< sien (`TK`, un parametre de type) : `float` en production, `double` quand
                     ///< on veut que le plancher du residu soit celui du solveur et pas celui de la
                     ///< geometrie.
using SI = int;      ///< un indice. `int` et non `int64` : 2e9 germes suffisent, et ca compte dans
                     ///< un noeud d'arbre, ou chaque octet est une fraction de ligne de cache.

/// UN POINT en `D` dimensions, passe PAR VALEUR : un agregat de deux ou trois `double` tient dans
/// les registres de l'ABI, la ou un pointeur forcerait a relire la memoire a chaque usage.
template<int D>
struct Vec {
    TF v[ D ];

    TF  operator[]( int d ) const { return v[ d ]; }
    TF &operator[]( int d )       { return v[ d ]; }

    friend TF dist2( Vec a, Vec b ) {
        TF s = 0;
        for ( int d = 0; d < D; ++d ) { const TF e = a[ d ] - b[ d ]; s += e * e; }
        return s;
    }
};

/// le meme, depuis un tableau. En initialisation d'agregat : toutes les lectures precedent la
/// construction, et le compilateur n'a pas a craindre qu'une ecriture aliase la source.
template<int D>
inline Vec<D> vec_of( const TF *a ) {
    if constexpr ( D == 2 ) return { a[ 0 ], a[ 1 ] };
    else                    return { a[ 0 ], a[ 1 ], a[ 2 ] };
}

/// l'horloge, en secondes.
inline double now() {
    using namespace std::chrono;
    return duration<double>( steady_clock::now().time_since_epoch() ).count();
}

} // namespace sf

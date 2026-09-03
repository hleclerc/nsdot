#pragma once

#include <cstdint>

namespace pd2d {

using TF = double;   ///< le type flottant de la GEOMETRIE. `float` est une variante a essayer.
using SI = int;      ///< un indice. `int` et non `int64` : 2e9 germes suffisent, et ca compte
                     ///< dans un noeud d'arbre, ou chaque octet est une fraction de ligne de cache.

struct Pt {
    TF x, y;
};

/// Ce qu'une coupe a fait de la cellule. Trois etats et non deux, parce que « le plan ne touche pas
/// la cellule » est le cas MAJORITAIRE : un accelerateur propose une feuille entiere de germes,
/// dont six seulement sont vraiment voisins en 2D. Le distinguer permet de ne RIEN ecrire.
enum class CutResult {
    unchanged,   ///< le demi-espace contient deja toute la cellule
    done,        ///< la cellule a ete coupee
    empty,       ///< il ne reste rien
    overflow,    ///< plus de `max_nb_vertices` sommets : la cellule est laissee TELLE QUELLE
};

} // namespace pd2d

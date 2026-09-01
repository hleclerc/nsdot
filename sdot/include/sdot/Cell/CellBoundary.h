#pragma once

namespace sdot {

namespace CellBoundary {
    enum {
        INFINITE = -2,
        BOUNDARY = -1,
    };
} //  namespace CellBoundary

/// Ce qu'une coupe a fait de la cellule.
///
/// Trois etats et non deux, parce que « le plan ne touche pas la cellule » est le cas le PLUS
/// FREQUENT : un accelerateur propose une feuille entiere de germes, dont quelques-uns seulement
/// sont vraiment voisins (6 en moyenne en 2D). Le distinguer permet a `cut_into` de ne rien ecrire
/// du tout -- ni sommets, ni coupes -- la ou il reecrivait la cellule entiere dans l'autre tampon
/// pour en produire une copie a l'identique.
enum class CutResult {
    unchanged,  ///< le demi-espace contient deja toute la cellule : RIEN n'a ete ecrit
    done,       ///< la cellule a ete coupee, le resultat est dans `res`
    overflow,   ///< la capacite n'a pas suffi : rien n'a ete ecrit, le compte voulu est enregistre
};

} // namespace sdot

#pragma once

#include <loom/support/common_macros.h>
#include "ConstantDensity.h"

namespace sdot {

// La distribution qui n'en est pas une : la mesure de Lebesgue, densité 1 partout.
//
// Ce que `PowerDiagram::integrate_into` reçoit quand l'appelant n'a donné aucune distribution, de
// sorte que « pas de distribution » soit un cas ORDINAIRE du même code et non une seconde
// implémentation -- exactement le rôle que `EverySeed` tient pour les accélérateurs.
//
// Un seul morceau, la cellule elle-même, et aucun découpage : ni scratch, ni copie, ni la moindre
// coupe. `measures` sans distribution calcule donc exactement ce qu'il calculait avant, à
// l'instruction près (le `TF( 1 ) *` se replie à la compilation).
struct UnitDensity {
    void for_each_piece( const auto &cell, auto &&/*ws*/, auto &&func ) const {
        // constante, et non paramétrée : le puits de gradient ne mène nulle part (voir
        // `ConstantDensity`). Le `TF( 1 ) *` de l'intégrateur se replie à la compilation.
        using TF = typename DECAYED_TYPE_OF( cell )::TF;
        func( cell, ConstantDensity{ TF( 1 ), []( auto &&/*grad_dist*/, auto /*g*/ ) {} } );
    }
};

} // namespace sdot

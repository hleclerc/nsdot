#pragma once

#include <loom/support/common_macros.h>

namespace sdot {

// La densité d'un MORCEAU sur lequel elle est constante -- ce que rend le découpage d'une image,
// et ce que rend `UnitDensity` pour la cellule entière.
//
// C'est le cas où l'intégration est exacte et gratuite : `valeur * mesure du morceau`, sans
// quadrature ni triangulation (voir `PowerDiagram::integrate_into`, qui branche là-dessus À LA
// COMPILATION sur `is_constant`).
//
// `sink` est ce qui rattache la valeur aux PARAMÈTRES de la distribution : l'intégrateur sait que
// `d masse / d valeur` est le volume du morceau, mais pas où cette valeur est rangée -- une case de
// `values` pour une image, rien du tout pour la densité unité. La fermeture est donc fabriquée là
// où l'indice est connu, et l'intégrateur ne fait que l'appeler.
template<class TF_,class Sink>
struct ConstantDensity {
    using TF = TF_;
    static constexpr bool is_constant = true;

    TF   value;
    Sink sink;

    void add_value_grad( auto &&grad_dist, TF g ) const { sink( grad_dist, g ); }
};

} // namespace sdot

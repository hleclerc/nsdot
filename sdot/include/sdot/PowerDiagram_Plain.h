#pragma once

// the axes this body names, as declared symbols (autocompletion, standalone compile) instead of
// globals the generated source happens to define around us. Written to the build include tree.
#include <sdot/generated/aggregates/PowerDiagram_Plain.h>
#include <loom/support/common_macros.h>
#include "diagram/Common.h"

namespace sdot {

// LES GERMES TELS QU'ILS SONT VENUS ( `PowerDiagram_Plain.py` ) : `positions [ n, d ]`, `weights
// [ n ]`, et aucune acceleration -- chaque cellule est coupee par les `n - 1` bissectrices, dans
// l'ordre. Le plancher contre lequel `PowerDiagram_Bsp` se mesure, et ce qui reste quand les
// positions sont un traceur ( pas d'arbre a batir dessus ).
SDOT_TEMPLATE_DECL_FOR_PowerDiagram_Plain
struct PowerDiagram_Plain {
    SDOT_ATTRIBUTES_OF_PowerDiagram_Plain
    SDOT_DIAGRAM_COMMON( positions, weights )

    HD SI user_id( SI k ) const { return k; }

    template<class TK>
    HD auto fournisseur( SI k0 ) const { return FournisseurTous<PowerDiagram_Plain,TK,ct_dim>( *this, k0 ); }
};

}

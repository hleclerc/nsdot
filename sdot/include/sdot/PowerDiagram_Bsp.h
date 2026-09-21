#pragma once

// the axes this body names, as declared symbols (autocompletion, standalone compile) instead of
// globals the generated source happens to define around us. Written to the build include tree.
#include <sdot/generated/aggregates/PowerDiagram_Bsp.h>
#include <loom/support/common_macros.h>
#include "diagram/Common.h"
#include "AaBsp.h"

namespace sdot {

// LES GERMES DANS L'ORDRE DE L'ARBRE ( `PowerDiagram_Bsp.py` ) : `sorted_positions` / `sorted_weights`
// sont les germes permutes comme `tree.seed_indices` les range, de sorte qu'une feuille se lit
// d'un seul tenant. Le fournisseur est l'arbre retourne ( `cell/Fournisseurs.h::FournisseurBsp` ),
// et tout ici -- rangs, mesures, gradients, identifiants de coupe -- est dans cet ordre-la ; c'est
// `PowerDiagram_Bsp.py` qui traduit vers l'ordre de l'utilisateur.
SDOT_TEMPLATE_DECL_FOR_PowerDiagram_Bsp
struct PowerDiagram_Bsp {
    SDOT_ATTRIBUTES_OF_PowerDiagram_Bsp
    SDOT_DIAGRAM_COMMON( sorted_positions, sorted_weights )

    HD SI user_id( SI k ) const { return SI( tree.seed_indices( k ) ); }

    /// la memoire ( `memo_nbrs / memo_counts`, voir `FournisseurBsp` ) : nommee ou `Unbound`, a la compilation
    static constexpr bool has_memo = DECAYED_TYPE_OF( std::declval<DECAYED_TYPE_OF( memo_counts )>().is_valid() )::value;

    template<class TK>
    HD auto fournisseur( SI k0 ) const { return FournisseurBsp<PowerDiagram_Bsp,TK,ct_dim,has_weights,has_memo>( *this, k0 ); }
};

}

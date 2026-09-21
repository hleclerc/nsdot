#pragma once

// the axes this body names, as declared symbols (autocompletion, standalone compile) instead of
// globals the generated source happens to define around us. Written to the build include tree.
#include <sdot/generated/aggregates/PowerDiagram_Bsp.h>
#include <loom/support/common_macros.h>
#include "diagram/Common.h"
#include "AaBsp.h"

namespace sdot {

/// les germes tries, tels que `bsp_build_level.h` les lit ( `nb_dims`, `positions`, `weights` )
template<class T_nb_dims,class T_positions,class T_weights>
struct BspCloudView {
    T_nb_dims   nb_dims;
    T_positions positions;
    T_weights   weights;
};

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

    /// LE MEME DIAGRAMME, les poids et les majorants de l'arbre lus AILLEURS -- des vues que
    /// l'appelant possede et ECRIT ( le solveur de `OtPlan`, qui pose des poids a chaque essai ) :
    /// les entrees d'un appel sont en lecture seule, ses sorties non. L'ordre des membres est
    /// celui de `PowerDiagram_Bsp.py` / `AaBsp.py` ( le meme que `kernel_form` ci-dessus ).
    HD auto with_weights( auto &&sorted_weights_, auto &&node_wa_, auto &&node_wb_ ) const {
        auto tree_ = ::sdot::AaBsp{ tree.seed_indices, tree.node_left, tree.node_right, tree.node_begin, tree.node_end,
                                    tree.node_box, node_wa_, node_wb_, tree.nb_bsp_seeds, tree.nb_bsp_nodes, tree.nb_lohi, tree.nb_dims };
        return ::sdot::PowerDiagram_Bsp{ box_min, box_max, bnd_directions, bnd_offsets, nb_points, nb_boundaries, nb_dims,
                                         tree_, sorted_positions, sorted_weights_, memo_nbrs, memo_counts, nb_memo };
    }
    /// ... et la memoire aussi lue ailleurs ( le solveur l'ecrit a chaque balayage )
    HD auto with_weights( auto &&sorted_weights_, auto &&node_wa_, auto &&node_wb_, auto &&memo_nbrs_, auto &&memo_counts_ ) const {
        auto tree_ = ::sdot::AaBsp{ tree.seed_indices, tree.node_left, tree.node_right, tree.node_begin, tree.node_end,
                                    tree.node_box, node_wa_, node_wb_, tree.nb_bsp_seeds, tree.nb_bsp_nodes, tree.nb_lohi, tree.nb_dims };
        return ::sdot::PowerDiagram_Bsp{ box_min, box_max, bnd_directions, bnd_offsets, nb_points, nb_boundaries, nb_dims,
                                         tree_, sorted_positions, sorted_weights_, memo_nbrs_, memo_counts_, nb_memo };
    }

    /// ce que `bsp_refresh_majorant` lit : les germes dans l'ordre de l'arbre
    HD auto sorted_cloud() const { return BspCloudView{ nb_dims, sorted_positions, sorted_weights }; }
};


}

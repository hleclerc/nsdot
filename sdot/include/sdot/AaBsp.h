#pragma once

// the axes this body names, as declared symbols (autocompletion, standalone compile) instead of
// globals the generated source happens to define around us. Written to the build include tree.
#include <sdot/generated/aggregates/AaBsp.h>
#include <loom/support/common_macros.h>
#include "bsp_build_level.h"

namespace sdot {

// An AXIS-ALIGNED BSP over the seeds: a binary tree of boxes, a handful of seeds per leaf. See
// `AaBsp.py` for what a node carries and why the weight majorant is AFFINE rather than constant.
//
// The WALK is not here: it is a FOURNISSEUR (`cell/Fournisseurs.h::FournisseurBsp`), suspended
// between two cuts, which the cell engine pulls plans from. What the C++ side keeps is the
// tensors (`node_box`, `node_begin` / `node_end`, `seed_indices`, the weight majorant). The one
// thing that has to be REDONE when the weights change and the positions do not -- the majorant --
// is `bsp_build_level.h::bsp_refresh_majorant`, a free function on a node's slice: it must not
// take the tree as an input, since the tree's current majorants are what it replaces.
SDOT_TEMPLATE_DECL_FOR_AaBsp
struct AaBsp {
    SDOT_ATTRIBUTES_OF_AaBsp

    static constexpr int ct_dim = DECAYED_TYPE_OF( nb_dims )::value;
    using TF = DECAYED_TYPE_OF( node_box )::TF;
};

}

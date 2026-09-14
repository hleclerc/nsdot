#pragma once

// the axes this body names, as declared symbols (autocompletion, standalone compile) instead of
// globals the generated source happens to define around us. Written to the build include tree.
#include <sdot/generated/aggregates/Cell_N.h>
#include <loom/support/common_macros.h>
#include "cell/LocalN.h"
#include "cell/Ops.h"

namespace sdot {

// LE POLYTOPE SIMPLE EN DIMENSION >= 3 ( `Cell_N.py` ) : `D` coupes et `D` voisins par sommet.
//
// Les tenseurs sont ceux de `Cell_N.py` -- le format PRATIQUE, un sommet par ligne, dans le flottant
// de l'appelant. Un kernel ne travaille jamais dessus directement : il pose la cellule dans sa
// forme locale ( `Local`, sur le scratch de l'item, dans le flottant du noyau ), travaille, et la
// repose. Les operations elles-memes sont dans `cell/Ops.h`, communes aux trois regimes.
SDOT_TEMPLATE_DECL_FOR_Cell_N
struct Cell_N {
    SDOT_ATTRIBUTES_OF_Cell_N

    static constexpr int ct_dim = DECAYED_TYPE_OF( nb_dims )::value;
    using TF = DECAYED_TYPE_OF( vertex_positions )::TF;
    template<class TK> using Local = LocalN<TK,ct_dim>;

    void init_as_hypercube( auto &&scratch, auto &&origin, auto &&axes, SI cut_id ) {
        cell_ops::init_as_hypercube<Local<KernelType<DECAYED_TYPE_OF( scratch )>>>( *this, scratch, origin, axes, cut_id );
    }
    void init_as_unbounded( auto &&scratch ) {
        cell_ops::init_as_unbounded<Local<KernelType<DECAYED_TYPE_OF( scratch )>>>( *this, scratch );
    }
    void cut( auto &&res, auto &&scratch, auto &&direction, auto &&offset, SI cut_id ) const {
        cell_ops::cut<Local<KernelType<DECAYED_TYPE_OF( scratch )>>>( *this, res, scratch, direction, offset, cut_id );
    }
    void measure( auto &&res, auto &&scratch ) const {
        cell_ops::measure<Local<KernelType<DECAYED_TYPE_OF( scratch )>>>( *this, res, scratch );
    }
    void measure_bwd( auto &&res, auto &&grad_res, auto &&grad_vertex_positions, auto &&scratch ) const {
        cell_ops::measure_bwd<Local<KernelType<DECAYED_TYPE_OF( scratch )>>>( *this, res, grad_res, grad_vertex_positions, scratch );
    }
};

}

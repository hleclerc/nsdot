#pragma once

// the members + generated methods (`operator()`, `make_available`, ...) as macros this struct
// drops in, plus the axes it names -- written to the build include tree by `CallArg_Aggregate`.
#include <sdot/generated/aggregates/Image.h>
#include <loom/support/common_macros.h>

namespace sdot {

SDOT_TEMPLATE_DECL_FOR_Image
struct Image {
    SDOT_ATTRIBUTES_OF_Image

    SCInt  ct_dim       = DECAYED_TYPE_OF( nb_dims )::value;
    using  TF           = DECAYED_TYPE_OF( values )::TF;

    // build a FULLY-POPULATED image -- each of `origin` / `frame` / `knots` that is unbound
    // (a `NoneTensor`) replaced by its documented default -- and hand it to `cont`. Lets the
    // methods below be written ONCE against a complete image instead of gating on `is_valid()`.
    auto   with_defaults( auto &&cont ) const;

    // total measure of the piecewise-constant function: sum over cells of value * cell volume.
    TF     measure      () const;

    // ---- intégrer une CELLULE contre cette image -------------------------------------------------
    // L'image comme DISTRIBUTION : ce que `PowerDiagram::measures` appelle quand on lui en donne
    // une. Le contrat (celui de toute distribution) est écrit dans `distributions/Distribution.py` ;
    // il tient en une phrase : découper `cell` en morceaux sur lesquels la densité est CONSTANTE,
    // et passer chacun à `func( morceau, valeur, add_value_grad )`.
    //
    // Ici un morceau est `cellule INTER pavé de la grille`. Le découpage n'est donc que 2d coupes
    // par pavé rencontré, faites dans les deux cellules de rechange que l'appelant fournit
    // (`PieceWorkspace.h`) -- la cellule d'origine, elle, n'est jamais touchée, ce qui permet
    // d'ouvrir un morceau après l'autre sans la copier.
    void   for_each_piece ( const auto &cell, auto &&ws, auto &&func ) const;
    void   _for_each_piece( const auto &cell, auto &&ws, auto &&func ) const;

    // le plus grand `i` tel que `knots( axis, i ) <= t`, borné dans `[ 0, nb_cells - 1 ]` (les
    // noeuds sont croissants, donc une dichotomie -- une image de production a beaucoup de pavés).
    SI     knot_index   ( SI axis, TF t, SI nb_cells ) const;

    // adjoint of `measure` w.r.t. `values`: since mass = Sum_c values(c) * cell_volume(c) is linear
    // in `values`, grad_values(c) = grad_mass * cell_volume(c). `grad_mass` is the scalar cotangent.
    void   measure_bwd  ( auto &&grad_values, auto &&grad_mass ) const;

    // unidimension piece
    struct Udp          { SI index; TF pos; TF mass; TF y; };
    auto   udp_start    () const;
    auto   udp_cont     ( auto &&udp, auto mass_to_take, auto &&cb_parts ) const;

    // jump directly to the `Udp` state a sequential walk would be in after having consumed exactly
    // `target_mass` -- via binary search over a PRECOMPUTED per-cell cumulative-mass array (built by
    // `fill_cell_cum_mass`), instead of walking there step by step. Lets independent work-items each
    // resume the walk at their own chunk's boundary in parallel.
    auto   udp_at       ( auto &&cell_cum_mass, auto target_mass ) const;

    // fills `cell_cum_mass[0..nb_cells]` (exclusive prefix of each cell's mass, sentinel = total
    // mass) -- cached on the PYTHON side as `Image.cell_cum_mass` (a `ComputedAttribute`, see
    // `distributions/Image.py`) so `OtPlan1d` reads it ready-made instead of rebuilding it on every
    // forward/backward call. Sequential (one thread does a whole angle): `nb_cells` is small enough
    // that no cooperative scan is worth the complexity here.
    void   fill_cell_cum_mass( auto &&cell_cum_mass ) const;
};

}

#include "Image.cxx"

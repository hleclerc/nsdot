#pragma once

#include <loom/support/common_macros.h> // HD

// CE QUE LES DEUX STOCKAGES ONT EN COMMUN, cote C++ : les points de `positions`, le poids ( absent
// -> zero, et le terme disparait a la compilation ), et les points d'entree que `PowerDiagram.py`
// appelle -- qui ne font que passer la main a `diagram/Ops.h` en s'ajoutant eux-memes.
//
// Un stockage l'instancie avec ses propres tenseurs : `DIAGRAM_COMMON( positions, weights )`.

#include <loom/support/containers/Vector.h>
#include "Ops.h"
#include "../Queue.h"

#define SDOT_DIAGRAM_COMMON( POSITIONS, WEIGHTS ) \
    static constexpr int  ct_dim = DECAYED_TYPE_OF( nb_dims )::value; \
    static constexpr bool on_cpu = std::is_same_v<Queue,CpuQueue>; \
    using TF = DECAYED_TYPE_OF( POSITIONS )::TF; \
    static constexpr bool has_weights = DECAYED_TYPE_OF( std::declval<DECAYED_TYPE_OF( WEIGHTS )>().is_valid() )::value; \
    \
    HD SI   nb_seeds() const { return SI( POSITIONS.shape( 0 ) ); } \
    HD auto point( SI k ) const { return Vector<TF,ct_dim>::with_func( [&]( PI d ) { return TF( POSITIONS( k, d ) ); } ); } \
    HD TF   weight( SI k ) const { if constexpr ( has_weights ) return TF( WEIGHTS( k ) ); else return TF( 0 ); } \
    \
    HD UnitDensity unit_density() const { return {}; } \
    \
    HD void measures( auto &&res, const auto &dom, auto &&scratch, const auto &dist, auto &&memo_nbrs, auto &&memo_counts, SI thread_index, SI nb_threads ) const { \
        diagram::measures( *this, res, dom, scratch, dist, memo_nbrs, memo_counts, thread_index, nb_threads ); \
    } \
    HD void measures_bwd( auto &&res, const auto &dom, auto &&grad_res, auto &&grad_positions, auto &&grad_weights, \
                       auto &&scratch, const auto &dist, auto &&grad_dist, SI thread_index, SI nb_threads ) const { \
        diagram::measures_bwd( *this, res, dom, grad_res, grad_positions, grad_weights, scratch, dist, grad_dist, thread_index, nb_threads ); \
    } \
    HD void moments( auto &&mass, auto &&first, auto &&second, const auto &dom, auto &&scratch, const auto &dist, SI thread_index, SI nb_threads ) const { \
        diagram::moments( *this, mass, first, second, dom, scratch, dist, thread_index, nb_threads ); \
    } \
    HD void hessian_row( SI k, const auto &dom, auto &&res, auto &&scratch, SI thread_index, const auto &dist ) const { \
        diagram::hessian_row( *this, k, dom, res, scratch, thread_index, dist ); \
    } \
    HD void build_cell( SI k, const auto &dom, auto &&res, auto &&scratch, SI thread_index ) const { \
        diagram::build_cell( *this, k, dom, res, scratch, thread_index ); \
    }

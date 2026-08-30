#pragma once

#include <loom/support/common_macros.h>
#include <loom/support/containers/Matrix.h>
#include <loom/support/containers/Vector.h>
#include <loom/support/atomic_add.h>
#include "PowerDiagram.h"
#include <cmath>

#define UTP SDOT_TEMPLATE_DECL_FOR_PowerDiagram
#define DTP PowerDiagram<SDOT_TEMPLATE_ARGS_FOR_PowerDiagram>

namespace sdot {

UTP auto DTP::point( SI i ) const {
    return Vector<TF,ct_dim>::with_func( [&]( PI d ) { return TF( positions( i, d ) ); } );
}

UTP auto DTP::weight( SI i ) const {
    // absent weights are the Euclidean case: `weights` is then a `NoneTensor`, the branch is
    // resolved at COMPILE time, and no zero is ever read.
    if constexpr ( CT_VALUE( weights.is_valid() ) )
        return TF( weights( i ) );
    else
        return TF( 0 );
}

UTP bool DTP::cut_by( bool &in_0, auto &&ws_0, auto &&ws_1, auto &&corr, const auto &dir, TF off, SI cut_id ) const {
    bool ok;
    if constexpr ( ct_dim > 2 ) {
        // d > 2 rewrites the face lattice and COMPACTS it, which is what `corr` is for (the old ->
        // new index tables, one row per work-item). Below 3D the clip is a single cyclic pass with
        // nothing tabulated, and `corr` is not even allocated (a `NoneTensor` -- see `PowerDiagram.py`).
        ok = in_0 ? ws_0.cut( ws_1, dir, off, cut_id, corr )
                  : ws_1.cut( ws_0, dir, off, cut_id, corr );
    } else {
        ok = in_0 ? ws_0.cut( ws_1, dir, off, cut_id )
                  : ws_1.cut( ws_0, dir, off, cut_id );
    }
    in_0 = ! in_0;

    // A cut that did not fit wrote NOTHING (see `Cell::cut`), so what is now "the" cell is the
    // stale content of the other buffer -- the previous seed's cell, or, for the first seed a
    // work-item builds, whatever the fresh output buffer happened to hold. Leave a well-defined
    // nothing instead: the run is doomed either way (the overflow is recorded, the host reserves
    // more and runs again, this result is thrown away), what matters is that the walk that reads
    // it afterwards stays on values it can trust.
    if ( ! ok ) {
        if ( in_0 ) make_empty( ws_0 );
        else        make_empty( ws_1 );
    }
    return ok;
}

UTP void DTP::make_empty( auto &&cell ) const {
    cell.is_fully_bounded = 1;
    cell.nb_vertices.set( 0 );
    cell.nb_cuts.set( 0 );
    if constexpr ( ct_dim > 2 )
        cell.nb_edges.set( 0 );
}

UTP bool DTP::cell_may_be_cut( const auto &cell, const auto &p0, TF w0,
                               const auto &lo, const auto &hi, const auto &wa, TF wb ) const {
    // The vertex sweep bounds the cell only when the cell IS the hull of its vertices, i.e. when
    // it is bounded. An unbounded one is a stand-in simplex whose corners are made up
    // (`Cell::init_as_unbounded`), so there is nothing to prune against and the honest answer is
    // "maybe". With a domain this never happens past the domain cuts, which `make_cell` does
    // first; without one the accelerator simply degenerates to the full sweep -- which is the
    // right answer, not a slow path anybody chose.
    if ( ! cell.is_fully_bounded )
        return true;

    const SI nv = cell.nb_vertices;
    for ( SI v = 0; v < nv; ++v ) {
        TF s = 0;                               // min over the box of `|p - y|^2 - wa . y`
        TF r0 = 0;                              // `|p - p0|^2`
        for ( PI d = 0; d < ct_dim; ++d ) {
            const TF p = TF( cell.vertex_positions( v, d ) );

            // the closest point of the box to `p`, SHIFTED by half the weight slope: the minimand
            // separates per axis and its free minimum sits at `p + wa/2`, so a clamp answers it.
            TF y = p;
            if constexpr ( CT_VALUE( weights.is_valid() ) )
                y += wa[ d ] / 2;
            y = y < lo[ d ] ? lo[ d ] : ( y > hi[ d ] ? hi[ d ] : y );

            const TF e = y - p;
            s += e * e;
            if constexpr ( CT_VALUE( weights.is_valid() ) )
                s -= wa[ d ] * y;

            const TF f = p - p0[ d ];
            r0 += f * f;
        }

        // `<=` and not `<`: a plane exactly through a vertex removes nothing, so admitting it
        // costs one useless cut, where skipping it on a rounding error would lose a real one.
        if ( s - wb - r0 + w0 <= 0 )
            return true;
    }
    return false;
}

UTP bool DTP::make_cell( SI i0, auto &&ws_0, auto &&ws_1, auto &&corr, const auto &acc, auto &&acc_ws ) const {
    bool in_0 = true;
    ws_0.init_as_unbounded();

    // ---- the domain. First, so that everything after it runs on a BOUNDED cell: an unbounded one
    // is only a stand-in whose artificial planes each cut has to push out again (`Cell.h`) -- and
    // it is also what the accelerator's prune test needs, having no hull to test against otherwise
    // (`cell_may_be_cut`).
    if constexpr ( CT_VALUE( bnd_directions.is_valid() ) ) {
        // read off the BUFFER, not off `nb_boundaries`: a bound input carries exactly its data, so
        // its extent is the count -- and the count then costs nothing to cross.
        const SI nb = SI( bnd_directions.shape( 0 ) );
        for ( SI b = 0; b < nb; ++b ) {
            const auto dir = Vector<TF,ct_dim>::with_func( [&]( PI d ) { return TF( bnd_directions( b, d ) ); } );
            if ( ! cut_by( in_0, ws_0, ws_1, corr, dir, TF( bnd_offsets( b ) ), CellBoundary::BOUNDARY ) )
                break;
        }
    }

    // ---- the seeds. WHICH ones is the accelerator's business (`SpatialAccelerator.py`); what a
    // cut IS, and whether a region can still reach the cell, is ours. `acc` is `EverySeed` when
    // the caller gave no accelerator, so there is one builder here and not two.
    const auto p0 = point( i0 );
    const TF w0 = weight( i0 );

    acc.for_each_candidate( p0, i0, acc_ws,
        [&]( const auto &lo, const auto &hi, const auto &wa, TF wb ) {
            return in_0 ? cell_may_be_cut( ws_0, p0, w0, lo, hi, wa, wb )
                        : cell_may_be_cut( ws_1, p0, w0, lo, hi, wa, wb );
        },
        [&]( SI i1 ) {
            // The cell of `i0` is where its power distance wins, so one half-space per other seed.
            // `|x - d_0|^2 - w_0 <= |x - d_1|^2 - w_1` loses its `|x|^2` on both sides and becomes
            // `( d_1 - d_0 ) . x <= ( d_1 - d_0 ) . ( d_0 + d_1 ) / 2 + ( w_0 - w_1 ) / 2`: the
            // Euclidean bisector, SHIFTED along its own normal by the weight gap. The plane is
            // written un-normalized (`direction` need not be -- `offset` is the dot product it is
            // compared to), which is also why the weight term is halved rather than divided by
            // `|d_1 - d_0|`. `cut_id` is the NEIGHBOUR's index, so a surviving cut says which seed
            // it faces.
            const auto p1 = point( i1 );
            const auto dir = p1 - p0;
            TF off = 0;
            for ( PI d = 0; d < ct_dim; ++d )
                off += dir[ d ] * ( p0[ d ] + p1[ d ] ) / 2;
            if constexpr ( CT_VALUE( weights.is_valid() ) )
                off += ( w0 - TF( weights( i1 ) ) ) / 2;

            if ( ! cut_by( in_0, ws_0, ws_1, corr, dir, off, i1 ) )
                return false;

            // nothing left to clip. Not an anomaly -- two seeds at the very same place leave one of
            // them with an empty cell -- and going on would cost a full walk for a cell that can
            // only stay empty.
            return ( in_0 ? SI( ws_0.nb_vertices ) : SI( ws_1.nb_vertices ) ) != 0;
        } );

    return in_0;
}

UTP void DTP::measure_into( auto &&res, auto &&cell, auto &&facet_apex ) const {
    // the two regimes of `Cell::measure`: the shoelace/segment formula reads the vertices and needs
    // nothing else, the fan triangulation walks the face lattice and needs its apex scratch.
    if constexpr ( ct_dim > 2 )
        cell.measure( res, facet_apex );
    else
        cell.measure( res );
}

UTP void DTP::for_each_simplex_of( const auto &cell, auto &&facet_apex, auto &&func ) const {
    // les deux régimes de `Cell::for_each_simplex` : au-delà de 2D c'est une marche sur le treillis
    // de faces, qui a besoin de son scratch d'apex ; en deçà, un éventail depuis le sommet 0, qui
    // n'a besoin de rien. Même callback des deux côtés (`ct_dim + 1` indices de sommets).
    if constexpr ( ct_dim > 2 )
        cell.for_each_simplex( facet_apex, func );
    else
        cell.for_each_simplex( func );
}

UTP auto DTP::simplex_points( const auto &cell, const auto &chain ) const {
    return Vector<Vector<TF,ct_dim>,ct_dim+1>( Function(), [&]( PI k ) {
        return Vector<TF,ct_dim>::with_func( [&]( PI c ) { return TF( cell.vertex_positions( chain[ k ], c ) ); } );
    } );
}

UTP void DTP::integrate_into( auto &&res, auto &&cell, auto &&facet_apex,
                              const auto &dist, auto &&piece_ws ) const {
    // La distribution DÉCOUPE, on INTÈGRE. Sans distribution (`UnitDensity`) il n'y a qu'un
    // morceau, la cellule elle-même, pas une coupe, et une densité constante : le `TF( 1 ) *` se
    // replie, donc ce chemin calcule exactement ce que `measure_into` calculait tout seul --
    // cellule infinie (`TF::max`) et cellule vide (0) comprises, qui traversent la somme telles
    // quelles.
    TF sum = 0;
    dist.for_each_piece( cell, piece_ws, [&]( const auto &piece, const auto &dens ) {
        if constexpr ( DECAYED_TYPE_OF( dens )::is_constant ) {
            TF m = 0;
            measure_into( m, piece, facet_apex );
            sum += dens.value * m;
        } else {
            // Densité non constante : on ne sait pas l'intégrer, ELLE sait. On ne lui donne que des
            // SIMPLEXES -- le découpage géométrique, la seule chose qu'on apporte ici -- et elle
            // rend l'intégrale sur chacun, par une formule fermée si elle en a une, par la
            // quadrature générique (`PointwiseDensity`) si elle n'est qu'une boîte noire. Aucune
            // règle d'intégration n'est écrite de ce côté-ci.
            //
            // Une cellule non bornée n'a pas de simplices qui veuillent dire quoi que ce soit : on
            // répond `TF::max`, comme `measure`, plutôt qu'un nombre inventé.
            if ( ! piece.is_fully_bounded ) {
                sum = std::numeric_limits<TF>::max();
                return;
            }

            for_each_simplex_of( piece, facet_apex, [&]( const auto &chain ) {
                sum += dens.integrate_over_simplex( simplex_points( piece, chain ) );
            } );
        }
    } );
    res = sum;
}

UTP void DTP::integrate_bwd_into( SI i0, auto &&grad_res, auto &&cell, auto &&facet_apex, auto &&grad_vp,
                                  auto &&grad_positions, auto &&grad_weights, auto &&grad_dist,
                                  const auto &dist, auto &&piece_ws ) const {
    const TF g = grad_res;
    dist.for_each_piece( cell, piece_ws, [&]( const auto &piece, const auto &dens ) {
        if constexpr ( DECAYED_TYPE_OF( dens )::is_constant ) {
            TF m = 0;
            measure_into( m, piece, facet_apex );

            // la part de la DENSITÉ : la masse est linéaire en elle, donc la dérivée par rapport à
            // la valeur portée par ce morceau EST le volume du morceau. Un morceau infini n'en a
            // pas (`measure_into` y répond `TF::max`) : il ne peut venir que d'une densité qui ne
            // découpe rien, donc qui n'a de toute façon aucune valeur où accumuler.
            if ( piece.is_fully_bounded )
                dens.add_value_grad( grad_dist, g * m );

            // ... et la part de la GÉOMÉTRIE, par la chaîne habituelle. Le morceau est un polytope
            // comme un autre : ses coupes portent l'indice du germe qu'elles font face, ou
            // `BOUNDARY` pour celles que la distribution a ajoutées, et `scatter_cell_grad` ne
            // demande rien de plus.
            measure_bwd_into( m, g * dens.value, piece, facet_apex, grad_vp );
            scatter_cell_grad( i0, piece, grad_vp, grad_positions, grad_weights );
        } else {
            // Le miroir : la densité rend la cotangente des `d + 1` SOMMETS du simplexe (et range
            // elle-même celle de ses propres paramètres), on la recolle sur les sommets du morceau.
            // De là, c'est la remontée de toujours -- `scatter_cell_grad` ne sait rien de plus ici
            // que dans le cas constant.
            if ( ! piece.is_fully_bounded )
                return;

            const SI nv = piece.nb_vertices;
            for ( SI v = 0; v < nv; ++v )
                for ( PI c = 0; c < ct_dim; ++c )
                    grad_vp( v, c ) = 0;

            for_each_simplex_of( piece, facet_apex, [&]( const auto &chain ) {
                auto grad_pts = Vector<Vector<TF,ct_dim>,ct_dim+1>( Function(), []( PI ) {
                    return Vector<TF,ct_dim>::zeros(); } );

                dens.integrate_over_simplex_bwd( simplex_points( piece, chain ), g, grad_pts, grad_dist );

                for ( SI k = 0; k <= ct_dim; ++k )
                    for ( PI c = 0; c < ct_dim; ++c )
                        grad_vp( chain[ k ], c ) += grad_pts[ k ][ c ];
            } );

            scatter_cell_grad( i0, piece, grad_vp, grad_positions, grad_weights );
        }
    } );
}

UTP void DTP::measures( auto &&res, auto &&ws_0, auto &&ws_1, auto &&corr, auto &&facet_apex,
                        const auto &acc, auto &&acc_ws, const auto &dist, auto &&piece_ws,
                        SI thread_index, SI nb_threads ) const {
    const SI n = nb_points;
    for ( SI i = thread_index; i < n; i += nb_threads ) {
        // built, measured, forgotten -- in that order, and never two cells alive at once (bar the
        // ping-pong pair the clip needs, and the pair a distribution cuts its pieces in). Writing
        // the volume straight into its slot is what keeps the diagram from ever existing as a whole.
        if ( make_cell( i, ws_0, ws_1, corr, acc, acc_ws ) )
            integrate_into( res( i ), ws_0, facet_apex, dist, piece_ws );
        else
            integrate_into( res( i ), ws_1, facet_apex, dist, piece_ws );
    }
}

UTP void DTP::measure_bwd_into( auto &&res, auto &&grad_res, auto &&cell, auto &&facet_apex, auto &&grad_vp ) const {
    // the mirror of `measure_into`: the same two regimes, the same scratch, one dimension test.
    if constexpr ( ct_dim > 2 )
        cell.measure_bwd( res, facet_apex, grad_res, grad_vp );
    else
        cell.measure_bwd( res, grad_res, grad_vp );
}

UTP void DTP::scatter_cell_grad( SI i0, auto &&cell, auto &&grad_vp,
                                 auto &&grad_positions, auto &&grad_weights ) const {
    // asked for neither gradient: there is nothing to compute, and saying so at COMPILE time is
    // what keeps the solves out of a kernel that does not need them.
    if constexpr ( CT_VALUE( grad_positions.surely_null() ) && CT_VALUE( grad_weights.surely_null() ) ) {
        return;
    } else {
        // an infinite cell measures `TF::max`, a constant -- `measure_bwd` wrote nothing, and there
        // is nothing here either. Same for an empty one. Both matter: the work cells are REUSED
        // from one seed to the next, so reading `grad_vp` past what was written this round would
        // read the previous seed's cotangents.
        const SI nv = cell.nb_vertices;
        if ( ! cell.is_fully_bounded || nv == 0 )
            return;

        auto atomic_add_to = []( auto &&dst, TF v ) {
            if constexpr ( ! CT_VALUE( dst.surely_null() ) )
                atomic_add( dst.ref(), v );
        };

        const auto p0 = point( i0 );

        // seed `i0` is on EVERY plane of its own cell, so its share arrives once per vertex and per
        // cut. Summed in registers and atomically added once, rather than `ct_dim * nb_vertices`
        // times onto the same slot.
        auto acc_p0 = Vector<TF,ct_dim>::zeros();
        TF acc_w0 = 0;

        for ( SI v = 0; v < nv; ++v ) {
            const auto q = Vector<TF,ct_dim>::with_func( [&]( PI d ) { return TF( grad_vp( v, d ) ); } );
            const auto x = Vector<TF,ct_dim>::with_func( [&]( PI d ) { return TF( cell.vertex_positions( v, d ) ); } );

            // Vertex `v` is the corner of its `ct_dim` cuts (`Cell::vertex_cut`): it solves
            // `A x = b`, rows = cut directions, right-hand side = cut offsets. Differentiating,
            // `A dx = db - dA x`, so a cotangent `q` on `x` reaches the planes through `u` with
            // `A^T u = q`:  d offset_r -> u[ r ],  d direction_r -> - u[ r ] * x.
            // One `d x d` solve per vertex, and the TRANSPOSE is free -- it is how we fill it.
            const auto At = Matrix<TF,ct_dim>::with_func( [&]( auto r, auto c ) {
                return TF( cell.cut_directions( cell.vertex_cut( v, c ), r ) );
            } );
            const auto u = Matrix<TF,ct_dim>::solve_ge( At, q );

            for ( PI r = 0; r < ct_dim; ++r ) {
                const SI c = cell.vertex_cut( v, r );
                const SI i1 = SI( cell.cut_ids( c ) );
                if ( i1 < 0 )       // the domain (BOUNDARY) or the stand-in simplex (INFINITE)
                    continue;

                // the plane of the pair ( i0, i1 ): `dir = p1 - p0`, `off = ( |p1|² - |p0|² ) / 2
                // + ( w0 - w1 ) / 2` (see `make_cell`). Both differentiate on the spot.
                const TF g_off = u[ r ];
                const auto p1 = point( i1 );
                for ( PI d = 0; d < ct_dim; ++d ) {
                    const TF g_dir = - g_off * x[ d ];
                    atomic_add_to( grad_positions( i1, d ), g_dir + g_off * p1[ d ] );
                    acc_p0[ d ] -= g_dir + g_off * p0[ d ];
                }
                atomic_add_to( grad_weights( i1 ), - g_off / 2 );
                acc_w0 += g_off / 2;
            }
        }

        for ( PI d = 0; d < ct_dim; ++d )
            atomic_add_to( grad_positions( i0, d ), acc_p0[ d ] );
        atomic_add_to( grad_weights( i0 ), acc_w0 );
    }
}

UTP void DTP::measures_bwd( auto &&res, auto &&grad_res, auto &&grad_positions, auto &&grad_weights,
                            auto &&ws_0, auto &&ws_1, auto &&corr, auto &&facet_apex, auto &&grad_vp,
                            const auto &acc, auto &&acc_ws, const auto &dist, auto &&grad_dist,
                            auto &&piece_ws, SI thread_index, SI nb_threads ) const {
    // the forward's loop, run again: same seeds for this work-item, same buffers, same builds. The
    // cells are not residuals -- they were never kept -- so the only way back through them is to
    // make them once more. The pieces are not kept either, for the same reason and at the same
    // price: the distribution cuts them a second time.
    const SI n = nb_points;
    for ( SI i = thread_index; i < n; i += nb_threads ) {
        if ( make_cell( i, ws_0, ws_1, corr, acc, acc_ws ) )
            integrate_bwd_into( i, grad_res( i ), ws_0, facet_apex, grad_vp,
                                grad_positions, grad_weights, grad_dist, dist, piece_ws );
        else
            integrate_bwd_into( i, grad_res( i ), ws_1, facet_apex, grad_vp,
                                grad_positions, grad_weights, grad_dist, dist, piece_ws );
    }
}

UTP void DTP::build_cell( SI i, auto &&res, auto &&ws_0, auto &&ws_1, auto &&corr, const auto &acc, auto &&acc_ws ) const {
    // `make_cell` reports WHICH of the two buffers the ping-pong ended on; the copy is what turns a
    // transient work cell into a kept one. An overflow in the copy is reported the same way as one
    // in a cut -- nothing written, the count recorded, the host runs again with more room.
    if ( make_cell( i, ws_0, ws_1, corr, acc, acc_ws ) )
        ws_0.copy_into( res );
    else
        ws_1.copy_into( res );
}

#undef UTP
#undef DTP

}

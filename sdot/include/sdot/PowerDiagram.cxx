#pragma once

#include <loom/support/common_macros.h>
#include <loom/support/containers/Vector.h>
#include "Voronoi.h"

#define UTP SDOT_TEMPLATE_DECL_FOR_Voronoi
#define DTP Voronoi<SDOT_TEMPLATE_ARGS_FOR_Voronoi>

namespace sdot {

UTP auto DTP::point( SI i ) const {
    return Vector<TF,ct_dim>::with_func( [&]( PI d ) { return TF( positions( i, d ) ); } );
}

UTP bool DTP::cut_by( bool &in_0, auto &&ws_0, auto &&ws_1, auto &&corr, const auto &dir, TF off, SI cut_id ) const {
    bool ok;
    if constexpr ( ct_dim > 2 ) {
        // d > 2 rewrites the face lattice and COMPACTS it, which is what `corr` is for (the old ->
        // new index tables, one row per work-item). Below 3D the clip is a single cyclic pass with
        // nothing tabulated, and `corr` is not even allocated (a `NoneTensor` -- see `Voronoi.py`).
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

UTP bool DTP::make_cell( SI i0, auto &&ws_0, auto &&ws_1, auto &&corr ) const {
    bool in_0 = true;
    ws_0.init_as_unbounded();

    // ---- the domain. First, so that everything after it runs on a BOUNDED cell: an unbounded one
    // is only a stand-in whose artificial planes each cut has to push out again (`Cell.h`).
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

    // ---- the seeds. The cell of `i0` is the set of points closer to it than to any other seed,
    // so one half-space per other seed: the perpendicular bisector, written un-normalized (
    // `direction` need not be, `offset` is the dot product it is compared to). `cut_id` is the
    // NEIGHBOUR's index, so a surviving cut says which seed it faces.
    const auto p0 = point( i0 );
    const SI n = nb_points;
    for ( SI i1 = 0; i1 < n; ++i1 ) {
        if ( i1 == i0 )
            continue;

        const auto p1 = point( i1 );
        const auto dir = p1 - p0;
        TF off = 0;
        for ( PI d = 0; d < ct_dim; ++d )
            off += dir[ d ] * ( p0[ d ] + p1[ d ] ) / 2;

        if ( ! cut_by( in_0, ws_0, ws_1, corr, dir, off, i1 ) )
            break;

        // nothing left to clip. Not an anomaly -- two seeds at the very same place leave one of
        // them with an empty cell -- and going on would cost a full pass over the seeds for a cell
        // that can only stay empty.
        if ( ( in_0 ? SI( ws_0.nb_vertices ) : SI( ws_1.nb_vertices ) ) == 0 )
            return in_0;
    }

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

UTP void DTP::measures( auto &&res, auto &&ws_0, auto &&ws_1, auto &&corr, auto &&facet_apex,
                        SI thread_index, SI nb_threads ) const {
    const SI n = nb_points;
    for ( SI i = thread_index; i < n; i += nb_threads ) {
        // built, measured, forgotten -- in that order, and never two cells alive at once (bar the
        // ping-pong pair the clip needs). Writing the volume straight into its slot is what keeps
        // the diagram from ever existing as a whole.
        if ( make_cell( i, ws_0, ws_1, corr ) )
            measure_into( res( i ), ws_0, facet_apex );
        else
            measure_into( res( i ), ws_1, facet_apex );
    }
}

UTP void DTP::build_cell( SI i, auto &&res, auto &&ws_0, auto &&ws_1, auto &&corr ) const {
    // `make_cell` reports WHICH of the two buffers the ping-pong ended on; the copy is what turns a
    // transient work cell into a kept one. An overflow in the copy is reported the same way as one
    // in a cut -- nothing written, the count recorded, the host runs again with more room.
    if ( make_cell( i, ws_0, ws_1, corr ) )
        ws_0.copy_into( res );
    else
        ws_1.copy_into( res );
}

#undef UTP
#undef DTP

}

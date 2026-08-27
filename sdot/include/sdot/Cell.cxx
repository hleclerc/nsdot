#pragma once

#include <loom/support/common_macros.h>
#include <loom/support/containers/Matrix.h>
#include <loom/support/atomic_add.h>
#include <type_traits>
#include <limits>
#include "Cell.h"

#define UTP SDOT_TEMPLATE_DECL_FOR_Cell
#define DTP Cell<SDOT_TEMPLATE_ARGS_FOR_Cell>

namespace sdot {

UTP void DTP::init_as_aligned_simplex( SI cut_id ) {
    // `nb_dims` is a compile-time `Ct` member, but a member is read through `this`, and `this` is
    // never a constant expression -- so bare `if constexpr ( nb_dims > 2 )` cannot compile (it is
    // `this->nb_dims`). Shadow it with its value, reached through the TYPE (`decltype` does not
    // touch `this`): from here `nb_dims` is a plain `constexpr` int, usable in `if constexpr`.
    is_fully_bounded = cut_id != CellBoundary::INFINITE;
    bool ok = nb_vertices.set( ct_dim + 1 );
    if constexpr ( ct_dim > 2 )
        ok &= nb_edges.set( ( ct_dim + 1 ) * ct_dim / 2 );
    ok &= nb_cuts.set( ct_dim + 1 );
    if ( ! ok )
        return;

    // vertex_positions
    for( PI n = 0; n < nb_vertices; ++n )
        for( PI d = 0; d < ct_dim; ++d )
            vertex_positions( num_vertex = n, dim = d ) = ( d + 1 == n );

    // The face lattice of the simplex. Vertex 0 is the ORIGIN, vertex n >= 1 is e_{n-1}; cut c < d
    // is `x_c >= 0` and cut d is `sum x <= 1`. So vertex 0 stands on cuts { 0 .. d-1 } and vertex
    // n >= 1 on { 0 .. d-1 } \ { n-1 }, plus cut d -- `ct_dim` cuts each, in increasing order,
    // which is the form the rest of the file relies on (`cut`'s key matching, `vertex_cut`).
    if constexpr ( ct_dim > 2 ) {
        for( PI n = 0; n < nb_vertices; ++n ) {
            if ( n == 0 ) {
                for( PI r = 0; r < ct_dim; ++r )
                    vertex_indices( num_vertex = n, dim = r ) = r;
            } else {
                for( PI r = 0; r + 1 < ct_dim; ++r )
                    vertex_indices( num_vertex = n, dim = r ) = r + ( r >= n - 1 );
                vertex_indices( num_vertex = n, dim = ct_dim - 1 ) = ct_dim;
            }
        }
    }

    // edge_indices: an edge carries the cuts of BOTH its ends -- `ct_dim - 1` of them, the two
    // vertex lists intersected. `o` runs over the geometry -> bound it by the CLAMPED `nb_edges`,
    // so an under-provisioned buffer stops rather than corrupting (the overflow is already
    // recorded).
    if constexpr ( ct_dim > 2 ) {
        const SI ne = nb_edges;
        for ( PI a = 0, o = 0; a < nb_vertices && SI( o ) < ne; ++a ) {
            for ( PI b = a + 1; b < nb_vertices; ++b ) {
                if ( SI( o ) >= ne )
                    break;
                edge_indices( num_edge = o, ein = 0 ) = a;
                edge_indices( num_edge = o, ein = 1 ) = b;
                if ( a == 0 ) {
                    // { 0 .. d-1 } minus the one cut vertex b is off
                    for( PI r = 0; r + 1 < ct_dim; ++r )
                        edge_indices( num_edge = o, ein = 2 + r ) = r + ( r >= b - 1 );
                } else {
                    // { 0 .. d-1 } minus the two cuts a and b are off, then the `sum x <= 1` one
                    for( PI r = 0; r + 2 < ct_dim; ++r ) {
                        PI x = r + ( r >= a - 1 );
                        edge_indices( num_edge = o, ein = 2 + r ) = x + ( x >= b - 1 );
                    }
                    edge_indices( num_edge = o, ein = ct_dim ) = ct_dim;
                }
                ++o;
            }
        }
    }

    // cut_planes
    if constexpr ( ct_dim > 2 ) {
        // cut c < d is `-x_c <= 0`, cut d is `sum x <= 1`. The LAST loop writes row `ct_dim`, not
        // row `d`: writing `cut_directions( num_cut = d, ... )` would overwrite the diagonal of the
        // first `d` cuts and leave the closing plane unwritten.
        for( PI n = 0; n < ct_dim; ++n ) {
            for( PI d = 0; d < ct_dim; ++d )
                cut_directions( num_cut = n, dim = d ) = - TF( d == n );
            cut_offsets( num_cut = n ) = 0;
        }
        for( PI d = 0; d < ct_dim; ++d )
            cut_directions( num_cut = ct_dim, dim = d ) = 1;
        cut_offsets( num_cut = ct_dim ) = 1;
    } else if constexpr ( ct_dim == 2 ) {
        // 2D: the triangle ( 0, 0 ), ( 1, 0 ), ( 0, 1 ), its 3 cuts ordered to follow the CYCLIC
        // vertex order the whole d <= 2 path relies on ( see `Cell.h::measure` ).
        cut_directions( num_cut = 0, dim = 0 ) =  0; cut_directions( num_cut = 0, dim = 1 ) = -1; cut_offsets( num_cut = 0 ) = 0;
        cut_directions( num_cut = 1, dim = 0 ) = +1; cut_directions( num_cut = 1, dim = 1 ) = +1; cut_offsets( num_cut = 1 ) = 1;
        cut_directions( num_cut = 2, dim = 0 ) = -1; cut_directions( num_cut = 2, dim = 1 ) =  0; cut_offsets( num_cut = 2 ) = 0;
    } else {
        // 1D: the segment [ 0, 1 ] -- the same simplex, with one cut per end. Spelled out rather
        // than folded into the 2D case: there is no `dim = 1` to write here, so the lines above
        // would index past the end of a 1D `cut_directions`.
        cut_directions( num_cut = 0, dim = 0 ) = -1; cut_offsets( num_cut = 0 ) = 0;
        cut_directions( num_cut = 1, dim = 0 ) = +1; cut_offsets( num_cut = 1 ) = 1;
    }

    // cut_ids
    for( PI n = 0; n < nb_cuts; ++n )
        cut_ids( num_cut = n ) = cut_id;
}

UTP void DTP::init_as_hypercube( auto &&origin, auto &&axes, SI cut_id ) {
    if constexpr ( ! CT_VALUE( origin.is_valid() ) ) {
        init_as_hypercube( Vector<TF,ct_dim>::zeros(), axes, cut_id );
    } else if constexpr ( ! CT_VALUE( axes.is_valid() ) ) {
        init_as_hypercube( origin, Matrix<TF,ct_dim,ct_dim>::identity(), cut_id );
    } else if constexpr ( ! std::is_same_v<DECAYED_TYPE_OF( axes ),Matrix<TF,ct_dim,ct_dim>> ) {
        init_as_hypercube( origin, Matrix<TF,ct_dim,ct_dim>::with_func( [&]( auto r, auto c ) { return axes( r, c ); } ), cut_id );
    } else {
        is_fully_bounded = cut_id != CellBoundary::INFINITE;
        bool ok = nb_vertices.set( PI( 1 ) << ct_dim );
        if constexpr ( ct_dim > 2 )
            ok &= nb_edges.set( ct_dim * ( PI( 1 ) << ( ct_dim - 1 ) ) );
        ok &= nb_cuts.set( 2 * ct_dim );
        if ( ! ok )
            return;

        // shared: F^T[r][c] = axis_c[r], used to compute rows of F^{-1} via solve_ge

        // vertex_positions: origin + sum of selected axes; vertex_indices: cut 2b or 2b+1 per axis
        const PI vertex_ordering_2D[] = { 0, 1, 3, 2 };
        for ( PI k = 0; k < nb_vertices; ++k ) {
            const PI l = ( ct_dim != 2 ? k : vertex_ordering_2D[ k ] );
            for ( PI d = 0; d < ct_dim; ++d ) {
                TF pos = origin( d );
                for ( PI b = 0; b < ct_dim; ++b )
                    if ( ( k >> b ) & 1 )
                        pos += axes( b, d );
                vertex_positions( l, d ) = pos;
            }
        }

        // vertex_indices
        if constexpr ( ct_dim > 2 ) {
            for ( PI k = 0; k < nb_vertices; ++k )
                for ( PI b = 0; b < ct_dim; ++b )
                    vertex_indices( k, b ) = 2 * b + ( ( k >> b ) & 1 );
        }

        // edge_indices: edges in direction b, from vertex k (bit b=0) to k|(1<<b)
        // `e` is computed from the geometry, so bound it by the CLAMPED `nb_edges`: an
        // under-provisioned buffer stops here (the overflow was recorded by the `nb_edges = ...`
        // above), rather than corrupting -- the edge writes then stay raw, no check on the tensor.
        if constexpr ( ct_dim > 2 ) {
            const SI ne = nb_edges;
            for ( PI b = 0, e = 0; b < ct_dim && SI( e ) < ne; ++b ) {
                for ( PI k = 0; k < nb_vertices; ++k ) {
                    if ( ( k >> b ) & 1 )
                        continue;
                    if ( SI( e ) >= ne )
                        break;
                    edge_indices( e, 0 ) = k;
                    edge_indices( e, 1 ) = k | ( PI( 1 ) << b );
                    for ( PI d = 0, col = 2; d < ct_dim; ++d ) {
                        if ( d == b )
                            continue;
                        edge_indices( e, col++ ) = 2 * d + ( ( k >> d ) & 1 );
                    }
                    ++e;
                }
            }
        }

        // cut planes: row d of F^{-1} via shared FT
        const PI cut_ordering_2D[] = { 3, 1, 0, 2 };
        for ( PI d = 0; d < ct_dim; ++d ) {
            auto e_d = Vector<TF,ct_dim>::with_func( [&] ( PI i ) {
                return i == d ? TF( 1 ) : TF( 0 );
            } );
            const auto row = Matrix<TF,ct_dim>::solve_ge( axes, e_d );

            TF row_dot_origin = 0;
            for ( PI c = 0; c < ct_dim; ++c )
                row_dot_origin += row[ c ] * origin( c );

            const PI r0 = ( ct_dim != 2 ? 2 * d + 0 : cut_ordering_2D[ 2 * d + 0 ] );
            for ( PI c = 0; c < ct_dim; ++c )
                cut_directions( r0, c ) = -row[ c ];
            cut_offsets( r0 ) = -row_dot_origin;
            cut_ids( r0 ) = cut_id;

            const PI r1 = ( ct_dim != 2 ? 2 * d + 1 : cut_ordering_2D[ 2 * d + 1 ] );
            for ( PI c = 0; c < ct_dim; ++c )
                cut_directions( r1, c ) = row[ c ];
            cut_offsets( r1 ) = row_dot_origin + 1;
            cut_ids( r1 ) = cut_id;
        }
    }
}

UTP void DTP::init_as_unbounded() {
    init_as_aligned_simplex( CellBoundary::INFINITE );
}

UTP SI DTP::vertex_cut( SI i, SI r ) const {
    // Below 3D the face lattice is not stored: it is the ORDER. In 2D cut i carries the edge
    // leaving v_i, so v_i is the corner of cuts i-1 and i; in 1D the segment's two ends are its two
    // cuts, in the same order. Above, it is read off `vertex_indices` -- which the whole d > 2 path
    // keeps in increasing order (see `init_as_aligned_simplex`, `init_as_hypercube`, `cut`).
    if constexpr ( ct_dim == 1 ) {
        return i;
    } else if constexpr ( ct_dim == 2 ) {
        const SI nb = nb_vertices;
        return r == 0 ? ( i + nb - 1 ) % nb : i;
    } else {
        return SI( vertex_indices( i, r ) );
    }
}

UTP auto DTP::growth_rate( SI i ) const {
    // Vertex `i` is the CORNER of its `ct_dim` cuts (`vertex_cut`). Pushing the infinite planes out
    // by `g` adds `g` to their offsets and leaves the real ones where they are, so the corner
    // travels at the rate that solves the very same d x d system with those RATES on the
    // right-hand side. A vertex with no infinite plane on it gets a zero rate: it does not move --
    // so "is this an infinite vertex" needs no test of its own, it IS a nonzero rate.
    auto planes = Matrix<TF,ct_dim>::with_func( [&]( auto r, auto c ) {
        return TF( cut_directions( vertex_cut( i, r ), c ) );
    } );
    auto rates = Vector<TF,ct_dim>::with_func( [&]( PI r ) {
        return TF( SI( cut_ids( vertex_cut( i, r ) ) ) == CellBoundary::INFINITE );
    } );

    return Matrix<TF,ct_dim>::solve_ge( planes, rates );
}

UTP auto DTP::grown_vertex( SI i, TF g ) const {
    auto res = Vector<TF,ct_dim>::with_func( [&]( PI d ) { return TF( vertex_positions( i, d ) ); } );
    if ( g != 0 )                       // a bounded cell never grows: not one solve is paid for
        res += g * growth_rate( i );
    return res;
}

UTP auto DTP::growth_for_cut( auto &&direction, auto off ) const {
    // A bounded cell has no artificial plane left: its geometry is the true one, nothing to push.
    if ( is_fully_bounded )
        return TF( 0 );

    const SI nb = nb_vertices;
    TF g = 0;

    for ( int round = 0; round < max_growth_rounds; ++round ) {
        // Would pushing further move a vertex to the OTHER side of the cutting plane? Each vertex
        // travels a straight ray, so its signed distance is AFFINE in the push and the answer is
        // one division: `root` is the extra push at which it would cross. A vertex whose distance
        // does not vary with the push (no infinite plane on it, or a ray parallel to the cut) can
        // never change side, and is what makes this terminate.
        bool push = false;
        TF grow = 0;
        for ( SI i = 0; i < nb; ++i ) {
            const auto ray = growth_rate( i );
            TF rate = 0;
            for ( PI d = 0; d < ct_dim; ++d )
                rate += TF( direction( d ) ) * ray[ d ];
            if ( rate == 0 )
                continue;

            const auto v = grown_vertex( i, g );
            TF s = - TF( off );
            for ( PI d = 0; d < ct_dim; ++d )
                s += TF( direction( d ) ) * v[ d ];

            // `>= 0`, not `> 0`: a vertex sitting exactly ON the plane is not yet on the side it
            // would be on at infinity -- that is precisely a case to push out of.
            const TF root = - s / rate;
            if ( root >= 0 ) {
                push = true;
                if ( root > grow )
                    grow = root;
            }
        }

        // nothing would change side any more: the configuration is the one it has at infinity
        if ( ! push )
            return g;

        // ... otherwise push past the FURTHEST of them, and a bit BEYOND, so a vertex lands
        // strictly on the far side rather than on the plane. Overshooting is safe -- a vertex whose
        // ray does not cross cannot be made to cross by pushing further -- but it inflates the
        // artificial coordinates, so the margin stays small. It is also the one number here that a
        // degenerate configuration could make too tight; the loop is what catches that.
        g += grow + ( grow + 1 ) * growth_margin;
    }

    return g;
}

UTP void DTP::cut( auto &&res, auto &&direction, auto &&offset, SI cut_id ) const {
    static_assert( ct_dim == 2, "`cut` is 2D-only for now (see Cell.py::cut)" );

    // Sutherland-Hodgman on a CONVEX polygon held in cyclic order, run on both representations at
    // once. The invariant that makes this a single pass -- and that the orderings in
    // `init_as_hypercube` / `init_as_aligned_simplex` exist to establish -- is:
    //
    //     cut i carries the edge [ v_i, v_i+1 ]   (so nb_cuts == nb_vertices)
    //
    // Each output vertex is therefore written together with the cut that carries the edge LEAVING
    // it: one of ours when that edge is (a piece of) an edge we already had, the new half-space
    // when it is the fresh one the cut opens. Nothing is tabulated -- no per-vertex side array, no
    // index list -- so a thread needs a handful of registers and no scratch at all.
    const SI nb = nb_vertices;
    const TF off = offset;

    // An UNBOUNDED cell is only a stand-in (see `Cell.h`): its infinite planes are pushed out until
    // this cut classifies its vertices the way it would at infinity, and everything below then runs
    // on the pushed geometry. `g` is 0 for a bounded cell, where all of this vanishes.
    const TF g = growth_for_cut( direction, off );

    // Reserve the WORST case up front: clipping a convex polygon by a half-space keeps at most its
    // vertices plus one. Getting it out of the way here is what lets the loop below write without a
    // single bound test. If the capacity does not fit, `set` has already recorded what was asked
    // for -- the host reserves more and runs us again -- so all that is left is to stop before
    // writing anything (`ok &=`, never `||`: we want BOTH counts recorded in one go, not one
    // re-run per count).
    bool ok = res.nb_vertices.set( nb + 1 );
    ok &= res.nb_cuts.set( nb + 1 );
    if ( ! ok )
        return;

    // signed distance to the cutting plane. `direction` is NOT normalized and `offset` is the dot
    // product it is compared to, so this is the plain `n . v - o`.
    //
    // ONE predicate is read off it -- `s > 0`, "outside" -- and nothing else, in particular never
    // `s == 0`. A vertex the plane passes exactly through comes out at the rounding noise of that
    // dot product, so no test could tell it from a vertex a hair off the plane; asking is the
    // mistake, not the answer. Sorting it with `s <= 0` costs at worst an edge of length ~1e-16,
    // and buys a clip whose two sides are complementary BY CONSTRUCTION: `sj - si` is never zero
    // where it is divided by, and no case is left over to special-case.
    auto sd = [&]( const auto &v ) {
        TF r = -off;
        for ( PI d = 0; d < ct_dim; ++d )
            r += direction( d ) * v[ d ];
        return r;
    };

    // A cell is unbounded as long as one of the placeholder cuts of `init_as_unbounded` is still
    // part of it; the clip is what removes them, so the flag is recomputed from what survives.
    bool still_infinite = false;

    // the cut carrying the edge that leaves output vertex `k`: ours (`c >= 0`) or the new one. An
    // infinite plane of ours carries the offset it was PUSHED to, the one its vertices now sit on.
    constexpr SI NEW_CUT = -1;
    auto write_cut = [&]( SI k, SI c ) {
        const SI id = ( c >= 0 ? SI( cut_ids( c ) ) : cut_id );
        const bool infinite = ( id == CellBoundary::INFINITE );
        for ( PI d = 0; d < ct_dim; ++d )
            res.cut_directions( k, d ) = ( c >= 0 ? TF( cut_directions( c, d ) ) : TF( direction( d ) ) );
        res.cut_offsets( k ) = ( c >= 0 ? TF( cut_offsets( c ) ) + ( infinite ? g : TF( 0 ) ) : off );
        res.cut_ids( k ) = id;
        still_infinite |= infinite;
    };

    auto write_vertex = [&]( SI k, const auto &v ) {
        for ( PI d = 0; d < ct_dim; ++d )
            res.vertex_positions( k, d ) = v[ d ];
    };

    // the point where the edge `vi -> vj` crosses the plane, from the signed distances of its ends.
    // Written as the weighted average `( sj vi - si vj ) / ( sj - si )` rather than
    // `vi + t ( vj - vi )`: the two are the same in exact arithmetic, but the average is SYMMETRIC
    // in i/j, so an edge gives the very same point whichever end one starts from (and it does not
    // lose the low bits of `vi` to a `1 - t` cancellation).
    auto write_crossing = [&]( SI k, const auto &vi, const auto &vj, TF si, TF sj ) {
        for ( PI d = 0; d < ct_dim; ++d )
            res.vertex_positions( k, d ) = ( sj * vi[ d ] - si * vj[ d ] ) / ( sj - si );
    };

    SI k = 0;
    auto vi = nb ? grown_vertex( 0, g ) : Vector<TF,ct_dim>::zeros();
    TF si = nb ? sd( vi ) : 0;
    for ( SI i = 0; i < nb; ++i ) {
        const SI j = ( i + 1 ) % nb;
        const auto vj = grown_vertex( j, g );
        const TF sj = sd( vj );

        // whatever leaves an output vertex, cut i carries it: from v_i it is (a piece of) edge i,
        // and from the crossing that OPENS the surviving tail of edge i it is that tail. Only the
        // crossing that CLOSES edge i is different -- what leaves it runs along the new plane.
        if ( ! ( si > 0 ) ) {
            write_vertex( k, vi );
            write_cut( k, i );
            ++k;

            if ( sj > 0 ) {   // leaving
                write_crossing( k, vi, vj, si, sj );
                write_cut( k, NEW_CUT );
                ++k;
            }
        } else if ( ! ( sj > 0 ) ) {   // entering
            write_crossing( k, vi, vj, si, sj );
            write_cut( k, i );
            ++k;
        }

        vi = vj;   // each vertex ends two edges: pushed and measured once, not twice
        si = sj;
    }

    // `k <= nb + 1`, the capacity accepted above, so these cannot fail
    res.nb_vertices.set( k );
    res.nb_cuts.set( k );
    res.is_fully_bounded = ! still_infinite;
}

UTP void DTP::cut( auto &&res, auto &&direction, auto &&offset, SI cut_id, auto &&corr ) const {
    static_assert( ct_dim > 2, "this overload is the d > 2 one (see Cell.h)" );

    const SI nv = nb_vertices;
    const SI ne = nb_edges;
    const SI nc = nb_cuts;
    const TF off = offset;

    // An UNBOUNDED cell is only a stand-in: its infinite planes are pushed out until this cut
    // classifies its vertices the way it would at infinity, and everything below runs on the pushed
    // geometry. `g` is exactly 0 for a bounded cell, where all of this vanishes (see `Cell.h`).
    const TF g = growth_for_cut( direction, off );

    // What the output buffers can actually hold. Asking for one more than that records the
    // overflow and stops: the host then reserves at least twice as much and runs us again (see
    // `driver.call`), so a capacity is only ever a guess, never a contract.
    const SI cap_v = SI( res.vertex_positions.shape( 0 ) );
    const SI cap_e = SI( res.edge_indices.shape( 0 ) );
    const SI cap_c = SI( res.cut_directions.shape( 0 ) );

    // signed distance to the cutting plane. `direction` is NOT normalized and `offset` is the dot
    // product it is compared to, so this is the plain `n . v - o`.
    //
    // ONE predicate is read off it -- `s > 0`, "outside" -- and nothing else, in particular never
    // `s == 0`. A vertex the plane passes exactly through comes out at the rounding noise of that
    // dot product, so no test could tell it from a vertex a hair off the plane; asking is the
    // mistake, not the answer. Sorting it with `s <= 0` costs at worst an edge (or a facet) of
    // size ~1e-16, and buys a clip whose two sides are complementary BY CONSTRUCTION: the two ends
    // of a crossing edge are never on the same side, `so - si` is never zero where it is divided
    // by, and every vertex the clip creates carries the new cut -- so nothing is left over to
    // special-case, anywhere below.
    auto sd = [&]( const auto &v ) {
        TF r = -off;
        for ( PI d = 0; d < ct_dim; ++d )
            r += TF( direction( d ) ) * v[ d ];
        return r;
    };

    // `corr` is this work-item's own row of scratch, in two halves: `[ 0, nv )` maps an old vertex
    // to its new index (-1 when dropped), `[ nv, nv + nc ]` an old cut to its new one. The cut half
    // doubles as the "is this cut still standing" flag while it is being filled, so it starts at 0.
    const SI cut_base = nv;
    for ( SI c = 0; c <= nc; ++c )
        corr( cut_base + c ) = 0;

    // ---- the vertices: keep those inside, COMPACTED (a stable order, so the cut lists that point
    // at cuts -- and the cut renumbering below -- stay increasing).
    SI kv = 0;
    for ( SI v = 0; v < nv; ++v ) {
        const auto p = grown_vertex( v, g );
        if ( sd( p ) > 0 ) {
            corr( v ) = -1;
            continue;
        }
        if ( kv >= cap_v ) { res.nb_vertices.set( cap_v + 1 ); return; }
        corr( v ) = kv;
        for ( PI d = 0; d < ct_dim; ++d )
            res.vertex_positions( kv, d ) = p[ d ];
        for ( PI r = 0; r < ct_dim; ++r ) {
            const SI c = SI( vertex_indices( v, r ) );
            res.vertex_indices( kv, r ) = c;
            corr( cut_base + c ) = 1;
        }
        ++kv;
    }
    const SI nb_kept = kv;

    // every vertex outside: the half-space misses the cell entirely. (For an unbounded cell this is
    // conclusive too -- the push has driven the stand-in out until its classification stopped
    // changing, so nothing further out would come back in.)
    if ( nb_kept == 0 ) {
        res.nb_vertices.set( 0 );
        res.nb_edges.set( 0 );
        res.nb_cuts.set( 0 );
        res.is_fully_bounded = 1;
        return;
    }

    // ---- the edges, and with them the vertices the cut CREATES. The new cut takes index `nc` in
    // the old numbering -- past every existing one, which is what keeps the cut lists increasing.
    const SI new_cut = nc;
    SI ke = 0;
    for ( SI e = 0; e < ne; ++e ) {
        const SI a = SI( edge_indices( e, 0 ) ), b = SI( edge_indices( e, 1 ) );
        const SI ca = SI( corr( a ) ), cb = SI( corr( b ) );

        if ( ca < 0 && cb < 0 )     // wholly outside: gone, along with both its ends
            continue;

        if ( ca >= 0 && cb >= 0 ) { // wholly inside: carried over, ends renumbered
            if ( ke >= cap_e ) { res.nb_edges.set( cap_e + 1 ); return; }
            res.edge_indices( ke, 0 ) = ca;
            res.edge_indices( ke, 1 ) = cb;
            for ( PI r = 0; r + 1 < ct_dim; ++r )
                res.edge_indices( ke, 2 + r ) = edge_indices( e, 2 + r );
            ++ke;
            continue;
        }

        // crossing: the edge is cut in two, and the point where it meets the plane becomes a
        // vertex. It stands on the edge's own `ct_dim - 1` cuts, plus the new one -- exactly the
        // `ct_dim` a vertex needs.
        const SI i = ( ca >= 0 ? a : b ), o = ( ca >= 0 ? b : a );
        const auto pi = grown_vertex( i, g ), po = grown_vertex( o, g );
        const TF si = sd( pi ), so = sd( po );

        if ( kv >= cap_v ) { res.nb_vertices.set( cap_v + 1 ); return; }
        if ( ke >= cap_e ) { res.nb_edges.set( cap_e + 1 ); return; }

        // the weighted average `( so pi - si po ) / ( so - si )` rather than `pi + t ( po - pi )`:
        // the same point in exact arithmetic, but SYMMETRIC in the two ends, so an edge gives the
        // very same point whichever end one starts from (and no `1 - t` cancellation).
        for ( PI d = 0; d < ct_dim; ++d )
            res.vertex_positions( kv, d ) = ( so * pi[ d ] - si * po[ d ] ) / ( so - si );
        for ( PI r = 0; r + 1 < ct_dim; ++r ) {
            const SI c = SI( edge_indices( e, 2 + r ) );
            res.vertex_indices( kv, r ) = c;
            corr( cut_base + c ) = 1;
        }
        res.vertex_indices( kv, ct_dim - 1 ) = new_cut;
        corr( cut_base + new_cut ) = 1;

        // the surviving stump keeps the edge's cuts; only its outside end moves
        res.edge_indices( ke, 0 ) = ( ca >= 0 ? ca : cb );
        res.edge_indices( ke, 1 ) = kv;
        for ( PI r = 0; r + 1 < ct_dim; ++r )
            res.edge_indices( ke, 2 + r ) = edge_indices( e, 2 + r );

        ++ke;
        ++kv;
    }

    // ---- the edges of the NEW facet. Its corners are exactly the vertices the clip just created
    // -- they sit contiguously, at `[ nb_kept, kv )`, and each carries `ct_dim - 1` old cuts and
    // then the new one. Two vertices of a polytope are adjacent when they share all but one of its
    // facets: on the new facet that means `ct_dim - 2` of the OLD cuts. That is the whole
    // stitching -- read straight off the cut lists just written, all increasing, so the shared
    // cuts come out of a single merge, and come out SORTED.
    //
    // No map, deliberately: a table keyed by the `ct_dim - 2` shared cuts would be `nb_cuts^(d-2)`
    // words of global scratch to allocate, zero and then random-access, where the cut lists are
    // already here and the corners to pair are a handful. It shares the map's assumption, though:
    // both need each `ct_dim - 2` old cuts to meet the plane in a single pair of points, which a
    // facet lying flush with the cutting plane breaks.
    for ( SI m = nb_kept; m < kv; ++m ) {
        for ( SI n = m + 1; n < kv; ++n ) {
            Vector<SI,ct_dim> common;
            SI nb_common = 0;
            for ( SI x = 0, y = 0; x + 1 < SI( ct_dim ) && y + 1 < SI( ct_dim ); ) {
                const SI cx = SI( res.vertex_indices( m, x ) ), cy = SI( res.vertex_indices( n, y ) );
                if ( cx == cy ) { common[ nb_common++ ] = cx; ++x; ++y; }
                else if ( cx < cy ) ++x;
                else ++y;
            }
            if ( nb_common != SI( ct_dim ) - 2 )
                continue;

            if ( ke >= cap_e ) { res.nb_edges.set( cap_e + 1 ); return; }
            res.edge_indices( ke, 0 ) = m;
            res.edge_indices( ke, 1 ) = n;
            for ( SI r = 0; r < nb_common; ++r )
                res.edge_indices( ke, 2 + r ) = common[ r ];
            res.edge_indices( ke, ct_dim ) = new_cut;
            ++ke;
        }
    }

    // ---- the cuts. A cut nothing stands on any more is gone -- otherwise a cell would carry every
    // plane it was ever cut by. The scan is in increasing order, so the renumbering is MONOTONE and
    // the cut lists written above stay increasing once remapped.
    // `nc + 1` cuts to look at, the new one included: whether it is really there is not a separate
    // question, it is its used flag -- every vertex the clip creates names it, and it creates one
    // per crossing edge.
    SI kc = 0;
    bool still_infinite = false;
    for ( SI c = 0; c <= nc; ++c ) {
        if ( SI( corr( cut_base + c ) ) == 0 ) {
            corr( cut_base + c ) = -1;
            continue;
        }
        if ( kc >= cap_c ) { res.nb_cuts.set( cap_c + 1 ); return; }

        const bool is_new = ( c == new_cut );
        const SI id = ( is_new ? cut_id : SI( cut_ids( c ) ) );
        const bool infinite = ( id == CellBoundary::INFINITE );
        for ( PI d = 0; d < ct_dim; ++d )
            res.cut_directions( kc, d ) = ( is_new ? TF( direction( d ) ) : TF( cut_directions( c, d ) ) );
        // an infinite plane of ours carries the offset it was PUSHED to, the one its vertices sit on
        res.cut_offsets( kc ) = ( is_new ? off : TF( cut_offsets( c ) ) + ( infinite ? g : TF( 0 ) ) );
        res.cut_ids( kc ) = id;
        still_infinite |= infinite;

        corr( cut_base + c ) = kc;
        ++kc;
    }

    // ---- and the references to them. Every cut named by a surviving vertex or edge was marked
    // used above, so none of these lands on a dropped one.
    for ( SI v = 0; v < kv; ++v )
        for ( PI r = 0; r < ct_dim; ++r )
            res.vertex_indices( v, r ) = SI( corr( cut_base + SI( res.vertex_indices( v, r ) ) ) );
    for ( SI e = 0; e < ke; ++e )
        for ( PI r = 0; r + 1 < ct_dim; ++r )
            res.edge_indices( e, 2 + r ) = SI( corr( cut_base + SI( res.edge_indices( e, 2 + r ) ) ) );

    // all three fitted in the capacities checked along the way, so these cannot fail
    res.nb_vertices.set( kv );
    res.nb_edges.set( ke );
    res.nb_cuts.set( kc );
    res.is_fully_bounded = ! still_infinite;
}

UTP void DTP::crossing_bwd( auto &&direction, SI i, SI j, const auto &vi, const auto &vj, TF si, TF sj,
                            const auto &q, auto &&grad_cell, auto &&grad_direction, auto &&grad_offset ) const {
    // With `si = n.vi - o`, `sj = n.vj - o`, `D = sj - si` and `p = ( sj vi - si vj ) / D`, a
    // cotangent `q` on `p` gives, writing `u = ( vj - vi ) . q` (the only way `q` reaches the
    // scalars) and `h = q - ( u / D ) n`:
    //
    //     dp/dvi -> ( sj / D ) h        dp/dn -> u si / D^2 ( vj - vi ) - ( u / D ) vi
    //     dp/dvj -> ( -si / D ) h       dp/do -> u / D
    //
    // The cut planes are absent on purpose: a crossing is built from the two VERTICES of the edge,
    // never from intersecting planes, so nothing flows back to `cut_directions` here.
    auto add_to = []( auto &&dst, TF v ) {
        if constexpr ( ! CT_VALUE( dst.surely_null() ) )
            dst.ref() += v;
    };
    // `direction` / `offset` are shared by every item of the batch: all of them land here.
    auto atomic_add_to = []( auto &&dst, TF v ) {
        if constexpr ( ! CT_VALUE( dst.surely_null() ) )
            atomic_add( dst.ref(), v );
    };

    const TF D = sj - si;

    TF u = 0;
    for ( PI d = 0; d < ct_dim; ++d )
        u += ( vj[ d ] - vi[ d ] ) * q[ d ];

    for ( PI d = 0; d < ct_dim; ++d ) {
        const TF h = q[ d ] - u / D * TF( direction( d ) );

        add_to( grad_cell.vertex_positions( i, d ),   sj / D * h );
        add_to( grad_cell.vertex_positions( j, d ), - si / D * h );
        atomic_add_to( grad_direction( d ), u * si / ( D * D ) * ( vj[ d ] - vi[ d ] ) - u / D * vi[ d ] );
    }

    atomic_add_to( grad_offset, u / D );
}

UTP void DTP::cut_bwd_setup( auto &&queue, auto &&grad_cell, auto &&grad_direction, auto &&grad_offset ) const {
    // `cut_bwd` accumulates (see `Cell.h`), so every slot it adds into must start at zero. A fresh
    // FFI output buffer does NOT: the platform only seeds a SHARED float output of a BATCHED call
    // (`CallArg_Tensor.cpp_seed_member`), which leaves out the per-item ones here -- and leaves out
    // everything when the call carries no batch at all. Hence this pre-pass, run ONCE before any
    // item's body; `fill_with( queue, ... )` goes through the queue, so it is ordered before them.
    auto zero = [&]( auto &&t ) {
        // nothing to zero where nothing is written: a cotangent that was not asked for is a
        // `NoneTensor` (and a `ZeroTensor` has no storage either) -- decided at COMPILE time,
        // neither has `fill_with`.
        if constexpr ( ! CT_VALUE( t.surely_null() ) )
            t.fill_with( queue, TF( 0 ) );
    };

    zero( grad_cell.vertex_positions );
    zero( grad_cell.cut_directions );
    zero( grad_cell.cut_offsets );
    zero( grad_direction );
    zero( grad_offset );
}

UTP void DTP::cut_bwd( auto &&direction, auto &&offset, auto &&grad_res, auto &&grad_cell,
                       auto &&grad_direction, auto &&grad_offset ) const {
    static_assert( ct_dim == 2, "`cut` is 2D-only for now (see Cell.py::cut)" );

    // The forward's walk is replayed rather than recorded: it is a handful of dot products, where
    // remembering which output came from which input would be a buffer -- exactly what a kernel
    // cannot have. So the loop below is `cut`'s, line for line, with each write replaced by its
    // adjoint. Keep the two in step.
    const SI nb = nb_vertices;
    const TF off = offset;

    // Same push as the forward's, and NOT differentiated: how far an unbounded cell's artificial
    // planes are moved is a property of the stand-in, not of the geometry -- and a cell that is
    // still unbounded has an infinite measure anyway, so nothing meaningful flows through it. For a
    // bounded cell `g` is exactly 0 and this whole business is absent from the adjoint.
    const TF g = growth_for_cut( direction, off );

    auto gV = grad_res.vertex_positions;
    auto gD = grad_res.cut_directions;
    auto gO = grad_res.cut_offsets;

    // a cotangent that is not there reads as zero (`ZeroTensor`), and one that was never asked for
    // is not even readable (`NoneTensor`) -- both answered at compile time, so neither reaches
    // the arithmetic below.
    auto read = []( auto &&src ) {
        if constexpr ( CT_VALUE( src.surely_null() ) )
            return TF( 0 );
        else
            return TF( src );
    };
    auto add_to = []( auto &&dst, TF v ) {
        if constexpr ( ! CT_VALUE( dst.surely_null() ) )
            dst.ref() += v;
    };
    // `direction` / `offset` are shared by every item of the batch: all of them land here.
    auto atomic_add_to = []( auto &&dst, TF v ) {
        if constexpr ( ! CT_VALUE( dst.surely_null() ) )
            atomic_add( dst.ref(), v );
    };

    auto sd = [&]( const auto &v ) {
        TF r = -off;
        for ( PI d = 0; d < ct_dim; ++d )
            r += direction( d ) * v[ d ];
        return r;
    };

    // adjoint of `write_cut`: an output cut is a COPY, so its cotangent goes straight back to
    // whichever input it was copied from (the push added to an infinite offset is a constant).
    constexpr SI NEW_CUT = -1;
    auto cut_grad = [&]( SI k, SI c ) {
        for ( PI d = 0; d < ct_dim; ++d ) {
            const TF g_d = read( gD( k, d ) );
            if ( c >= 0 )
                add_to( grad_cell.cut_directions( c, d ), g_d );
            else
                atomic_add_to( grad_direction( d ), g_d );
        }

        const TF g_o = read( gO( k ) );
        if ( c >= 0 )
            add_to( grad_cell.cut_offsets( c ), g_o );
        else
            atomic_add_to( grad_offset, g_o );
    };

    // adjoint of `write_crossing` -- the arithmetic itself is `crossing_bwd` (shared with the
    // d > 2 clip, which meets the very same expression), so all this does is read the cotangent.
    auto crossing_grad = [&]( SI k, SI i, SI j, const auto &vi, const auto &vj, TF si, TF sj ) {
        auto q = Vector<TF,ct_dim>::with_func( [&]( PI d ) { return read( gV( k, d ) ); } );
        crossing_bwd( direction, i, j, vi, vj, si, sj, q, grad_cell, grad_direction, grad_offset );
    };

    SI k = 0;
    auto vi = nb ? grown_vertex( 0, g ) : Vector<TF,ct_dim>::zeros();
    TF si = nb ? sd( vi ) : 0;
    for ( SI i = 0; i < nb; ++i ) {
        const SI j = ( i + 1 ) % nb;
        const auto vj = grown_vertex( j, g );
        const TF sj = sd( vj );

        if ( ! ( si > 0 ) ) {
            for ( PI d = 0; d < ct_dim; ++d )               // v_i was copied over: identity
                add_to( grad_cell.vertex_positions( i, d ), read( gV( k, d ) ) );
            cut_grad( k, i );
            ++k;

            if ( sj > 0 ) {
                crossing_grad( k, i, j, vi, vj, si, sj );
                cut_grad( k, NEW_CUT );
                ++k;
            }
        } else if ( ! ( sj > 0 ) ) {
            crossing_grad( k, i, j, vi, vj, si, sj );
            cut_grad( k, i );
            ++k;
        }

        vi = vj;
        si = sj;
    }
}

UTP void DTP::cut_bwd( auto &&direction, auto &&offset, auto &&grad_res, auto &&grad_cell,
                       auto &&grad_direction, auto &&grad_offset, auto &&corr ) const {
    static_assert( ct_dim > 2, "this overload is the d > 2 one (see Cell.h)" );

    // `cut`'s walk, replayed line for line with each write replaced by its adjoint -- keep the two
    // in step. Two of its passes are absent, and only those: the NEW FACET's edges (an edge carries
    // no gradient, and stitching them changes neither `kv` nor `kc`) and the final renumbering of
    // the cut references (integers again -- but the renumbering's `kc` counter IS replayed, since
    // that is what says which input cut an output cut came from).
    const SI nv = nb_vertices;
    const SI ne = nb_edges;
    const SI nc = nb_cuts;
    const TF off = offset;

    // Same push as the forward's, and NOT differentiated: how far an unbounded cell's artificial
    // planes are moved is a property of the stand-in, not of the geometry -- and a cell that is
    // still unbounded has an infinite measure anyway. For a bounded cell `g` is exactly 0 and this
    // whole business is absent from the adjoint.
    const TF g = growth_for_cut( direction, off );

    auto gV = grad_res.vertex_positions;
    auto gD = grad_res.cut_directions;
    auto gO = grad_res.cut_offsets;

    // a cotangent that is not there reads as zero (`ZeroTensor`), and one that was never asked for
    // is not even readable (`NoneTensor`) -- both answered at compile time.
    auto read = []( auto &&src ) {
        if constexpr ( CT_VALUE( src.surely_null() ) )
            return TF( 0 );
        else
            return TF( src );
    };
    auto add_to = []( auto &&dst, TF v ) {
        if constexpr ( ! CT_VALUE( dst.surely_null() ) )
            dst.ref() += v;
    };
    auto atomic_add_to = []( auto &&dst, TF v ) {
        if constexpr ( ! CT_VALUE( dst.surely_null() ) )
            atomic_add( dst.ref(), v );
    };

    auto sd = [&]( const auto &v ) {
        TF r = -off;
        for ( PI d = 0; d < ct_dim; ++d )
            r += TF( direction( d ) ) * v[ d ];
        return r;
    };

    const SI cut_base = nv;
    for ( SI c = 0; c <= nc; ++c )
        corr( cut_base + c ) = 0;

    // ---- the vertices carried over: a plain copy, so the cotangent goes straight back
    SI kv = 0;
    for ( SI v = 0; v < nv; ++v ) {
        const auto p = grown_vertex( v, g );
        if ( sd( p ) > 0 ) {
            corr( v ) = -1;
            continue;
        }
        corr( v ) = kv;
        for ( PI d = 0; d < ct_dim; ++d )
            add_to( grad_cell.vertex_positions( v, d ), read( gV( kv, d ) ) );
        for ( PI r = 0; r < ct_dim; ++r )
            corr( cut_base + SI( vertex_indices( v, r ) ) ) = 1;
        ++kv;
    }
    if ( kv == 0 )      // empty result: nothing downstream, nothing to send back
        return;

    // ---- the crossings
    const SI new_cut = nc;
    for ( SI e = 0; e < ne; ++e ) {
        const SI a = SI( edge_indices( e, 0 ) ), b = SI( edge_indices( e, 1 ) );
        const SI ca = SI( corr( a ) ), cb = SI( corr( b ) );
        if ( ( ca < 0 ) == ( cb < 0 ) )
            continue;

        const SI i = ( ca >= 0 ? a : b ), o = ( ca >= 0 ? b : a );
        const auto pi = grown_vertex( i, g ), po = grown_vertex( o, g );
        const TF si = sd( pi ), so = sd( po );

        auto q = Vector<TF,ct_dim>::with_func( [&]( PI d ) { return read( gV( kv, d ) ); } );
        crossing_bwd( direction, i, o, pi, po, si, so, q, grad_cell, grad_direction, grad_offset );

        for ( PI r = 0; r + 1 < ct_dim; ++r )
            corr( cut_base + SI( edge_indices( e, 2 + r ) ) ) = 1;
        corr( cut_base + new_cut ) = 1;

        ++kv;
    }

    // ---- the cuts: an output cut is a COPY, so its cotangent goes straight back to whichever
    // input it was copied from (the push added to an infinite offset is a constant).
    SI kc = 0;
    for ( SI c = 0; c <= nc; ++c ) {
        if ( SI( corr( cut_base + c ) ) == 0 )
            continue;

        if ( c == new_cut ) {
            for ( PI d = 0; d < ct_dim; ++d )
                atomic_add_to( grad_direction( d ), read( gD( kc, d ) ) );
            atomic_add_to( grad_offset, read( gO( kc ) ) );
        } else {
            for ( PI d = 0; d < ct_dim; ++d )
                add_to( grad_cell.cut_directions( c, d ), read( gD( kc, d ) ) );
            add_to( grad_cell.cut_offsets( c ), read( gO( kc ) ) );
        }
        ++kc;
    }
}

UTP void DTP::init_as_hypercube_bwd( auto &&origin, auto &&axes, auto &&grad_cell, auto &&grad_for_origin, auto &&grad_for_axes ) const {
    // Adjoint of `init_as_hypercube`. Fwd (per axis d, with B = axes^{-1}, column d = axes.solve_ge(e_d)):
    //   vertex_positions(l(k), c) = origin(c) + Σ_{b: bit b of k} axes(b, c)
    //   cut_directions(r0,c) = -B(c,d),  cut_offsets(r0) = -P_d
    //   cut_directions(r1,c) = +B(c,d),  cut_offsets(r1) = +P_d + 1,   with P_d = Σ_c B(c,d) origin(c).
    //
    // With cotangents gV / gD / gO on those three outputs, and
    //   G_d    = gO(r1) - gO(r0)            (sensitivity through the offset)
    //   H(d,c) = gD(r1,c) - gD(r0,c)        (sensitivity through the direction)
    //   W(c,d) = G_d * origin(c) + H(d,c)   (total sensitivity through B(c,d), since P_d carries origin)
    // the gradients are, using dB = -B daxes B :
    //   grad_origin(c) = Σ_l gV(l,c)                       +  Σ_d G_d * B(c,d)
    //   grad_axes      = Σ_{k: bit b} gV(l(k),·) [vertex]  +  ( -B^T W B^T ) [cuts]
    //
    // The stale draft below solved with axes^T (a transpose off from the current forward, which uses
    // `axes.solve_ge` directly); this version rebuilds B exactly as the forward does.

    using Mat = Matrix<TF,ct_dim>;
    using Vec = Vector<TF,ct_dim>;

    auto gV = grad_cell.vertex_positions;   // cotangent of vertex_positions (real or ZeroTensor)
    auto gD = grad_cell.cut_directions;     // cotangent of cut_directions
    auto gO = grad_cell.cut_offsets;        // cotangent of cut_offsets

    const PI vertex_ordering_2D[] = { 0, 1, 3, 2 };
    const PI cut_ordering_2D[]    = { 3, 1, 0, 2 };

    // B = axes^{-1}, column d rebuilt exactly like the forward's `row` (so a degenerate cell, where
    // solve_ge leaves a zero pivot at 0, differentiates consistently with how it was built).
    // `solve_ge` is static and copies its argument, so it takes `axes` as a bare TensorView directly.
    Mat B;
    for ( PI d = 0; d < ct_dim; ++d ) {
        auto e_d = Vec::with_func( [&]( PI i ) { return i == d ? TF( 1 ) : TF( 0 ); } );
        const auto col = Mat::solve_ge( axes, e_d );
        for ( PI c = 0; c < ct_dim; ++c )
            B( c, d ) = col[ c ];
    }

    // per-axis cut sensitivities G_d and H(d,·)
    Vec G;
    Mat H;
    for ( PI d = 0; d < ct_dim; ++d ) {
        const PI r0 = ( ct_dim != 2 ? 2 * d + 0 : cut_ordering_2D[ 2 * d + 0 ] );
        const PI r1 = ( ct_dim != 2 ? 2 * d + 1 : cut_ordering_2D[ 2 * d + 1 ] );
        G[ d ] = gO( r1 ) - gO( r0 );
        for ( PI c = 0; c < ct_dim; ++c )
            H( d, c ) = gD( r1, c ) - gD( r0, c );
    }

    // grad wrt origin. A NoneTensor (origin not perturbed) has no `operator=`, so the whole block
    // must be dropped at COMPILE time -- `if constexpr` on `is_valid()`, never a runtime `if`.
    if constexpr ( CT_VALUE( grad_for_origin.is_valid() ) ) {
        for ( PI c = 0; c < ct_dim; ++c ) {
            TF g = 0;                                   // cut part: Σ_d G_d * B(c,d)
            for ( PI d = 0; d < ct_dim; ++d )
                g += G[ d ] * B( c, d );
            grad_for_origin( c ) = g;
        }
        for ( PI k = 0; k < ( PI( 1 ) << ct_dim ); ++k ) {   // vertex part: Σ_l gV(l,c)
            const PI l = ( ct_dim != 2 ? k : vertex_ordering_2D[ k ] );
            for ( PI c = 0; c < ct_dim; ++c )
                grad_for_origin( c ) += gV( l, c );
        }
    }

    // grad wrt axes = -B^T W B^T (cuts) + the vertex contribution. Same compile-time guard.
    if constexpr ( CT_VALUE( grad_for_axes.is_valid() ) ) {
        Mat W;
        for ( PI c = 0; c < ct_dim; ++c )
            for ( PI d = 0; d < ct_dim; ++d )
                W( c, d ) = G[ d ] * origin( c ) + H( d, c );

        // Y = W B^T : Y(c,b) = Σ_d W(c,d) B(b,d)
        Mat Y;
        for ( PI c = 0; c < ct_dim; ++c )
            for ( PI b = 0; b < ct_dim; ++b ) {
                TF y = 0;
                for ( PI d = 0; d < ct_dim; ++d )
                    y += W( c, d ) * B( b, d );
                Y( c, b ) = y;
            }

        // grad_axes(a,b) = -(B^T Y)(a,b) = -Σ_c B(c,a) Y(c,b)
        for ( PI a = 0; a < ct_dim; ++a )
            for ( PI b = 0; b < ct_dim; ++b ) {
                TF s = 0;
                for ( PI c = 0; c < ct_dim; ++c )
                    s += B( c, a ) * Y( c, b );
                grad_for_axes( a, b ) = -s;
            }

        // vertex part: grad_axes(b,c) += Σ_{k: bit b of k} gV(l(k), c)
        for ( PI k = 0; k < ( PI( 1 ) << ct_dim ); ++k ) {
            const PI l = ( ct_dim != 2 ? k : vertex_ordering_2D[ k ] );
            for ( PI b = 0; b < ct_dim; ++b )
                if ( ( k >> b ) & 1 )
                    for ( PI c = 0; c < ct_dim; ++c )
                        grad_for_axes( b, c ) += gV( l, c );
        }
    }
}

// ---- d <= 2 -----------------------------------------------------------------------------------
// `vertex_positions` IS the geometry here: a segment in 1D, a CYCLICALLY ordered polygon in 2D
// (`init_as_hypercube` lays the vertices out in that order, see `vertex_ordering_2D`). Nothing has
// to be enumerated, so this pair takes NO scratch -- `Cell.py::measure` short-paths to it and does
// not even build the `item_map` / `nb_map_items` the d > 2 pair below needs.

UTP void DTP::measure_bwd( auto &&res, auto &&grad_res, auto &&grad_vertex_positions ) const {
    static_assert( ct_dim <= 2, "this overload is the d <= 2 one (see Cell.h)" );

    // infinite cell: the measure is a constant, so it carries no gradient
    if ( ! is_fully_bounded )
        return;

    // a `ZeroTensor` / `NoneTensor` cotangent has no `operator=`: the whole block has to go at
    // COMPILE time, never behind a runtime `if`.
    if constexpr ( ! CT_VALUE( grad_vertex_positions.surely_null() ) ) {
        const SI nb_vertices = this->nb_vertices;
        if ( nb_vertices == 0 )
            return;

        // Both branches write EVERY vertex of the output (one expression per element), rather than
        // accumulating into a shared slot: this call carries no batch axis, so
        // `CallArg_Tensor.cpp_seed_member` does not zero-seed `grad_vertex_positions` (its rule
        // assumes a per-item output is written once) -- a `+=` would add onto whatever was left in
        // the freshly-allocated device buffer.
        if constexpr ( ct_dim == 1 ) {
            // 1D: measure = x(i_max) - x(i_min), so only those two endpoints move it.
            SI i_min = 0, i_max = 0;
            for ( SI i = 1; i < nb_vertices; ++i ) {
                if ( vertex_positions( i, 0 ) < vertex_positions( i_min, 0 ) ) i_min = i;
                if ( vertex_positions( i, 0 ) > vertex_positions( i_max, 0 ) ) i_max = i;
            }
            for ( SI i = 0; i < nb_vertices; ++i )
                grad_vertex_positions( i, 0 ) = grad_res * ( TF( i == i_max ) - TF( i == i_min ) );
        } else {
            // 2D: shoelace adjoint. Each vertex gets a contribution from its two neighboring edges.
            for ( SI i = 0; i < nb_vertices; ++i ) {
                const SI p = ( i + nb_vertices - 1 ) % nb_vertices;
                const SI n = ( i + 1 ) % nb_vertices;
                grad_vertex_positions( i, 0 ) = grad_res * ( vertex_positions( n, 1 ) - vertex_positions( p, 1 ) ) / 2;
                grad_vertex_positions( i, 1 ) = grad_res * ( vertex_positions( p, 0 ) - vertex_positions( n, 0 ) ) / 2;
            }
        }
    }
}

UTP void DTP::measure( auto &&res ) const {
    static_assert( ct_dim <= 2, "this overload is the d <= 2 one (see Cell.h)" );

    // infinite cell
    if ( ! is_fully_bounded ) {
        res = std::numeric_limits<TF>::max();
        return;
    }

    const SI nb_vertices = this->nb_vertices;
    if ( nb_vertices == 0 ) {
        res = 0;
        return;
    }

    if constexpr ( ct_dim == 1 ) {
        // 1D: the cell is a segment -- its measure is the gap between its two endpoints. Taken as
        // max - min rather than |v(1) - v(0)|, so it does not depend on which end came first.
        TF mi = vertex_positions( 0, 0 ), ma = mi;
        for ( SI i = 1; i < nb_vertices; ++i ) {
            const TF x = vertex_positions( i, 0 );
            if ( x < mi ) mi = x;
            if ( x > ma ) ma = x;
        }
        res = ma - mi;
    } else {
        // 2D: shoelace formula, straight on the cyclically ordered vertices.
        TF sum = 0;
        for ( SI i = 0; i < nb_vertices; ++i ) {
            const SI j = ( i + 1 ) % nb_vertices;
            sum += vertex_positions( i, 0 ) * vertex_positions( j, 1 )
                 - vertex_positions( j, 0 ) * vertex_positions( i, 1 );
        }
        res = sum / 2;
    }
}

// ---- d > 2 ------------------------------------------------------------------------------------
// No formula to read off the vertices any more: the cell has to be CUT INTO SIMPLICES first, and
// that is a walk on the face lattice (`vertex_indices`). See `Cell.h` for what `facet_apex` is and
// why it is `ct_dim * nb_cuts` words instead of the `nb_cuts^(d-1)` a map keyed by cut sets takes.

UTP bool DTP::has_cut( SI v, SI c ) const {
    for ( PI r = 0; r < ct_dim; ++r )
        if ( SI( vertex_indices( v, r ) ) == c )
            return true;
    return false;
}

UTP void DTP::for_each_simplex( auto &&facet_apex, auto &&func ) const {
    static_assert( ct_dim > 2, "this walk is the d > 2 one (see Cell.h)" );

    if ( nb_vertices == 0 )
        return;

    // `chain` collects the apexes as the recursion goes down -- `ct_dim + 1` of them by the time it
    // reaches a vertex, which is exactly a simplex. `face_cuts` is the current face: the cuts
    // picked so far, in the order they were picked.
    Vector<SI,ct_dim+1> chain;
    Vector<SI,ct_dim> face_cuts;
    chain[ 0 ] = 0;   // ANY vertex of the cell will do as the apex of the cell itself
    for_each_simplex_rec( facet_apex, chain, face_cuts, func, Ct<int,ct_dim>() );
}

UTP void DTP::for_each_simplex_rec( auto &&facet_apex, auto &chain, auto &face_cuts, auto &&func, auto face_dim ) const {
    constexpr int k = DECAYED_TYPE_OF( face_dim )::value;   // the dimension of the current face G
    constexpr SI depth = ct_dim - k;                        // == the number of cuts that define it

    // G is a vertex: `chain` holds `ct_dim + 1` apexes, one per dimension on the way down -- a
    // simplex of the triangulation.
    if constexpr ( k == 0 ) {
        func( chain );
    } else {
        const SI nv = nb_vertices;
        const SI nc = SI( facet_apex.shape( 1 ) );   // >= nb_cuts: the slots past it stay empty
        const SI p = chain[ depth ];                 // the apex of G, fixed for this whole subtree

        // The facets of G, and one vertex of each, in ONE pass over the vertices of G: a vertex of
        // G carries the cuts of G plus `k` others, and each of those others names a facet. The
        // first vertex seen carrying a cut sits on that cut, hence on that facet -- and being on it
        // is all an apex has to be, since the triangulation of a face only needs its apex to be
        // FIXED, not canonical.
        for ( SI c = 0; c < nc; ++c )
            facet_apex( depth, c ) = -1;

        for ( SI v = 0; v < nv; ++v ) {
            bool on_g = true;
            for ( SI m = 0; m < depth; ++m )
                on_g &= has_cut( v, face_cuts[ m ] );
            if ( ! on_g )
                continue;

            for ( PI r = 0; r < ct_dim; ++r ) {
                const SI c = SI( vertex_indices( v, r ) );
                bool in_g = false;
                for ( SI m = 0; m < depth; ++m )
                    in_g |= ( face_cuts[ m ] == c );
                if ( ! in_g && SI( facet_apex( depth, c ) ) < 0 )
                    facet_apex( depth, c ) = v;
            }
        }

        // ... then cone `p` over the facets that do NOT contain it. Those that do would give
        // flat simplices: they are the ones the cone from `p` covers with zero volume, and
        // dropping them is what makes this a triangulation rather than an overlapping cover.
        for ( SI c = 0; c < nc; ++c ) {
            const SI a = SI( facet_apex( depth, c ) );
            if ( a < 0 || has_cut( p, c ) )
                continue;

            face_cuts[ depth ] = c;
            chain[ depth + 1 ] = a;
            for_each_simplex_rec( facet_apex, chain, face_cuts, func, Ct<int,k-1>() );
        }
    }
}

UTP void DTP::measure_bwd( auto &&res, auto &&facet_apex, auto &&grad_res, auto &&grad_vertex_positions ) const {
    static_assert( ct_dim > 2, "this overload is the d > 2 one (see Cell.h)" );

    // infinite cell: the measure is a constant, so it carries no gradient
    if ( ! is_fully_bounded )
        return;

    // a `ZeroTensor` / `NoneTensor` cotangent has no `operator=`: the whole block has to go at
    // COMPILE time, never behind a runtime `if`.
    if constexpr ( ! CT_VALUE( grad_vertex_positions.surely_null() ) ) {
        // this ACCUMULATES ( a vertex belongs to many simplices ), and the call carries no batch
        // axis, so nothing zero-seeds the buffer for us -- see `cut_bwd_setup` for the same story.
        // The whole buffer, padding included: a cotangent has to have a value everywhere its
        // primal does.
        const SI cap_v = SI( grad_vertex_positions.shape( 0 ) );
        for ( SI v = 0; v < cap_v; ++v )
            for ( PI d = 0; d < ct_dim; ++d )
                grad_vertex_positions( v, d ) = 0;

        TF fact = 1;
        for ( int i = 2; i <= ct_dim; ++i )
            fact *= i;
        const TF g = TF( grad_res ) / fact;

        // Adjoint of `abs( det M )` with `M( r, c ) = v_{chain[c+1]}[r] - v_{chain[0]}[r]`:
        // `d|det|/dM = sign( det ) * cofactor( M )`, and each column of M is one apex minus the
        // first, so the first apex collects MINUS the sum of the columns.
        for_each_simplex( facet_apex, [&]( const auto &chain ) {
            auto M = Matrix<TF,ct_dim>::with_func( [&]( auto r, auto c ) {
                return TF( vertex_positions( chain[ c + 1 ], r ) ) - TF( vertex_positions( chain[ 0 ], r ) );
            } );
            const TF det = M.determinant();
            const TF s = ( det < 0 ? -g : g );

            for ( PI r = 0; r < ct_dim; ++r ) {
                TF row_sum = 0;
                for ( PI c = 0; c < ct_dim; ++c ) {
                    const TF minor = M.without_row_and_col( r, c ).determinant();
                    const TF cof = ( ( r + c ) % 2 ? -minor : minor ) * s;
                    grad_vertex_positions( chain[ c + 1 ], r ) += cof;
                    row_sum += cof;
                }
                grad_vertex_positions( chain[ 0 ], r ) -= row_sum;
            }
        } );
    }
}

UTP void DTP::measure( auto &&res, auto &&facet_apex ) const {
    static_assert( ct_dim > 2, "this overload is the d > 2 one (see Cell.h)" );

    // infinite cell
    if ( ! is_fully_bounded ) {
        res = std::numeric_limits<TF>::max();
        return;
    }

    // the simplices tile the cell and do not overlap, so this sum has no cancellation in it: every
    // term is a volume, taken positive.
    TF sum = 0;
    for_each_simplex( facet_apex, [&]( const auto &chain ) {
        auto M = Matrix<TF,ct_dim>::with_func( [&]( auto r, auto c ) {
            return TF( vertex_positions( chain[ c + 1 ], r ) ) - TF( vertex_positions( chain[ 0 ], r ) );
        } );
        const TF det = M.determinant();
        sum += ( det < 0 ? -det : det );
    } );

    TF fact = 1;
    for ( int i = 2; i <= ct_dim; ++i )
        fact *= i;
    res = sum / fact;
}

}

#pragma once

#include "sdot/support/common_macros.h"
#include "support/containers/Matrix.h"
#include <type_traits>
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

    // vertex_inds
    if constexpr ( ct_dim > 2 )
        for( PI num_vertex = 0; num_vertex < nb_vertices; ++num_vertex )
            for( PI d = 0; d < ct_dim; ++d )
                vertex_indices( num_vertex, d ) = d + ( d >= num_vertex );

    // edge_indices ( `o` runs over the geometry -> bound it by the CLAMPED `nb_edges`, so an
    // under-provisioned buffer stops rather than corrupting; the overflow is already recorded )
    if constexpr ( ct_dim > 2 ) {
        const SI ne = nb_edges;
        for ( PI a = 0, o = 0; a < nb_vertices && SI( o ) < ne; ++a ) {
            for ( PI b = a + 1; b < nb_vertices; ++b ) {
                if ( a != b ) {
                    if ( SI( o ) >= ne )
                        break;
                    edge_indices( num_edge = o, ein = 0 ) = a;
                    edge_indices( num_edge = o, ein = 1 ) = b;
                    for( PI d = 0; d < ct_dim - 1; ++d )
                        edge_indices( o, ein = 2 + d ) = d + ( d >= a ) + ( d >= b - 1 );
                    ++o;
                }
            }
        }
    }

    // cut_planes
    if constexpr ( ct_dim > 2 ) {
        for( PI n = 0; n < ct_dim; ++n ) {
            for( PI d = 0; d < ct_dim; ++d )
                cut_directions( num_cut = n, dim = d ) = - ( d == n );
            cut_offsets( num_cut = n ) = 0;
        }
        for( PI d = 0; d < ct_dim; ++d )
            cut_directions( num_cut = d, dim = d ) = 1;
        cut_offsets( num_cut = dim ) = 1;
    } else {
        cut_directions( num_cut = 0, dim = 0 ) =  0; cut_directions( num_cut = 0, dim = 1 ) = -1; cut_offsets( num_cut = 0 ) = 0;
        cut_directions( num_cut = 1, dim = 0 ) = +1; cut_directions( num_cut = 1, dim = 1 ) = +1; cut_offsets( num_cut = 1 ) = 1;
        cut_directions( num_cut = 2, dim = 0 ) = -1; cut_directions( num_cut = 2, dim = 1 ) =  0; cut_offsets( num_cut = 2 ) = 0;
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

UTP void DTP::measure( auto &&res, auto &&item_map, auto &&nb_map_items ) const {
    // infinite cell
    if ( ! is_fully_bounded ) {
        res = std::numeric_limits<TF>::max();
        return;
    }

    // 2D: shoelace formula
    if ( ct_dim == 2 ) {
        const SI nb_vertices = this->nb_vertices();
        TF sum = 0;
        for ( SI i = 0; i < nb_vertices; ++i ) {
            const SI j = ( i + 1 ) % nb_vertices;
            sum += vertex_positions( i, 0 ) * vertex_positions( j, 1 )
                 - vertex_positions( j, 0 ) * vertex_positions( i, 1 );
        }
        res = sum / 2;
        return;
    }

    // nD: fan triangulation
    INFO( item_map.shape(), nb_map_items );
    res = 32;
    // TF sum = 0;
    // for_each_simplex( item_map, [&] ( const auto &simplex_indices ) {
    //     const TI v0 = simplex_indices[ 0 ];
    //     Matrix<TF,ct_dim> M = Matrix<TF,ct_dim>::with_func( [&]( auto row, auto col ) {
    //         return vertex_positions( simplex_indices[ col + 1 ], row ) - vertex_positions( v0, row );
    //     } );
    //     sum += std::abs( M.determinant() );
    // } );

    // return sum / factorial( ct_dim );
}

}

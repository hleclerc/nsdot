#pragma once

#include <loom/support/algorithms/CartesianIndices.h> // iterate the (dynamic-rank) cell grid
#include <loom/support/containers/IotaTensor.h>       // IotaTensor<TF> (knots = 0,1,2) -- default `knots`
#include <loom/support/containers/Vector.h>           // Vector<TF,d>::zeros()          -- default `origin`
#include <loom/support/containers/Matrix.h>
#include <loom/support/atomic_add.h>
#include "Cell/CellBoundary.h"
#include "ConstantDensity.h"
#include "CstUdPiece.h"
#include "Image.h"
#include <cmath>

#define UTP SDOT_TEMPLATE_DECL_FOR_Image
#define DTP Image<SDOT_TEMPLATE_ARGS_FOR_Image>

namespace sdot {

UTP auto DTP::with_defaults( auto &&cont ) const {
    // Substitute one absent member per step, then recurse -- exactly the shape of
    // `Cell::init_as_hypercube`'s default handling, but rebuilding the WHOLE aggregate (a new
    // template instantiation, deduced by C++20 aggregate CTAD, like the generated `make_available`)
    // rather than passing individual defaults down. We must rebuild, not mutate: a `NoneTensor` has
    // a fixed type and no `operator=`, so replacing its VALUE means replacing its TYPE.
    //
    // The brace-init is POSITIONAL, so its order must match `SDOT_ATTRIBUTES_OF_Image` (the field
    // declaration order across the MRO, axes skipped): target_mass, nb_dims, shape, values, origin,
    // frame, knots, current_mass, nb_cells_cum, cell_cum_mass. `::sdot::Image` (qualified) names the
    // TEMPLATE so CTAD re-deduces; bare `Image` would mean the current instantiation and defeat the
    // substitution.
    if constexpr ( ! CT_VALUE( origin.is_valid() ) )
        return ::sdot::Image{ target_mass, nb_dims, shape, values, Vector<TF,ct_dim>::zeros(), frame, knots, current_mass, nb_cells_cum, cell_cum_mass }.with_defaults( FORWARD( cont ) );
    else if constexpr ( ! CT_VALUE( frame.is_valid() ) )
        return ::sdot::Image{ target_mass, nb_dims, shape, values, origin, Matrix<TF,ct_dim>::identity(), knots, current_mass, nb_cells_cum, cell_cum_mass }.with_defaults( FORWARD( cont ) );
    else if constexpr ( ! CT_VALUE( knots.is_valid() ) )
        return ::sdot::Image{ target_mass, nb_dims, shape, values, origin, frame, IotaTensor<TF>{}, current_mass, nb_cells_cum, cell_cum_mass }.with_defaults( FORWARD( cont ) );
    else
        return cont( *this );
}

UTP typename DTP::TF DTP::measure() const {
    return with_defaults( []( auto &&img ) {
        using ImgT = DECAYED_TYPE_OF( img );
        using TF = typename ImgT::TF;
        constexpr int d = ImgT::ct_dim;

        // |det(frame)| -- rebuild a dense d*d matrix so this works whether `frame` is a `Matrix`
        // (the identity default) or a user-given `TensorView`.
        const auto F = Matrix<TF,d>::with_func( [&]( auto r, auto c ) { return TF( img.frame( r, c ) ); } );
        const TF fdet = std::abs( F.determinant() );

        // sum over the cells (extent per axis = number of cells along it = `values.shape()`).
        // Each cell contributes value * |det(frame)| * Prod_axis ( knots(axis, i+1) - knots(axis, i) ).
        // `values` is an ordinary dense rank-d view (the `img_pos` AxisList unrolls into d named
        // dimensions), so it and `knots` are both indexed positionally.
        auto shape = img.values.shape();
        CartesianIndices<DECAYED_TYPE_OF( shape )> cells{ shape };
        TF sum = 0;
        for ( PI flat = 0; flat < cells.size(); ++flat ) {
            const TF cell = cells[ flat ].apply_values( [&]( auto ...i ) {
                // running axis counter over the index pack (axes 0..d-1, in order)
                PI axis = 0;
                TF spacing = 1;
                ( ( spacing *= TF( img.knots( axis, i + 1 ) ) - TF( img.knots( axis, i ) ), ++axis ), ... );
                return TF( img.values( i... ) ) * spacing;
            } );
            sum += cell * fdet;
        }
        return sum;
    } );
}

namespace detail {
    /// Un multi-indice RUNTIME (`Vector<SI,d>`) rendu en `Tuple`, que `TensorView::offset` déplie
    /// tout seul -- de quoi écrire `values( k )` avec un `k` dont les composantes sont calculées.
    /// Même récursion `Ct` que `unravel_index` (voir `CartesianIndices.h`).
    auto image_index_tuple( const auto &k, auto &&acc, auto done ) {
        constexpr int i = DECAYED_TYPE_OF( done )::value;
        constexpr int n = DECAYED_TYPE_OF( k.size() )::value;
        if constexpr ( i == n )
            return acc;
        else
            return image_index_tuple( k, acc.with_appended_value( SI( k[ i ] ) ), Ct<int,i+1>() );
    }
}

UTP SI DTP::knot_index( SI axis, TF t, SI nb_cells ) const {
    SI b = 0, e = nb_cells;                 // on cherche dans [ b, e ), invariant : la réponse y est
    while ( e - b > 1 ) {
        const SI m = ( b + e ) / 2;
        if ( TF( knots( axis, m ) ) <= t ) b = m;
        else                               e = m;
    }
    return b;
}

UTP void DTP::for_each_piece( const auto &cell, auto &&ws, auto &&func ) const {
    with_defaults( [&]( auto &&img ) { img._for_each_piece( cell, ws, func ); } );
}

UTP void DTP::_for_each_piece( const auto &cell, auto &&ws, auto &&func ) const {
    constexpr int d = ct_dim;
    static_assert( d >= 1 );

    if ( SI( cell.nb_vertices ) == 0 )
        return;

    // ---- les plans de la grille, en coordonnées PHYSIQUES.
    // La convention est celle de `measure` : `x = origin + F^T t`, donc `t = ( F^T )^-1 ( x - origin )`
    // et la ligne `a` de `( F^T )^-1` est la COLONNE `a` de `F^-1`, c.-à-d. `solve_ge( F, e_a )`.
    // On n'inverse donc rien : d résolutions, une par axe, et le pavé `k` est l'intersection des
    // bandes `knots( a, k_a ) <= n_a . ( x - origin ) <= knots( a, k_a + 1 )`.
    const auto F = Matrix<TF,d>::with_func( [&]( auto r, auto c ) { return TF( frame( r, c ) ); } );
    const auto og = Vector<TF,d>::with_func( [&]( PI c ) { return TF( origin( c ) ); } );

    Vector<Vector<TF,d>,d> nrm;
    Vector<TF,d> shift;                     // `n_a . origin`, ce qui décale l'offset du plan
    for ( PI a = 0; a < d; ++a ) {
        nrm[ a ] = Matrix<TF,d>::solve_ge( F, Vector<TF,d>::with_value_at( a, TF( 1 ) ) );
        shift[ a ] = dot( nrm[ a ], og );
    }

    // ---- QUELS pavés essayer : ceux que la boîte englobante de la cellule rencontre.
    // Une cellule NON BORNÉE n'a pas de boîte -- ses sommets sont ceux d'un simplexe bouche-trou
    // (`Cell::init_as_unbounded`) et ne bornent rien. On balaie alors toute l'image, ce qui reste
    // JUSTE (l'image est à support compact, donc l'intégrale est finie même sur une cellule
    // infinie) et se paie en temps seulement -- un domaine (`box = ...`) supprime le cas.
    Vector<SI,d> k0, k1;
    const bool bounded = bool( cell.is_fully_bounded );
    for ( PI a = 0; a < d; ++a ) {
        const SI nb = SI( values.shape( a ) );
        if ( nb <= 0 )
            return;
        if ( ! bounded ) {
            k0[ a ] = 0;
            k1[ a ] = nb;
            continue;
        }

        const SI nv = cell.nb_vertices;
        TF t_min = 0, t_max = 0;
        for ( SI v = 0; v < nv; ++v ) {
            TF t = - shift[ a ];
            for ( PI c = 0; c < d; ++c )
                t += nrm[ a ][ c ] * TF( cell.vertex_positions( v, c ) );
            if ( v == 0 || t < t_min ) t_min = t;
            if ( v == 0 || t > t_max ) t_max = t;
        }

        // hors de l'image de ce côté-là : rien à intégrer du tout.
        if ( t_max < TF( knots( a, 0 ) ) || t_min > TF( knots( a, nb ) ) )
            return;

        k0[ a ] = knot_index( a, t_min, nb );
        k1[ a ] = knot_index( a, t_max, nb ) + 1;
    }

    // ---- un morceau par pavé. Compteur kilométrique sur `[ k0, k1 )`, et 2d coupes par pavé.
    //
    // Une descente RÉCURSIVE (couper d'abord la tranche de l'axe 0, puis subdiviser dedans) ferait
    // ~2 coupes par pavé au lieu de 2d, et élaguerait les tranches vides d'un coup ; elle demande
    // en revanche une cellule de rechange PAR NIVEAU, là où celle-ci en demande deux en tout. À
    // faire le jour où le profil le réclame -- le contrat côté Python (`extra_cuts_per_piece`) ne
    // change pas.
    Vector<SI,d> k = k0;
    for ( PI a = 0; a < d; ++a )
        if ( k0[ a ] >= k1[ a ] )
            return;

    while ( true ) {
        bool alive = true, fitted = true;
        for ( PI a = 0; a < d && alive; ++a ) {
            const TF lo = TF( knots( a, k[ a ] ) ) + shift[ a ];
            const TF hi = TF( knots( a, k[ a ] + 1 ) ) + shift[ a ];

            // `cut_id = BOUNDARY` : ces plans-là ne font face à aucun germe, et c'est exactement ce
            // que l'adjoint lit pour savoir que leur part ne va nulle part (voir
            // `PowerDiagram::scatter_cell_grad`).
            fitted = ( a == 0 ) ? ws.start( cell, nrm[ a ], hi, CellBoundary::BOUNDARY )
                                : ws.cut  (       nrm[ a ], hi, CellBoundary::BOUNDARY );
            if ( ! fitted ) break;
            if ( ws.nb_vertices() == 0 ) { alive = false; break; }

            fitted = ws.cut( - nrm[ a ], - lo, CellBoundary::BOUNDARY );
            if ( ! fitted ) break;
            if ( ws.nb_vertices() == 0 ) { alive = false; break; }
        }

        // une coupe qui n'a pas tenu : la capacité manquante est enregistrée, l'hôte relancera avec
        // le double et ce résultat-ci sera jeté. On s'arrête là plutôt que de continuer sur une
        // géométrie qui n'existe plus.
        if ( ! fitted )
            return;

        if ( alive ) {
            const auto idx = detail::image_index_tuple( k, tuple(), Ct<int,0>() );
            const TF value = TF( values( idx ) );
            ws.with_current( [&]( const auto &piece ) {
                func( piece, ConstantDensity{ value, [&]( auto &&grad_dist, TF g ) {
                    // le puits de gradient du morceau : `d masse / d values( k )` est le volume du
                    // morceau, et `k` est ce que cette fermeture-ci sait et que l'appelant ignore.
                    // Atomique : plusieurs work-items intègrent des cellules qui touchent le même pavé.
                    if constexpr ( CT_VALUE( grad_dist.values.is_valid() ) )
                        atomic_add( grad_dist.values( idx ).ref(), g );
                } } );
            } );
        }

        PI a = 0;
        while ( a < d && ++k[ a ] >= k1[ a ] ) { k[ a ] = k0[ a ]; ++a; }
        if ( a == PI( d ) )
            break;
    }
}

UTP void DTP::measure_bwd( auto &&grad_values, auto &&grad_mass ) const {
    // `mass` is linear in `values` (see `measure`): mass = Sum_c values(c) * |det(frame)| * spacing(c),
    // so grad_values(c) = grad_mass * |det(frame)| * spacing(c). Guarded at compile time: an
    // unperturbed `values` arrives as a `NoneTensor` (no `operator=`), so the block must vanish.
    if constexpr ( CT_VALUE( grad_values.is_valid() ) ) {
        with_defaults( [&]( auto &&img ) {
            using ImgT = DECAYED_TYPE_OF( img );
            using TF = typename ImgT::TF;
            constexpr int d = ImgT::ct_dim;

            const auto F = Matrix<TF,d>::with_func( [&]( auto r, auto c ) { return TF( img.frame( r, c ) ); } );
            const TF fdet = std::abs( F.determinant() );
            const TF gm = grad_mass; // scalar cotangent of `current_mass`

            auto shape = img.values.shape();
            CartesianIndices<DECAYED_TYPE_OF( shape )> cells{ shape };
            for ( PI flat = 0; flat < cells.size(); ++flat ) {
                cells[ flat ].apply_values( [&]( auto ...i ) {
                    PI axis = 0;
                    TF spacing = 1;
                    ( ( spacing *= TF( img.knots( axis, i + 1 ) ) - TF( img.knots( axis, i ) ), ++axis ), ... );
                    grad_values( i... ) = gm * fdet * spacing;
                } );
            }
        } );
    }
}

// The unidimensional-piece walk runs in PHYSICAL detector coordinates: cell edge i sits at
// `x(i) = origin(0) + frame(0,0) * knots(i)` (1D), so `origin`/`frame` are honored exactly as in
// `measure` (whose cell volume is |det(frame)| * knot spacing). A cell's mass is therefore
// `frame(0,0) * ( knots(i+1) - knots(i) ) * value` -- integral of the density over the physical
// width -- while `y` stays the DENSITY (so `mass / y` is a physical length). The default
// origin=0 / frame=identity collapses `x(i)` back to `knots(i)`, reproducing the plain-knot walk.
// Callers reach these through `with_defaults`, so `origin`/`frame`/`knots` are always populated.
// Assumes `frame(0,0) > 0` (an increasing detector axis), like the sorted-by-position diracs.
UTP auto DTP::udp_start() const {
    const TF sc = frame( 0, 0 );
    const TF of = origin( 0 );
    return Udp{
        .index = 0,
        .pos = of + sc * knots( 0 ),
        .mass = sc * ( knots( 1 ) - knots( 0 ) ) * values[ 0 ],
        .y = values[ 0 ]
    };
}

UTP auto DTP::udp_cont( auto &&udp, auto mass_to_take, auto &&cb_parts ) const {
    const SI nb_cells = values.size();
    const TF sc = frame( 0, 0 );
    const TF of = origin( 0 );

    // Consume whole cells while the current one cannot satisfy the whole request.
    // The `udp.index + 1 < nb_cells` guard keeps every access in bounds: the caller
    // only ever asks for the total remaining mass, so on the last cell we fall through
    // to the partial take below (and a floating-point overshoot cannot walk past the end).
    while( mass_to_take >= udp.mass && udp.index + 1 < nb_cells ) {
        mass_to_take -= udp.mass;

        const SI new_index = udp.index + 1;
        const TF x1 = of + sc * knots( new_index );
        // Emit even when y == 0: it contributes nothing to the cost / barycenter (`w2_dist` and
        // `first_moment` carry the y factor) but its `second_moment_about` feeds the direct term of
        // d cost / d y_c, which is non-zero on a zero-density cell.
        cb_parts( CstUdPiece{ .x0 = udp.pos, .x1 = x1, .y = udp.y } );

        udp.index = new_index;
        udp.pos = x1;
        udp.y = values[ new_index ];

        udp.mass = sc * ( knots( new_index + 1 ) - knots( new_index ) ) * udp.y;
    }

    // Partial take inside the current cell. Interior zero-density cells are already SKIPPED by the
    // loop above (their mass is 0, so `mass_to_take >= udp.mass` always advances past them); the only
    // way `udp.y` can be 0 here is a TRAILING run of empty cells at the target's right end. There is
    // no mass to place there -- the residual `mass_to_take` is the floating-point remainder of an
    // exhausted target -- so we skip the take rather than divide by zero. The dirac's cost and
    // barycenter come entirely from the (nonzero) pieces already emitted, as a zero-`y` piece
    // contributes nothing to either (`w2_dist` / `first_moment` carry the `y` factor).
    if ( udp.y != 0 ) {
        const TF x1 = udp.pos + mass_to_take / udp.y;
        cb_parts( CstUdPiece{ .x0 = udp.pos, .x1 = x1, .y = udp.y } );
        udp.pos = x1;
        udp.mass -= mass_to_take; // remaining mass of the current cell, for the next dirac
    }
}

UTP auto DTP::udp_at( auto &&cell_cum_mass, auto target_mass ) const {
    const SI nb_cells = values.size();
    const TF sc = frame( 0, 0 );
    const TF of = origin( 0 );

    // Binary search for the SMALLEST cell index `c` with `cell_cum_mass(c+1) >= target_mass` -- NOT
    // the largest `c` with `cell_cum_mass(c) <= target_mass`. The two rules agree everywhere EXCEPT
    // across a run of zero-mass cells sharing the same cumulative value as `target_mass`, and only the
    // smallest-c rule reproduces what the sequential walk actually does there: `udp_start()` never
    // pre-advances past a leading zero-density cell (advancing only ever happens inside `udp_cont`'s
    // own while-loop, driven by an actual dirac's `mass_to_take`), and a boundary landing exactly on a
    // cell edge resolves to the LAST cell whose mass ended there (`udp.y` possibly nonzero, `udp.mass`
    // exactly 0), matching `udp_cont`'s own state right after a partial-take finishes exactly on an
    // edge -- the following zero-mass run is only entered by the NEXT `udp_cont` call, exactly like the
    // sequential algorithm's own next call would. For any `c > 0` chosen by this rule, `y(c)` is
    // PROVABLY nonzero (`cell_cum_mass(c) < target_mass` forces `cell_cum_mass(c) < cell_cum_mass(c+1)`,
    // i.e. cell `c` itself carries mass) -- only `c == 0` can have `y(c) == 0`, and only when
    // `target_mass == 0` exactly, guarded below like `udp_cont`'s own partial-take guard.
    SI lo = 0, hi = nb_cells - 1; // invariant: answer in [lo,hi]
    while ( lo < hi ) {
        const SI mid = lo + ( hi - lo ) / 2;
        if ( TF( cell_cum_mass( mid + 1 ) ) >= TF( target_mass ) ) hi = mid;
        else                                                       lo = mid + 1;
    }
    const SI c = lo; // a float overshoot past the array's own total (independent chunked-summation
                      // rounding between `cell_cum_mass` and the caller's own target mass) cannot walk
                      // past the end -- same bound-safety reasoning as `udp_cont`'s `index+1 < nb_cells`.

    const TF base = TF( cell_cum_mass( c ) );
    const TF y    = values[ c ];
    const TF x0   = of + sc * knots( c );
    return Udp{
        .index = c,
        .pos   = ( y != 0 ) ? x0 + ( TF( target_mass ) - base ) / y : x0,
        .mass  = TF( cell_cum_mass( c + 1 ) ) - TF( target_mass ), // array-consistent (not recomputed
                                                                    // from knots*y), stays coherent with
                                                                    // whatever rounding built the array
        .y     = y
    };
}

// Sequential build of `cell_cum_mass` -- exactly the cell-mass formula `udp_start`/`udp_cont` use
// inline above, but walked once end-to-end and cached (see `Image.py`'s `cell_cum_mass`
// `ComputedAttribute`) instead of being rebuilt by every `OtPlan1d` forward/backward call. One
// thread does a whole angle -- `nb_cells` is small enough that no cooperative scan is worth it.
UTP void DTP::fill_cell_cum_mass( auto &&cell_cum_mass ) const {
    with_defaults( [&]( auto &&img ) {
        const SI nb_cells = img.values.size();
        const TF sc = img.frame( 0, 0 );
        auto cell_mass = [&]( SI c ) { return sc * ( TF( img.knots( c + 1 ) ) - TF( img.knots( c ) ) ) * TF( img.values[ c ] ); };

        TF running = 0;
        for ( SI c = 0; c < nb_cells; ++c ) {
            cell_cum_mass( c ) = running;
            running += cell_mass( c );
        }
        cell_cum_mass( nb_cells ) = running; // sentinel: total target mass
    } );
}

}

#undef UTP
#undef DTP

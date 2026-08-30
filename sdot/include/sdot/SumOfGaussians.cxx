#pragma once

#include <loom/support/common_macros.h>
#include <loom/support/containers/Vector.h>
#include "SumOfGaussians.h"
#include <SYCL/sycl.hpp>

// Les mathématiques passent par `sycl::`, JAMAIS par `std::`.
//
// Ce n'est pas une préférence de style : `std::exp` / `std::atan` / `std::erf` sur un `float`
// abaissent vers des intrinsèques LLVM que la cible CUDA AOT ne sait pas résoudre --
// `error: no libcall available for fexp` / `fatan`, et la compilation s'arrête là. Le JIT
// `generic` et le CPU y arrivent, eux, ce qui rend le piège invisible tant qu'on ne compile pas
// en AOT (`LOOM_GPU_AOT=1`). Les surcharges SYCL, elles, sont définies pour tous les backends.
//
// `sqrt` y est inclus bien qu'il passe (c'est une instruction native) : une règle qui souffre une
// exception n'est pas une règle qu'on suit.

#define UTP SDOT_TEMPLATE_DECL_FOR_SumOfGaussians
#define DTP SumOfGaussians<SDOT_TEMPLATE_ARGS_FOR_SumOfGaussians>

namespace sdot {

UTP auto DTP::kernel_at( SI i, const auto &x ) const {
    struct Kernel { TF phi; TF r2; TF s; };

    const TF s = TF( sigmas( i ) );
    TF r2 = 0;
    for ( PI c = 0; c < ct_dim; ++c ) {
        const TF e = TF( x[ c ] ) - TF( positions( i, c ) );
        r2 += e * e;
    }

    // `( 2 pi s^2 ) ^ ( -d/2 )`, écrit comme une puissance entière de `1 / ( s sqrt( 2 pi ) )` : d
    // multiplications au lieu d'un `pow`, et `ct_dim` étant connu à la compilation la boucle
    // disparaît.
    const TF two_pi = TF( 6.283185307179586476925286766559 );
    const TF inv = TF( 1 ) / ( s * sycl::sqrt( two_pi ) );
    TF norm = 1;
    for ( int k = 0; k < ct_dim; ++k )
        norm *= inv;

    return Kernel{ norm * sycl::exp( - r2 / ( 2 * s * s ) ), r2, s };
}

UTP typename DTP::TF DTP::value_at( const auto &x ) const {
    const SI n = nb_gaussians;
    TF res = 0;
    for ( SI i = 0; i < n; ++i )
        res += TF( weights( i ) ) * kernel_at( i, x ).phi;
    return res;
}

UTP auto DTP::gradient_at( const auto &x ) const {
    // `d/dx exp( -r^2 / 2s^2 ) = - ( x - c ) / s^2 * ...` : le gradient d'une gaussienne pointe vers
    // son centre, avec le facteur `1 / s^2`.
    const SI n = nb_gaussians;
    auto res = Vector<TF,ct_dim>::zeros();
    for ( SI i = 0; i < n; ++i ) {
        const auto k = kernel_at( i, x );
        const TF f = TF( weights( i ) ) * k.phi / ( k.s * k.s );
        for ( PI c = 0; c < ct_dim; ++c )
            res[ c ] -= f * ( TF( x[ c ] ) - TF( positions( i, c ) ) );
    }
    return res;
}

UTP void DTP::add_value_grad_at( auto &&grad_dist, const auto &x, TF g ) const {
    auto add_to = []( auto &&dst, TF v ) {
        if constexpr ( ! CT_VALUE( dst.surely_null() ) )
            atomic_add( dst.ref(), v );
    };

    // rien de demandé : pas une lecture, pas une exponentielle. Le test est à la COMPILATION, donc
    // un forward pur ne paie pas l'existence de ce bloc.
    if constexpr ( CT_VALUE( grad_dist.weights.surely_null() )
                && CT_VALUE( grad_dist.positions.surely_null() )
                && CT_VALUE( grad_dist.sigmas.surely_null() ) ) {
        return;
    } else {
        const SI n = nb_gaussians;
        for ( SI i = 0; i < n; ++i ) {
            const auto k = kernel_at( i, x );
            const TF w = TF( weights( i ) );

            // d rho / d w_i = le noyau normalisé
            add_to( grad_dist.weights( i ), g * k.phi );

            // d rho / d c_i = w_i * phi * ( x - c_i ) / s^2   ( le gradient EN x, changé de signe )
            if constexpr ( ! CT_VALUE( grad_dist.positions.surely_null() ) ) {
                const TF f = g * w * k.phi / ( k.s * k.s );
                for ( PI c = 0; c < ct_dim; ++c )
                    add_to( grad_dist.positions( i, c ), f * ( TF( x[ c ] ) - TF( positions( i, c ) ) ) );
            }

            // d rho / d s_i = w_i * phi * ( r^2 / s^3 - d / s ) : le premier terme vient de
            // l'exponentielle, le second de la constante de normalisation `s^-d`.
            add_to( grad_dist.sigmas( i ),
                    g * w * k.phi * ( k.r2 / ( k.s * k.s * k.s ) - TF( ct_dim ) / k.s ) );
        }
    }
}

// ---- l'intégration exacte en 2D ---------------------------------------------------------------
// Voir `SumOfGaussians.h` pour la réduction. Ici, les 8 noeuds de Gauss-Legendre (symétriques,
// donnés en demi-table) et les quatre méthodes.

namespace detail {
    // Gauss-Legendre à 8 points sur [ -1, 1 ], moitié positive
    inline constexpr double gl8_x[ 4 ] = { 0.1834346424956498, 0.5255324099163290,
                                           0.7966664774136267, 0.9602898564975363 };
    inline constexpr double gl8_w[ 4 ] = { 0.3626837833783620, 0.3137066458778873,
                                           0.2223810344533745, 0.1012285362903763 };

    // `Phi`, la fonction de répartition normale standard
    template<class TF> TF std_normal_cdf( TF u ) {
        return TF( 0.5 ) * ( 1 + sycl::erf( u * TF( 0.70710678118654752440 ) ) );
    }
}

UTP typename DTP::TF DTP::wedge_measure( const auto &P, const auto &Q ) const {
    static_assert( ct_dim == 2, "le coin polaire est la réduction 2D (voir SumOfGaussians.h)" );
    const TF two_pi = TF( 6.283185307179586476925286766559 );

    const TF dx = Q[ 0 ] - P[ 0 ], dy = Q[ 1 ] - P[ 1 ];
    const TF L = sycl::sqrt( dx * dx + dy * dy );
    if ( ! ( L > 0 ) )
        return 0;

    // `( n, u )` DIRECT, de sorte que `cross( P, Q ) = p * L` : le signe de `p` est celui de l'aire
    // du coin, et `t` croît de `P` vers `Q`.
    const TF ux = dx / L, uy = dy / L;
    const TF nx = uy, ny = -ux;
    const TF p = nx * P[ 0 ] + ny * P[ 1 ];
    const TF ap = p < 0 ? -p : p;
    if ( ! ( ap > 0 ) )                     // l'origine EST sur la droite : coin plat
        return 0;

    const TF t0 = ux * P[ 0 ] + uy * P[ 1 ];
    const TF t1 = ux * Q[ 0 ] + uy * Q[ 1 ];

    TF acc = 0;
    if ( ap >= tail_cut ) {
        // la gaussienne ne vaut plus rien sur toute la droite : il ne reste que la lorentzienne
        acc = sycl::atan( t1 / ap ) - sycl::atan( t0 / ap );
    } else {
        // Les QUEUES d'abord, chacune bornée par le segment lui-même : un segment entièrement
        // au-delà de `tail_cut` d'un seul côté n'a pas de coeur du tout, et sa queue va de `t0` à
        // `t1`, pas de `tail_cut` à `t1`. Clipper « symétriquement » compterait `[ tail_cut, t0 ]`
        // en trop -- une part d'angle bien visible quand l'arête est longue et rase l'origine.
        if ( t0 < -tail_cut ) {
            const TF e = t1 < -tail_cut ? t1 : -tail_cut;
            acc += sycl::atan( e / ap ) - sycl::atan( t0 / ap );
        }
        if ( t1 > tail_cut ) {
            const TF b = t0 > tail_cut ? t0 : tail_cut;
            acc += sycl::atan( t1 / ap ) - sycl::atan( b / ap );
        }

        const TF c0 = t0 > -tail_cut ? t0 : -tail_cut;
        const TF c1 = t1 <  tail_cut ? t1 :  tail_cut;

        // le coeur, par Gauss-Legendre composite. L'intégrande y a une échelle `>= 1` (le facteur
        // `1 - exp` annule le pic de la lorentzienne quand `p` est petit), donc quelques panneaux
        // suffisent quelle que soit la configuration.
        if ( c1 > c0 ) {
            const TF h = ( c1 - c0 ) / ( 2 * nb_panels );
            for ( int k = 0; k < nb_panels; ++k ) {
                const TF m = c0 + ( 2 * k + 1 ) * h;
                for ( int j = 0; j < 4; ++j ) {
                    for ( int sg = -1; sg <= 1; sg += 2 ) {
                        const TF t = m + sg * h * TF( detail::gl8_x[ j ] );
                        const TF r2 = ap * ap + t * t;
                        acc += h * TF( detail::gl8_w[ j ] ) * ( 1 - sycl::exp( - r2 / 2 ) ) * ap / r2;
                    }
                }
            }
        }
    }

    return ( p < 0 ? -acc : acc ) / two_pi;
}

UTP typename DTP::TF DTP::std_triangle_measure( const auto &ys ) const {
    const TF s = wedge_measure( ys[ 0 ], ys[ 1 ] )
               + wedge_measure( ys[ 1 ], ys[ 2 ] )
               + wedge_measure( ys[ 2 ], ys[ 0 ] );
    // la somme signée porte l'ORIENTATION du triangle ; la mesure, elle, n'en a pas.
    return s < 0 ? -s : s;
}

UTP typename DTP::EdgeInfo DTP::edge_info( const auto &A, const auto &B, const auto &C ) const {
    const TF two_pi = TF( 6.283185307179586476925286766559 );
    const TF sq_2pi = TF( 2.5066282746310005024157652848110 );

    EdgeInfo res{ Vector<TF,2>::zeros(), 0, 0, 0 };

    const TF dx = B[ 0 ] - A[ 0 ], dy = B[ 1 ] - A[ 1 ];
    const TF L2 = dx * dx + dy * dy;
    if ( ! ( L2 > 0 ) )
        return res;
    const TF L = sycl::sqrt( L2 );

    // la normale SORTANTE : celle qui s'éloigne du troisième sommet
    TF nx = dy / L, ny = -dx / L;
    if ( nx * ( C[ 0 ] - A[ 0 ] ) + ny * ( C[ 1 ] - A[ 1 ] ) > 0 ) { nx = -nx; ny = -ny; }
    res.n[ 0 ] = nx;
    res.n[ 1 ] = ny;
    res.p = nx * A[ 0 ] + ny * A[ 1 ];      // constant le long de l'arête

    // `| A + s ( B - A ) |^2 = L^2 ( s - s0 )^2 + p^2` : le pied de la perpendiculaire, et la
    // distance à la droite. C'est ce qui fait sortir un `exp( -p^2/2 )` en facteur et laisse une
    // gaussienne 1D, donc des `erf`.
    const TF s0 = - ( A[ 0 ] * dx + A[ 1 ] * dy ) / L2;
    TF p2 = A[ 0 ] * A[ 0 ] + A[ 1 ] * A[ 1 ] - L2 * s0 * s0;
    if ( p2 < 0 ) p2 = 0;

    const TF u0 = - L * s0, u1 = L * ( 1 - s0 );
    const TF d_phi = detail::std_normal_cdf( u1 ) - detail::std_normal_cdf( u0 );
    const TF e = sycl::exp( - p2 / 2 );

    res.j0  = e * d_phi / sq_2pi;
    res.j1a = ( e / two_pi ) * ( ( 1 - s0 ) * sq_2pi * d_phi
                               - ( sycl::exp( - u0 * u0 / 2 ) - sycl::exp( - u1 * u1 / 2 ) ) / L );
    return res;
}

UTP typename DTP::TF DTP::integrate_over_simplex( const auto &pts ) const {
    static_assert( ct_dim == 2, "le chemin exact est le 2D ; au-delà on passe par PointwiseDensity" );

    const SI n = nb_gaussians;
    TF res = 0;
    for ( SI i = 0; i < n; ++i ) {
        const TF s = TF( sigmas( i ) );
        const auto ys = Vector<Vector<TF,2>,3>( Function(), [&]( PI k ) {
            return Vector<TF,2>( Function(), [&]( PI c ) { return ( pts[ k ][ c ] - TF( positions( i, c ) ) ) / s; } );
        } );
        res += TF( weights( i ) ) * std_triangle_measure( ys );
    }
    return res;
}

UTP void DTP::integrate_over_simplex_bwd( const auto &pts, TF g, auto &&grad_pts, auto &&grad_dist ) const {
    static_assert( ct_dim == 2, "le chemin exact est le 2D ; au-delà on passe par PointwiseDensity" );

    auto add_to = []( auto &&dst, TF v ) {
        if constexpr ( ! CT_VALUE( dst.surely_null() ) )
            atomic_add( dst.ref(), v );
    };

    const SI n = nb_gaussians;
    for ( SI i = 0; i < n; ++i ) {
        const TF s = TF( sigmas( i ) );
        const TF w = TF( weights( i ) );
        const auto ys = Vector<Vector<TF,2>,3>( Function(), [&]( PI k ) {
            return Vector<TF,2>( Function(), [&]( PI c ) { return ( pts[ k ][ c ] - TF( positions( i, c ) ) ) / s; } );
        } );

        // une passe sur les trois arêtes : chacune verse à ses DEUX sommets (avec le poids
        // barycentrique qui vaut 1 chez l'un et 0 chez l'autre), et au sigma via `y . n`.
        auto dm = Vector<Vector<TF,2>,3>( Function(), []( PI ) { return Vector<TF,2>::zeros(); } );
        TF dsig = 0;
        for ( SI k = 0; k < 3; ++k ) {
            const SI k1 = ( k + 1 ) % 3, k2 = ( k + 2 ) % 3;
            const auto ei = edge_info( ys[ k ], ys[ k1 ], ys[ k2 ] );
            for ( PI c = 0; c < 2; ++c ) {
                dm[ k  ][ c ] += ei.j1a * ei.n[ c ];
                dm[ k1 ][ c ] += ( ei.j0 - ei.j1a ) * ei.n[ c ];
            }
            dsig += ei.p * ei.j0;
        }

        // les sommets, puis le centre -- qui est MOINS leur somme (translater le triangle et la
        // gaussienne ensemble ne change rien), ce qui évite une seconde dérivation.
        const TF f = g * w / s;
        auto dc = Vector<TF,2>::zeros();
        for ( SI k = 0; k < 3; ++k ) {
            for ( PI c = 0; c < 2; ++c ) {
                grad_pts[ k ][ c ] += f * dm[ k ][ c ];
                dc[ c ] -= f * dm[ k ][ c ];
            }
        }

        if constexpr ( ! CT_VALUE( grad_dist.positions.surely_null() ) )
            for ( PI c = 0; c < 2; ++c )
                add_to( grad_dist.positions( i, c ), dc[ c ] );

        add_to( grad_dist.sigmas( i ), - f * dsig );
        add_to( grad_dist.weights( i ), g * std_triangle_measure( ys ) );
    }
}

#undef UTP
#undef DTP

}

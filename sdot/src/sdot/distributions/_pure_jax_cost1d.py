"""Closed-form 1D optimal-transport cost between a Dirac source and a piecewise-constant
target density, computed with plain (differentiable) JAX ops -- no custom kernel, no
`driver.call`. See `Image.try_update_otplan1d` for the dispatch that picks this path over
the general driver.call/C++ one, and [[pure-jax-otplan1d]] for the evaluation that led here
(1.2x-3.7x faster than the C++ kernel at n=1e6-3e7, matches it to float32 precision).

Moment trick: for a piecewise-CONSTANT target density, the cumulative mass/first/second
moment M0/M1/M2 are piecewise linear/quadratic/cubic in position, so
`integral_{t_k}^{t_{k+1}} y(x)(x-p)^2 dx` is a difference of these evaluated at the two
ends, with `t_k = target_CDF^{-1}( W_k )` (`W_k` = cumulative dirac weight up to sorted rank
k) found via `jnp.searchsorted` (vectorized over every dirac-rank boundary at once) plus an
exact linear in-bin inversion. Only the diracs need sorting (`jnp.argsort`); the target's
CDF inversion needs no sort of its own (its bins are already ordered).

Known caveat (not introduced here, matches a latent bug in the C++ reference too): the
gradient wrt `values` is inexact for a TRAILING run of exactly-zero-density target cells at
the very end of the array -- a narrow, pre-existing edge case, not a blocker for realistic
(measured, non-exactly-zero) density data.
"""
import jax.numpy as jnp


def cost_1d_ot( proj, weights, values, s_min, dw ):
    """`proj`: [n] projected 1D dirac positions. `weights`: [n] dirac masses (any positive
    scale). `values`: [m] target bin densities (any positive scale, piecewise-constant over
    bins of width `dw` starting at `s_min`). Both sides are normalized to mass 1 internally,
    matching `OtPlan1d.cost`'s semantics. Returns a scalar."""
    m = values.shape[ 0 ]
    w_norm = weights / jnp.sum( weights )
    v_norm = values / ( jnp.sum( values ) * dw )
    edges = s_min + dw * jnp.arange( m + 1, dtype = proj.dtype )

    C   = jnp.concatenate( [ jnp.zeros( 1, dtype = proj.dtype ), jnp.cumsum( v_norm * dw ) ] )
    M1e = jnp.concatenate( [ jnp.zeros( 1, dtype = proj.dtype ), jnp.cumsum( v_norm * ( edges[ 1: ] ** 2 - edges[ :-1 ] ** 2 ) / 2 ) ] )
    M2e = jnp.concatenate( [ jnp.zeros( 1, dtype = proj.dtype ), jnp.cumsum( v_norm * ( edges[ 1: ] ** 3 - edges[ :-1 ] ** 3 ) / 3 ) ] )

    order = jnp.argsort( proj )
    p_sorted = proj[ order ]
    w_sorted = w_norm[ order ]
    W = jnp.concatenate( [ jnp.zeros( 1, dtype = proj.dtype ), jnp.cumsum( w_sorted ) ] )

    c_idx = jnp.clip( jnp.searchsorted( C[ 1: ], W, side = "left" ), 0, m - 1 )
    y_c = v_norm[ c_idx ]
    e_c = edges[ c_idx ]
    safe_y = jnp.where( y_c != 0, y_c, 1.0 )
    t = jnp.where( y_c != 0, e_c + ( W - C[ c_idx ] ) / safe_y, e_c )

    M0_t = C[ c_idx ] + y_c * ( t - e_c )
    M1_t = M1e[ c_idx ] + y_c * ( t ** 2 - e_c ** 2 ) / 2
    M2_t = M2e[ c_idx ] + y_c * ( t ** 3 - e_c ** 3 ) / 3

    dM0 = M0_t[ 1: ] - M0_t[ :-1 ]
    dM1 = M1_t[ 1: ] - M1_t[ :-1 ]
    dM2 = M2_t[ 1: ] - M2_t[ :-1 ]

    return jnp.sum( dM2 - 2 * p_sorted * dM1 + p_sorted ** 2 * dM0 )

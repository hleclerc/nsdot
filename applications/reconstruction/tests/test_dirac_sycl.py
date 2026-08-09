"""Vérifie la référence SYCL fusionnée (`dirac_sycl.diracs_cost_grad`) contre le chemin PUR JAX
existant (`models.DiracModel.cost` + `jax.grad`) : même formule (transport optimal 1D
semi-discret, diracs de masse égale), donc coût ET gradient doivent coïncider à la précision
flottante près -- voir `dirac_sycl.py` pour pourquoi un seul kernel fwd-only suffit ici (pas de
`bwd_code`, le gradient est écrit directement par la formule fermée).
"""
import numpy as np

from reconstruction.Sinogram import Sinogram
from reconstruction.models import DiracModel
from reconstruction.dirac_sycl import diracs_cost_grad
from sdot import driver
from sdot.testing import test


def _disk_sinogram( nb_angles = 8, nb_bins = 201, extent = 6.0, center = ( 0.3, -0.2 ), radius = 1.0 ):
    s = Sinogram( nb_angles = nb_angles, nb_bins = nb_bins, extent = extent )
    s.add_disk( center = list( center ), radius = radius )
    return s


if test( "diracs_cost_grad_matches_pure_jax" ):
    sino = _disk_sinogram()
    rng = np.random.default_rng( 0 )
    pts = ( rng.random( ( 137, 2 ) ) - 0.5 ) * 2

    cost_sycl, grad_sycl = diracs_cost_grad( pts, sino )

    model = DiracModel( sino )
    def scalar_loss( p ):
        return model.cost( model.wrap( p ) ).tensor
    cost_jax = float( driver.jit( scalar_loss )( pts ) )
    grad_jax = np.asarray( driver.jit( driver.grad( scalar_loss ) )( pts ) )

    assert np.isfinite( cost_sycl )
    assert abs( cost_sycl - cost_jax ) < 1e-8 * max( 1.0, abs( cost_jax ) ), \
        f"coût SYCL { cost_sycl } != coût Jax { cost_jax }"
    assert np.allclose( grad_sycl, grad_jax, atol = 1e-6, rtol = 1e-5 ), \
        f"gradient SYCL != gradient Jax, écart max { np.max( np.abs( grad_sycl - grad_jax ) ) }"


if test( "diracs_cost_grad_single_angle" ):
    # cas limite : un seul angle -- vérifie que le batching sur `num_angle` dégénère correctement.
    sino = _disk_sinogram( nb_angles = 1 )
    rng = np.random.default_rng( 1 )
    pts = ( rng.random( ( 23, 2 ) ) - 0.5 ) * 2

    cost_sycl, grad_sycl = diracs_cost_grad( pts, sino )

    model = DiracModel( sino )
    def scalar_loss( p ):
        return model.cost( model.wrap( p ) ).tensor
    cost_jax = float( driver.jit( scalar_loss )( pts ) )
    grad_jax = np.asarray( driver.jit( driver.grad( scalar_loss ) )( pts ) )

    assert abs( cost_sycl - cost_jax ) < 1e-8 * max( 1.0, abs( cost_jax ) )
    assert np.allclose( grad_sycl, grad_jax, atol = 1e-6, rtol = 1e-5 )

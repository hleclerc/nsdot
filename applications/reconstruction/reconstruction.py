import numpy as np

from sdot import ProjectedSumOfDiracs, OtPlan1d, Tensor, driver

from .Sinogram import Sinogram
from .optimizers import GradientDescent


def loss( sinogram: Sinogram, positions ):
    """Coût de reconstruction : somme, sur les angles, du coût de transport optimal
    1D entre les diracs projetés et le profil mesuré.

    `positions` : particules 2D (Tensor ou tableau [ n, 2 ]) de la densité reconstruite.

    Tout est BATCHÉ sur `num_angle` (l'axe de batch du sinogramme) : un seul `OtPlan1d`
    traite les `nb_angles` transports d'un coup (`cost` est de rang 1, un coût par angle),
    au lieu d'une boucle Python. Chaque tranche normalise ses deux distributions à la masse
    1 (diracs uniformes, image `target_mass = 1`).

    La projection `s = point·normale` N'EST PAS matérialisée : `ProjectedSumOfDiracs` garde les
    `points` 2D PARTAGÉS (une seule copie pour tous les angles) et la `normale` PAR ANGLE, et le
    kernel calcule la position 1D à la volée -- au lieu d'un tenseur `[ nb_angles, n ]` (80 Go à
    1e7 diracs x 1000 angles). Reste différentiable par rapport à `positions` : le backward
    scatter-atomique le gradient de la position projetée sur les points 2D partagés.
    """
    points = positions if isinstance( positions, Tensor ) else Tensor( positions )
    src = ProjectedSumOfDiracs( points = points, normal = sinogram.normals_t,
                                batch_axes = [ sinogram.num_angle ] )
    dst = sinogram.batched_image()
    return OtPlan1d( src, dst ).cost.sum()                   # somme sur les angles


def random_positions( nb_diracs: int, extent: float, seed: int = 0 ) -> Tensor:
    """`nb_diracs` positions 2D tirées uniformément dans [ −extent/2, extent/2 ]²."""
    rng = np.random.default_rng( seed )
    return Tensor( ( rng.random( ( nb_diracs, 2 ) ) - 0.5 ) * extent )


def reconstruct( sinogram: Sinogram, positions, optimizer = None, lr: float = None, nb_steps: int = None, callback = None ):
    """Descente de gradient sur `positions` pour diminuer `loss`, sinogramme FIXÉ.

    `optimizer` : instance d'Optimizer. Si None, utilise GradientDescent(lr, nb_steps).
    `lr` et `nb_steps` : paramètres rétro-compatibles pour GradientDescent (défaut 0.2 et 100).
    À chaque pas on suit l'opposé du gradient de la perte par rapport aux positions,
    obtenu par `driver.grad` (mode adjoint, qui traverse le backward d'`OtPlan1d`).
    `positions` : Tensor ou [ n, 2 ].

    `callback( step, positions_tensor )` est appelé après chaque pas si fourni.
    Renvoie les positions optimisées, sous forme de Tensor.
    """
    if optimizer is None:
        if lr is None:
            lr = 0.2
        if nb_steps is None:
            nb_steps = 100
        optimizer = GradientDescent(lr=lr, nb_steps=nb_steps)

    # Extract raw JAX array from Tensor or use directly
    if isinstance( positions, Tensor ):
        p = positions.raw
    else:
        p = driver.array( positions )

    # Verify initial state by computing loss
    if callback is not None:
        callback( -1, Tensor.wrap( p, [ "num_dirac", "dim" ] ) )  # Report initial state

    def scalar_loss( q ):
        return loss( sinogram, Tensor.wrap( q, [ "num_dirac", "dim" ] ) ).tensor

    def wrap_callback( step, x ):
        if callback is not None:
            callback( step, Tensor.wrap( x, [ "num_dirac", "dim" ] ) )

    p_opt = optimizer.minimize( scalar_loss, p, callback=wrap_callback )

    return Tensor.wrap( p_opt, [ "num_dirac", "dim" ] )

import numpy as np

from sdot import SumOfDiracs1d, OtPlan1d, Tensor, driver

from .Sinogram import Sinogram


def loss( sinogram: Sinogram, positions ):
    """Coût de reconstruction : somme, sur les angles, du coût de transport optimal
    1D entre les diracs projetés et le profil mesuré.

    `positions` : particules 2D (Tensor ou tableau [ n, 2 ]) de la densité reconstruite.

    Tout est BATCHÉ sur `num_angle` (l'axe de batch du sinogramme) : un seul `OtPlan1d`
    traite les `nb_angles` transports d'un coup (`cost` est de rang 1, un coût par angle),
    au lieu d'une boucle Python. Chaque tranche normalise ses deux distributions à la masse
    1 (diracs uniformes, image `target_mass = 1`). La projection passe par l'algèbre `Tensor`,
    donc le coût reste différentiable par rapport à `positions`.
    """
    projected = sinogram.project_points( positions )        # Tensor [ num_angle, n ]

    src = SumOfDiracs1d( positions = projected, batch_axes = [ sinogram.num_angle ] )
    dst = sinogram.batched_image()
    return OtPlan1d( src, dst ).cost.sum()                   # somme sur les angles


def random_positions( nb_diracs: int, extent: float, seed: int = 0 ) -> Tensor:
    """`nb_diracs` positions 2D tirées uniformément dans [ −extent/2, extent/2 ]²."""
    rng = np.random.default_rng( seed )
    return Tensor( ( rng.random( ( nb_diracs, 2 ) ) - 0.5 ) * extent )


def reconstruct( sinogram: Sinogram, positions, lr: float = 0.2, nb_steps: int = 100, callback = None ):
    """Descente de gradient sur `positions` pour diminuer `loss`, sinogramme FIXÉ.

    Boucle simple (le passage en batch viendra) : à chaque pas on suit l'opposé du
    gradient de la perte par rapport aux positions, obtenu par `driver.grad` (mode
    adjoint, qui traverse le backward d'`OtPlan1d`). `positions` : Tensor ou [ n, 2 ].

    `callback( step, positions_tensor )` est appelé après chaque pas si fourni.
    Renvoie les positions optimisées, sous forme de Tensor.
    """
    p = positions.raw if isinstance( positions, Tensor ) else driver.array( positions )

    def scalar_loss( q ):
        return loss( sinogram, Tensor.wrap( q, [ "num_dirac", "dim" ] ) ).tensor

    grad = driver.grad( scalar_loss )
    for step in range( nb_steps ):
        p = p - lr * grad( p )
        if callback is not None:
            callback( step, Tensor.wrap( p, [ "num_dirac", "dim" ] ) )

    return Tensor.wrap( p, [ "num_dirac", "dim" ] )

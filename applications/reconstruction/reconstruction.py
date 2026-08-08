import numpy as np

from sdot import ProjectedSumOfDiracs, OtPlan1d, Tensor, driver

from .Sinogram import Sinogram
from .optimizers import GradientDescent, LBFGS


def loss( sinogram: Sinogram, positions, with_barycenters: bool = False ):
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

    `with_barycenters` : transmis tel quel à `OtPlan1d` -- quand le SEUL gradient demandé est
    celui des positions (le cas de `reconstruct`/le benchmark de gradient), stocker les
    barycentres évite au backward de re-trier + re-balayer chaque angle (voir
    `OtPlan1d.__init__`'s docstring). Coûte un buffer `[nb_angles, n]` en plus ; à activer si ce
    coût mémoire est acceptable pour le gain de vitesse du backward.
    """
    points = positions if isinstance( positions, Tensor ) else Tensor( positions )
    src = ProjectedSumOfDiracs( points = points, normal = sinogram.normals_t,
                                batch_axes = [ sinogram.num_angle ] )
    dst = sinogram.batched_image()
    return OtPlan1d( src, dst, with_barycenters = with_barycenters ).cost.sum()  # somme sur les angles


def random_positions( nb_diracs: int, extent: float, seed: int = 0 ) -> Tensor:
    """`nb_diracs` positions 2D tirées uniformément dans [ −extent/2, extent/2 ]²."""
    rng = np.random.default_rng( seed )
    return Tensor( ( rng.random( ( nb_diracs, 2 ) ) - 0.5 ) * extent )


def split_positions( positions, factor: int, noise_scale: float, seed: int = 0 ) -> Tensor:
    """Remplace chaque point par `factor` enfants, superposés au parent puis dispersés d'un
    bruit uniforme d'amplitude `noise_scale` (par axe). Utilisé par `reconstruct_multiscale`
    pour RAFFINER une solution convergée à basse résolution, plutôt que repartir d'un tirage
    uniforme à la résolution cible."""
    p = positions.raw if isinstance( positions, Tensor ) else np.asarray( positions )
    rng = np.random.default_rng( seed )
    tiled = np.repeat( p, factor, axis = 0 )                                  # [ n*factor, dim ]
    noise = ( rng.random( tiled.shape ) - 0.5 ) * 2 * noise_scale
    return Tensor.wrap( tiled + noise, [ "num_dirac", "dim" ] )


def reconstruct_multiscale(
    sinogram: Sinogram, extent: float, nb_diracs_final: int, nb_diracs_init: int = 1000,
    factor: int = 4, optimizer_factory = None, noise_frac: float = 0.5, seed: int = 0,
    stage_callback = None,
):
    """Reconstruction grossier -> fin : converge `reconstruct` à `nb_diracs_init` points (tirage
    uniforme), puis répète { `split_positions` (chaque point -> `factor` enfants bruités),
    reconverger } jusqu'à atteindre `nb_diracs_final`.

    Motivation : à `nb_diracs_final` directement (ex. 1e7), un tirage initial UNIFORME sur toute
    l'étendue doit être globalement réarrangé d'un coup par l'optimiseur (la masse cible est
    concentrée sur une toute petite fraction du support) -- observé à provoquer un échec de la
    line search de L-BFGS-B (arrêt après 1-2 itérations, loin de la convergence). En procédant
    par étapes, chaque niveau part déjà de la bonne structure GLOBALE (héritée du niveau
    précédent) et n'a plus qu'à raffiner LOCALEMENT -- un problème beaucoup mieux conditionné.

    `noise_frac` : amplitude du bruit de split, en fraction de l'espacement typique au niveau
    COURANT (`extent / sqrt( n )`) -- les enfants doivent explorer le voisinage immédiat du
    parent, pas plus (sinon on retombe sur un réarrangement global à chaque étape).

    `stage_callback( stage, n, positions )`, si fourni, est appelé après convergence de chaque
    étage (avant le split suivant) -- utile pour tracer/illustrer la progression.
    """
    if optimizer_factory is None:
        optimizer_factory = lambda n: LBFGS( max_iter = 60, ftol = 1e-10 )

    n = nb_diracs_init
    positions = random_positions( n, extent = extent, seed = seed )
    stage = 0
    while True:
        positions = reconstruct( sinogram, positions, optimizer = optimizer_factory( n ) )
        if stage_callback is not None:
            stage_callback( stage, n, positions )
        if n >= nb_diracs_final:
            return positions

        noise_scale = noise_frac * extent / np.sqrt( n )
        next_n = min( n * factor, nb_diracs_final )
        positions = split_positions( positions, factor, noise_scale, seed = seed + stage + 1 )
        if len( positions.raw ) > next_n:
            # dernier étage : `factor` déborde la cible -- sous-échantillonne au lieu de résoudre
            # un facteur non entier.
            idx = np.random.default_rng( seed + stage + 1 ).choice( len( positions.raw ), next_n, replace = False )
            positions = Tensor.wrap( positions.raw[ idx ], [ "num_dirac", "dim" ] )
        n = next_n
        stage += 1


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

import time
from functools import partial
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
# from loom.testing import Param, bench
from gpu_mem import jax_mem_budget_bytes
from tracker import GradTimer

# Needed for the float64 promotion in `_w2_1d` below -- disabled by default,
# JAX otherwise SILENTLY truncates any float64 array back to float32.
jax.config.update("jax_enable_x64", True)

# See `loss`'s docstring for how this was measured.
_BYTES_PER_CHUNK_ELEMENT = 256


def _w2_1d(proj, bin_mass,bin_edges):
    """
    Calcule la distance de Wasserstein quadratique W₂² en 1D.

    Résumé :
        Compare la mesure empirique formée par les projections `proj`
        avec la mesure discrète définie par `bin_mass` et `bin_edges`.

        Les projections sont triées pour obtenir les quantiles de la
        mesure empirique. Pour chaque tranche de masse 1/n, on calcule
        ensuite le barycentre correspondant de la mesure cible.

        La fonction renvoie finalement W₂² sous une forme quadratique
        permettant à JAX de calculer automatiquement le gradient.

    Args:
        proj:
            Positions 1D des n Dirac après projection sur une direction.
            Shape : (n,).

        bin_mass:
            Masse contenue dans chaque bin de la distribution cible.
            Shape : (nb_bins,), avec somme des masses = 1.

        bin_edges:
            Frontières des bins.
            Shape : (nb_bins + 1,).

    Returns:
        La distance de Wasserstein quadratique entre les deux mesures.
    """
    n = proj.shape[0]                    # Nombre de Dirac dans la projection
    w = 1.0 / n                          # Chaque Dirac porte la masse 1/n

    dw = bin_edges[1] - bin_edges[0]     # Largeur commune des bins
    bin_center = bin_edges[:-1] + dw / 2 # Centre de chaque bin

    cum = jnp.cumsum(bin_mass)            # CDF discrète : masse cumulée
    cum_start = cum - bin_mass            # Quantile auquel commence chaque bin

    # Intégrale cumulée du quantile :
    # prefix_M[j] = ∫ jusqu'au début du bin j de Q(t) dt
    prefix_M = jnp.cumsum(bin_mass * bin_center) - bin_mass * bin_center

    def M(q):
        """
         Calcule M(q) = ∫₀ᑫ Q(t) dt.

         Résumé :
             Intègre la fonction quantile de la distribution cible
             jusqu'au niveau de masse q. Le calcul est analytique
             à l'intérieur de chaque bin.
        """
        # Cherche dans quel bin tombe le quantile q
        j = jnp.clip(jnp.searchsorted(cum, q, side="right"), 0, bin_mass.shape[0] - 1)
        # Position normalisée de q à l'intérieur du bin [0,1]
        f = jnp.where(bin_mass[j] > 0, (q - cum_start[j]) / bin_mass[j], 0.0)
        # Intégrale du quantile jusqu'au point q
        return prefix_M[j] + bin_mass[j] * (bin_edges[j] * f + dw * f * f / 2)


    # Trie les projections : elles deviennent les quantiles  de la mesure empirique.
    s = jnp.sort(proj).astype(jnp.float64)
    q = jnp.arange(n, dtype=jnp.float64) * w     # Début de chaque intervalle de quantile de largeur w = 1/n
    # Barycentre de la masse cible correspondant à chaque Dirac.
    # C'est la position vers laquelle le Dirac devrait idéalement aller pour minimiser W₂.
    bary = (M(q + w) - M(q)) / w
    # Second moment de la distribution cible. dw²/12 est la variance d'une loi uniforme sur un bin.
    target_second_moment = jnp.sum(bin_mass * bin_center ** 2) + dw * dw / 12
    return w * jnp.sum(s ** 2) - 2 * w * jnp.sum(s * bary) + target_second_moment


def _sino_arrays(sino):
    """`(normals, bin_edges, bin_mass)` for `sino`, dtypes as `loss` needs
    them. Split out of `loss` so `optimize` can compute this ONCE and pass
    the results into `step` as actual `jax.jit` ARGUMENTS rather than
    closed-over free variables -- see `loss`'s docstring for why that
    distinction matters here."""
    g = sino.geometry
    normals = jnp.asarray(g.normals, dtype=jnp.float32)
    bin_edges = jnp.asarray(g.bin_edges, dtype=jnp.float64)
    bin_mass = jnp.asarray(sino.values, dtype=jnp.float64)
    bin_mass = bin_mass / bin_mass.sum(axis=1, keepdims=True)
    return normals, bin_edges, bin_mass


def loss(points, normals, bin_edges, bin_mass, mem_budget_bytes=-1):
    """Sum over angles of the 1D Wasserstein distance between `points`
    projected on that angle and the sinogram's line for that angle. Takes
    raw arrays (see `_sino_arrays`) rather than `sino` itself -- see below.

    Résumé :
        Pour chaque angle de projection, les points 2D sont projetés
        sur la normale correspondante. La distribution 1D obtenue est
        comparée à la ligne correspondante du sinogramme via W₂².

        La loss finale est la somme des distances de Wasserstein
        sur tous les angles.
    Args:
            points:
                Coordonnées 2D des Dirac à optimiser, shape (N, 2).

            normals:
                Normales des directions de projection, shape (A, 2).

            bin_edges:
                Frontières communes des bins du sinogramme.

            bin_mass:
                Masses du sinogramme normalisées, shape (A, B).

            mem_budget_bytes:
                Budget mémoire utilisé pour choisir combien d'angles traiter
                simultanément.
    Returns:
        La somme des W₂² sur tous les angles.
    """
    # `-1` (not `None`) as the sentinel: `mem_budget_bytes` flows through
    # `jax.jit` alongside `points`/`normals`/... in `optimize.step`, and a
    # `None` default would make jit treat "was it passed" as part of the
    # traced signature -- a plain Python int stays a compile-time constant
    # either way (it's never turned into a jnp array).
    if mem_budget_bytes == -1:        # Si aucun budget n'est fourni, on interroge le gestionnaire mémoire.
        mem_budget_bytes = jax_mem_budget_bytes()

    mem_budget_mb = mem_budget_bytes / (1024 ** 2)  # Conversion en Mo
    print(f"{mem_budget_mb=:.2f} Mb")  # Affiche avec 2 décimales
    n, A = points.shape[0], normals.shape[0]    # Nombre de points 2D, # Nombre d'angles
    # Détermine combien d'angles peuvent être traités simultanément sans dépasser le budget mémoire.
    if mem_budget_bytes is None:
        chunk_size = 1
    else :
        # Optimisation mémoire :
        # Si tu as un budget mémoire limité (mem_budget_bytes), tu ne peux pas charger tous les angles (A) en une seule fois.
        # Tu dois donc diviser le travail en chunks (morceaux) de taille chunk_size.

        memory_for_chunk = _BYTES_PER_CHUNK_ELEMENT * max(n, 1)
        nb_chunk_in_memory = mem_budget_bytes // memory_for_chunk
        nb_chunk = min(A, nb_chunk_in_memory) #  ne pas dépasser le nombre total d'angles
        chunk_size = max(1,nb_chunk)

    def angle_cost(normal_and_mass):
        """
        Calcule le coût Wasserstein pour un seul angle.

        Résumé :
          Projette tous les points sur la normale donnée puis compare
          la distribution projetée à la projection correspondante
          du sinogramme.
        """
        normal, mass = normal_and_mass
        # p_i · n : projection scalaire de chaque point sur la normale
        projections = points @ normal
        return _w2_1d(projections, mass, bin_edges)
    # fin angle cost , retour sur la loss

    # Applique angle_cost à tous les angles.
    # checkpoint économise de la mémoire pendant le calcul du gradient.
    batch_size = int(chunk_size) if chunk_size > 1 else None # TODO see if None
    print(f"{batch_size=}")

    costs = jax.lax.map(jax.checkpoint(angle_cost),
                        (normals, bin_mass),
                        batch_size=batch_size)

    return costs.sum().astype(jnp.float32)    # Somme des coûts de tous les angles = fonction objectif globale


def optimize(points,
             sino,
             max_iter=15,
             tracker=None,
             grad_timer=None,
             max_linesearch_steps=8):
    """L-BFGS (optax) on `loss`. Returns the optimized points.

    Runs a plain Python loop (not `lax.scan`) with a single jitted step: the
    step compiles once and every iteration then actually runs (and can be
    reported via `tracker`) instead of vanishing inside one big opaque
    compiled program for the whole `max_iter` budget.
    Optimise les positions des Dirac avec L-BFGS. Résumé : À partir d'une configuration initiale de points 2D,
    minimise la fonction `loss`, qui mesure l'écart entre les projections des points et les projections du sinogramme
     via une distance de Wasserstein. L'optimisation utilise L-BFGS d'Optax. Une seule fonction `step` est compilée
     avec `jax.jit`, puis réutilisée à chaque itération. Avant l'optimisation réelle, quelques itérations de "warmup"
     sont exécutées afin de déclencher la compilation JIT et de vérifier que la consommation mémoire est compatible
     avec le GPU disponible. En cas de dépassement mémoire (OOM), le nombre d'angles traités simultanément
      est progressivement réduit. La boucle principale reste une boucle Python plutôt qu'un `jax.lax.scan`,
       afin de pouvoir mesurer chaque itération et enregistrer son évolution dans `tracker`.
   points: Positions initiales des Dirac, shape `(N, 2)`.
    max_iter: Nombre d'itérations L-BFGS de l'optimisation réelle.
    max_linesearch_steps: Nombre maximal d'essais effectués par le line search pour déterminer une longueur de pas satisfaisante.
    Returns: Les positions 2D optimisées des Dirac, shape `(N, 2)`.
    """
    normals, bin_edges, bin_mass = _sino_arrays(sino) #Conversion de l'objet `sino` en tableaux JAX directement utilisables
    # par la loss et par la fonction compilée avec `jax.jit`
    linesearch = optax.scale_by_zoom_linesearch(max_linesearch_steps=max_linesearch_steps, initial_guess_strategy="one")
    # L-BFGS propose une direction de déplacement pour les points, # mais il faut encore déterminer quelle distance parcourir dans # cette direction.
    # Le line search teste donc plusieurs longueurs de pas et cherche # une valeur donnant une diminution suffisamment intéressante de la loss.
    solver = optax.lbfgs(linesearch=linesearch)
    # L-BFGS est une méthode quasi-Newton : # elle utilise les gradients successifs pour construire implicitement
    # une approximation de l'inverse du Hessien, sans jamais stocker  explicitement une énorme matrice Hessienne.
    state = solver.init(points)
    #`state` contient toutes les informations dont L-BFGS a besoin
    # entre deux itérations : historique des gradients/déplacements,

    def make_step(mem_budget_bytes):
        # Toute la fonction `step` est compilée par JAX. # # L'objectif est que la séquence : # # loss → gradient → L-BFGS → déplacement
        # # soit exécutée comme un seul programme compilé, plutôt que # de faire de nombreux allers-retours Python ↔ GPU.
        @jax.jit
        def step(p, state, normals, bin_edges, bin_mass):
            # Fixe les données du problème qui ne changent pas pendant
            # l'optimisation. `fun` devient ainsi une fonction uniquement, fun(p) = loss(p, normals, bin_edges, bin_mass, ...)
            # de `loss` :
            fun = partial(
                loss,
                normals=normals,  bin_edges=bin_edges,  bin_mass=bin_mass, mem_budget_bytes=mem_budget_bytes)
            # Construit une fonction qui calcule à la fois la valeur de la loss
            # et son gradient par rapport aux paramètres `p`.
            #
            #     value = fun(p)
            #     grad  = ∂fun/∂p
            #
            # `state` permet à Optax de tenir compte de l'état courant
            # du solveur L-BFGS, notamment pour le line search.
            value_and_grad = optax.value_and_grad_from_state(fun)
            # Évalue la fonction objectif et calcule automatiquement
            # son gradient avec JAX.
            #
            # `p`     : positions actuelles des Dirac, shape (N, 2)
            # `value` : valeur scalaire de la loss
            # `grad`  : gradient de même shape que `p`, donc (N, 2)
            value, grad = value_and_grad(p, state=state)
            # L-BFGS utilise le gradient actuel ainsi que son état interne
            # pour calculer le déplacement à appliquer aux points.
            #
            # `value_fn=fun` est nécessaire au line search : celui-ci peut
            # tester plusieurs longueurs de pas et donc réévaluer la loss
            # pour différentes configurations candidates.
            updates, state = solver.update(grad, state, p, value=value, grad=grad, value_fn=fun)
            # Applique le déplacement calculé par L-BFGS aux positions.
            #
            # Conceptuellement :
            #
            #     p_new = p + updates
            #
            # `state` est séparé de `p` car il contient l'état de l'optimiseur,
            # pas les coordonnées physiques des Dirac.
            p = optax.apply_updates(p, updates)
            # Retourne :
            #   - les nouvelles positions des Dirac,
            #   - le nouvel état de L-BFGS,
            #   - la valeur de la loss calculée pour cette étape.
            return p, state, value

        # Retourne la fonction `step` compilable par JAX.
        return step

    # Demande le budget mémoire disponible pour le calcul.
    # Ce budget sera utilisé dans `loss` pour déterminer combien d'angles peuvent être traités simultanément sur le GPU.
    mem_budget_bytes = jax_mem_budget_bytes()
    # Construit la fonction `step` avec le budget mémoire initial.
    step = make_step(mem_budget_bytes)
    # Affiche le début de la phase de warmup. Cette phase sert principalement à déclencher la compilation JIT
    # et à vérifier que le calcul tient dans la mémoire disponible.
    print(f"  [warmup] compiling/stabilizing JIT (n={points.shape[0]})...", end="", flush=True)
    t_warmup = time.time()# Démarre le chronomètre du warmup.
    # On utilise une boucle car le premier appel peut provoquer un OOM.
    # Dans ce cas, le budget mémoire sera réduit puis `step` sera recompilé.
    while True:
        try:
            # Utilise des variables temporaires pour le warmup.
            # IMPORTANT :
            # les itérations de warmup ne doivent pas modifier les vrais
            # `points` et `state` qui seront utilisés ensuite.
            wp, ws = points, state
            # Effectue quelques itérations fictives. # Le but n'est PAS d'améliorer la reconstruction :
            # on cherche surtout à déclencher la compilation JIT et à # tester la consommation mémoire du calcul complet.
            for _ in range(4):
                wp, ws, wv = step(wp, ws, normals, bin_edges, bin_mass)
            # Les calculs JAX sur GPU peuvent être asynchrones. `block_until_ready()` force donc l'attente de la fin réelle
            # du calcul avant de continuer.
            wv.block_until_ready()
            # Si on arrive ici, la compilation et les calculs de warmup ont réussi avec le budget mémoire actuel.
            break
        except jax.errors.JaxRuntimeError as e:
            # Si l'erreur n'est pas liée à un dépassement de mémoire, on ne tente pas de la corriger automatiquement.
            # L'erreur est donc propagée normalement.
            if mem_budget_bytes is None or "RESOURCE_EXHAUSTED" not in str(e):
                raise
            # Un OOM a été détecté. On divise le budget mémoire par deux afin de forcer `loss`
            # à traiter moins d'angles simultanément. Cela réduit la consommation mémoire au prix d'un calcul
            # potentiellement plus long.
            mem_budget_bytes = mem_budget_bytes // 2 if mem_budget_bytes >= 2 else None
            print(f" OOM, shrinking angle-chunk budget...", end="", flush=True)
            # Reconstruit `step` avec le nouveau budget mémoire.
            #
            # Comme le budget influence le chunking utilisé dans `loss`,
            # JAX devra compiler à nouveau cette nouvelle version.
            step = make_step(mem_budget_bytes)

    print(f" done ({time.time() - t_warmup:.2f}s)")
    # Boucle principale de l'optimisation L-BFGS.
    #
    # Contrairement au warmup, les modifications de `points` et `state`
    # sont ici conservées d'une itération à l'autre.
    for i in range(max_iter):
        if tracker is not None:
            tracker.start()
        if grad_timer is not None:
            t0 = time.time()
        # Effectue une itération complète de l'optimisation :
        #
        #     points actuels
        #          ↓
        #     calcul de la loss
        #          ↓
        #     calcul du gradient
        #          ↓
        #     L-BFGS + line search
        #          ↓
        #     nouveaux points
        #
        # `state` est également mis à jour pour conserver l'historique
        # nécessaire aux prochaines étapes de L-BFGS.
        points, state, value = step(points, state, normals, bin_edges, bin_mass)

        if grad_timer is not None:
            # Attend la fin réelle du calcul GPU.
            value.block_until_ready()

            elapsed_ms = (time.time() - t0) * 1000
            # Le line search peut effectuer plusieurs évaluations de la loss
            # avant de choisir la longueur de pas finale.
            #
            # On récupère le nombre d'évaluations effectuées :
            #
            #     1 évaluation principale
            #     + les évaluations supplémentaires du line search
            nb_evals = 1 + int(state[2].info.num_linesearch_steps)
            # Répartit approximativement le temps total entre les différentes
            # évaluations afin d'obtenir une estimation du temps moyen
            # par appel de gradient.
            for _ in range(nb_evals):
                grad_timer.record(elapsed_ms / nb_evals)

        if tracker is not None: # Si un tracker est fourni, enregistre l'état de la reconstruction
    # après cette itération.
            tracker.step(i, value, points)

    return points# Une fois toutes les itérations terminées, retourne les positions  finales des Dirac.


def _split(points, n, key, jitter):
    """Grow `points` to `n` rows by tiling (cyclic repeat) + jitter noise.
    Augmente le nombre de points de `points` jusqu'à atteindre `n`.
    Résumé :
        Les points existants sont d'abord répétés cycliquement jusqu'à
        obtenir suffisamment de points, puis un petit bruit aléatoire
        (`jitter`) est ajouté à chaque point.

        Cette fonction est utilisée dans l'approche multiscale :
        on part d'une reconstruction avec peu de Dirac, puis on augmente
        progressivement leur nombre en créant plusieurs copies légèrement
        décalées des points déjà optimisés.

    Args:
        points:
            Positions actuelles des Dirac, de shape `(N, 2)`.

        n:
            Nombre de Dirac souhaité après le raffinement.

        key:
            Clé pseudo-aléatoire JAX utilisée pour générer le bruit.

        jitter:
            Amplitude du bruit aléatoire ajouté aux points.

    Returns:
        Un tableau de shape `(n, 2)` contenant les nouveaux points.
    """
    reps = -(-n // points.shape[0])  # ceil div ,     # On obtient donc au moins `n` points après répétition.
    # Répète les points `reps` fois suivant l'axe des lignes.
    #
    # Si :
    #
    #     points = [A, B, C]
    #
    # alors :
    #
    #     jnp.tile(points, (2, 1))
    #
    # donne :
    #
    #     [A]
    #     [B]
    #     [C]
    #     [A]
    #     [B]
    #     [C]
    #
    # Le `[:n]` permet ensuite de conserver exactement `n` points.
    tiled = jnp.tile(points, (reps, 1))[:n]
    # Génère un bruit gaussien indépendant pour chacun des `n` points.
    # Cela évite que les nouveaux Dirac soient exactement superposés.
    noises =  jitter * jr.normal(key, (n, 2), dtype=points.dtype)
    new_points = tiled + noises
    return new_points


def multiscale_optimize(sino,
                        nb_points_final,
                        nb_points_init=200,
                        factor=4,
                        seed=0,
                        tracker=None,
                        timings=None,
                        **kwargs):
    """Coarse-to-fine `optimize`: converge on `nb_points_init` random diracs,
    then repeatedly split each point into `factor` jittered children and
    re-converge, until reaching `nb_points_final`. Each early stage is much
    cheaper (fewer points to sort per angle, see `optimize`'s docstring) and
    already gives the next stage a good warm start instead of a random one.

    `timings`, if given a dict, is filled with `{n: mean_ms_per_grad_call}`
    per stage (and each stage's mean is printed) -- see `GradTimer`.

    Effectue une reconstruction multiscale par Dirac.

    Résumé :
        Commence avec un petit nombre de points aléatoires, optimise
        leur position, puis augmente progressivement leur nombre.

        À chaque changement d'échelle, chaque point est dupliqué plusieurs
        fois avec un petit bruit aléatoire. La solution précédente sert
        donc d'initialisation à l'étape suivante.

        Cette stratégie réduit le coût initial et fournit un meilleur
        point de départ qu'une initialisation aléatoire complète avec
        beaucoup de Dirac.

    Args:
        sino:
            Sinogramme cible.

        nb_points_final:
            Nombre final de Dirac.

        nb_points_init:
            Nombre de Dirac utilisés à la première échelle.

        factor:
            Facteur de multiplication du nombre de points entre deux
            niveaux.

        seed:
            Graine aléatoire.

    Returns:
        Les coordonnées 2D finales des Dirac.

    """
    extent = sino.geometry.extent
    key, sub = jr.split(jr.PRNGKey(seed))    # Initialise le générateur pseudo-aléatoire JAX
    points = jr.uniform(sub,    # Initialise les premiers points uniformément dans le domaine spatial
                        (nb_points_init, 2),
                        minval=-extent / 2,
                        maxval=extent / 2,
                        dtype=jnp.float32)

    n = nb_points_init

    while True:
        grad_timer = GradTimer() if timings is not None else None
        # Optimise les positions des n Dirac actuels
        points = optimize(points, sino, tracker=tracker, grad_timer=grad_timer, **kwargs)

        if grad_timer is not None:
            timings[n] = grad_timer.mean_ms
            print(f"  n={n:8d}: {grad_timer.mean_ms:.3f} ms/grad ({len(grad_timer.times_ms)} calls)")

        if n >= nb_points_final:        # Si on a atteint le nombre demandé, terminé.
            return points
        # Augmente le nombre de points pour l'échelle suivante.
        n = min(n * factor, nb_points_final)
        key, sub = jr.split(key)         # Nouvelle sous-clé aléatoire pour le jitter
        # Duplique les anciens points et les sépare légèrement.
        points = _split(points, n, sub, jitter=sino.geometry.dw / 1e6)


if __name__=='__main__':

    nb_diracs = 1_000
    # p = bench( "multiscale", nb_diracs = Param( 1_000, help = "nb diracs" ) )
    from geometry import CtGeometry
    from sinogram import Sinogram
    from tracker import Tracker

    sino = Sinogram( CtGeometry( nb_angles = 600, nb_bins = 4096, extent = 2.0 ) )
    sino.add_disk( center = [ 0, 0 ], radius = 0.9, density = + 1.0 )
    sino.add_disk( center = [ 0, 0 ], radius = 0.7, density = - 1.0 )

    from unidim.plots import plot_sinogram, plot_final_points

    plot_sinogram(sino,'input_sinogram.png')


    tracker = Tracker( record_frames = True )
    timings = {}

    points = multiscale_optimize( sino,
                                  nb_points_final = nb_diracs,
                                  tracker = tracker,
                                  timings = timings )

    # p.results[ "ms_per_grad_by_n" ] = timings
    tracker.export_html("unidim_reconstruction.html", sino.geometry.extent )
    plot_final_points(points, 'final_points.png')

    # import subprocess
    # subprocess.Popen(["firefox", "unidim_reconstruction.html"])
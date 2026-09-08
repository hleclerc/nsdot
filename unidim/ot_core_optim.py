import time
import jax
import jax.numpy as jnp
import optax
import tqdm



def _w2_1d(proj, bin_mass, bin_edges, ext_dtype):
    """
    Calcule W2² entre une mesure empirique uniforme portée par `proj` et une
    mesure cible discrétisée par `bin_mass` / `bin_edges`.
    """
    n = proj.shape[0]
    w = jnp.asarray(1.0 / n, dtype=ext_dtype)    # Masse portée par chaque point.
    dw = bin_edges[1] - bin_edges[0]    # Largeur des bins.
    bin_center = bin_edges[:-1] + dw / 2    # Centre des bins.
    cum = jnp.cumsum(bin_mass)     # Fonction de répartition de la cible.
    cum_start = cum - bin_mass    # Masse cumulée avant chaque bin.
    # Premier moment cumulé avant chaque bin.
    prefix_M = jnp.cumsum(bin_mass * bin_center) - bin_mass * bin_center

    def M(q):
        j = jnp.clip(jnp.searchsorted(cum, q, side="right"), 0, bin_mass.shape[0] - 1)
        f = jnp.where(bin_mass[j] > 0, (q - cum_start[j]) / bin_mass[j], 0.0)
        return prefix_M[j] + bin_mass[j] * (bin_edges[j] * f + dw * f * f / 2)

    s = jnp.sort(proj)    # Quantiles de la mesure empirique.
    q = jnp.arange(n, dtype=ext_dtype) * w    # Positions des quantiles.
    bary = (M(q + w) - M(q)) / w    # Quantile moyen de la cible sur chaque intervalle de masse.
    target_second_moment = jnp.sum(bin_mass * bin_center * bin_center) + dw * dw / 12
    wasserstein2 = w * jnp.sum(s * s) - 2 * w * jnp.sum(s * bary) + target_second_moment #mal conditionnée en float32.
    return wasserstein2


def loss_lax_map(points, normals, bin_mass, bin_edges, batch_size=None,  ext_dtype=jnp.float32):
    """
    Même loss que `loss_vmap`, mais les angles sont traités par blocs.
    `batch_size` contrôle le nombre d'angles traités simultanément.
    batch_size = 1,-> très faible mémoire batch_size = 16 -> 16 angles simultanément
    batch_size = 600-> parallélisme proche de vmap
    """

    def angle_cost(normal):
        projections = points @ normal
        return _w2_1d(projections, bin_mass, bin_edges, ext_dtype=ext_dtype)

    # Le checkpoint limite la quantité d'information conservée pour calculer
    # le gradient. Le gradient peut donc recalculer certaines opérations au
    # lieu de stocker énormément de données en mémoire.


    if batch_size is None:  #    Toutes les directions sont traitées simultanément.

        costs = jax.vmap(angle_cost)(normals)
    else:
        angle_cost = jax.checkpoint(angle_cost)
        costs = jax.lax.map(angle_cost, normals, batch_size=batch_size)

    return jnp.sum(costs)


def optimize(points, normals, batch_size, max_iter=15, max_linesearch_steps=8, ext_dtype=jnp.float32):
    # Loss capturée dans une fermeture. `batch_size` reste une constante Python et contrôle la structure du
    # programme XLA.
    def fun(points):
        return loss_lax_map(points, normals, bin_mass, bin_edges, batch_size,  ext_dtype)

    linesearch = optax.scale_by_zoom_linesearch(max_linesearch_steps=max_linesearch_steps,
                                                initial_guess_strategy="one",)
    solver = optax.lbfgs(linesearch=linesearch)
    state = solver.init(points)
    # Une seule fonction JIT pour englober loss, gradient, line search,
    # L-BFGS et mise à jour des points.
    @jax.jit
    def step(points, state):
        value_and_grad = optax.value_and_grad_from_state(fun)
        value, grad = value_and_grad(points, state=state)
        updates, state = solver.update(grad,state,points,value=value,grad=grad,value_fn=fun,)
        points = optax.apply_updates(points, updates)

        return points, state, value
    # Warm-up : la première exécution déclenche la compilation XLA.
    points, state, value = step(points, state)
    value.block_until_ready()

    for i in tqdm.tqdm(range(max_iter)):
        points, state, value = step(points, state)
        value.block_until_ready()

        tqdm.tqdm.write(f"iteration={i:02d}  loss={float(value):.8e}")

    return points

def benchmark_loss(points, normals, bin_mass, bin_edges,ext_dtype):
    """
    Chaque valeur de batch_size peut entraîner une compilation XLA différente.
    """
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, NB_ANGLES]

    for batch_size in batch_sizes:
        print(f"batch_size = {batch_size}")
        fun = jax.jit(lambda p: loss_lax_map(p, normals, bin_mass, bin_edges,batch_size,  ext_dtype))
        # Première exécution : compilation.
        value = fun(points)
        value.block_until_ready()
        n_repeat = 3         # Mesure de plusieurs exécutions.
        start = time.perf_counter()

        for _ in range(n_repeat):
            value = fun(points)
            value.block_until_ready()

        elapsed = (time.perf_counter() - start) / n_repeat
        print(f"loss = {float(value):.8e}    time = {elapsed:.4f} s")

if __name__ == "__main__":
    NB_POINTS = 10_000
    NB_ANGLES = 600
    NB_BINS = 4096
    BATCH_SIZE = 16
    MAX_ITER = 15
    MAX_LINESEARCH_STEPS = 8
    ext_dtype = jnp.float64

    # On construit une mesure uniforme sur [-1, 1].
    # Toutes les masses sont identiques.
    # Directions de projection.
    # Pour rester proche de ton problème CT, on prend 600 normales  régulièrement réparties sur le cercle.
    bin_edges = jnp.linspace(-1.0, 1.0, NB_BINS + 1, dtype=ext_dtype)
    bin_mass = jnp.ones(NB_BINS, dtype=ext_dtype)
    bin_mass = bin_mass / jnp.sum(bin_mass)
    theta = jnp.linspace(0.0,jnp.pi,NB_ANGLES,endpoint=False,dtype=ext_dtype,)
    normals = jnp.stack([jnp.cos(theta),jnp.sin(theta),],axis=1,)
    # Initialisation aléatoire des points.Les points sont initialement dans [-1, 1]².
    key = jax.random.PRNGKey(0)
    points = jax.random.uniform(key,shape=(NB_POINTS, 2),minval=-1.0,maxval=1.0,dtype=ext_dtype,)
    benchmark_loss(points, normals,bin_mass, bin_edges,ext_dtype=ext_dtype)
    points = optimize(points, normals, None, max_iter=15, max_linesearch_steps=8, ext_dtype=jnp.float32)
    print("Optimisation terminée.")
    print("points.shape =", points.shape)
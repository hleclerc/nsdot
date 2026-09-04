import jax
import jax.numpy as jnp
import optax
from functools import partial
# TODO from jax.tree_util import Partial
import tqdm

from unidim.plots import plot_final_points
import nvidia_smi

nvidia_smi.nvmlInit()
handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
# input
NB_ANGLES = 600
NB_BINS = 4096
NB_POINTS_FINAL = 100_000
INNER_RADIUS = 0.7
OUTER_RADIUS = 0.9
# reconstruction
JAX_ENABLE_X64 = False
BYTES_PER_CHUNK_ELEMENT = 256  # no effect on gpu mem
SAFETY_FRACTION = 0.5  # leave headroom for everything else alive on the device
FALLBACK_BYTES = 512 * 1024 * 1024  # no CUDA visible -- a conservative default chunk budget
# linesearch = optax.scale_by_zoom_linesearch(... args
MAX_ITER=15
MAX_LINESEARCH_STEPS=8 # TODO 4
INITIAL_GUESS_STRATEGY="one" # TODO "quadratic"  # ou "backtracking"
# multiscale_optimize, gestion des point
NB_POINTS_INIT=200
FACTOR=4
SEED=0

jax.config.update("jax_enable_x64", JAX_ENABLE_X64)
ext_dtype = jnp.float64 if jax.config.x64_enabled else jnp.float32

def _w2_1d(proj, bin_mass, bin_edges):
    n = proj.shape[0]
    w = 1.0 / n
    dw = bin_edges[1] - bin_edges[0]
    bin_center = bin_edges[:-1] + dw / 2
    cum = jnp.cumsum(bin_mass)
    cum_start = cum - bin_mass
    prefix_M = jnp.cumsum(bin_mass * bin_center) - bin_mass * bin_center

    def M(q):
        j = jnp.clip(jnp.searchsorted(cum, q, side="right"), 0, bin_mass.shape[0] - 1)
        f = jnp.where(bin_mass[j] > 0, (q - cum_start[j]) / bin_mass[j], 0.0)
        return prefix_M[j] + bin_mass[j] * (bin_edges[j] * f + dw * f * f / 2)

    s = jnp.sort(proj).astype(ext_dtype)
    q = jnp.arange(n, dtype=ext_dtype) * w
    bary = (M(q + w) - M(q)) / w
    target_second_moment = jnp.sum(bin_mass * bin_center * bin_center) + dw * dw / 12
    wasserstein2 = w * jnp.sum(s * s) - 2 * w * jnp.sum(s * bary) + target_second_moment
    return wasserstein2

# def _w2_1d(proj, bin_mass, bin_edges):
#     n = proj.shape[0]
#     w = 1.0 / n
#     dw = bin_edges[1] - bin_edges[0]
#     bin_center = bin_edges[:-1] + dw / 2
#     cum = jnp.cumsum(bin_mass)
#     cum_start = cum - bin_mass
#     prefix_M = jnp.cumsum(bin_mass * bin_center) - bin_mass * bin_center
#
#     s = jnp.sort(proj).astype(ext_dtype)
#     q = jnp.arange(n, dtype=ext_dtype) * w
#
#     # Pré-calculer j et f pour q et q + w
#     j0 = jnp.clip(jnp.searchsorted(cum, q, side="right"), 0, bin_mass.shape[0] - 1)
#     j1 = jnp.clip(jnp.searchsorted(cum, q + w, side="right"), 0, bin_mass.shape[0] - 1)
#     f0 = jnp.where(bin_mass[j0] > 0, (q - cum_start[j0]) / bin_mass[j0], 0.0)
#     f1 = jnp.where(bin_mass[j1] > 0, (q + w - cum_start[j1]) / bin_mass[j1], 0.0)
#
#     # Calculer M0 et M1 en une seule passe
#     M0 = prefix_M[j0] + bin_mass[j0] * (bin_edges[j0] * f0 + dw * f0 * f0 / 2)
#     M1 = prefix_M[j1] + bin_mass[j1] * (bin_edges[j1] * f1 + dw * f1 * f1 / 2)
#     bary = (M1 - M0) / w
#
#     # Calculer target_second_moment une seule fois
#     target_second_moment = jnp.sum(bin_mass * bin_center * bin_center) + dw * dw / 12
#     wasserstein2 = w * jnp.sum(s * s) - 2 * w * jnp.sum(s * bary) + target_second_moment
#     return wasserstein2

def _get_chunk_size(nb_points, nb_angles, mem_budget_bytes):
    if mem_budget_bytes is None:
        max_batch_size= 1
    else:
        memory_for_one_angle = BYTES_PER_CHUNK_ELEMENT * max(nb_points, 1)
        max_batch_size = mem_budget_bytes // memory_for_one_angle
        max_batch_size = max(1, min(nb_angles, max_batch_size))
    return max_batch_size #

def loss(points, normals, bin_edges, bin_mass, mem_budget_bytes=-1):
    def angle_cost(normal_and_mass):
        normal, mass = normal_and_mass
        projections = points @ normal
        return _w2_1d(projections, mass, bin_edges)

    n, A = points.shape[0], normals.shape[0]
    batch_size = _get_chunk_size(n, A, mem_budget_bytes)
    costs = jax.lax.map(jax.checkpoint(angle_cost), (normals, bin_mass), batch_size=batch_size)
    # costs = jax.vmap(angle_cost)((normals, bin_mass))
    # jax.checkpoint (gradient checkpointing) est utile pour économiser de la mémoire, mais ralentit l'exécution si mal utilisé.
#     Pourquoi ? vmap est plus rapide que lax.map car il fusionne les opérations.
# Attention : vmap consomme plus de mémoire. Utilisez-le seulement si batch_size est grand et que la mémoire est disponible.

    return costs.sum().astype(ext_dtype)

def optimize(points, sino, max_iter=15, max_linesearch_steps=8, initial_guess_strategy='one'):
    def make_step(mem_budget_bytes):
        @jax.jit
        def step(p, state, normals, bin_edges, bin_mass):
            fun = partial(loss, normals=normals, bin_edges=bin_edges, bin_mass=bin_mass,
                          mem_budget_bytes=mem_budget_bytes)
            value_and_grad = optax.value_and_grad_from_state(fun)
            # grad = jax.grad(fun)(p) désactive les fonctionnalités avancées d'optax (comme le linesearch
            # updates = -0.01 * grad  # Remplacer par un pas fixe ou un optimiseur simple
            value, grad = value_and_grad(p, state=state)
            updates, state = solver.update(grad, state, p, value=value, grad=grad, value_fn=fun)
            p = optax.apply_updates(p, updates)

            return p, state, value

        return step

    g = sino.geometry
    normals = jnp.asarray(g.normals, dtype=ext_dtype)
    bin_edges = jnp.asarray(g.bin_edges, dtype=ext_dtype)
    bin_mass = jnp.asarray(sino.values, dtype=ext_dtype)
    bin_mass = bin_mass / bin_mass.sum(axis=1, keepdims=True)


    linesearch = optax.scale_by_zoom_linesearch(max_linesearch_steps=max_linesearch_steps,
                                                initial_guess_strategy=initial_guess_strategy)
    solver = optax.lbfgs(linesearch=linesearch)
    state = solver.init(points)

    device = jax.devices()[0]
    stats = device.memory_stats()
    # Récupère les informations de mémoire

    if not stats or "bytes_limit" not in stats:
        mem_budget_bytes = FALLBACK_BYTES
    else:
        bytes_limit = stats.get("bytes_limit", 0)
        bytes_in_use = stats.get("bytes_in_use", 0)
        free_bytes = bytes_limit - bytes_in_use
        safe_free_bytes = int(max(free_bytes, 0) * SAFETY_FRACTION)
        mem_budget_bytes = safe_free_bytes if safe_free_bytes > 0 else FALLBACK_BYTES

    chunk_size = _get_chunk_size(points.shape[0], normals.shape[0], mem_budget_bytes)
    step = make_step(mem_budget_bytes)

    for i in (pbar:= tqdm.tqdm(range(max_iter))):

        points, state, value = step(points, state, normals, bin_edges, bin_mass)
        nvidia_mem = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)

        pbar.set_description(
        f"for n = {points.shape[0]} Step: {i}| mem_budget_bytes: {mem_budget_bytes / 1024 ** 3:.2f} GiB, "
        f"VRAM jax: {bytes_in_use / 1024 ** 3:.2f}/{bytes_limit / 1024 ** 3:.2f} GiB, "
        f"VRAM nvidia: {nvidia_mem.used / 1024 ** 3:.2f}/{nvidia_mem.total / 1024 ** 3:.2f} GiB, "
        f" Chunk size: {chunk_size}") #,  in ring {compute_percent_in_ring(points):.1f} %")
    return points

def multiscale_optimize(sino,
                        nb_points_final,
                        nb_points_init=200,
                        factor=4,
                        seed=0,
                        **kwargs):
    extent = sino.geometry.extent
    key, sub = jax.random.split(jax.random.PRNGKey(seed))
    points = jax.random.uniform(sub, (nb_points_init, 2),
                                minval=-extent / 2, maxval=extent / 2,
                                dtype=ext_dtype)
    n = nb_points_init
    jitter = sino.geometry.dw / 1e6  # self.extent / nb_bins
    while n < nb_points_final:
        points = optimize(points, sino, **kwargs)
        n_next = min(n * factor, nb_points_final)
        key, sub = jax.random.split(key)
        reps = -(-n_next // points.shape[0])
        tiled = jnp.tile(points, (reps, 1))[:n_next]
        noises = jitter * jax.random.normal(sub, (n_next, 2), dtype=points.dtype)
        points = tiled + noises
        n = n_next
        # La boucle s'arrête dès que n >= nb_points_final. Il reste alors à
        # optimiser cette dernière population de `nb_points_final` points.

    points = optimize(points, sino, **kwargs)

    return points

if __name__ == '__main__':


    from geometry import CtGeometry
    from sinogram import Sinogram
    sino = Sinogram(CtGeometry(nb_angles=NB_ANGLES, nb_bins=NB_BINS, extent= 2.0 ))
    sino.add_disk(center=[0, 0], radius=OUTER_RADIUS, density=+1.0)
    sino.add_disk(center=[0, 0], radius=INNER_RADIUS, density=-1.0)

    points = multiscale_optimize(sino,
                                 nb_points_final=NB_POINTS_FINAL,
                                 nb_points_init=NB_POINTS_INIT,
                                 factor=FACTOR,
                                 seed=SEED,
                                 max_iter=MAX_ITER,
                                 max_linesearch_steps=MAX_LINESEARCH_STEPS,
                                 initial_guess_strategy=INITIAL_GUESS_STRATEGY)

    plot_final_points(points, 'final_points.png')



# def compute_percent_in_ring(points):
#     distances = jnp.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2)
#     in_ring = (distances >= INNER_RADIUS) & (distances <= OUTER_RADIUS)
#     percent_in_ring = jnp.mean(in_ring) * 100.  # Moyenne = proportion de True
#     return float(percent_in_ring)
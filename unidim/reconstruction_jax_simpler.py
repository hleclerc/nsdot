import jax
import jax.numpy as jnp
import optax
from functools import partial
import tqdm

JAX_ENABLE_X64 = True
jax.config.update("jax_enable_x64", JAX_ENABLE_X64)
BYTES_PER_CHUNK_ELEMENT = 256  # no effect on gpu mem
SAFETY_FRACTION = 0.5  # leave headroom for everything else alive on the device
FALLBACK_BYTES = 512 * 1024 * 1024  # no CUDA visible -- a conservative default chunk budget
# linesearch = optax.scale_by_zoom_linesearch(... args
MAX_ITER=15
MAX_LINESEARCH_STEPS=8
INITIAL_GUESS_STRATEGY="one"
# multiscale_optimize, gestion des point
NB_POINTS_INIT=200
FACTOR=4
SEED=0


def _w2_1d(proj, bin_mass, bin_edges):
    ext_dtype = jnp.float64 if jax.config.x64_enabled else jnp.float32
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

def _get_chunk_size(nb_points, nb_angles, mem_budget_bytes):
    if mem_budget_bytes is None:
        return 1
    else:
        memory_for_one_angle = BYTES_PER_CHUNK_ELEMENT * max(nb_points, 1)
        max_batch_size = mem_budget_bytes // memory_for_one_angle
        max_batch_size = jnp.maximum(1, jnp.minimum(nb_angles, max_batch_size))
    return max_batch_size

def loss(points, normals, bin_edges, bin_mass, mem_budget_bytes=-1):
    def angle_cost(normal_and_mass):
        normal, mass = normal_and_mass
        projections = points @ normal
        return _w2_1d(projections, mass, bin_edges)

    n, A = points.shape[0], normals.shape[0]
    chunk_size = _get_chunk_size(n, A, mem_budget_bytes)
    batch_size = jnp.where(chunk_size > 1, chunk_size, 0)

    try:
        batch_size = int(batch_size.item()) if batch_size > 0 else None
    except:
        batch_size = None

    costs = jax.lax.map(jax.checkpoint(angle_cost), (normals, bin_mass), batch_size=batch_size)
    cost_32 = costs.sum().astype(jnp.float32)
    return cost_32

def optimize(points, sino, max_iter=15, max_linesearch_steps=8, initial_guess_strategy='one'):
    def make_step(mem_budget_bytes):
        @jax.jit
        def step(p, state, normals, bin_edges, bin_mass):
            fun = partial(loss, normals=normals, bin_edges=bin_edges, bin_mass=bin_mass,
                          mem_budget_bytes=mem_budget_bytes)
            value_and_grad = optax.value_and_grad_from_state(fun)
            value, grad = value_and_grad(p, state=state)
            updates, state = solver.update(grad, state, p, value=value, grad=grad, value_fn=fun)
            p = optax.apply_updates(p, updates)
            return p, state, value

        return step

    ext_dtype = jnp.float64 if jax.config.x64_enabled else jnp.float32
    g = sino.geometry
    normals = jnp.asarray(g.normals, dtype=jnp.float32)
    bin_edges = jnp.asarray(g.bin_edges, dtype=ext_dtype)
    bin_mass = jnp.asarray(sino.values, dtype=ext_dtype)
    bin_mass = bin_mass / bin_mass.sum(axis=1, keepdims=True)


    linesearch = optax.scale_by_zoom_linesearch(max_linesearch_steps=max_linesearch_steps, initial_guess_strategy=initial_guess_strategy)
    solver = optax.lbfgs(linesearch=linesearch)
    state = solver.init(points)

    device = jax.devices()[0]
    stats = device.memory_stats()

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
        pbar.set_description(f"for n = {points.shape[0]} Step: {i}| mem_budget_bytes: {mem_budget_bytes/1024**3:.2f} GiB VRAM used: {bytes_in_use/1024**3:.2f} GiB, VRAM tot: {bytes_limit/1024**3:.2f} GiB, Chunk size: {chunk_size}")
        points, state, value = step(points, state, normals, bin_edges, bin_mass)

    return points

def multiscale_optimize(sino, nb_points_final,
                        nb_points_init=200, factor=4, seed=0, **kwargs):
    extent = sino.geometry.extent
    key, sub = jax.random.split(jax.random.PRNGKey(seed))
    points = jax.random.uniform(sub, (nb_points_init, 2),
                                minval=-extent / 2, maxval=extent / 2,
                                dtype=jnp.float32)
    n = nb_points_init
    jitter = sino.geometry.dw / 1e6  # self.extent / nb_bins
    steps = []
    while n < nb_points_final:
        steps.append(n)
        n = min(n * factor, nb_points_final)

    for step in steps:
        points = optimize(points, sino, **kwargs)
        key, sub = jax.random.split(key)
        reps = -(-step // points.shape[0])
        tiled = jnp.tile(points, (reps, 1))[:step]
        noises = jitter * jax.random.normal(sub, (step, 2), dtype=points.dtype)
        points = tiled + noises

    return points

if __name__ == '__main__':
    nb_points_final = 100_000
    from geometry import CtGeometry
    from sinogram import Sinogram
    sino = Sinogram(CtGeometry(nb_angles=6000, nb_bins=4096, extent=2.0))
    sino.add_disk(center=[0, 0], radius=0.9, density=+1.0)
    sino.add_disk(center=[0, 0], radius=0.7, density=-1.0)

    points = multiscale_optimize(sino,
                                 nb_points_final=nb_points_final,
                                 nb_points_init=NB_POINTS_INIT,
                                 factor=NB_POINTS_INIT,
                                 seed=SEED,
                                 max_iter=MAX_ITER,
                                 max_linesearch_steps=MAX_LINESEARCH_STEPS,
                                 initial_guess_strategy=INITIAL_GUESS_STRATEGY)

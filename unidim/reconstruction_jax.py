import time

import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from loom.testing import Param, bench
from gpu_mem import jax_mem_budget_bytes
from tracker import GradTimer

# Needed for the float64 promotion in `_w2_1d` below -- disabled by default,
# JAX otherwise SILENTLY truncates any float64 array back to float32.
jax.config.update("jax_enable_x64", True)

# See `loss`'s docstring for how this was measured.
_BYTES_PER_CHUNK_ELEMENT = 256


def _w2_1d(proj,
           bin_mass,
           bin_edges):
    """Squared 1D Wasserstein distance between the empirical measure of
    `proj` (n equal-mass diracs) and the piecewise-constant measure
    `bin_mass` over `bin_edges` (both total mass 1).

    Closed form via the target's integrated quantile function
    M(q) = int_0^q Q(t) dt: the barycentric projection of dirac i is
    (M(q1) - M(q0)) / w over its quantile interval [q0, q1). `bary` only
    depends on the target and on each dirac's RANK (not its value), so
    plain autodiff through `jnp.sort` already gives the correct
    envelope-theorem gradient wrt `proj` — no hand-derived backward pass.

    """
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

    s = jnp.sort(proj).astype(jnp.float64)
    q = jnp.arange(n, dtype=jnp.float64) * w
    bary = (M(q + w) - M(q)) / w

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

    """
    # `-1` (not `None`) as the sentinel: `mem_budget_bytes` flows through
    # `jax.jit` alongside `points`/`normals`/... in `optimize.step`, and a
    # `None` default would make jit treat "was it passed" as part of the
    # traced signature -- a plain Python int stays a compile-time constant
    # either way (it's never turned into a jnp array).
    if mem_budget_bytes == -1:
        mem_budget_bytes = jax_mem_budget_bytes()
    n, A = points.shape[0], normals.shape[0]
    chunk_size = 1 if mem_budget_bytes is None else max(1,
                                                        min(A,
                                                            mem_budget_bytes // (_BYTES_PER_CHUNK_ELEMENT * max(n, 1))
                                                            )
                                                        )

    def angle_cost(normal_and_mass):
        normal, mass = normal_and_mass
        return _w2_1d(points @ normal, mass, bin_edges)

    costs = jax.lax.map(jax.checkpoint(angle_cost), (normals, bin_mass),
                        batch_size=int(chunk_size) if chunk_size > 1 else None)
    return costs.sum().astype(jnp.float32)


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

    """
    normals, bin_edges, bin_mass = _sino_arrays(sino)
    linesearch = optax.scale_by_zoom_linesearch(
        max_linesearch_steps=max_linesearch_steps,
        initial_guess_strategy="one")
    solver = optax.lbfgs(linesearch=linesearch)
    state = solver.init(points)

    def make_step(mem_budget_bytes):
        @jax.jit
        def step(p, state, normals, bin_edges, bin_mass):
            fun = lambda pp: loss(pp, normals, bin_edges, bin_mass, mem_budget_bytes=mem_budget_bytes)
            value_and_grad = optax.value_and_grad_from_state(fun)
            value, grad = value_and_grad(p, state=state)
            updates, state = solver.update(grad, state, p, value=value, grad=grad, value_fn=fun)
            p = optax.apply_updates(p, updates)
            return p, state, value
        return step

    mem_budget_bytes = jax_mem_budget_bytes()
    step = make_step(mem_budget_bytes)

    print(f"  [warmup] compiling/stabilizing JIT (n={points.shape[0]})...", end="", flush=True)
    t_warmup = time.time()
    while True:
        try:
            wp, ws = points, state
            for _ in range(4):
                wp, ws, wv = step(wp, ws, normals, bin_edges, bin_mass)
            wv.block_until_ready()
            break
        except jax.errors.JaxRuntimeError as e:
            if mem_budget_bytes is None or "RESOURCE_EXHAUSTED" not in str(e):
                raise
            mem_budget_bytes = mem_budget_bytes // 2 if mem_budget_bytes >= 2 else None
            print(f" OOM, shrinking angle-chunk budget...", end="", flush=True)
            step = make_step(mem_budget_bytes)
    print(f" done ({time.time() - t_warmup:.2f}s)")

    for i in range(max_iter):
        if tracker is not None:
            tracker.start()
        if grad_timer is not None:
            t0 = time.time()
        points, state, value = step(points, state, normals, bin_edges, bin_mass)
        if grad_timer is not None:
            value.block_until_ready()
            elapsed_ms = (time.time() - t0) * 1000
            nb_evals = 1 + int(state[2].info.num_linesearch_steps)
            for _ in range(nb_evals):
                grad_timer.record(elapsed_ms / nb_evals)
        if tracker is not None:
            tracker.step(i, value, points)
    return points


def _split(points, n, key, jitter):
    """Grow `points` to `n` rows by tiling (cyclic repeat) + jitter noise."""
    reps = -(-n // points.shape[0])  # ceil div
    tiled = jnp.tile(points, (reps, 1))[:n]
    return tiled + jitter * jr.normal(key, (n, 2), dtype=points.dtype)


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
    """
    extent = sino.geometry.extent
    key, sub = jr.split(jr.PRNGKey(seed))
    points = jr.uniform(sub,
                        (nb_points_init, 2),
                        minval=-extent / 2,
                        maxval=extent / 2,
                        dtype=jnp.float32)

    n = nb_points_init

    while True:
        grad_timer = GradTimer() if timings is not None else None
        points = optimize(points, sino, tracker=tracker, grad_timer=grad_timer, **kwargs)
        if grad_timer is not None:
            timings[n] = grad_timer.mean_ms
            print(f"  n={n:8d}: {grad_timer.mean_ms:.3f} ms/grad "
                  f"({len(grad_timer.times_ms)} calls)")
        if n >= nb_points_final:
            return points
        n = min(n * factor, nb_points_final)
        key, sub = jr.split(key)
        points = _split(points, n, sub, jitter=sino.geometry.dw / 1e6)


if __name__=='__main__':

    nb_diracs = 1_000
    p = bench( "multiscale", nb_diracs = Param( 1_000, help = "nb diracs" ) )
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
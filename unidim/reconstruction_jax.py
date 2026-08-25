import time

import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from loom.testing import Param, bench

from .tracker import GradTimer

# Needed for the float64 promotion in `_w2_1d` below -- disabled by default,
# JAX otherwise SILENTLY truncates any float64 array back to float32.
jax.config.update("jax_enable_x64", True)


def _w2_1d(proj, bin_mass, bin_edges):
    """Squared 1D Wasserstein distance between the empirical measure of
    `proj` (n equal-mass diracs) and the piecewise-constant measure
    `bin_mass` over `bin_edges` (both total mass 1).

    Closed form via the target's integrated quantile function
    M(q) = int_0^q Q(t) dt: the barycentric projection of dirac i is
    (M(q1) - M(q0)) / w over its quantile interval [q0, q1). `bary` only
    depends on the target and on each dirac's RANK (not its value), so
    plain autodiff through `jnp.sort` already gives the correct
    envelope-theorem gradient wrt `proj` — no hand-derived backward pass.

    `s`/`q`/`bary` are float64 despite `proj` being float32: at convergence
    `s` and `bary` agree to several digits, so `s - bary` (inside the
    `w*sum(s**2) - 2*w*sum(s*bary)` expansion below) is a near-cancellation
    of two O(1) quantities -- in float32 its rounding error stops shrinking
    once the true residual drops near float32's ~7-digit floor. Verified:
    an exact-duplicate point split, which must leave the true loss EXACTLY
    unchanged (see `_split`'s docstring), moved the float32 loss by up to
    ~250%; the same computation promoted to float64 left it bit-exact.
    `proj` itself (and its `jnp.sort`, the expensive O(n log n) part) stays
    float32 -- only the cheap per-quantile scalar math promotes.
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


def loss(points, sino):
    """Sum over angles of the 1D Wasserstein distance between `points`
    projected on that angle and the sinogram's line for that angle.

    Angles are folded with `lax.map` (not `vmap`) so only ONE angle's
    [n]-sized projection is ever materialized at a time — with nb_angles in
    the hundreds and nb_diracs up to ~1e11, a stacked [nb_angles, n] tensor
    is not an option.

    `bin_edges`/`bin_mass` are float64 (see `_w2_1d`'s docstring); `normals`
    stays float32 so `points @ normal` (and so the sort inside `_w2_1d`)
    doesn't get promoted along with them -- JAX would upcast `proj` to
    float64 too if `normal` were float64, doubling the cost of the one part
    of this computation that's actually expensive. The float64 SUM (over up
    to nb_angles terms) is only cast back to float32 at the very end: optax
    caches this value internally and requires it to stay the same dtype as
    `points` across calls (a `lax.cond` branch dtype mismatch otherwise) --
    a plain downcast is safe here since by this point the delicate
    cancellation is already done and the result is a well-conditioned small
    number, which float32 represents just fine.
    """
    g = sino.geometry
    normals = jnp.asarray(g.normals, dtype=jnp.float32)
    bin_edges = jnp.asarray(g.bin_edges, dtype=jnp.float64)
    bin_mass = jnp.asarray(sino.values, dtype=jnp.float64)
    bin_mass = bin_mass / bin_mass.sum(axis=1, keepdims=True)

    def angle_cost(normal_and_mass):
        normal, mass = normal_and_mass
        return _w2_1d(points @ normal, mass, bin_edges)

    costs = jax.lax.map(angle_cost, (normals, bin_mass))
    return costs.sum().astype(jnp.float32)


def optimize(points, sino, max_iter=15, tracker=None, grad_timer=None, max_linesearch_steps=8):
    """L-BFGS (optax) on `loss`. Returns the optimized points.

    Runs a plain Python loop (not `lax.scan`) with a single jitted step: the
    step compiles once and every iteration then actually runs (and can be
    reported via `tracker`) instead of vanishing inside one big opaque
    compiled program for the whole `max_iter` budget.

    `max_linesearch_steps` caps optax's zoom linesearch (default 20) at a
    lower value: each `loss` evaluation here does 600 SEQUENTIAL `jnp.sort`
    calls (one per angle, ~15ms each on CPU with no cross-angle batching
    speedup -- verified: batching them together via `lax.map`'s `batch_size`
    was actually slower here), so every extra linesearch trial is ~9s. A
    smaller cap trades a bit of per-step step-size accuracy for far fewer
    `loss` evaluations.

    `grad_timer`, if given, is fed one entry per ACTUAL `loss`/grad
    evaluation, not per outer step: `step` is a single jitted call, so the
    only thing timeable from Python is the whole call, but optax's own
    `state[2].info.num_linesearch_steps` (plus the one eval `value_and_grad`
    itself always does) says exactly how many evaluations that call made --
    the elapsed time is divided by that count.

    `step` is a FRESH closure every `optimize()` call (a new stage's
    shapes/solver state), so its first calls compile AND run -- and it
    takes the first FOUR calls, not just the first one: `state`'s
    weak-typed fields (e.g. `num_linesearch_steps`, a weak int32 zero at
    init) settle into their concrete, non-weak dtype pattern only after a
    few real updates, and each distinct abstract signature along the way
    gets its own compile (verified: 1 warmup call left 2 of the real
    iterations slow, 4 left all of them fast). Compile time is dominated by
    graph structure, barely by n, so left in it swamps the timing and makes
    it look almost independent of n -- warming up on a THROWAWAY copy of
    the initial state, before starting to record from the real one, avoids
    that without spending any of `max_iter`'s real optimization budget on
    unrecorded steps.
    """
    fun = lambda p: loss(p, sino)
    value_and_grad = optax.value_and_grad_from_state(fun)
    linesearch = optax.scale_by_zoom_linesearch(
        max_linesearch_steps=max_linesearch_steps, initial_guess_strategy="one")
    solver = optax.lbfgs(linesearch=linesearch)
    state = solver.init(points)

    @jax.jit
    def step(p, state):
        value, grad = value_and_grad(p, state=state)
        updates, state = solver.update(grad, state, p, value=value, grad=grad, value_fn=fun)
        p = optax.apply_updates(p, updates)
        return p, state, value

    if grad_timer is not None:
        print(f"  [warmup] compiling/stabilizing JIT (n={points.shape[0]})...", end="", flush=True)
        t_warmup = time.time()
        wp, ws = points, state
        for _ in range(4):
            wp, ws, wv = step(wp, ws)
        wv.block_until_ready()
        print(f" done ({time.time() - t_warmup:.2f}s)")

    for i in range(max_iter):
        if tracker is not None:
            tracker.start()
        if grad_timer is not None:
            t0 = time.time()
        points, state, value = step(points, state)
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


def multiscale_optimize(sino, nb_points_final, nb_points_init=200, factor=4,
                        seed=0, tracker=None, timings=None, **kwargs):
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
    points = jr.uniform(sub, (nb_points_init, 2), minval=-extent / 2, maxval=extent / 2,
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


if p := bench( "multiscale", nb_diracs = Param( 100_000, help = "nb diracs" ) ):
    from .geometry import CtGeometry
    from .sinogram import Sinogram
    from .tracker import Tracker

    sino = Sinogram( CtGeometry( nb_angles = 600, nb_bins = 4096, extent = 2.0 ) )
    sino.add_disk( center = [ 0, 0 ], radius = 0.9, density = + 1.0 )
    sino.add_disk( center = [ 0, 0 ], radius = 0.7, density = - 1.0 )

    tracker = Tracker( record_frames = True )
    timings = {}
    points = multiscale_optimize( sino, nb_points_final = p.nb_diracs, tracker = tracker, timings = timings )
    p.results[ "ms_per_grad_by_n" ] = timings
    tracker.export_html( p.out_dir / "unidim_reconstruction.html", sino.geometry.extent )

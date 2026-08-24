import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from loom.testing import Param, bench


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

    s = jnp.sort(proj)
    q = jnp.arange(n) * w
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
    """
    g = sino.geometry
    normals = jnp.asarray(g.normals)
    bin_edges = jnp.asarray(g.bin_edges)
    bin_mass = jnp.asarray(sino.values)
    bin_mass = bin_mass / bin_mass.sum(axis=1, keepdims=True)

    def angle_cost(normal_and_mass):
        normal, mass = normal_and_mass
        return _w2_1d(points @ normal, mass, bin_edges)

    costs = jax.lax.map(angle_cost, (normals, bin_mass))
    return costs.sum()


def optimize(points, sino, max_iter=15, tracker=None, max_linesearch_steps=8):
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

    for i in range(max_iter):
        if tracker is not None:
            tracker.start()
        points, state, value = step(points, state)
        if tracker is not None:
            tracker.step(i, value, points)
    return points


def _split(points, n, key, jitter):
    """Grow `points` to `n` rows by tiling (cyclic repeat) + jitter noise."""
    reps = -(-n // points.shape[0])  # ceil div
    tiled = jnp.tile(points, (reps, 1))[:n]
    return tiled + jitter * jr.normal(key, (n, 2))


def multiscale_optimize(sino, nb_points_final, nb_points_init=200, factor=4,
                        seed=0, tracker=None, **kwargs):
    """Coarse-to-fine `optimize`: converge on `nb_points_init` random diracs,
    then repeatedly split each point into `factor` jittered children and
    re-converge, until reaching `nb_points_final`. Each early stage is much
    cheaper (fewer points to sort per angle, see `optimize`'s docstring) and
    already gives the next stage a good warm start instead of a random one.
    """
    extent = sino.geometry.extent
    key, sub = jr.split(jr.PRNGKey(seed))
    points = jr.uniform(sub, (nb_points_init, 2), minval=-extent / 2, maxval=extent / 2)

    print( points.device )
    
    n = nb_points_init
    while True:
        print( f'nb_diracs={ points.shape[ 0 ] }' )
        points = optimize(points, sino, tracker=tracker, **kwargs)
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
    points = multiscale_optimize( sino, nb_points_final = p.nb_diracs, tracker = tracker )
    tracker.export_html( p.out_dir / "unidim_reconstruction.html", sino.geometry.extent )

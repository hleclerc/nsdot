import time
import jax
import jax.numpy as jnp
import optax
from lab_wasser import _w2_1d as wasser_dist

def loss_lax_map(points, normals, bin_mass, bin_edges, batch_size, ext_dtype, use_checkpoint=True):
    def angle_cost(normal_and_mass):
        normal, mass = normal_and_mass
        projections = points @ normal
        w = wasser_dist(projections, mass, bin_edges, ext_dtype)
        return w

    if use_checkpoint:
        angle_cost = jax.checkpoint(angle_cost)
    costs = jax.lax.map(angle_cost,
                        (normals,bin_mass),
                        batch_size=batch_size,)

    return jnp.sum(costs).astype(jnp.float32)

def optimize(points, sino,max_iter=15,max_linesearch_steps=8,
    initial_guess_strategy="one"):
    batch_size = 16
    ext_dtype = jnp.float64
    use_checkpoint = True

    g = sino.geometry

    normals = jnp.asarray(g.normals, dtype=ext_dtype)
    bin_edges = jnp.asarray(g.bin_edges, dtype=ext_dtype)
    bin_mass = jnp.asarray(sino.values, dtype=ext_dtype)
    bin_mass = bin_mass / bin_mass.sum(axis=1, keepdims=True)

    # ------------------------------------------------------------------
    # Fonction objectif
    # ------------------------------------------------------------------

    def fun(p):
        return loss_lax_map(p, normals, bin_mass, bin_edges, batch_size,
            ext_dtype, use_checkpoint)

    # ------------------------------------------------------------------
    # Optimiseur L-BFGS + zoom line search
    # ------------------------------------------------------------------

    linesearch = optax.scale_by_zoom_linesearch(
        max_linesearch_steps=max_linesearch_steps,
        initial_guess_strategy=initial_guess_strategy,
    )

    solver = optax.lbfgs(linesearch=linesearch)
    state = solver.init(points)

    # ------------------------------------------------------------------
    # Une itération : loss + gradient + L-BFGS
    # ------------------------------------------------------------------

    @jax.jit
    def step(p, state):
        value_and_grad = optax.value_and_grad_from_state(fun)
        value, grad = value_and_grad(p,state=state)
        updates, state = solver.update(grad,state, p,value=value,grad=grad,value_fn=fun)
        p = optax.apply_updates(p, updates)

        return p, state, value, grad

    # ------------------------------------------------------------------
    # Boucle d'optimisation
    # ------------------------------------------------------------------

    print("Starting L-BFGS")
    print(f"  max_iter             : {max_iter}")
    print(f"  max_linesearch_steps : {max_linesearch_steps}")
    print(f"  batch_size           : {batch_size}")
    print(f"  ext_dtype            : {ext_dtype}")
    print(f"  checkpoint            : {use_checkpoint}")
    print()

    t0 = time.perf_counter()

    for i in range(max_iter):
        points, state, value, grad = step(points, state)

        # Synchronisation nécessaire pour mesurer correctement le temps.
        value.block_until_ready()
        grad_norm = jnp.linalg.norm(grad)

        # print(
        #     f"iter {i + 1:3d} | "
        #     f"loss = {float(value):.12e} | "
        #     f"|grad| = {float(grad_norm):.6e}"
        # )

    points.block_until_ready()

    elapsed = time.perf_counter() - t0

    print()
    print(f"L-BFGS time : {elapsed:.3f} s")
    print(f"Final loss  : {float(value):.12e}")

    return points, state

from geometry import CtGeometry
from sinogram import Sinogram
sino = Sinogram(CtGeometry(nb_angles=600, nb_bins=4096, extent= 2.0 ))
sino.add_disk(center=[0, 0], radius=.9, density=+1.0)
sino.add_disk(center=[0, 0], radius=0.7, density=-1.0)
key = jax.random.PRNGKey(0)

points0 = jax.random.uniform(
    key,
    shape=(10_000, 2),
    minval=-1.0,
    maxval=1.0,
    dtype=jnp.float32,
)
for max_linesearch_steps in [1, 4, 8 ]:
    points = points0
    print("-"*60)
    # print(max_linesearch_steps)
    optimize(points, sino,max_iter=50,max_linesearch_steps=max_linesearch_steps,initial_guess_strategy="one")


"""

| Itérations |    LS |        Temps |   Loss finale |
| ---------: | ----: | -----------: | ------------: |
|         15 |     1 | **13.891 s** |     7.3243e-4 |
|         15 | **4** | **13.727 s** | **6.5477e-4** |
|         15 |     8 |     13.983 s |     6.5477e-4 |
|         50 |     1 | **22.792 s** |     7.6384e-5 |
|         50 |     4 |     24.155 s | **7.5413e-5** |
|         50 |     8 |     24.377 s | **7.5413e-5** |
Donc ça dépend de ton objectif :

rapidité → max_linesearch_steps=1
meilleure convergence → max_linesearch_steps=4

Le coût moyen par itération diminue énormément :
15 itérations :  ~0.915 s/it
50 itérations :  ~0.483 s/it
"""
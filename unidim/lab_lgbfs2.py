import time
from plots import plot_points
import jax
import jax.numpy as jnp
import optax

from geometry import CtGeometry
from sinogram import Sinogram
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


print("JAX version :", jax.__version__)
print("JAX x64     :", jax.config.read("jax_enable_x64"))
print()



NB_POINTS = 10_000
NB_ANGLES = 600
NB_BINS = 4096

BATCH_SIZE = 16
EXT_DTYPE = jnp.float64
USE_CHECKPOINT = True

MAX_ITER = 15
MAX_LINESEARCH_STEPS = 4
INITIAL_GUESS_STRATEGY = "one"


# ============================================================
# Construction du sinogramme cible
# ============================================================

sino = Sinogram(CtGeometry(nb_angles=NB_ANGLES,nb_bins=NB_BINS,extent=2.0))

sino.add_disk(center=[0, 0],radius=0.9,density=+1.0)

sino.add_disk(center=[0, 0],radius=0.7,density=-1.0)


# ============================================================
# Initialisation des points
# ============================================================

key = jax.random.PRNGKey(0)

points0 = jax.random.uniform(key,shape=(NB_POINTS, 2),
    minval=-1.0,maxval=1.0,dtype=jnp.float32)

print("Problem")
print("  points shape : ", points0.shape)
print("  points dtype : ", points0.dtype)
print("  sino shape   : ", sino.values.shape)

print("sino min :", float(sino.values.min()))
print("sino max :", float(sino.values.max()))

mass_sum = sino.values.sum(axis=1)

print("mass sum min/max :",float(mass_sum.min()),float(mass_sum.max()))

# ============================================================
# Optimisation L-BFGS
# ============================================================

def optimize(points,sino,max_iter=50, max_linesearch_steps=4,
    initial_guess_strategy="one",):
    g = sino.geometry

    normals = jnp.asarray(g.normals,dtype=EXT_DTYPE)
    bin_edges = jnp.asarray(g.bin_edges,dtype=EXT_DTYPE)
    bin_mass = jnp.asarray(sino.values,dtype=EXT_DTYPE)
    bin_mass = bin_mass / bin_mass.sum(axis=1,keepdims=True)

    # --------------------------------------------------------
    # Fonction objectif
    # --------------------------------------------------------

    def fun(p):
        return loss_lax_map(
            p,
            normals,
            bin_mass,
            bin_edges,
            BATCH_SIZE,
            EXT_DTYPE,
            USE_CHECKPOINT,
        )

    # --------------------------------------------------------
    # L-BFGS + zoom line search
    # --------------------------------------------------------

    linesearch = optax.scale_by_zoom_linesearch(
        max_linesearch_steps=max_linesearch_steps,
        initial_guess_strategy=initial_guess_strategy,
    )

    solver = optax.lbfgs(linesearch=linesearch)
    state = solver.init(points)

    # --------------------------------------------------------
    # Une étape L-BFGS
    # --------------------------------------------------------

    @jax.jit
    def step(p, state):
        value_and_grad = optax.value_and_grad_from_state(fun)
        value, grad = value_and_grad(p,state=state)
        updates, state = solver.update(grad,state, p,value=value, grad=grad, value_fn=fun)
        p = optax.apply_updates(p,updates)

        return p, state, value, grad

    # --------------------------------------------------------
    # Première étape : compilation + exécution
    # --------------------------------------------------------

    print("Starting L-BFGS")
    print(f"  max_iter             : {max_iter}")
    print(f"  max_linesearch_steps : {max_linesearch_steps}")
    print(f"  batch_size           : {BATCH_SIZE}")
    print(f"  ext_dtype            : {EXT_DTYPE}")
    print(f"  checkpoint            : {USE_CHECKPOINT}")
    print()

    t0 = time.perf_counter()

    points, state, value, grad = step(points,state)

    value.block_until_ready()
    grad.block_until_ready()
    compile_time = time.perf_counter() - t0
    grad_norm = jnp.linalg.norm(grad)

    print(f"iter {1:3d} | loss = {float(value):.12e} | |grad| = {float(grad_norm):.6e}")

    # --------------------------------------------------------
    # Itérations restantes
    # --------------------------------------------------------

    t1 = time.perf_counter()

    for i in range(1, max_iter):
        points, state, value, grad = step(points,state)

        value.block_until_ready()
        grad.block_until_ready()

        grad_norm = jnp.linalg.norm(grad)

        print(f"iter {i + 1:3d} | loss = {float(value):.12e} | |grad| = {float(grad_norm):.6e}")
        plot_points(points, step=i, exp_dir='visu')

    optimization_time = time.perf_counter() - t1
    total_time = compile_time + optimization_time


    print("-" * 60)
    print(f"Compilation + première itération : {compile_time:.3f} s")
    print(f"Optimisation après compilation  : {optimization_time:.3f} s")
    print(f"Temps total                     : {total_time:.3f} s")
    print(f"Temps moyen après compilation   :{optimization_time / max(1, max_iter - 1):.3f} s/it")
    print(f"Loss finale                     : {float(value):.12e}")
    print(f"Gradient final                  : {float(grad_norm):.6e}")
    print("-" * 60)

    return points, state


plot_points(points0, step='init', exp_dir='visu')

points, state = optimize(points0,sino,max_iter=MAX_ITER,
                         max_linesearch_steps=MAX_LINESEARCH_STEPS,
                        initial_guess_strategy=INITIAL_GUESS_STRATEGY,)
plot_points(points, step='optim', exp_dir='visu')


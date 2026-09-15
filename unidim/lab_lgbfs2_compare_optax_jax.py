import gc
import time

import jax
import jax.numpy as jnp
import optax

from geometry import CtGeometry
from sinogram import Sinogram
from lab_wasser import _w2_1d as wasser_dist


# ============================================================
# Configuration
# ============================================================

NB_POINTS = 10_000
NB_ANGLES = 600
NB_BINS = 4096

BATCH_SIZE = 16
EXT_DTYPE = jnp.float64
USE_CHECKPOINT = True

SEED = 0


# ============================================================
# Loss
# ============================================================

def loss_lax_map(
    points,
    normals,
    bin_mass,
    bin_edges,
    batch_size,
    ext_dtype,
    use_checkpoint=True,
):
    def angle_cost(normal_and_mass):
        normal, mass = normal_and_mass

        projections = points @ normal

        return wasser_dist(
            projections,
            mass,
            bin_edges,
            ext_dtype,
        )

    if use_checkpoint:
        angle_cost = jax.checkpoint(angle_cost)

    costs = jax.lax.map(
        angle_cost,
        (normals, bin_mass),
        batch_size=batch_size,
    )

    return jnp.sum(costs).astype(jnp.float32)


# ============================================================
# Affichage mémoire GPU
# ============================================================

def print_gpu_memory(label):
    device = jax.devices()[0]

    stats = device.memory_stats()

    if stats is None:
        print(f"{label}: memory_stats() indisponible")
        return

    bytes_in_use = stats.get("bytes_in_use", None)
    bytes_limit = stats.get("bytes_limit", None)

    print()
    print(f"[GPU memory] {label}")

    if bytes_in_use is not None:
        print(
            f"  bytes_in_use : "
            f"{bytes_in_use / 1024**2:.2f} MiB"
        )

    if bytes_limit is not None:
        print(
            f"  bytes_limit  : "
            f"{bytes_limit / 1024**3:.2f} GiB"
        )

    if bytes_in_use is not None and bytes_limit is not None:
        print(
            f"  utilization  : "
            f"{100 * bytes_in_use / bytes_limit:.2f} %"
        )


# ============================================================
# Synchronisation
# ============================================================

def block_tree(tree):
    leaves = jax.tree_util.tree_leaves(tree)

    for leaf in leaves:
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("COMPARAISON JAX / OPTAX")
    print("=" * 70)

    print()
    print("JAX version :", jax.__version__)
    print(
        "JAX x64     :",
        jax.config.read("jax_enable_x64"),
    )
    print(
        "Device      :",
        jax.devices()[0],
    )

    # --------------------------------------------------------
    # Problème
    # --------------------------------------------------------

    sino = Sinogram(
        CtGeometry(
            nb_angles=NB_ANGLES,
            nb_bins=NB_BINS,
            extent=2.0,
        )
    )

    sino.add_disk(
        center=[0, 0],
        radius=0.9,
        density=+1.0,
    )

    sino.add_disk(
        center=[0, 0],
        radius=0.7,
        density=-1.0,
    )

    # --------------------------------------------------------
    # Points initiaux
    # --------------------------------------------------------

    key = jax.random.PRNGKey(SEED)

    points0 = jax.random.uniform(
        key,
        shape=(NB_POINTS, 2),
        minval=-1.0,
        maxval=1.0,
        dtype=jnp.float32,
    )

    # --------------------------------------------------------
    # Données cible
    # --------------------------------------------------------

    g = sino.geometry

    normals = jnp.asarray(
        g.normals,
        dtype=EXT_DTYPE,
    )

    bin_edges = jnp.asarray(
        g.bin_edges,
        dtype=EXT_DTYPE,
    )

    bin_mass = jnp.asarray(
        sino.values,
        dtype=EXT_DTYPE,
    )

    bin_mass = bin_mass / bin_mass.sum(
        axis=1,
        keepdims=True,
    )

    # --------------------------------------------------------
    # Informations
    # --------------------------------------------------------

    print()
    print("Problem")
    print(
        f"  points    : {points0.shape} "
        f"{points0.dtype}"
    )
    print(
        f"  normals   : {normals.shape} "
        f"{normals.dtype}"
    )
    print(
        f"  bin_mass  : {bin_mass.shape} "
        f"{bin_mass.dtype}"
    )
    print(
        f"  bin_edges : {bin_edges.shape} "
        f"{bin_edges.dtype}"
    )

    print()
    print("Configuration")
    print(
        f"  batch_size : {BATCH_SIZE}"
    )
    print(
        f"  ext_dtype  : {EXT_DTYPE}"
    )
    print(
        f"  checkpoint : {USE_CHECKPOINT}"
    )

    # --------------------------------------------------------
    # Fonction objectif
    # --------------------------------------------------------

    def fun(points):
        return loss_lax_map(
            points,
            normals,
            bin_mass,
            bin_edges,
            BATCH_SIZE,
            EXT_DTYPE,
            USE_CHECKPOINT,
        )

    # ========================================================
    # 1. JAX value_and_grad
    # ========================================================

    print()
    print("=" * 70)
    print("1. JAX value_and_grad")
    print("=" * 70)

    jax_value_and_grad = jax.jit(
        jax.value_and_grad(fun)
    )

    print_gpu_memory("avant compilation JAX")

    # Compilation + première exécution
    start = time.perf_counter()

    value_jax, grad_jax = jax_value_and_grad(
        points0
    )

    value_jax.block_until_ready()
    grad_jax.block_until_ready()

    time_jax_compile_run = (
        time.perf_counter() - start
    )

    print(
        f"Compile + run : "
        f"{time_jax_compile_run:.3f} s"
    )

    print_gpu_memory(
        "après compilation/exécution JAX"
    )

    # --------------------------------------------------------
    # Exécution compilée
    # --------------------------------------------------------

    start = time.perf_counter()

    value_jax, grad_jax = jax_value_and_grad(
        points0
    )

    value_jax.block_until_ready()
    grad_jax.block_until_ready()

    time_jax_run = (
        time.perf_counter() - start
    )

    print(
        f"Run compilé   : "
        f"{time_jax_run:.3f} s"
    )

    print(
        f"Loss          : "
        f"{float(value_jax):.12e}"
    )

    print(
        f"Grad norm     : "
        f"{float(jnp.linalg.norm(grad_jax)):.12e}"
    )

    # --------------------------------------------------------
    # Compilation explicite pour memory_analysis
    # --------------------------------------------------------

    jax_compiled = jax_value_and_grad.lower(
        points0
    ).compile()

    jax_memory = jax_compiled.memory_analysis()

    print()
    print("Mémoire XLA JAX")
    print(jax_memory)

    # ========================================================
    # 2. Optax value_and_grad_from_state
    # ========================================================

    print()
    print("=" * 70)
    print("2. Optax value_and_grad_from_state")
    print("=" * 70)

    # --------------------------------------------------------
    # Solver L-BFGS
    # --------------------------------------------------------

    solver = optax.lbfgs()

    state = solver.init(
        points0
    )

    optax_value_and_grad = (
        optax.value_and_grad_from_state(fun)
    )

    @jax.jit
    def optax_grad_step(
        points,
        state,
    ):
        value, grad = optax_value_and_grad(
            points,
            state=state,
        )

        return value, grad, state

    print_gpu_memory(
        "avant compilation Optax"
    )

    # --------------------------------------------------------
    # Compilation + première exécution
    # --------------------------------------------------------

    start = time.perf_counter()

    value_optax, grad_optax, state_out = (
        optax_grad_step(
            points0,
            state,
        )
    )

    value_optax.block_until_ready()
    grad_optax.block_until_ready()

    time_optax_compile_run = (
        time.perf_counter() - start
    )

    print(
        f"Compile + run : "
        f"{time_optax_compile_run:.3f} s"
    )

    print_gpu_memory(
        "après compilation/exécution Optax"
    )

    # --------------------------------------------------------
    # Exécution compilée
    # --------------------------------------------------------

    start = time.perf_counter()

    value_optax, grad_optax, state_out = (
        optax_grad_step(
            points0,
            state,
        )
    )

    value_optax.block_until_ready()
    grad_optax.block_until_ready()

    time_optax_run = (
        time.perf_counter() - start
    )

    print(
        f"Run compilé   : "
        f"{time_optax_run:.3f} s"
    )

    print(
        f"Loss          : "
        f"{float(value_optax):.12e}"
    )

    print(
        f"Grad norm     : "
        f"{float(jnp.linalg.norm(grad_optax)):.12e}"
    )

    # --------------------------------------------------------
    # Compilation explicite pour memory_analysis
    # --------------------------------------------------------

    optax_compiled = optax_grad_step.lower(
        points0,
        state,
    ).compile()

    optax_memory = (
        optax_compiled.memory_analysis()
    )

    print()
    print("Mémoire XLA Optax")
    print(optax_memory)

    # ========================================================
    # 3. Comparaison numérique
    # ========================================================

    print()
    print("=" * 70)
    print("3. COMPARAISON NUMÉRIQUE")
    print("=" * 70)

    loss_diff = jnp.abs(
        value_jax - value_optax
    )

    grad_diff = jnp.linalg.norm(
        grad_jax - grad_optax
    )

    grad_jax_norm = jnp.linalg.norm(
        grad_jax
    )

    grad_optax_norm = jnp.linalg.norm(
        grad_optax
    )

    relative_grad_diff = (
        grad_diff
        / jnp.maximum(
            grad_jax_norm,
            1e-30,
        )
    )

    print()
    print(
        f"JAX   loss : "
        f"{float(value_jax):.12e}"
    )

    print(
        f"Optax loss : "
        f"{float(value_optax):.12e}"
    )

    print(
        f"Diff loss  : "
        f"{float(loss_diff):.12e}"
    )

    print()
    print(
        f"JAX   |grad| : "
        f"{float(grad_jax_norm):.12e}"
    )

    print(
        f"Optax |grad| : "
        f"{float(grad_optax_norm):.12e}"
    )

    print(
        f"Diff |grad|  : "
        f"{float(grad_diff):.12e}"
    )

    print(
        f"Relative diff : "
        f"{float(relative_grad_diff):.12e}"
    )

    # --------------------------------------------------------
    # Vérification élément par élément
    # --------------------------------------------------------

    max_grad_diff = jnp.max(
        jnp.abs(
            grad_jax - grad_optax
        )
    )

    print(
        f"Max abs grad diff : "
        f"{float(max_grad_diff):.12e}"
    )

    # ========================================================
    # 4. Comparaison des temps
    # ========================================================

    print()
    print("=" * 70)
    print("4. COMPARAISON TEMPS")
    print("=" * 70)

    print()
    print(
        f"{'Méthode':35s}"
        f"{'Compile + run':>18s}"
        f"{'Run compilé':>18s}"
    )

    print("-" * 70)

    print(
        f"{'jax.value_and_grad':35s}"
        f"{time_jax_compile_run:>18.3f} s"
        f"{time_jax_run:>18.3f} s"
    )

    print(
        f"{'optax.value_and_grad_from_state':35s}"
        f"{time_optax_compile_run:>18.3f} s"
        f"{time_optax_run:>18.3f} s"
    )

    print()
    print(
        f"Ratio run Optax / JAX : "
        f"{time_optax_run / time_jax_run:.3f}"
    )

    # ========================================================
    # 5. Mémoire
    # ========================================================

    print()
    print("=" * 70)
    print("5. MÉMOIRE")
    print("=" * 70)

    print()
    print("JAX")
    print(jax_memory)

    print()
    print("Optax")
    print(optax_memory)

    print()
    print("=" * 70)
    print("FIN")
    print("=" * 70)
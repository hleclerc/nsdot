# OPTI A: vectorized projection layout [C, N] with configurable angle batching.
# Keeps the Optax L-BFGS/zoom pipeline; checkpoint remains optional.

import time
import os
from pprint import pprint
import pandas as pd
import tqdm

XLA_PYTHON_CLIENT_PREALLOCATE = True
XLA_PYTHON_CLIENT_MEM_FRACTION = 0.75
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = str(XLA_PYTHON_CLIENT_PREALLOCATE).lower()
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(XLA_PYTHON_CLIENT_MEM_FRACTION)

import jax
import jax.numpy as jnp
import optax
import nvidia_smi
from geometry import CtGeometry
from sinogram import Sinogram
from gpu_mem import jax_mem_budget_bytes
from peak_memory_monitor import measure_gpu_peak

import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("wasserstein-lbfgs")
nvidia_smi.nvmlInit()
handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
nvidia_mem = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)

# Needed for the float64 promotion in `_w2_1d` below -- disabled by default,
# JAX otherwise SILENTLY truncates any float64 array back to float32.
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")

# See `loss`'s docstring for how this was measured.
_BYTES_PER_CHUNK_ELEMENT = 256

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


def _choose_chunk_size(n, A, mem_budget_bytes, batch_size="auto"):
    if batch_size == "auto":
        if mem_budget_bytes is None:
            return 1
        return max(
            1,
            min(
                A,
                mem_budget_bytes
                // (_BYTES_PER_CHUNK_ELEMENT * max(n, 1)),
            ),
        )

    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError('batch_size must be a positive integer or "auto"')

    return min(A, batch_size)


def _loss_chunk(
    points,
    normals,
    bin_edges,
    bin_mass,
    use_checkpoint=True,
):
    """Vectorized projection and Wasserstein evaluation for one angle chunk."""

    def chunk_cost(p):
        proj = normals @ p.T
        costs = jax.vmap(
            lambda pr, mass: _w2_1d(pr, mass, bin_edges)
        )(
            proj,
            bin_mass,
        )
        return costs.sum().astype(jnp.float32)

    if use_checkpoint:
        chunk_cost = jax.checkpoint(chunk_cost)

    return chunk_cost(points)


def loss(
    points,
    normals,
    bin_edges,
    bin_mass,
    mem_budget_bytes=-1,
    batch_size="auto",
    use_checkpoint=True,
):
    """Sum the Wasserstein loss over angle chunks.

    `batch_size` is either a fixed number of angles per chunk or "auto" to
    choose the chunk size from the available accelerator memory.
    """
    if batch_size == "auto" and mem_budget_bytes == -1:
        mem_budget_bytes = jax_mem_budget_bytes()

    n = points.shape[0]
    A = normals.shape[0]
    C = _choose_chunk_size(n, A, mem_budget_bytes, batch_size)

    n_full = (A // C) * C

    normals_chunks = normals[:n_full].reshape(
        n_full // C, C, normals.shape[1]
    )
    mass_chunks = bin_mass[:n_full].reshape(
        n_full // C, C, bin_mass.shape[1]
    )

    def scan_body(value, xs):
        normals_chunk, mass_chunk = xs
        value = value + _loss_chunk(
            points,
            normals_chunk,
            bin_edges,
            mass_chunk,
            use_checkpoint=use_checkpoint,
        )
        return value, None

    value = jnp.array(0.0, dtype=jnp.float32)

    value, _ = jax.lax.scan(
        scan_body,
        value,
        (normals_chunks, mass_chunks),
    )

    if n_full < A:
        value = value + _loss_chunk(
            points,
            normals[n_full:],
            bin_edges,
            bin_mass[n_full:],
            use_checkpoint=use_checkpoint,
        )

    return value


def optimize(points, sino,
             max_iter=50,
             max_linesearch_steps=4,
             initial_guess_strategy="one",
             ext_dtype=jnp.float64,
             batch_size="auto",
             use_checkpoint=True,
             target_loss=1e-3,
             avg_last_n=10):
    normals, bin_edges, bin_mass = _sino_arrays(sino)
    linesearch = optax.scale_by_zoom_linesearch(
        max_linesearch_steps=max_linesearch_steps,
        initial_guess_strategy=initial_guess_strategy,
    )
    solver = optax.lbfgs(linesearch=linesearch)
    state = solver.init(points)

    def make_step(mem_budget_bytes):
        chunk_size = _choose_chunk_size(
            points.shape[0], normals.shape[0], mem_budget_bytes, batch_size
        )

        @jax.jit
        def step(p, state, normals, bin_edges, bin_mass):
            fun = lambda pp: loss(
                pp,
                normals,
                bin_edges,
                bin_mass,
                mem_budget_bytes=mem_budget_bytes,
                batch_size=batch_size,
                use_checkpoint=use_checkpoint,
            )
            value_and_grad = optax.value_and_grad_from_state(fun)
            value, grad = value_and_grad(p, state=state)
            updates, state = solver.update(grad, state, p, value=value, grad=grad, value_fn=fun)
            p = optax.apply_updates(p, updates)
            return p, state, value, grad

        return step, chunk_size

    mem_budget_bytes = jax_mem_budget_bytes() if batch_size == "auto" else None
    step, chunk_size = make_step(mem_budget_bytes)

    print(f"  [warmup] compiling/stabilizing JIT (n={points.shape[0]})...", end="", flush=True)
    t_warmup = time.perf_counter()
    while True:
        try:
            wp, ws = points, state
            for _ in range(4):
                wp, ws, wv, wg = step(wp, ws, normals, bin_edges, bin_mass)
            wg.block_until_ready()
            break
        except jax.errors.JaxRuntimeError as e:
            if (
                batch_size != "auto"
                or mem_budget_bytes is None
                or "RESOURCE_EXHAUSTED" not in str(e)
            ):
                raise
            mem_budget_bytes = mem_budget_bytes // 2 if mem_budget_bytes >= 2 else None
            print(" OOM, shrinking angle-chunk budget...", end="", flush=True)
            step, chunk_size = make_step(mem_budget_bytes)
    warmup_time = time.perf_counter() - t_warmup
    print(f" done ({warmup_time:.2f}s, C={chunk_size})")

    history = []
    start_optimization = time.perf_counter()
    time_to_loss = -1.0
    peak = {"jax_peak_used": 0, "jax_peak_pool": 0, "nvidia_peak": 0}

    for idx in tqdm.tqdm(range(max_iter)):
        iteration_start = time.perf_counter()
        if idx == max_iter - 1:
            (points, state, value, grad), peak = measure_gpu_peak(
                step, points, state, normals, bin_edges, bin_mass, interval=0.02,
            )
        else:
            points, state, value, grad = step(points, state, normals, bin_edges, bin_mass)

        value.block_until_ready()
        grad.block_until_ready()
        grad_norm = float(jnp.linalg.norm(grad))
        loss_value = float(value)
        elapsed_time = round(time.perf_counter() - start_optimization, 2)

        if time_to_loss < 0 and loss_value <= target_loss:
            time_to_loss = elapsed_time

        raw_metrics = {
            "iteration": idx,
            "loss": loss_value,
            "grad_norm": grad_norm,
            "elapsed_time": elapsed_time,
            "iteration_time": round(time.perf_counter() - iteration_start, 4),
            "num_linesearch_steps": int(state[2].info.num_linesearch_steps),
        }
        history.append(raw_metrics)
        mlflow.log_metrics(raw_metrics, step=idx)

    end_optimization = time.perf_counter()
    optimization_time = end_optimization - start_optimization
    total_time = warmup_time + optimization_time
    df_history = pd.DataFrame(history)
    last_5 = df_history.tail(min(5, len(df_history)))
    last_n = df_history.tail(min(avg_last_n, len(df_history)))

    results = {
        "time_1st_run_compile_": round(warmup_time, 3),
        "time_2_to_end_run": round(optimization_time, 3),
        "total_time": round(total_time, 3),
        "avg_iteration_time_last_5": round(float(last_5["iteration_time"].mean()), 3),
        "avg_iteration_time_last_n": round(float(last_n["iteration_time"].mean()), 3),
        "final_loss": float(value),
        "final_grad_norm": float(jnp.linalg.norm(grad)),
        "mean_linesearch_steps": round(float(df_history["num_linesearch_steps"].mean()), 2),
        "max_linesearch_steps": int(df_history["num_linesearch_steps"].max()),
        "time_to_loss": float(time_to_loss),
        "jax_peak_used_last_iter_MB": round(peak["jax_peak_used"] / 1024**2, 2),
        "jax_peak_pool_last_iter_MB": round(peak["jax_peak_pool"] / 1024**2, 2),
        "nvidia_peak_last_iter_MB": round(peak["nvidia_peak"] / 1024**2, 2),
        "chunk_size": int(chunk_size),
    }
    mlflow.log_metrics(results)
    mlflow.log_text(df_history.to_csv(index=False), "history.csv")
    return points, state, results, df_history

def run_experiments(params):
    pprint(params)

    with mlflow.start_run():
        mlflow.log_params(params)

        sino = Sinogram(CtGeometry(nb_angles=params['nb_angles'], nb_bins=params['nb_bins'], extent=2.0))
        sino.add_disk(center=[0, 0], radius=0.9, density=+1.0)
        sino.add_disk(center=[0.5, 0], radius=0.3, density=-1.0)
        sino.add_disk(center=[-0.5, 0], radius=0.3, density=-1.0)

        key = jax.random.PRNGKey(params['seed'])
        points0 = jax.random.uniform(key, shape=(params['nb_points'], 2), minval=-1.0, maxval=1.0,
                                     dtype=jnp.float32)

        points, state, results, df_history = optimize(
            points0,
            sino,
            max_iter=params['max_iter'],
            max_linesearch_steps=params['max_linesearch_steps'],
            initial_guess_strategy=params['initial_guess_strategy'],
            ext_dtype=params["ext_dtype"],
            batch_size=params['batch_size'],
            use_checkpoint=params['use_checkpoint'],
            target_loss=params['target_loss'],
            avg_last_n=10,
        )

        print(df_history.to_markdown(index=False))
        pprint(results)


if __name__ == '__main__':
    print("JAX version :", jax.__version__)
    print("JAX x64 :", jax.config.read("jax_enable_x64"))
    print("Device :", jax.devices()[0])

    base_params = dict(XLA_PYTHON_CLIENT_PREALLOCATE=XLA_PYTHON_CLIENT_PREALLOCATE,
                       XLA_PYTHON_CLIENT_MEM_FRACTION=XLA_PYTHON_CLIENT_MEM_FRACTION,
                       nb_points=100_000,
                       nb_angles=600,
                       nb_bins=4096,
                       batch_size="auto",  # use 2 to match Hermann's fixed batching
                       ext_dtype=jnp.float64,
                       use_checkpoint=True,
                       max_iter=15,
                       max_linesearch_steps=4,
                       initial_guess_strategy="one",
                       seed=27,
                       target_loss=1e-3)

    base_params['backend'] = "jax_A"
    base_params['exp'] = "big_nb_pts"
    base_params['variant'] = "A"

    run_experiments(base_params)

# OPTI B: split regular numerical work into small JIT kernels.
# Python orchestrates the L-BFGS/Wolfe line-search; angle batching is configurable.

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

# Numerical agreement with the CUDA reference requires x64 for the
# target-side Wasserstein computations and "highest" matmul precision.
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")

# Provisional conservative value for the new direct/vectorized kernel.
#
# Measurements on H100:
#   live buffers       ~= 40-41 B / (N*C)
#   actual HBM usage   ~= 73-80 B / (N*C) + fixed overhead
#
# 128 keeps substantial safety margin. It is intentionally NOT yet
# presented as the final calibrated value.
_BYTES_PER_CHUNK_ELEMENT = 128

def _w2_1d(proj, bin_mass, bin_edges):
    """Squared 1D Wasserstein distance."""
    n = proj.shape[0]
    w = 1.0 / n
    dw = bin_edges[1] - bin_edges[0]
    bin_center = bin_edges[:-1] + dw / 2

    cum = jnp.cumsum(bin_mass)
    cum_start = cum - bin_mass
    prefix_M = (
        jnp.cumsum(bin_mass * bin_center)
        - bin_mass * bin_center
    )

    def M(q):
        j = jnp.clip(
            jnp.searchsorted(cum, q, side="right"),
            0,
            bin_mass.shape[0] - 1,
        )
        f = jnp.where(
            bin_mass[j] > 0,
            (q - cum_start[j]) / bin_mass[j],
            0.0,
        )
        return (
            prefix_M[j]
            + bin_mass[j]
            * (
                bin_edges[j] * f
                + dw * f * f / 2
            )
        )

    s = jnp.sort(proj).astype(jnp.float64)
    q = jnp.arange(n, dtype=jnp.float64) * w
    bary = (M(q + w) - M(q)) / w

    target_second_moment = (
        jnp.sum(bin_mass * bin_center ** 2)
        + dw * dw / 12
    )

    return (
        w * jnp.sum(s ** 2)
        - 2 * w * jnp.sum(s * bary)
        + target_second_moment
    )


def _sino_arrays(sino):
    g = sino.geometry

    normals = jnp.asarray(
        g.normals,
        dtype=jnp.float32,
    )

    bin_edges = jnp.asarray(
        g.bin_edges,
        dtype=jnp.float64,
    )

    bin_mass = jnp.asarray(
        sino.values,
        dtype=jnp.float64,
    )

    bin_mass = (
        bin_mass
        / bin_mass.sum(axis=1, keepdims=True)
    )

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
                // (
                    _BYTES_PER_CHUNK_ELEMENT
                    * max(n, 1)
                ),
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
):
    """Direct vectorized kernel for one angle chunk."""
    proj = normals @ points.T

    costs = jax.vmap(
        lambda pr, mass:
            _w2_1d(
                pr,
                mass,
                bin_edges,
            )
    )(
        proj,
        bin_mass,
    )

    return costs.sum().astype(jnp.float32)


def loss(
    points,
    normals,
    bin_edges,
    bin_mass,
    mem_budget_bytes=-1,
    batch_size="auto",
):
    """Same mathematical loss, chunked over angles."""
    if batch_size == "auto" and mem_budget_bytes == -1:
        mem_budget_bytes = jax_mem_budget_bytes()

    n = points.shape[0]
    A = normals.shape[0]

    C = _choose_chunk_size(
        n,
        A,
        mem_budget_bytes,
        batch_size,
    )

    value = jnp.array(
        0.0,
        dtype=jnp.float32,
    )

    for start in range(0, A, C):
        stop = min(
            start + C,
            A,
        )

        value = value + _loss_chunk(
            points,
            normals[start:stop],
            bin_edges,
            bin_mass[start:stop],
        )

    return value



def optimize(points, sino,
             max_iter=50,
             max_linesearch_steps=4,
             initial_guess_strategy="one",
             ext_dtype=jnp.float64,
             batch_size="auto",
             target_loss=1e-3,
             avg_last_n=10):
    """L-BFGS with JIT numerical kernels and Python-controlled Wolfe search."""

    normals, bin_edges, bin_mass = _sino_arrays(sino)

    n = points.shape[0]
    A = normals.shape[0]

    direction_tx = optax.chain(
        optax.scale_by_lbfgs(memory_size=10),
        optax.scale(-1.0),
    )
    state = direction_tx.init(points)

    @jax.jit
    def make_direction(grad, state, p):
        direction, new_state = direction_tx.update(grad, state, p)
        slope0 = jnp.vdot(grad, direction)
        return direction, new_state, slope0

    @jax.jit
    def make_trial(base, direction, alpha):
        return base + alpha * direction

    @jax.jit
    def directional_slope(grad, direction):
        return jnp.vdot(grad, direction)

    vg_chunk = jax.jit(jax.value_and_grad(_loss_chunk))

    def make_value_grad(mem_budget_bytes):
        C = _choose_chunk_size(n, A, mem_budget_bytes, batch_size)

        def value_grad(p):
            value = jnp.array(0.0, dtype=jnp.float32)
            grad = jnp.zeros_like(p)

            for start in range(0, A, C):
                stop = min(start + C, A)
                v, g = vg_chunk(
                    p,
                    normals[start:stop],
                    bin_edges,
                    bin_mass[start:stop],
                )
                value = value + v
                grad = grad + g

            grad.block_until_ready()
            return value, grad

        return C, value_grad

    def python_wolfe(base, f0, g0, direction, dphi0, value_grad):
        c1 = 1e-4
        c2 = 0.9

        phi0 = float(f0)
        dphi0 = float(dphi0)

        if dphi0 >= 0.0:
            direction = -g0
            dphi0_arr = directional_slope(g0, direction)
            dphi0_arr.block_until_ready()
            dphi0 = float(dphi0_arr)

        evals = 0

        def evaluate(alpha):
            nonlocal evals

            trial = make_trial(base, direction, alpha)
            f, g = value_grad(trial)
            slope = directional_slope(g, direction)
            slope.block_until_ready()
            evals += 1

            return trial, f, g, float(f), float(slope)

        def zoom(a_lo, a_hi, phi_lo):
            best = None

            while evals < max_linesearch_steps:
                a = 0.5 * (a_lo + a_hi)
                trial, f, g, phi, dphi = evaluate(a)

                armijo = phi <= phi0 + c1 * a * dphi0
                curvature = abs(dphi) <= c2 * abs(dphi0)
                best = trial, f, g, a

                if (not armijo) or phi >= phi_lo:
                    a_hi = a
                else:
                    if curvature:
                        return best
                    if dphi * (a_hi - a_lo) >= 0:
                        a_hi = a_lo
                    a_lo = a
                    phi_lo = phi

            return best

        a_prev = 0.0
        phi_prev = phi0
        a = 1.0
        best = None

        while evals < max_linesearch_steps:
            trial, f, g, phi, dphi = evaluate(a)

            armijo = phi <= phi0 + c1 * a * dphi0
            curvature = abs(dphi) <= c2 * abs(dphi0)
            best = trial, f, g, a

            if (not armijo) or (evals > 1 and phi >= phi_prev):
                result = zoom(a_prev, a, phi_prev)
                return *result, evals

            if curvature:
                return trial, f, g, a, evals

            if dphi >= 0:
                result = zoom(a, a_prev, phi)
                return *result, evals

            a_prev = a
            phi_prev = phi
            a *= 2.0

        return *best, evals

    mem_budget_bytes = jax_mem_budget_bytes() if batch_size == "auto" else None

    print(
        f"  [warmup] compiling/stabilizing JIT (n={n})...",
        end="",
        flush=True,
    )
    t_warmup = time.perf_counter()

    while True:
        C, value_grad = make_value_grad(mem_budget_bytes)

        try:
            wv, wg = value_grad(points)
            wp = points
            ws = state

            for _ in range(4):
                wd, ws, wslope = make_direction(wg, ws, wp)
                wslope.block_until_ready()
                wp, wv, wg, _, _ = python_wolfe(
                    wp, wv, wg, wd, wslope, value_grad,
                )

            wg.block_until_ready()
            break

        except jax.errors.JaxRuntimeError as e:
            if (
                batch_size != "auto"
                or mem_budget_bytes is None
                or "RESOURCE_EXHAUSTED" not in str(e)
            ):
                raise

            mem_budget_bytes = (
                mem_budget_bytes // 2
                if mem_budget_bytes >= 2
                else None
            )
            print(" OOM, shrinking angle-chunk budget...", end="", flush=True)

    warmup_time = time.perf_counter() - t_warmup
    print(f" done ({warmup_time:.2f}s, C={C})")

    initial_eval_start = time.perf_counter()
    value, grad = value_grad(points)
    initial_eval_ms = (time.perf_counter() - initial_eval_start) * 1000

    def one_iteration(points, state, value, grad):
        direction, state, slope0 = make_direction(grad, state, points)
        slope0.block_until_ready()

        points, value, grad, alpha, nb_evals = python_wolfe(
            points,
            value,
            grad,
            direction,
            slope0,
            value_grad,
        )
        return points, state, value, grad, alpha, nb_evals

    history = []
    start_optimization = time.perf_counter()
    time_to_loss = -1.0
    peak = {"jax_peak_used": 0, "jax_peak_pool": 0, "nvidia_peak": 0}

    for idx in tqdm.tqdm(range(max_iter)):
        iteration_start = time.perf_counter()

        if idx == max_iter - 1:
            (points, state, value, grad, alpha, nb_evals), peak = measure_gpu_peak(
                one_iteration,
                points,
                state,
                value,
                grad,
                interval=0.02,
            )
        else:
            points, state, value, grad, alpha, nb_evals = one_iteration(
                points, state, value, grad,
            )

        elapsed_iteration = time.perf_counter() - iteration_start
        if idx == 0:
            elapsed_iteration += initial_eval_ms / 1000.0

        value.block_until_ready()
        grad.block_until_ready()
        grad_norm = float(jnp.linalg.norm(grad))
        loss_value = float(value)
        elapsed_time = round(time.perf_counter() - start_optimization + initial_eval_ms / 1000.0, 2)

        if time_to_loss < 0 and loss_value <= target_loss:
            time_to_loss = elapsed_time

        raw_metrics = {
            "iteration": idx,
            "loss": loss_value,
            "grad_norm": grad_norm,
            "elapsed_time": elapsed_time,
            "iteration_time": round(elapsed_iteration, 4),
            "num_linesearch_steps": int(nb_evals),
            "line_search_alpha": float(alpha),
        }
        history.append(raw_metrics)
        mlflow.log_metrics(raw_metrics, step=idx)

    end_optimization = time.perf_counter()
    optimization_time = end_optimization - start_optimization + initial_eval_ms / 1000.0
    total_time = warmup_time + optimization_time
    df_history = pd.DataFrame(history)
    last_5 = df_history.tail(min(5, len(df_history)))
    last_n = df_history.tail(min(avg_last_n, len(df_history)))

    stats = jax.devices()[0].memory_stats() or {}
    budget_gib = None if mem_budget_bytes is None else mem_budget_bytes / 2**30
    peak_live_gib = stats.get("peak_bytes_in_use", 0) / 2**30
    peak_pool_gib = stats.get("peak_pool_bytes", 0) / 2**30

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
        "chunk_size": int(C),
        "budget_gib": -1.0 if budget_gib is None else float(budget_gib),
        "process_peak_live_gib": float(peak_live_gib),
        "process_peak_pool_gib": float(peak_pool_gib),
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
                       max_iter=15,
                       max_linesearch_steps=4,
                       initial_guess_strategy="one",
                       seed=27,
                       target_loss=1e-3)

    base_params['backend'] = "jax_B"
    base_params['exp'] = "big_nb_pts"
    base_params['variant'] = "B"

    run_experiments(base_params)

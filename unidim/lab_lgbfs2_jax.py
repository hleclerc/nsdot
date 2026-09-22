import time
import os
from pprint import pprint
import pandas as pd
import tqdm

XLA_PYTHON_CLIENT_PREALLOCATE = True
XLA_PYTHON_CLIENT_MEM_FRACTION = 0.75
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = str(XLA_PYTHON_CLIENT_PREALLOCATE).lower()
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]= str(XLA_PYTHON_CLIENT_MEM_FRACTION)

import jax
import jax.numpy as jnp
import optax
import nvidia_smi
from geometry import CtGeometry
from sinogram import Sinogram
from lab_wasser import _w2_1d as wasser_dist
from peak_memory_monitor import measure_gpu_peak
# from plots import plot_points

import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("wasserstein-lbfgs")
nvidia_smi.nvmlInit()
handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
nvidia_mem = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)

def loss_lax_map(points, normals, bin_mass, bin_edges, batch_size, ext_dtype, use_checkpoint=True):
    def angle_cost(normal_and_mass):
        normal, mass = normal_and_mass
        mass = jax.lax.stop_gradient(mass)
        normal = jax.lax.stop_gradient(normal)
        projections = points @ normal
        w = wasser_dist(projections, mass, bin_edges, ext_dtype)
        return w

    if use_checkpoint:
        angle_cost = jax.checkpoint(angle_cost)

    if batch_size is None:
        costs = jax.vmap(angle_cost,in_axes=0,)((normals, bin_mass))
    else:
        costs = jax.lax.map(angle_cost,(normals,bin_mass),batch_size=batch_size,)

    return jnp.sum(costs).astype(jnp.float32)

def optimize(points,sino,
             max_iter=50,
             max_linesearch_steps=4,
             initial_guess_strategy="one",
             ext_dtype=jnp.float64,
             batch_size=16,
             use_checkpoint=True,
             target_loss = 1e-3,
             avg_last_n=10):
    g = sino.geometry

    normals = jnp.asarray(g.normals,dtype=ext_dtype)
    bin_edges = jnp.asarray(g.bin_edges,dtype=ext_dtype)
    bin_mass = jnp.asarray(sino.values,dtype=ext_dtype)
    bin_mass = bin_mass / bin_mass.sum(axis=1,keepdims=True)
    history = []
    start_total = time.perf_counter()
    time_to_loss = -1

    def fun(p):
        # jax.debug.callback(count_fun_call,p,ordered=True,)
        return loss_lax_map(p, normals, bin_mass,bin_edges,batch_size, ext_dtype,use_checkpoint)
    # --------------------------------------------------------
    # L-BFGS + zoom line search
    # --------------------------------------------------------
    linesearch = optax.scale_by_zoom_linesearch(
        max_linesearch_steps=max_linesearch_steps,
        initial_guess_strategy=initial_guess_strategy,
    )

    solver = optax.lbfgs(linesearch=linesearch)
    state = solver.init(points)

    value_and_grad = optax.value_and_grad_from_state(fun)

    @jax.jit
    def step(p, state):
        value, grad = value_and_grad(p,state=state)
        updates, state = solver.update(grad,state, p,value=value, grad=grad, value_fn=fun)
        p = optax.apply_updates(p,updates)

        return p, state, value, grad


    points, state, value, grad = step( points, state)

    value.block_until_ready()
    grad.block_until_ready()
    end_compile = time.perf_counter()
    history.append({"iteration": 1, "loss": float(value), "grad_norm": float(jnp.linalg.norm(grad)),
                    "elapsed_time": round(time.perf_counter() - start_total,2),
                    'iteration_time': round(end_compile - start_total,4),
                    "num_linesearch_steps": int(state[2].info.num_linesearch_steps)})


    start_optimization = time.perf_counter()
    for idx in tqdm.tqdm(range(1, max_iter)):
        iteration_start = time.perf_counter()
        if idx == max_iter - 1:
            (points, state, value, grad), peak = measure_gpu_peak(
                step, points, state, interval=0.02,)
        else:
            points, state, value, grad = step(points, state)

        value.block_until_ready()
        grad.block_until_ready()
        grad_norm = float(jnp.linalg.norm(grad))
        loss_value = float(value)
        elapsed_time = round(time.perf_counter() - start_total,3)

        if time_to_loss < 0 and loss_value <= target_loss:
            time_to_loss = elapsed_time

        raw_metrics = {"iteration": idx, "loss": loss_value,
                        "grad_norm": grad_norm,
                        "elapsed_time" : elapsed_time,
                        "iteration_time": round(time.perf_counter() - iteration_start,4),
                        "num_linesearch_steps": int(state[2].info.num_linesearch_steps),
                       }
        history.append(raw_metrics)
        mlflow.log_metrics(raw_metrics, step=idx)

        # print(f"{iteration:5d} " f"{float(value):18.12e} " f"{float(grad_norm):16.6e}")
        # plot_points(points, step=iteration, exp_dir='visu')

    end_optimization = time.perf_counter()
    optimization_time = end_optimization - start_optimization
    total_time = end_optimization - start_total
    # average_time = optimization_time / (max_iter - 2)
    avg_iteration_time_last_5 = sum(h["iteration_time"] for h in history[-5:]) / 5
    df_history = pd.DataFrame(history)
    results = {"time_1st_run_compile_": round(end_compile - start_total,3),
                "time_2_to_end_run": round(optimization_time,3),
                "total_time": round(total_time,3),
                "avg_iteration_time_last_5": round(avg_iteration_time_last_5,3),
                "final_loss": float(value), "final_grad_norm": float(jnp.linalg.norm(grad)),
                "mean_linesearch_steps": round(float(df_history["num_linesearch_steps"].mean()),2),
                "max_linesearch_steps" : int(df_history["num_linesearch_steps"].max()),
                "time_to_loss" : float(time_to_loss),
                "jax_peak_used_last_iter_MB":  round(peak["jax_peak_used"] / 1024**2, 2),
                "jax_peak_pool_last_iter_MB":  round(peak["jax_peak_pool"] / 1024**2, 2),
                "nvidia_peak_last_iter_MB":  round(peak["nvidia_peak"] / 1024**2,2),
              }
    mlflow.log_metrics(results)
    mlflow.log_text(df_history.to_csv(index=False),"history.csv",)
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


        # plot_points(points0, step=0, exp_dir='visu')
        points, state, results, df_history = optimize(points0,
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
        # plot_points(points, step=MAX_ITER, exp_dir='visu')

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
                       batch_size=1,
                       ext_dtype=jnp.float64,
                       use_checkpoint=False,
                       max_iter=15,
                       max_linesearch_steps=4,
                       initial_guess_strategy="one",
                       seed=27,
                       target_loss=1e-3)

    base_params['backend'] = "jax_multi"
    base_params['exp_type'] = "big_nb_pts"
    run_experiments(base_params)
    # nb_angles_exp = [100, 200, 400, 600, 1000, 2000]
    # nb_bins_exp = [512, 1024, 2048, 4096, 8192]
    # nb_points_exp = [1_000, 5_000, 10_000, 20_000, 50_000]
    # batch_sizes = [1, 4, 8, 16, 32, 64, 128, 256, 600]
    # run_experiments(params)


    # from itertools import product
    #
    # batch_sizes = [1, 16, 64, 600]
    # checkpoints = [True, False]
    # line_searches = [1, 4, 8]
    # experiments = []
    # for nb_bins in nb_bins_exp : #product( batch_sizes, checkpoints, line_searches, ):
    #     params = base_params.copy()
    #     params["nb_bins"] = nb_bins
    #     experiments.append(params)
    #
    # for i, params in enumerate(experiments):
    #     print( f"\n\n########## " f"EXPERIMENT {i + 1}/{len(experiments)} " f"##########" )
    #     run_experiments(params)

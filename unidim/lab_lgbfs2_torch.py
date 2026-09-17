import time
from pprint import pprint
import torch
import tqdm

import pandas as pd
from geometry import CtGeometry
from sinogram import Sinogram
from unidim.plots import plot_points, make_gif_from_png

import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("wasserstein-lbfgs")

XLA_PYTHON_CLIENT_PREALLOCATE = False
XLA_PYTHON_CLIENT_MEM_FRACTION = 1.0
torch.cuda.set_per_process_memory_fraction(XLA_PYTHON_CLIENT_MEM_FRACTION)

DEVICE = torch.device("cuda")
img_dir= None

def get_torch_gpu_memory(device=DEVICE):

    device_index = device.index

    torch_allocated = torch.cuda.memory_allocated(device_index)
    torch_reserved = torch.cuda.memory_reserved(device_index)
    torch_peak_allocated = torch.cuda.max_memory_allocated(device_index)
    torch_peak_reserved = torch.cuda.max_memory_reserved(device_index)

    gpu_free, gpu_total = torch.cuda.mem_get_info(device_index)
    gpu_used = gpu_total - gpu_free
    mem_info = {
        "torch_allocated": torch_allocated,
        "torch_reserved": torch_reserved ,
        "torch_peak_allocated": torch_peak_allocated ,
        "torch_peak_reserved": torch_peak_reserved ,
        "gpu_used": gpu_used ,
        "gpu_total": gpu_total ,
        "gpu_free": gpu_free,
    }
    mem_info_mb = {k:v/1024**2 for k,v in mem_info.items()}

    return mem_info_mb


def _w2_1d(proj, bin_mass, bin_edges, ext_dtype=torch.float64):
    proj = proj.to(dtype=ext_dtype)
    bin_mass = bin_mass.to(dtype=ext_dtype)
    bin_edges = bin_edges.to(dtype=ext_dtype)
    n = proj.shape[0]
    w = 1.0 / n
    dw = bin_edges[1] - bin_edges[0]
    bin_center = bin_edges[:-1] + dw / 2
    cum = torch.cumsum(bin_mass, dim=0)
    cum_start = cum - bin_mass
    prefix_M = torch.cumsum(bin_mass * bin_center, dim=0) - bin_mass * bin_center

    def M(q):
        j = torch.searchsorted(cum, q, right=True)
        j = torch.clamp(j, 0, bin_mass.shape[0] - 1)
        mass_j = bin_mass[j]
        cum_start_j = cum_start[j]
        prefix_M_j = prefix_M[j]
        bin_edges_j = bin_edges[j]
        f = torch.where(mass_j > 0,(q - cum_start_j) / mass_j,torch.zeros_like(q),)
        return prefix_M_j + mass_j * (bin_edges_j * f + dw * f * f / 2)

    s = torch.sort(proj).values
    q = torch.arange(n,device=proj.device,dtype=ext_dtype,) * w
    bary = (M(q + w) - M(q)) / w
    source_M2 = w * torch.sum(s * s)
    target_M2 = (torch.sum(bin_mass * bin_center * bin_center)+ dw * dw / 12)
    cross_term = 2 * w * torch.sum(s * bary)
    wasserstein2 = source_M2 + target_M2 - cross_term
    return wasserstein2

def loss_torch_map(points, normals, bin_mass, bin_edges, batch_size, ext_dtype, use_checkpoint=True):
    def angle_cost(normal, mass):
        # normal, mass = normal_and_mass
        mass = mass.detach()
        normal = normal.detach()
        projections = points @ normal
        w = _w2_1d(projections, mass, bin_edges, ext_dtype)
        return w

    costs = torch.vmap(angle_cost, in_dims=(0, 0), out_dims=0,
                       chunk_size=batch_size, randomness="error",)(normals, bin_mass)

    return costs.sum().to(torch.float32)


def optimize(points, sino, max_iter=50, line_search_fn="strong_wolfe",
             ext_dtype=torch.float64, batch_size=16, use_checkpoint=True, target_loss=1e-3, avg_last_n=10):

    g = sino.geometry
    points = points.to(dtype=ext_dtype, device=DEVICE)
    points.requires_grad_(True)

    normals = torch.as_tensor(g.normals, dtype=ext_dtype, device=DEVICE)
    bin_edges = torch.as_tensor(g.bin_edges, dtype=ext_dtype, device=DEVICE)
    bin_mass = torch.as_tensor(sino.values, dtype=ext_dtype, device=DEVICE)
    bin_mass = bin_mass / bin_mass.sum(dim=1, keepdim=True)

    def fun(p):
        return loss_torch_map(p, normals, bin_mass, bin_edges, batch_size, ext_dtype, use_checkpoint)

    # ------------------------------------------------------------------------------------------------
    # Équivalent conceptuel du @jax.jit appliqué à la fonction de calcul de la loss.
    # On compile la fonction numérique plutôt que torch.optim.LBFGS lui-même, car LBFGS effectue
    # plusieurs appels dynamiques à sa closure pendant la recherche linéaire.
    # ------------------------------------------------------------------------------------------------


    compiled_fun = fun # torch.compile(fun, fullgraph=False, dynamic=False)
    # L-BFGS
    # PyTorch ne possède pas scale_by_zoom_linesearch d'Optax. strong_wolfe est la recherche
    # linéaire native la plus proche disponible dans torch.optim.LBFGS.
    optimizer = torch.optim.LBFGS(
        [points],
        lr=1.0,
        max_iter=1,
        max_eval=5,
        tolerance_grad= 1e-7,
        tolerance_change=1e-9,
        history_size=5,
        line_search_fn=line_search_fn,
    )
    value_holder = {}
    grad_holder = {}

    def step(p):
        def closure():
            optimizer.zero_grad(set_to_none=True)
            value = compiled_fun(p)
            value.backward()
            value_holder["value"] = value.detach()
            grad_holder["grad"] = p.grad.detach().clone()
            return value

        optimizer.step(closure)

        value = value_holder["value"]
        grad = grad_holder["grad"]

        return p, value, grad
    torch.cuda.synchronize(DEVICE)
    torch.cuda.reset_peak_memory_stats(DEVICE.index)
    start_optimization = time.perf_counter()
    history = []
    time_to_loss = -1

    for idx in tqdm.tqdm(range(max_iter)):
        iteration_start = time.perf_counter()
        points, value, grad = step(points)
        torch.cuda.synchronize(DEVICE)
        grad_norm = float(grad.norm())
        loss_value = float(value)
        if img_dir is not None:

            points_cpu = points.detach().cpu().numpy()
            plot_points(points_cpu, step=idx, img_dir=img_dir)

        iteration_time = time.perf_counter() - iteration_start
        elapsed_time = time.perf_counter() - start_optimization
        raw_metrics = {
            "iteration": idx,
            "loss": loss_value,
            "grad_norm": grad_norm,
            "elapsed_time": round(elapsed_time, 3),
            "iteration_time": round(iteration_time, 4),
        }
        history.append(raw_metrics)
        mlflow.log_metrics(raw_metrics, step=idx)

        if time_to_loss < 0 and loss_value <= target_loss:
            time_to_loss = elapsed_time

    torch.cuda.synchronize(DEVICE)
    mem_info_mb = get_torch_gpu_memory(DEVICE)
    end_optimization = time.perf_counter()
    total_time = end_optimization - start_optimization
    avg_iteration_time_last_5 = sum(h["iteration_time"] for h in history[-5:]) / 5
    df_history = pd.DataFrame(history)

    print(f"time : {round(total_time,1)} s")
    results = {"compile_time_1st_run": round(history[0]["iteration_time"], 3),
               "total_time": round(total_time, 2),
               "avg_iteration_time_last_5": round(avg_iteration_time_last_5, 3),
               "final_loss": loss_value, "final_grad_norm": grad_norm,
               "time_to_loss": round(time_to_loss,1),
               "torch_peak_allocated": round(mem_info_mb['torch_peak_allocated']),
               "gpu_used": round(mem_info_mb["gpu_used"]),
               }
    mlflow.log_metrics(results)
    mlflow.log_text(df_history.to_csv(index=False), "history.csv", )
    return points, optimizer, results, df_history

def run_experiments(params):
    pprint(params)
    with mlflow.start_run():
        mlflow.log_params(params)
        sino = Sinogram(CtGeometry(nb_angles=params["nb_angles"], nb_bins=params["nb_bins"], extent=2.0))

        sino.add_disk(center=[0, 0], radius=0.9, density=+1.0)
        sino.add_disk(center=[0.5, 0], radius=0.3, density=-1.0)
        sino.add_disk(center=[-0.5, 0], radius=0.3, density=-1.0)

        torch.manual_seed(params["seed"])

        points0 = torch.rand((params["nb_points"], 2), dtype=torch.float32, device=DEVICE) * 2.0 - 1.0
        # points0.requires_grad_(True)

        points, optimizer, results, df_history = optimize(
            points0,
            sino,
            max_iter=params["max_iter"],
            line_search_fn=params['line_search_fn'],
            ext_dtype=params["ext_dtype"],
            batch_size=params["batch_size"],
            use_checkpoint=params["use_checkpoint"],
            target_loss=params["target_loss"],
            avg_last_n=10,
        )
        print(df_history.to_markdown(index=False))

        pprint(results)


base_params = dict(XLA_PYTHON_CLIENT_PREALLOCATE=XLA_PYTHON_CLIENT_PREALLOCATE,
                       XLA_PYTHON_CLIENT_MEM_FRACTION=XLA_PYTHON_CLIENT_MEM_FRACTION,
                       nb_points=10_000,
                       nb_angles=600,
                       nb_bins=4096,
                       batch_size=1,
                       ext_dtype=torch.float64,
                       max_iter=15,
                       use_checkpoint = False,
                       line_search_fn="strong_wolfe",
                       seed=27,
                       target_loss=1e-3)

base_params['backend'] = "torch"
run_experiments(base_params)



if img_dir is not None :
    make_gif_from_png(img_dir)

# PyTorch fonctionne différemment : il utilise déjà un CUDA caching allocator. Il ne préalloue normalement pas 75 % de la VRAM au démarrage.

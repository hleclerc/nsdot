import time
import torch
print(f"torch.__version__=={torch.__version__}")
import torch.optim as optim
import subprocess
from geometry import CtGeometry
from sinogram import Sinogram
from tracker import Tracker, GradTimer

# --- Définition du device (GPU ou CPU) ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"CUDA disponible : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Nom du GPU : {torch.cuda.get_device_name(0)}")
    print(f"Utilisation du device : {device}")
else:
    print("Aucun GPU détecté. Utilisation du CPU.")

# Configuration pour éviter les problèmes de précision
torch.set_default_dtype(torch.float64)

# Constante pour la gestion de la mémoire
_BYTES_PER_CHUNK_ELEMENT = 256

def _w2_1d(proj, bin_mass, bin_edges):
    n = proj.shape[0]
    w = 1.0 / n
    dw = bin_edges[1] - bin_edges[0]
    bin_center = bin_edges[:-1] + dw / 2

    cum = torch.cumsum(bin_mass, dim=0)
    cum_start = cum - bin_mass
    prefix_M = torch.cumsum(bin_mass * bin_center, dim=0) - bin_mass * bin_center

    def M(q):
        # print(f"cum.is_contiguous(): {cum.is_contiguous()} is view {cum._is_view()} cum.stride(): {cum.stride()}")  # <-- Ajoute ceci
        # print(f"q.is_contiguous(): {q.is_contiguous()} is view {q._is_view()} q.stride(): {q.stride()}")  # <-- Ajoute ceci
        j = torch.clamp(torch.searchsorted(cum, q, right=True), 0, bin_mass.shape[0] - 1)
        f = torch.where(bin_mass[j] > 0, (q - cum_start[j]) / bin_mass[j], 0.0)
        return prefix_M[j] + bin_mass[j] * (bin_edges[j] * f + dw * f * f / 2)

    s = torch.sort(proj).values.to(torch.float64)
    q = torch.arange(n, dtype=torch.float64, device=proj.device) * w
    bary = (M(q + w) - M(q)) / w

    target_second_moment = torch.sum(bin_mass * bin_center ** 2) + dw * dw / 12
    return w * torch.sum(s ** 2) - 2 * w * torch.sum(s * bary) + target_second_moment

def _sino_arrays(sino):
    g = sino.geometry
    normals = torch.tensor(g.normals, dtype=torch.float32, device=device)
    bin_edges = torch.tensor(g.bin_edges, dtype=torch.float64, device=device)
    bin_mass = torch.tensor(sino.values, dtype=torch.float64, device=device)
    bin_mass = bin_mass / bin_mass.sum(dim=1, keepdim=True)
    return normals, bin_edges, bin_mass

def loss(points, normals, bin_edges, bin_mass, mem_budget_bytes=-1):
    n, A = points.shape[0], normals.shape[0]

    def angle_cost(normal, mass):
        proj = points @ normal
        return _w2_1d(proj, mass, bin_edges)

    # Vectorise angle_cost sur les normales et bin_mass
    vectorized_cost = torch.vmap(angle_cost)

    # Applique à toutes les normales/mass
    costs = vectorized_cost(normals, bin_mass)
    return costs.sum().float()

def optimize(points,
             sino,
             max_iter=15,
             tracker=None,
             grad_timer=None,
             max_linesearch_steps=8):

    normals, bin_edges, bin_mass = _sino_arrays(sino)
    points = points.clone().requires_grad_(True).to(device)

    optimizer = optim.LBFGS([points], lr=1.0, max_iter=max_iter, line_search_fn='strong_wolfe')

    def closure():
        optimizer.zero_grad()
        loss_val = loss(points, normals, bin_edges, bin_mass)
        loss_val.backward()
        return loss_val

    print(f"  [warmup] compiling/stabilizing (n={points.shape[0]})...", end="", flush=True)
    t_warmup = time.time()

    for _ in range(4):
        loss_val = closure()
        optimizer.step(closure)

    print(f" done ({time.time() - t_warmup:.2f}s)")

    for i in range(max_iter):
        if tracker is not None:
            tracker.start()
        if grad_timer is not None:
            t0 = time.time()

        loss_val = closure()
        optimizer.step(closure)

        if grad_timer is not None:
            elapsed_ms = (time.time() - t0) * 1000
            grad_timer.record(elapsed_ms)

        if tracker is not None:
            tracker.step(i, loss_val.item(), points.detach().cpu())

    return points.detach().cpu()

def _split(points, n, key, jitter):
    reps = -(-n // points.shape[0])
    tiled = points.repeat(reps, 1)[:n]
    noise = jitter * torch.randn_like(tiled)  # Utilise `tiled` comme référence pour le device
    return tiled + noise

def multiscale_optimize(sino, nb_points_final, nb_points_init=200, factor=4, seed=0, tracker=None, timings=None, **kwargs):
    extent = sino.geometry.extent
    torch.manual_seed(seed)
    points = torch.rand((nb_points_init, 2), dtype=torch.float32, device=device) * extent - extent / 2

    n = nb_points_init
    while True:
        grad_timer = GradTimer() if timings is not None else None
        points = optimize(points, sino, tracker=tracker, grad_timer=grad_timer, **kwargs)

        if grad_timer is not None:
            timings[n] = grad_timer.mean_ms
            print(f"  n={n:8d}: {grad_timer.mean_ms:.3f} ms/grad ({len(grad_timer.times_ms)} calls)")

        if n >= nb_points_final:
            return points

        n = min(n * factor, nb_points_final)
        points = _split(points, n, None, jitter=sino.geometry.dw / 1e6)

if __name__ == "__main__":
    nb_diracs = 1_000

    sino = Sinogram(CtGeometry(nb_angles=600, nb_bins=4096, extent=2.0))
    sino.add_disk(center=[0, 0], radius=0.9, density=+1.0)
    sino.add_disk(center=[0, 0], radius=0.7, density=-1.0)

    tracker = Tracker(record_frames=True)
    timings = {}

    points = multiscale_optimize(
        sino,
        nb_points_final=nb_diracs,
        tracker=tracker,
        timings=timings
    )

    tracker.export_html("unidim_reconstruction.html", sino.geometry.extent)
    subprocess.Popen(["firefox", "unidim_reconstruction.html"])

""" ERROR
reconstruction_torch.py:38: UserWarning: torch.searchsorted(): input value tensor is non-contiguous, this will lower the performance due to extra data copy when converting non-contiguous tensor to contiguous, please use contiguous input value tensor if possible. This message will only appear once per program. (Triggered internally at /__w/pytorch/pytorch/aten/src/ATen/native/BucketizationUtils.h:32.)
  j = torch.clamp(torch.searchsorted(cum, q, right=True), 0, bin_mass.shape[0] - 1)
  
optimizer = optim.LBFGS([points], lr=1.0, max_iter=max_iter, line_search_fn='strong_wolfe')

 File "/home/hcourtei/miniconda3/envs/cuda_recons/lib/python3.12/site-packages/torch/optim/optimizer.py", line 1155, in add_param_group
    raise ValueError("can't optimize a non-leaf Tensor")
ValueError: can't optimize a non-leaf Tensor
"""
import torch
from torch import optim
import tqdm
import nvidia_smi

from unidim.plots import plot_final_points

import mlflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("reconstruction_tomo")

# Initialisation de NVIDIA SMI
nvidia_smi.nvmlInit()
handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)

# Paramètres
NB_ANGLES = 600
NB_BINS = 4096
NB_POINTS_FINAL = 10_000
INNER_RADIUS = 0.7
OUTER_RADIUS = 0.9

# Reconstruction
BYTES_PER_CHUNK_ELEMENT = 256
SAFETY_FRACTION = 0.5
FALLBACK_BYTES = 512 * 1024 * 1024  # 512 Mo
MAX_ITER = 15
MAX_LINESEARCH_STEPS = 8
NB_POINTS_INIT = 200
FACTOR = 4
SEED = 0

# Type de données
ext_dtype = torch.float64  # ou torch.float32

# Fonction pour calculer la distance de Wasserstein 1D
def _w2_1d(proj, bin_mass, bin_edges):
    # Assure que tous les tenseurs sont contigus
    proj = proj.contiguous()
    bin_mass = bin_mass.contiguous()
    bin_edges = bin_edges.contiguous()
    n = proj.shape[0]
    w = 1.0 / n
    dw = bin_edges[1] - bin_edges[0]
    bin_center = bin_edges[:-1] + dw / 2
    cum = torch.cumsum(bin_mass, dim=0).contiguous()  # Rendre le tenseur contigu
    cum_start = (cum - bin_mass).contiguous()
    prefix_M = (torch.cumsum(bin_mass * bin_center, dim=0) - bin_mass * bin_center).contiguous()

    def M(q):
        q_contiguous = q.contiguous()  # Assurer la contiguïté
        j = torch.clamp(torch.searchsorted(cum, q_contiguous, side="right"), 0, bin_mass.shape[0] - 1)
        f = torch.where(bin_mass[j] > 0, (q_contiguous - cum_start[j]) / bin_mass[j], 0.0)
        return prefix_M[j] + bin_mass[j] * (bin_edges[j] * f + dw * f * f / 2)

    s = torch.sort(proj)[0].to(ext_dtype).contiguous()
    q = (torch.arange(n, dtype=ext_dtype, device=proj.device) * w).contiguous()

    bary = (M(q + w) - M(q)) / w
    target_second_moment = torch.sum(bin_mass * bin_center * bin_center) + dw * dw / 12
    wasserstein2 = w * torch.sum(s * s) - 2 * w * torch.sum(s * bary) + target_second_moment
    return wasserstein2

# Fonction pour calculer la taille des chunks
def _get_chunk_size(nb_points, nb_angles, mem_budget_bytes):
    if mem_budget_bytes is None:
        return 1
    memory_for_one_angle = BYTES_PER_CHUNK_ELEMENT * max(nb_points, 1)
    max_batch_size = mem_budget_bytes // memory_for_one_angle
    return max(1, min(nb_angles, max_batch_size))

# Fonction de perte avec traitement par chunks
def loss(points, normals, bin_edges, bin_mass, mem_budget_bytes=-1):
    def angle_cost(normal, mass):
        projections = points @ normal
        return _w2_1d(projections, mass, bin_edges)

    n, A = points.shape[0], normals.shape[0]
    batch_size = _get_chunk_size(n, A, mem_budget_bytes)

    # Traiter les données par chunks
    total_cost = 0.0
    for i in range(0, A, batch_size):
        chunk_normals = normals[i:i + batch_size]
        chunk_bin_mass = bin_mass[i:i + batch_size]
        costs = torch.vmap(angle_cost)(chunk_normals, chunk_bin_mass)
        total_cost += costs.sum()

    return total_cost.to(ext_dtype)

stage = 0
# Fonction d'optimisation
def optimize(points, sino, max_iter=15):
    g = sino.geometry
    global stage
    normals = torch.as_tensor(g.normals, dtype=ext_dtype, device=points.device)
    bin_edges = torch.as_tensor(g.bin_edges, dtype=ext_dtype, device=points.device)
    bin_mass = torch.as_tensor(sino.values, dtype=ext_dtype, device=points.device)
    bin_mass = bin_mass / bin_mass.sum(dim=1, keepdim=True)

    # Activer requires_grad pour les points
    points = points.clone().detach().requires_grad_(True)

    # Initialisation de l'optimiseur L-BFGS
    optimizer = optim.LBFGS([points], lr=1.0, max_iter=max_iter)

    # Calcul du budget mémoire
    device = torch.cuda.current_device()
    bytes_limit = torch.cuda.memory_reserved(device)
    bytes_in_use = torch.cuda.memory_allocated(device)
    free_bytes = bytes_limit - bytes_in_use
    safe_free_bytes = int(max(free_bytes, 0) * SAFETY_FRACTION)
    mem_budget_bytes = safe_free_bytes if safe_free_bytes > 0 else FALLBACK_BYTES

    chunk_size = _get_chunk_size(points.shape[0], normals.shape[0], mem_budget_bytes)

    def closure():
        optimizer.zero_grad()
        current_loss = loss(points, normals, bin_edges, bin_mass, mem_budget_bytes)
        current_loss.backward()
        return current_loss
    for i in (pbar := tqdm.tqdm(range(max_iter))):
        global_step = stage * max_iter + i
        current_loss = optimizer.step(closure)

        # Mise à jour de l'affichage
        nvidia_mem = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        pbar.set_description(
            f"for n = {points.shape[0]} Step: {i}| mem_budget_bytes: {mem_budget_bytes / 1024 ** 3:.2f} GiB, "
            f"VRAM torch: {bytes_in_use / 1024 ** 3:.2f}/{bytes_limit / 1024 ** 3:.2f} GiB, "
            f"VRAM nvidia: {nvidia_mem.used / 1024 ** 3:.2f}/{nvidia_mem.total / 1024 ** 3:.2f} GiB, "
            f"Chunk size: {chunk_size}"
        )
        mlflow.log_metric("loss", current_loss.item(), step=global_step)
        mlflow.log_metric("vram_torch_used_GB", bytes_in_use / 1024 ** 3, step=global_step)
        mlflow.log_metric("vram_torch_reserved_GB", bytes_limit / 1024 ** 3, step=global_step)
        mlflow.log_metric("vram_nvidia_used_GB", nvidia_mem.used / 1024 ** 3, step=global_step)
        mlflow.log_metric("vram_nvidia_total_GB", nvidia_mem.total / 1024 ** 3, step=global_step)
        mlflow.log_metric("mem_budget_bytes_GB", mem_budget_bytes / 1024 ** 3, step=global_step)
        mlflow.log_metric("chunk_size", chunk_size, step=global_step)
    stage = stage + 1
    return points.detach()

# Fonction d'optimisation multi-échelle
def multiscale_optimize(sino,
                        nb_points_final,
                        nb_points_init=200,
                        factor=4,
                        seed=0, **kwargs):
    extent = sino.geometry.extent
    torch.manual_seed(seed)
    points = torch.rand((nb_points_init, 2), dtype=ext_dtype, device="cuda") * extent - extent / 2
    points.requires_grad_(True)

    n = nb_points_init
    jitter = sino.geometry.dw / 1e6
    while n < nb_points_final:
        points = optimize(points, sino, max_iter = MAX_ITER)
        n_next = min(n * factor, nb_points_final)
        reps = -(-n_next // points.shape[0])
        tiled = points.repeat(reps, 1)[:n_next]
        noises = jitter * torch.randn((n_next, 2), dtype=points.dtype, device=points.device)
        points = tiled + noises
        points.requires_grad_(True)
        n = n_next

    points = optimize(points, sino, max_iter = MAX_ITER )
    return points.detach()

# Exemple d'utilisation
if __name__ == "__main__":
    # Démarrer une run MLflow globale
    with mlflow.start_run():
        # Enregistrer les paramètres initiaux
        mlflow.log_params({
            "NB_ANGLES": NB_ANGLES,
            "NB_BINS": NB_BINS,
            "NB_POINTS_FINAL": NB_POINTS_FINAL,
            "INNER_RADIUS": INNER_RADIUS,
            "OUTER_RADIUS": OUTER_RADIUS,
            "MAX_ITER": MAX_ITER,
            "FACTOR": FACTOR,
            "SEED": SEED,
        })


        from geometry import CtGeometry
        from sinogram import Sinogram

        # Création d'un sinogramme
        sino = Sinogram(CtGeometry(nb_angles=NB_ANGLES, nb_bins=NB_BINS, extent=2.0))
        sino.add_disk(center=[0, 0], radius=OUTER_RADIUS, density=+1.0)
        sino.add_disk(center=[0, 0], radius=INNER_RADIUS, density=-1.0)

        # Optimisation multi-échelle
        points = multiscale_optimize(
            sino,
            nb_points_final=NB_POINTS_FINAL,
            nb_points_init=NB_POINTS_INIT,
            factor=FACTOR,
            seed=SEED,
            max_iter=MAX_ITER,
        )

        plot_final_points(points, 'final_points.png')
        mlflow.log_artifact('final_points.png')



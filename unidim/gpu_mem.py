"""Available-GPU-memory helpers for angle-chunk sizing (see
`reconstruction_jax.loss`'s and `reconstruction_cuda._cost_grad`'s
`mem_budget_bytes`) -- read the ACTUAL free memory on the active device
instead of a fixed guess, so chunk_size shrinks safely on a small GPU and
grows to use more of a big one. Each backend's query lives behind its own
function, importing its backend lazily inside, so importing this module
doesn't pull in a dependency (torch/jax) the caller isn't already using.
"""

_SAFETY_FRACTION = 0.5  # leave headroom for everything else alive on the device (bin_mass/cum/prefix_* tensors, LBFGS/optax history, points, ...) -- chunk scratch isn't the only allocation
_FALLBACK_BYTES = 512 * 1024 * 1024  # no CUDA visible -- a conservative default chunk budget


def jax_mem_budget_bytes():
    """Free memory on JAX's default device, scaled by `_SAFETY_FRACTION`, or
    None if that device isn't a GPU/TPU (a plain CPU run). `loss` treats
    None as "keep chunk_size at 1", its previous, always-correct behavior --
    batching angles together was verified SLOWER on CPU (see `optimize`'s
    docstring: no parallelism gain from a batched sort there, only worse
    cache behavior), so there's no point chunking on CPU at all, memory or
    not. `memory_stats()` itself also only reports anything on a GPU/TPU
    backend -- None on CPU -- which this treats the same way."""
    import jax
    device = jax.devices()[0]
    # Sur CPU, pas de gestion de chunks :
    # les benchmarks ont montré que le batching des angles est plus lent.
    if device.platform == "cpu":
        return None
    stats = device.memory_stats()
    # Si JAX ne fournit pas ces statistiques, on retombe sur
    # le comportement CPU : un seul angle à la fois.
    if not stats or "bytes_limit" not in stats:
        return None
    # On n'utilise qu'une fraction de la mémoire disponible afin
    # de conserver une marge de sécurité et d'éviter les OOM.
    free_bytes = stats["bytes_limit"] - stats.get("bytes_in_use", 0)
    return int(max(free_bytes, 0) * _SAFETY_FRACTION)

def torch_mem_budget_bytes():
    """Free CUDA memory right now (`torch.cuda.mem_get_info`), scaled by
    `_SAFETY_FRACTION`. Falls back to `_FALLBACK_BYTES` if no CUDA device is
    visible."""
    import torch
    if not torch.cuda.is_available():
        return _FALLBACK_BYTES
    free_bytes, _total_bytes = torch.cuda.mem_get_info()
    return int(free_bytes * _SAFETY_FRACTION)

def _get_chunk_size(nb_points, nb_angles, nb_bins, safety_fraction=0.8):
    import jax # Mémoire par angle (en bytes)
    memory_per_angle = 96 * nb_points + 48 * nb_bins + 48

    # Mémoire disponible sur le GPU
    device = jax.devices()[0]
    stats = device.memory_stats()
    available_jax_bytes = stats["bytes_limit"] - stats["bytes_in_use"]

    # Appliquer une marge de sécurité
    safe_mem_budget = int(available_jax_bytes * safety_fraction)

    # Calculer le batch_size maximal
    max_batch_size = safe_mem_budget // memory_per_angle
    max_batch_size = max(1, min(nb_angles, max_batch_size))
    return max_batch_size

def get_jax_gpu_memory():
    import jax
    """Retourne la mémoire disponible sur le GPU en bytes."""
    # Obtenir le device GPU (suppose un seul GPU)
    device = jax.devices()[0]
    # Obtenir les statistiques de mémoire
    stats = device.memory_stats()

    bytes_limit = stats["bytes_limit"]
    bytes_in_use = stats["bytes_in_use"]
    print(f" (JAX) used / total  : {bytes_in_use/1024**3:.2f} / {bytes_limit/1024**3:.2f} GiB")
    return  bytes_in_use, bytes_limit



import time
import threading
import jax
import nvidia_smi


# Initialisation une seule fois
nvidia_smi.nvmlInit()
handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)

def get_gpu_memory(handle=handle):
    result = {
        "jax_used": None,
        "jax_pool": None,
        "jax_total": None,
        "nvidia_used": None,
        "nvidia_total": None,
    }

    try:
        stats = jax.devices()[0].memory_stats()

        if stats is not None:
            result["jax_used"] = stats.get("bytes_in_use")
            result["jax_pool"] = stats.get("pool_bytes")
            result["jax_total"] = stats.get("bytes_limit")

    except Exception:
        pass

    try:
        mem = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)

        result["nvidia_used"] = mem.used
        result["nvidia_total"] = mem.total

    except Exception:
        pass

    return result


def monitor_gpu_memory(stop_event, handle=handle, interval=0.02):
    peak = {
        "jax_peak_used": 0,
        "jax_peak_pool": 0,
        "nvidia_peak": 0,
        "jax_total": None,
        "nvidia_total": None,
    }

    while not stop_event.is_set():

        mem = get_gpu_memory(handle)

        if mem["jax_used"] is not None:
            peak["jax_peak_used"] = max(
                peak["jax_peak_used"],
                mem["jax_used"],
            )

        if mem["jax_pool"] is not None:
            peak["jax_peak_pool"] = max(
                peak["jax_peak_pool"],
                mem["jax_pool"],
            )

        if mem["nvidia_used"] is not None:
            peak["nvidia_peak"] = max(
                peak["nvidia_peak"],
                mem["nvidia_used"],
            )

        peak["jax_total"] = mem["jax_total"]
        peak["nvidia_total"] = mem["nvidia_total"]

        time.sleep(interval)

    return peak


def measure_gpu_peak(fn, *args, interval=0.02, **kwargs):

    stop_event = threading.Event()
    peak_result = {}

    def monitor():
        nonlocal peak_result
        peak_result = monitor_gpu_memory(
            stop_event,
            handle=handle,
            interval=interval,
        )

    thread = threading.Thread(target=monitor)
    thread.start()

    try:
        output = fn(*args, **kwargs)

        # Fonctionne aussi si output est un tuple/list/dict de JAX arrays
        jax.tree_util.tree_map(
            lambda x: x.block_until_ready()
            if hasattr(x, "block_until_ready")
            else x,
            output,
        )

    finally:
        stop_event.set()
        thread.join()

    return output, peak_result
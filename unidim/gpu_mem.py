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


def torch_cuda_mem_budget_bytes():
    """Free CUDA memory right now (`torch.cuda.mem_get_info`), scaled by
    `_SAFETY_FRACTION`. Falls back to `_FALLBACK_BYTES` if no CUDA device is
    visible."""
    import torch
    if not torch.cuda.is_available():
        return _FALLBACK_BYTES
    free_bytes, _total_bytes = torch.cuda.mem_get_info()
    return int(free_bytes * _SAFETY_FRACTION)


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
    if device.platform == "cpu":
        return None
    stats = device.memory_stats()
    if not stats or "bytes_limit" not in stats:
        return None
    free_bytes = stats["bytes_limit"] - stats.get("bytes_in_use", 0)
    return int(max(free_bytes, 0) * _SAFETY_FRACTION)

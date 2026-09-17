
import time
import jax
import threading
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
        peak["jax_peak_used"] = max(peak["jax_peak_used"],mem["jax_used"],)
        peak["jax_peak_pool"] = max(peak["jax_peak_pool"],mem["jax_pool"],)
        peak["nvidia_peak"] = max(peak["nvidia_peak"],mem["nvidia_used"],)
        peak["jax_total"] = mem["jax_total"]
        peak["nvidia_total"] = mem["nvidia_total"]

        time.sleep(interval)

    return peak


def measure_gpu_peak(fn, *args, interval=0.02, **kwargs):

    stop_event = threading.Event()
    peak_result = {}

    def monitor():
        nonlocal peak_result
        peak_result = monitor_gpu_memory(stop_event,handle=handle,interval=interval,)

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
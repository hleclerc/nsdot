import time

import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)


def bytes_to_kib(x):
    return x / 1024.0


def bytes_to_mib(x):
    return x / (1024.0 ** 2)

def print_memory_stats(memory):
    """
    Affiche les statistiques mémoire XLA de manière lisible.
    """

    print("  Memory:")
    print(
        f"    generated code : "
        f"{memory.generated_code_size_in_bytes:>10,} B "
        f"({bytes_to_kib(memory.generated_code_size_in_bytes):8.2f} KiB)"
    )

    print(
        f"    arguments      : "
        f"{memory.argument_size_in_bytes:>10,} B "
        f"({bytes_to_kib(memory.argument_size_in_bytes):8.2f} KiB)"
    )

    print(
        f"    output         : "
        f"{memory.output_size_in_bytes:>10,} B "
        f"({bytes_to_kib(memory.output_size_in_bytes):8.2f} KiB)"
    )

    print(
        f"    temporary      : "
        f"{memory.temp_size_in_bytes:>10,} B "
        f"({bytes_to_kib(memory.temp_size_in_bytes):8.2f} KiB)"
    )

    print(
        f"    alias          : "
        f"{memory.alias_size_in_bytes:>10,} B "
        f"({bytes_to_kib(memory.alias_size_in_bytes):8.2f} KiB)"
    )

    total = (
        memory.generated_code_size_in_bytes
        + memory.argument_size_in_bytes
        + memory.output_size_in_bytes
        + memory.temp_size_in_bytes
    )

    print(
        f"    TOTAL          : "
        f"{total:>10,} B "
        f"({bytes_to_mib(total):8.4f} MiB)"
    )


def _w2_1d(proj, bin_mass, bin_edges, ext_dtype):
    proj = proj.astype(ext_dtype) #(N,2) float64 : N x 2 x 8 bytes
    bin_mass = bin_mass.astype(ext_dtype) #(B) float64 : B  x 8
    bin_edges = bin_edges.astype(ext_dtype) #(B+1) float64: (B+1) x 8
    # TOTAL entrées 16*N + 8*B + 8
    n = proj.shape[0]
    w = 1.0 / n
    dw = bin_edges[1] - bin_edges[0]
    bin_center = bin_edges[:-1] + dw / 2 # B  x 8
    cum = jnp.cumsum(bin_mass) #(B) float64 # B  x 8
    cum_start = cum - bin_mass # B  x 8
    prefix_M = jnp.cumsum(bin_mass * bin_center) - bin_mass * bin_center #B  x 8

    def M(q):
        j = jnp.clip(jnp.searchsorted(cum, q, side="right"), 0, bin_mass.shape[0] - 1)
        # N int32 :   N x 4
        f = jnp.where(bin_mass[j] > 0, (q - cum_start[j]) / bin_mass[j], 0.0) # N float64 : N * 8

        return prefix_M[j] + bin_mass[j] * (bin_edges[j] * f + dw * f * f / 2) # 12 N

    s = jnp.sort(proj) # N * 8
    q = jnp.arange(n) * w # N * 8
    bary = (M(q + w) - M(q)) / w # N * 8
    # 24 N
    # + 2 appel fun M : 2 x 12 N = 24 N

    source_M2 = w * jnp.sum(s * s)  # s*s : N * 8
    target_M2 = jnp.sum(bin_mass * bin_center * bin_center) + dw * dw / 12 # B*8
    cross_term = 2 * w * jnp.sum(s * bary) #s*bary : N * 8
    wasserstein2 = source_M2 + target_M2 - cross_term  # mémoire  : 80*N + 48*B + 8 bytes
    return wasserstein2 #
    # mémoire par appel à angle cost : 96*N + 48*B + 40


def _w2_1d_new(proj, bin_mass, bin_edges, ext_dtype):
    proj = proj.astype(ext_dtype)
    bin_mass = bin_mass.astype(ext_dtype)
    bin_edges = bin_edges.astype(ext_dtype)

    n = proj.shape[0]
    w = jnp.asarray(1.0 / n)
    dw = bin_edges[1] - bin_edges[0]
    bin_left = bin_edges[:-1]
    bin_center = bin_left + dw / 2
    cum = jnp.cumsum(bin_mass)
    cum_start = cum - bin_mass
    bin_M1 = bin_mass * bin_center
    prefix_M1 = jnp.cumsum(bin_M1) - bin_M1
    bin_M2 = bin_mass * (bin_center * bin_center + dw * dw / 12)
    prefix_M2 = jnp.cumsum(bin_M2) - bin_M2

    def moments(q):
        j = jnp.clip(jnp.searchsorted(cum, q, side="right"),0,bin_mass.shape[0] - 1,)
        mass_j = bin_mass[j]
        left_j = bin_left[j]
        f = jnp.where(mass_j > 0,(q - cum_start[j]) / mass_j,0.0,)
        partial_M1 = mass_j * (left_j * f+ dw * f * f / 2)
        M1 = prefix_M1[j] + partial_M1
        partial_M2 = mass_j * (left_j * left_j * f + left_j * dw * f * f+ dw * dw * f * f * f / 3)
        M2 = prefix_M2[j] + partial_M2
        return M1, M2

    s = jnp.sort(proj)
    q = jnp.arange(n) * w
    M1_0, M2_0 = moments(q)
    M1_1, M2_1 = moments(q + w)
    dM1 = M1_1 - M1_0
    dM2 = M2_1 - M2_0

    bary = dM1 / w
    squared_distance = w * (s - bary) ** 2
    target_variance = dM2 - dM1 * dM1 / w
    # target_variance = jnp.maximum(target_variance, 0.0)
    wasserstein2 = jnp.sum(squared_distance + target_variance)

    return wasserstein2

def _w2_1d_new2(proj, bin_mass, bin_edges, ext_dtype):
    proj = proj.astype(ext_dtype)
    bin_mass = bin_mass.astype(ext_dtype)
    bin_edges = bin_edges.astype(ext_dtype)

    n = proj.shape[0]
    w = jnp.asarray(1.0 / n)

    dw = bin_edges[1] - bin_edges[0]
    bin_left = bin_edges[:-1]
    bin_center = bin_left + dw / 2

    cum = jnp.cumsum(bin_mass)
    cum_start = cum - bin_mass

    bin_M1 = bin_mass * bin_center
    prefix_M1 = jnp.cumsum(bin_M1) - bin_M1

    bin_M2 = bin_mass * (bin_center * bin_center + dw * dw / 12)
    prefix_M2 = jnp.cumsum(bin_M2) - bin_M2

    def moments(q):
        j = jnp.clip(jnp.searchsorted(cum, q, side="right"),0,bin_mass.shape[0] - 1,)
        mass_j = bin_mass[j]
        left_j = bin_left[j]
        f = jnp.where(mass_j > 0,(q - cum_start[j]) / mass_j,0.0,)
        M1 = (prefix_M1[j]+ mass_j * (left_j * f+ dw * f * f / 2))
        M2 = (prefix_M2[j]+ mass_j * (left_j * left_j * f+ left_j * dw * f * f+ dw * dw * f * f * f / 3))

        return M1, M2

    s = jnp.sort(proj)
    q = jnp.arange(n) * w
    M1_0, M2_0 = moments(q)
    M1_1, M2_1 = moments(q + w)
    dM1 = M1_1 - M1_0
    dM2 = M2_1 - M2_0
    source_M2 = w * jnp.sum(s * s)
    cross_term = 2 * jnp.sum(s * dM1)

    wasserstein2 = source_M2+ jnp.sum(dM2) - cross_term

    return wasserstein2
#

def benchmark_one(wasser_dist, proj, bin_mass, bin_edges, ext_dtype):
    """
    Benchmark d'une implémentation de W2.
    """

    print()
    print("=" * 80)
    print(wasser_dist.__name__)
    print("=" * 80)

    # ------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------

    wasser_jit = jax.jit(
        lambda p: wasser_dist(p,bin_mass,bin_edges,ext_dtype=ext_dtype,)
    )

    # Warm-up / compilation
    wasser_value = wasser_jit(proj)
    wasser_value.block_until_ready()

    wasser_compiled = wasser_jit.lower(proj).compile()
    memory = wasser_compiled.memory_analysis()

    print(f"\nW2² = {float(wasser_value):.12e}")

    # print_memory_stats(memory)

    # ------------------------------------------------------------
    # Timing loss
    # ------------------------------------------------------------

    n_repeat = 10

    start = time.perf_counter()

    for _ in range(n_repeat):
        value = wasser_jit(proj)
        value.block_until_ready()

    elapsed = (time.perf_counter() - start) / n_repeat

    print(f"\n  loss time       : {elapsed:.6f} s")

    # ------------------------------------------------------------
    # Gradient
    # ------------------------------------------------------------

    value_and_grad_jit = jax.jit(
        jax.value_and_grad(
            lambda p: wasser_dist(p,bin_mass,bin_edges,ext_dtype=ext_dtype,)
        ))

    # Compilation + warm-up
    value, grad = value_and_grad_jit(proj)

    value.block_until_ready()
    grad.block_until_ready()

    # ------------------------------------------------------------
    # Timing gradient
    # ------------------------------------------------------------

    start = time.perf_counter()

    for _ in range(n_repeat):
        value, grad = value_and_grad_jit(proj)
        value.block_until_ready()
        grad.block_until_ready()

    elapsed_grad = (time.perf_counter() - start) / n_repeat

    print(f"  loss + grad time : {elapsed_grad:.6f} s")

    # ------------------------------------------------------------
    # Gradient statistics
    # ------------------------------------------------------------

    grad_norm = jnp.linalg.norm(grad)
    grad_min = jnp.min(grad)
    grad_max = jnp.max(grad)

    print("\n  Gradient:")
    print(f"    shape          : {grad.shape}")
    print(f"    dtype          : {grad.dtype}")
    print(f"    norm           : {float(grad_norm):.12e}")
    print(f"    min            : {float(grad_min):.12e}")
    print(f"    max            : {float(grad_max):.12e}")

    # ------------------------------------------------------------
    # Memory of loss + gradient
    # ------------------------------------------------------------

    grad_compiled = value_and_grad_jit.lower(proj).compile()
    grad_memory = grad_compiled.memory_analysis()

    print("\n  Memory loss + gradient:")
    print_memory_stats(grad_memory)

    return {
        "name": wasser_dist.__name__,
        "value": float(wasser_value),
        "loss_time": elapsed,
        "grad_time": elapsed_grad,
        "grad_norm": float(grad_norm),
        "memory_loss": memory,
        "memory_grad": grad_memory,
    }


if __name__ == "__main__":

    # ------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------

    ext_dtype = jnp.float64

    print("ext_dtype =", ext_dtype)

    NB_POINTS = 10_000
    NB_BINS = 4096

    bin_edges = jnp.linspace(-1.0,1.0,NB_BINS + 1, dtype=ext_dtype,)
    bin_mass = jnp.ones(NB_BINS, dtype=ext_dtype,)
    bin_mass = bin_mass / jnp.sum(bin_mass)


    proj = jnp.linspace(-1.0,1.0,NB_POINTS,dtype=ext_dtype,)

    # ------------------------------------------------------------
    # Benchmark
    # ------------------------------------------------------------

    results = []

    for wasser_dist in [_w2_1d, _w2_1d_new, _w2_1d_new2,]:
        result = benchmark_one(wasser_dist, proj, bin_mass, bin_edges,ext_dtype,)

        results.append(result)

    # ------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------

    print()
    print()
    print("#" * 120)
    print("FINAL SUMMARY")
    print("#" * 120)

    print(
        f"{'implementation':<20}"
        f"{'W2²':>18}"
        f"{'loss [ms]':>14}"
        f"{'loss+grad [ms]':>18}"
        f"{'temp loss [KiB]':>18}"
        f"{'temp grad [KiB]':>18}"
    )

    print("-" * 120)

    for r in results:
        mem_loss = r["memory_loss"]
        mem_grad = r["memory_grad"]

        print(
            f"{r['name']:<20}"
            f"{r['value']:>18.10e}"
            f"{1000 * r['loss_time']:>14.3f}"
            f"{1000 * r['grad_time']:>18.3f}"
            f"{(mem_loss.temp_size_in_bytes/1024):>18.2f}"
            f"{(mem_grad.temp_size_in_bytes/1024):>18.2f}"
        )

    print("-" * 120)

    # Donc _w2_1d_new a plus de code compilé, mais moins de mémoire temporaire.
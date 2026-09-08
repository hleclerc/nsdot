import jax.numpy as jnp
import jax
import time
jax.config.update("jax_enable_x64", True)


def print_memory_stats(label):
    """Affiche les statistiques mémoire disponibles sur le device JAX."""
    device = jax.devices()[0]
    print(f"\n--- {label} --")
    print(f"Device : {device}")
    stats = device.memory_stats()
    if stats is None:
        print("Les statistiques mémoire ne sont pas disponibles sur ce device.")
        return

    for key, value in stats.items():
        print(f"{key:30s}: {value / 1024**2:10.2f} MiB")

def _w2_1d(proj, bin_mass, bin_edges, ext_dtype):
    proj = proj.astype(ext_dtype)
    bin_mass = bin_mass.astype(ext_dtype)
    bin_edges = bin_edges.astype(ext_dtype)

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

    s = jnp.sort(proj)
    q = jnp.arange(n) * w
    bary = (M(q + w) - M(q)) / w

    source_M2 = w * jnp.sum(s * s)
    target_M2 = jnp.sum(bin_mass * bin_center * bin_center) + dw * dw / 12
    cross_term = 2 * w * jnp.sum(s * bary)
    wasserstein2 = source_M2 + target_M2 - cross_term
    return wasserstein2

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

    bin_M2 = bin_mass * (
        bin_center * bin_center + dw * dw / 12
    )
    prefix_M2 = jnp.cumsum(bin_M2) - bin_M2

    def moments(q):
        j = jnp.clip(
            jnp.searchsorted(cum, q, side="right"),
            0,
            bin_mass.shape[0] - 1,
        )

        mass_j = bin_mass[j]
        left_j = bin_left[j]

        f = jnp.where(
            mass_j > 0,
            (q - cum_start[j]) / mass_j,
            0.0,
        )

        M1 = (
            prefix_M1[j]
            + mass_j * (
                left_j * f
                + dw * f * f / 2
            )
        )

        M2 = (
            prefix_M2[j]
            + mass_j * (
                left_j * left_j * f
                + left_j * dw * f * f
                + dw * dw * f * f * f / 3
            )
        )

        return M1, M2

    s = jnp.sort(proj)
    q = jnp.arange(n) * w

    M1_0, M2_0 = moments(q)
    M1_1, M2_1 = moments(q + w)

    dM1 = M1_1 - M1_0
    dM2 = M2_1 - M2_0

    source_M2 = w * jnp.sum(s * s)
    cross_term = 2 * jnp.sum(s * dM1)

    wasserstein2 = (
        source_M2
        + jnp.sum(dM2)
        - cross_term
    )

    return wasserstein2
#
if __name__ =='__main__':
    ext_dtype = jnp.float64
    print("ext_dtype" , ext_dtype)
    NB_POINTS = 10_000
    NB_BINS = 4096
    bin_edges = jnp.linspace( -1.0,1.0, NB_BINS + 1, dtype=jnp.float32)

    bin_mass = jnp.ones(NB_BINS,dtype=jnp.float32)
    bin_mass = bin_mass / jnp.sum(bin_mass)
    proj = jnp.linspace( -1.0,1.0,NB_POINTS,dtype=jnp.float32)

    old_jit = jax.jit(lambda p: _w2_1d(p,bin_mass,bin_edges, ext_dtype=ext_dtype))

    wasser = old_jit(proj)
    wasser.block_until_ready()
    old_compiled = old_jit.lower(proj).compile()

    new_jit = jax.jit(lambda p: _w2_1d_new(p,bin_mass,bin_edges,  ext_dtype=ext_dtype))
    wasser_new = new_jit(proj)
    wasser_new.block_until_ready()
    new_compiled = new_jit.lower(proj).compile()
    #
    old_memory = old_compiled.memory_analysis()
    new_memory = new_compiled.memory_analysis()

    print("Ancienne formulation")
    print(old_memory)

    print("\nNouvelle formulation")
    print(new_memory)

    print(f" wasser = {float(wasser):.12e}")
    print(f" wasser new = {float(wasser_new):.12e}")

    old_value_grad = jax.jit(
        jax.value_and_grad(lambda p: _w2_1d(p, bin_mass,bin_edges,ext_dtype) ))

    new_value_grad = jax.jit(
        jax.value_and_grad(lambda p: _w2_1d_new(p,bin_mass,bin_edges,ext_dtype)))

    old_value, old_grad = old_value_grad(proj)
    new_value, new_grad = new_value_grad(proj)

    old_value.block_until_ready()
    new_value.block_until_ready()

    print("W2 old :", old_value)
    print("W2 new :", new_value)


    # # =============================================================================
    # # Comparaison des temps d'exécution
    # # =============================================================================
    # n_repeat = 100
    # start = time.perf_counter()
    #
    # for _ in range(n_repeat):
    #     value = old_jit(proj)
    #     value.block_until_ready()
    #
    # old_time = (time.perf_counter() - start) / n_repeat
    #
    # start = time.perf_counter()
    #
    # for _ in range(n_repeat):
    #     value = new_jit(proj)
    #     value.block_until_ready()
    #
    # new_time = (time.perf_counter() - start) / n_repeat
    #
    # print(f"10x wasser = {old_time:.6f} s")
    # print(f"10 x wasser new = {new_time:.6f} s")
    #

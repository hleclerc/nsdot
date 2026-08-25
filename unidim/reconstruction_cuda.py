"""Same reconstruction as `reconstruction_jax`, but `loss`+grad are computed
by ONE fused CUDA kernel (cost and gradient in a single pass, no autodiff) --
the CUDA counterpart of `otrec/src/otrec/HcReconstruction/cost/hc_ot_sycl.cpp`
(`hc_ot_cost_grad`, the diracs model).

Compilation/GPU tensors go through standard PyTorch tooling, not loom/acpp:
`torch.utils.cpp_extension.load_inline` JIT-compiles the kernel below with
nvcc (cached after the first call), and `torch.Tensor` is the GPU array type
in and out of it. `torch.optim.LBFGS` (strong-Wolfe line search) plays the
role `optax.lbfgs` plays in `reconstruction_jax`, with the kernel's gradient
assigned to `points.grad` directly instead of going through autograd.

Algorithm: identical closed form to `reconstruction_jax._w2_1d` (validated
against it numerically before this file was written) -- for angle `a` and a
dirac at sorted rank `k`, its target quantile interval `[k/n, (k+1)/n)` is
known without looking at any other dirac, so the barycentric projection is
an O(log m) binary search into that angle's target CDF, independent per
`(a, k)` pair. Only the SORT (one per angle, giving each dirac its rank) is
a genuine cross-dirac dependency.

Memory/batching: `points` ([n, 2]) is the only thing sized by nb_diracs
that's ever fully materialized -- a [nb_angles, n] projection tensor (as a
naive batched `torch.sort` over all angles at once would need) is exactly
what nb_diracs in the 1e11 range rules out (see `reconstruction_jax.loss`'s
docstring). So the C++ side below loops over CHUNKS of angles, projecting +
sorting + reducing a whole chunk per iteration into scratch buffers sized
`chunk_size * n` and reused every iteration -- `chunk_size` is picked from
`mem_budget_bytes` (default 512 MB) so it shrinks toward 1 automatically as
n grows, and grows toward nb_angles automatically when n is small (fewer,
bigger kernel launches instead of one launch triplet per angle -- kernel
launch overhead, not compute, dominates at small n). The sort itself is
`cub::DeviceSegmentedRadixSort` (not `cub::BlockRadixSort`): a DEVICE-WIDE,
many-block primitive, so a single angle's segment already scales past what
one block could hold once nb_diracs reaches its final, large value -- the
cost/grad kernel is launched the same way (2D grid: one axis over diracs,
sized to n, the other over the angles in the current chunk).
"""
import time

import torch
import torch.utils.cpp_extension
from loom.testing import Param, bench

from .tracker import GradTimer

_CPP_DECL = r"""
std::vector<torch::Tensor> ot_cost_grad_cuda(
    torch::Tensor points, torch::Tensor normals,
    torch::Tensor bin_mass, torch::Tensor cum, torch::Tensor cum_start,
    torch::Tensor prefix_M, torch::Tensor prefix_M2, torch::Tensor bin_edges,
    double dw, int64_t mem_budget_bytes);
"""

_CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cub/cub.cuh>
#include <algorithm>
#include <vector>

constexpr int THREADS = 256;

__device__ __forceinline__ int searchsorted_right(const double* cum, int m, double q) {
    int lo = 0, hi = m;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (cum[mid] <= q) lo = mid + 1; else hi = mid;
    }
    return lo;
}

// Project points onto every angle of the current chunk -- the (key, value)
// pair CUB then segment-sorts in place: key = projected position, value =
// original point index (so the sorted order can scatter the gradient back
// to the right point). 2D grid: x over diracs, y over the chunk's angles.
__global__ void project_kernel(
    const float*   __restrict__ points,        // [n, 2]
    const float*   __restrict__ normals_chunk, // [chunk_size, 2]
    int n,
    float*   __restrict__ proj,  // [chunk_size, n] out
    int64_t* __restrict__ idx    // [chunk_size, n] out
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int local_a = blockIdx.y;
    if (i < n) {
        const float* normal = normals_chunk + local_a * 2;
        int64_t base = (int64_t)local_a * n;
        proj[base + i] = points[i * 2] * normal[0] + points[i * 2 + 1] * normal[1];
        idx[base + i] = i;
    }
}

// Cost + grad for a whole chunk of angles, already segment-sorted. 2D grid:
// x sized to n (multi-block per angle, required once n is large -- see
// module docstring), y over the chunk's angles. Dirac at sorted rank k has
// quantile interval [k/n, (k+1)/n), so its barycenter/local-variance is an
// independent binary search into ITS angle's target CDF, no dependency on
// other diracs. `grad` is accumulated with atomics (scattered by
// `sorted_idx` across angles); `cost_out` (one angle's scalar) too, since
// multiple blocks cover one angle.
__global__ void ot_cost_grad_chunk_kernel(
    const float*   __restrict__ sorted_proj,   // [chunk_size, n]
    const int64_t* __restrict__ sorted_idx,    // [chunk_size, n]
    const float*   __restrict__ normals_chunk, // [chunk_size, 2]
    const double*  __restrict__ bin_mass,      // [chunk_size, m] -- float64, see below
    const double*  __restrict__ cum,           // [chunk_size, m]
    const double*  __restrict__ cum_start,     // [chunk_size, m]
    const double*  __restrict__ prefix_M,      // [chunk_size, m]
    const double*  __restrict__ prefix_M2,     // [chunk_size, m]
    const double*  __restrict__ bin_edges,     // [m + 1]
    int n, int m, double dw,
    float* __restrict__ grad,                  // [n, 2]
    double* __restrict__ cost_per_angle        // [chunk_size], pre-zeroed, float64 (see below)
) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int local_a = blockIdx.y;
    double w = 1.0 / n;
    double local_cost = 0.0;

    // `bary`/`M`/`M2` below are DOUBLE despite float32 inputs: at
    // convergence `s` and `bary` agree to several digits, so `s - bary` is
    // a near-cancellation -- in float32 its rounding error stops shrinking
    // once the true residual drops near float32's ~7-digit floor, so the
    // reported cost stops meaning anything below that floor (verified: an
    // exact-duplicate point split, which must leave the true loss EXACTLY
    // unchanged -- see `_split`'s docstring -- moved the float32 loss by
    // 50-140%, but the SAME computation in float64 left it bit-exact).
    // This is register-only scalar arithmetic (the bulk arrays stay
    // float32 for memory bandwidth), so the extra cost is negligible next
    // to the sort/searchsorted memory traffic this kernel is bound by.
    if (k < n) {
        const float   *normal    = normals_chunk + local_a * 2;
        const double  *row_mass  = bin_mass   + (size_t)local_a * m;
        const double  *row_cum   = cum        + (size_t)local_a * m;
        const double  *row_cs    = cum_start  + (size_t)local_a * m;
        const double  *row_pM    = prefix_M   + (size_t)local_a * m;
        const double  *row_pM2   = prefix_M2  + (size_t)local_a * m;
        const float   *row_proj  = sorted_proj + (size_t)local_a * n;
        const int64_t *row_idx   = sorted_idx  + (size_t)local_a * n;

        double q0 = k * w;
        double q1 = q0 + w;

        int j0 = min(searchsorted_right(row_cum, m, q0), m - 1);
        int j1 = min(searchsorted_right(row_cum, m, q1), m - 1);

        double bm0 = row_mass[j0], e0 = bin_edges[j0];
        double f0 = bm0 > 0.0 ? (q0 - row_cs[j0]) / bm0 : 0.0;
        double M0  = row_pM[j0]  + bm0 * (e0 * f0 + dw * f0 * f0 * 0.5);
        double M2_0 = row_pM2[j0] + bm0 * (e0 * e0 * f0 + e0 * f0 * f0 * dw
                                           + f0 * f0 * f0 * dw * dw / 3.0);

        double bm1 = row_mass[j1], e1 = bin_edges[j1];
        double f1 = bm1 > 0.0 ? (q1 - row_cs[j1]) / bm1 : 0.0;
        double M1  = row_pM[j1]  + bm1 * (e1 * f1 + dw * f1 * f1 * 0.5);
        double M2_1 = row_pM2[j1] + bm1 * (e1 * e1 * f1 + e1 * f1 * f1 * dw
                                           + f1 * f1 * f1 * dw * dw / 3.0);

        double bary = (M1 - M0) / w;
        double local_var = fmax((M2_1 - M2_0) - w * bary * bary, 0.0);

        double s = row_proj[k];
        double diff = s - bary;
        local_cost = w * diff * diff + local_var;

        double grad_s = 2.0 * w * diff;
        int64_t orig = row_idx[k];
        atomicAdd(&grad[orig * 2],     (float)(grad_s * normal[0]));
        atomicAdd(&grad[orig * 2 + 1], (float)(grad_s * normal[1]));
    }

    typedef cub::BlockReduce<double, THREADS> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    double block_sum = BlockReduce(temp_storage).Sum(local_cost);
    if (threadIdx.x == 0) atomicAdd(&cost_per_angle[local_a], block_sum);
}

std::vector<torch::Tensor> ot_cost_grad_cuda(
    torch::Tensor points, torch::Tensor normals,
    torch::Tensor bin_mass, torch::Tensor cum, torch::Tensor cum_start,
    torch::Tensor prefix_M, torch::Tensor prefix_M2, torch::Tensor bin_edges,
    double dw, int64_t mem_budget_bytes
) {
    int64_t n = points.size(0);
    int64_t A = normals.size(0);
    int64_t m = bin_mass.size(1);

    auto grad = torch::zeros({n, 2}, points.options());
    // float64: the cost's `s - bary` is a near-cancellation at convergence
    // (see `ot_cost_grad_chunk_kernel`'s comment) -- float32 accumulation
    // here would throw that precision away right after computing it.
    auto cost_per_angle = torch::zeros({A}, points.options().dtype(torch::kFloat64));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // chunk_size angles' worth of (proj, idx) double-buffered scratch costs
    // 2 (double-buffer) * (4 + 8) bytes per element -- pick the largest
    // chunk_size that fits mem_budget_bytes, clamped to [1, A]. Shrinks
    // toward 1 as n grows (bounded working memory), grows toward A when n
    // is small (fewer, bigger kernel launches -- see module docstring).
    int64_t chunk_size = std::max<int64_t>(1, mem_budget_bytes / (24 * std::max<int64_t>(n, 1)));
    chunk_size = std::min(chunk_size, A);

    auto proj_a = torch::empty({chunk_size * n}, points.options());
    auto proj_b = torch::empty({chunk_size * n}, points.options());
    auto idx_a = torch::empty({chunk_size * n}, points.options().dtype(torch::kInt64));
    auto idx_b = torch::empty({chunk_size * n}, points.options().dtype(torch::kInt64));

    cub::DoubleBuffer<float>   d_keys(proj_a.data_ptr<float>(), proj_b.data_ptr<float>());
    cub::DoubleBuffer<int64_t> d_values(idx_a.data_ptr<int64_t>(), idx_b.data_ptr<int64_t>());

    // Segment [begin, end) offsets for a FULL chunk_size-segment sort --
    // segment s is [s*n, (s+1)*n). A smaller last chunk (cs < chunk_size)
    // just uses the first cs+1 entries of this same array (still correct:
    // offsets[0..cs] = [0, n, ..., cs*n]).
    auto seg_offsets = (torch::arange(chunk_size + 1, points.options().dtype(torch::kInt32)) * (int)n);
    const int *seg_offsets_ptr = seg_offsets.data_ptr<int>();

    // cub::DeviceSegmentedRadixSort is DEVICE-WIDE (many blocks), unlike
    // cub::BlockRadixSort -- required once n is large: a single angle's
    // segment must scale past what one block's threads/shared memory can
    // hold. Sized for a FULL chunk (the largest case); reused as-is for a
    // smaller last chunk (needs no more temp storage than the max case).
    size_t temp_storage_bytes = 0;
    cub::DeviceSegmentedRadixSort::SortPairs(
        nullptr, temp_storage_bytes, d_keys, d_values,
        (int)(chunk_size * n), (int)chunk_size, seg_offsets_ptr, seg_offsets_ptr + 1,
        0, sizeof(float) * 8, stream);
    auto temp_storage = torch::empty(
        { (int64_t)temp_storage_bytes }, points.options().dtype(torch::kUInt8));

    const float *points_ptr  = points.data_ptr<float>();
    const float *normals_ptr = normals.data_ptr<float>();
    const double *bin_edges_ptr = bin_edges.data_ptr<double>();
    const double *bin_mass_ptr  = bin_mass.data_ptr<double>();
    const double *cum_ptr       = cum.data_ptr<double>();
    const double *cum_start_ptr = cum_start.data_ptr<double>();
    const double *prefix_M_ptr  = prefix_M.data_ptr<double>();
    const double *prefix_M2_ptr = prefix_M2.data_ptr<double>();
    float *grad_ptr = grad.data_ptr<float>();
    double *cost_per_angle_ptr = cost_per_angle.data_ptr<double>();

    const int proj_blocks = (int)((n + THREADS - 1) / THREADS);

    for (int64_t chunk_start = 0; chunk_start < A; chunk_start += chunk_size) {
        int64_t cs = std::min(chunk_size, A - chunk_start);
        dim3 grid(proj_blocks, (unsigned)cs);

        // Write this chunk's fresh (unsorted) projections into WHATEVER
        // buffer the DoubleBuffer currently considers "current" -- it may
        // have flipped after the previous chunk's sort, and doing this
        // avoids an extra copy every iteration.
        project_kernel<<<grid, THREADS, 0, stream>>>(
            points_ptr, normals_ptr + chunk_start * 2, (int)n,
            d_keys.Current(), d_values.Current());

        size_t tsb = temp_storage_bytes;
        cub::DeviceSegmentedRadixSort::SortPairs(
            temp_storage.data_ptr(), tsb, d_keys, d_values,
            (int)(cs * n), (int)cs, seg_offsets_ptr, seg_offsets_ptr + 1,
            0, sizeof(float) * 8, stream);

        ot_cost_grad_chunk_kernel<<<grid, THREADS, 0, stream>>>(
            d_keys.Current(), d_values.Current(), normals_ptr + chunk_start * 2,
            bin_mass_ptr + chunk_start * m, cum_ptr + chunk_start * m, cum_start_ptr + chunk_start * m,
            prefix_M_ptr + chunk_start * m, prefix_M2_ptr + chunk_start * m, bin_edges_ptr,
            (int)n, (int)m, dw,
            grad_ptr, cost_per_angle_ptr + chunk_start);
    }

    return {cost_per_angle, grad};
}
"""

_ext = None


def _load_extension():
    global _ext
    if _ext is None:
        _ext = torch.utils.cpp_extension.load_inline(
            name="unidim_ot_cuda",
            cpp_sources=_CPP_DECL,
            cuda_sources=_CUDA_SOURCE,
            functions=["ot_cost_grad_cuda"],
            with_cuda=True,
        )
    return _ext


def _cost_grad(points, sino, grad_timer=None, mem_budget_bytes=512 * 1024 * 1024):
    """(cost, grad [n, 2]) for `points` ([n, 2], CUDA float32) against
    `sino` -- projection, sort and the cost/grad reduction are ALL inside
    the fused kernel now (see module docstring); only the small [A, m]
    per-angle target-CDF prefix sums are plain `torch.cumsum` calls, since
    they're O(nb_angles * nb_bins), not O(nb_angles * nb_diracs).

    These prefix sums (and the kernel's own `bary` computation) are float64:
    `cum`/`prefix_M`/`prefix_M2` are CUMULATIVE sums over nb_bins terms, and
    with nb_bins in the thousands their float32 accumulation error alone is
    enough to noticeably move the reported cost (verified: fixing only the
    kernel's internal arithmetic, leaving these `torch.cumsum` calls in
    float32, cut a spurious 47% cost jump from an exact-duplicate point
    split down to 23% -- only computing THESE in float64 too closed the
    rest of the gap). `points`/`sorted_proj` stay float32: they're leaf
    values, not sums-of-thousands, so they carry no such accumulated error.
    """
    ext = _load_extension()
    g = sino.geometry
    device, dtype = points.device, points.dtype

    normals = torch.as_tensor(g.normals, dtype=dtype, device=device)
    bin_edges = torch.as_tensor(g.bin_edges, dtype=torch.float64, device=device)
    bin_mass = torch.as_tensor(sino.values, dtype=torch.float64, device=device)
    bin_mass = bin_mass / bin_mass.sum(dim=1, keepdim=True)

    dw = float(g.dw)
    bin_center = bin_edges[:-1] + dw / 2
    cum = torch.cumsum(bin_mass, dim=1)
    cum_start = cum - bin_mass
    prefix_M = torch.cumsum(bin_mass * bin_center, dim=1) - bin_mass * bin_center
    bin_second_moment = bin_mass * (bin_center ** 2 + dw * dw / 12)
    prefix_M2 = torch.cumsum(bin_second_moment, dim=1) - bin_second_moment

    if grad_timer is not None:
        torch.cuda.synchronize()
        t0 = time.time()
    cost_per_angle, grad = ext.ot_cost_grad_cuda(
        points.contiguous(), normals.contiguous(), bin_mass.contiguous(),
        cum.contiguous(), cum_start.contiguous(),
        prefix_M.contiguous(), prefix_M2.contiguous(), bin_edges.contiguous(),
        dw, mem_budget_bytes)
    if grad_timer is not None:
        torch.cuda.synchronize()
        grad_timer.record((time.time() - t0) * 1000)
    return cost_per_angle.sum(), grad


def loss(points, sino):
    """Same value as `reconstruction_jax.loss` -- cost only (see `_cost_grad`)."""
    cost, _ = _cost_grad(points, sino)
    return cost


def optimize(points, sino, max_iter=15, tracker=None, grad_timer=None, max_eval=25):
    """L-BFGS (`torch.optim.LBFGS`, strong-Wolfe) using the fused kernel's
    gradient directly -- no autograd graph, `points.grad` is assigned by
    hand each closure call, same one-shot cost+grad the kernel computes.

    `max_eval` MUST be set explicitly: `torch.optim.LBFGS` defaults it to
    `max_iter * 5 // 4`, and with the `max_iter=1` this function uses (one
    L-BFGS iteration per outer loop step, so `tracker`/`grad_timer` see
    every step) that default is 1 -- giving the strong-Wolfe line search a
    budget of `max_eval - 1 == 0` extra evaluations. With no room to
    bracket/zoom, it takes the raw, unchecked quasi-Newton step, which can
    corrupt the curvature memory (a bad (s, y) pair) and then STALL
    completely (identical loss/grad every step, zero displacement) for the
    rest of that call -- confirmed on lmo: multiscale's stage 1 froze at
    exactly this pattern until `max_eval` was set explicitly.
    """
    points = points.detach().clone().requires_grad_(True)
    opt = torch.optim.LBFGS([points], max_iter=1, max_eval=max_eval,
                            line_search_fn="strong_wolfe", history_size=10)

    def closure():
        opt.zero_grad()
        cost, grad = _cost_grad(points.detach(), sino, grad_timer=grad_timer)
        points.grad = grad
        return cost

    if tracker is not None:
        tracker.start()
    for i in range(max_iter):
        value = opt.step(closure)
        if tracker is not None:
            tracker.step(i, value, points.detach().cpu())
    return points.detach()


def _split(points, n, jitter):
    """Grow `points` to `n` rows by tiling (cyclic repeat) + jitter noise."""
    reps = -(-n // points.shape[0])  # ceil div
    tiled = points.repeat(reps, 1)[:n]
    return tiled + jitter * torch.randn(n, 2, device=points.device, dtype=points.dtype)


def multiscale_optimize(sino, nb_points_final, nb_points_init=200, factor=4,
                        seed=0, tracker=None, timings=None, device="cuda", **kwargs):
    """Same coarse-to-fine schedule as `reconstruction_jax.multiscale_optimize`.

    `timings`, if given a dict, is filled with `{n: mean_ms_per_grad_call}`
    per stage (and each stage's mean is printed) -- see `GradTimer`.
    """
    torch.manual_seed(seed)
    extent = sino.geometry.extent
    points = (torch.rand(nb_points_init, 2, device=device) - 0.5) * extent

    n = nb_points_init
    while True:
        grad_timer = GradTimer() if timings is not None else None
        points = optimize(points, sino, tracker=tracker, grad_timer=grad_timer, **kwargs)
        if grad_timer is not None:
            timings[n] = grad_timer.mean_ms
            print(f"  n={n:8d}: {grad_timer.mean_ms:.3f} ms/grad "
                  f"({len(grad_timer.times_ms)} calls)")
        if n >= nb_points_final:
            return points
        n = min(n * factor, nb_points_final)
        points = _split(points, n, jitter=sino.geometry.dw)


if p := bench( "multiscale_cuda", nb_diracs = Param( 100_000, help = "nb diracs" ) ):
    from .geometry import CtGeometry
    from .sinogram import Sinogram
    from .tracker import Tracker

    sino = Sinogram( CtGeometry( nb_angles = 600, nb_bins = 4096, extent = 2.0 ) )
    sino.add_disk( center = [ 0, 0 ], radius = 0.9, density = + 1.0 )
    sino.add_disk( center = [ 0, 0 ], radius = 0.7, density = - 1.0 )

    tracker = Tracker( record_frames = True )
    timings = {}
    points = multiscale_optimize( sino, nb_points_final = p.nb_diracs, tracker = tracker, timings = timings )
    p.results[ "ms_per_grad_by_n" ] = timings
    tracker.export_html( p.out_dir / "unidim_reconstruction_cuda.html", sino.geometry.extent )

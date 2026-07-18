---
layout: home

hero:
  name: SDOT
  text: Semi-Discrete Optimal Transport
  tagline: Fast · Differentiable · N-dimensional — works natively with JAX and PyTorch.
  image:
    src: /examples/img/patate.png
    alt: VitePress
  actions:
    - theme: brand
      text: Get Started
      link: /guide/getting-started
    - theme: alt
      text: Examples
      link: /examples/
    - theme: alt
      text: GitHub
      link: https://github.com/sdot-team/sdot

features:
  - icon: 📦
    title: Many target distributions
    details: SplineGrid, PolynomialGrid, SumOfGaussians, Mesh, and more — all supporting smoothing for faster convergence
  - icon: 🔗
    title: Differentiable end-to-end
    details: Use sdot as a differentiable layer inside JAX jit / grad or a PyTorch autograd graph.
  - icon: 📐
    title: Any number of dimensions
    details: 2D, 6D... — the same API and the same convergence guarantees.
  - icon: 🔄
    title: Flexible ground metrics
    details: Euclidean (default), periodic, entropic regularization, and custom radial kernels.
  - icon: 🚀
    title: GPU-ready
    details: CUDA, Metal, and SIMD-optimized cell construction. Batch variants for data-parallel workloads.
  - icon: ⚡
    title: O(n log n) — not O(n²)
    details: Fast newton solvers on power diagrams. Memory efficient. Parallel version (Dask)
---

## Hello, World

Here is a computation of the Wasserstein 2 distance between a sum of dirac and a spline grid:

```python
from sdot import SplineGrid, SumOfDiracs, distance
import numpy as np

f = SumOfDiracs( np.random.rand( 1000, 2 ) ) # 1000 equal-weight diracs in 2D
g = SplineGrid( np.random.rand( 10, 10 ) )   # random 10×10 spline density, C0 by default (auto-normalized)

print( distance( f, g ) ) # by default, used the 2 norm as the ground metric
```

If not already loaded, SDOT automatically picks the best available backend (JAX, then PyTorch) and device. [Configure it →](/guide/backends)

## What is Semi-Discrete Optimal Transport?

Given a discrete measure **f** (a weighted sum of Dirac masses) and a continuous density **g**, semi-discrete OT finds the _power diagram_ (Laguerre tessellation) that partitions space so that the mass of each cell matches the weight of the corresponding Dirac.

The solution is unique, and can be computed via a provably-convergent Newton algorithm in **O(n log n)** time — making it practical for very large number of points in 2D, 3D or more, and can easily achieves machine precision if required.

$$f = \sum_i m_i \, \delta_{y_i}, \qquad g = \text{continuous density on } \Omega$$

$$\text{Find } w \text{ such that } \int_{C_i(w)} g(x) \, dx = m_i \quad \forall i$$

> **Reference:** Kitagawa, Mérigot, Thibert —
> _Convergence of a Newton algorithm for semi-discrete optimal transport_, JEMS 2019.

## Extract quantities from OT plan

Of courses, sdot allows to access quantities of the transport plan.

```python
from sdot import SplineGrid, SumOfDiracs, optimal_transport_plan
import numpy as np

f = np.random.random( [ 30, 2 ] ) * 2 - 1 # dirac positions
g = Box( frame = [ [ 0, 0 ], [ 2, 0 ], [ 2, 0 ] ] )

plan = optimal_transport_plan( f, g )

print( plan.barycenters ) # centroid of each transport cell
print( plan.cells )       # batch of cells, with possible access to vertices, edges, ... with gradient enabled
print( plan.second_order_moments ) # ...

plan.backward_map.brenier_potential.plot()  # the dual potential ψ
```

![Computed Tomography Reconstruction](/examples/img/brenier.png){ width="200" style="display: block; margin: 0 auto"  }

### Use inside JAX/Torch

Virtually all the outputs can generated gradients. Here is an example with `distance`:

```python
@jax.jit
def loss( positions ):
    f = SumOfGaussians( positions )
    g = SumOfDiracs( diracs )
    return distance( f, g )

diracs = np.random.normal( [ 1, 2 ], 0.3, size = ( 50, 2 ) )
grad = jax.grad( loss )( jnp.array( [ 0.0, 0.0 ] ) )
```

Of course, it applies on all the other quantities (e.g. barycenters, cell vertices) with respect to all input quantities (images values, ...).

## Applications

SDOT has been used for:

- **Quantization & sampling** — optimal discretization of continuous densities
- **Meshing** — isotropic remeshing via Lloyd-like iterations
- **Machine learning** — fitting Gaussian mixture models, learning distributions
- **PDEs** — Fokker-Planck, incompressible Euler, crowd motion (Wasserstein gradient flows)
- **Registration** — matching point clouds to images or meshes

See the [Examples gallery →](/examples/)

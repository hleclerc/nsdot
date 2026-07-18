# Optimal Transport Plans

This tutorial builds intuition for what SDOT computes, how the Newton solver works, and what quantities you can extract from an OT plan.

---

## The Problem

Given:
- **f** — a discrete measure: a finite collection of weighted Dirac masses at positions $y_1, \ldots, y_n$ with masses $m_1, \ldots, m_n$ (with $\sum_i m_i = 1$)
- **g** — a continuous density on some domain $\Omega$

Find the **transport map** $T : \Omega \to \{y_1, \ldots, y_n\}$ that moves $g$ onto $f$ at minimum cost:

$$W_2^2(f, g) = \min_T \int_\Omega \|x - T(x)\|^2 \, g(x) \, dx \quad \text{s.t.} \quad \int_{T^{-1}(y_i)} g(x) \, dx = m_i \quad \forall i$$

---

## Power Diagrams (Laguerre Cells)

The optimal transport map in the semi-discrete setting is always a **power diagram** (also called a Laguerre tessellation): a partition of $\Omega$ into cells $C_i(w)$, each associated to a Dirac $y_i$, where membership is defined by a weighted distance:

$$x \in C_i(w) \iff \|x - y_i\|^2 - w_i \leq \|x - y_j\|^2 - w_j \quad \forall j \neq i$$

When all weights $w_i = 0$, this reduces to the standard Voronoi diagram. The weights "inflate" or "deflate" each cell to match the target mass.

<!-- INSERT IMAGE: power diagram with colored cells -->

The transport plan then simply sends every point $x$ in cell $C_i$ to position $y_i$.

---

## The Newton Solver

SDOT finds the optimal weights by solving:

$$\int_{C_i(w)} g(x) \, dx = m_i \quad \forall i$$

This is a smooth, strictly concave system — Newton's method converges in **a handful of iterations** in practice (typically 5–20). The key cost is computing the power diagram and its integrals at each step, which runs in **O(n log n)** time.

> **Reference:** Kitagawa, Mérigot, Thibert —
> _Convergence of a Newton algorithm for semi-discrete optimal transport_, JEMS 2019.

---

## Computing a Plan

```python
from sdot import SplineGrid, SumOfDiracs, optimal_transport_plan
import numpy as np

f = SumOfDiracs( np.random.rand( 300, 2 ) )
g = SplineGrid( np.random.rand( 10, 10 ) )

plan = optimal_transport_plan( f, g )
```

### Extracting quantities

```python
plan.distance          # W2^2 transport cost (scalar)
plan.barycenters       # centroid of each Laguerre cell — shape (n, d)
plan.cell_masses       # mass of each cell             — shape (n,)
plan.brenier_potential # dual potential psi            — shape (n,)
plan.power_diagram     # the underlying PowerDiagram object
```

The **Brenier potential** $\psi$ is the dual variable: $T(x) = x - \frac{1}{2}\nabla\psi(x)$, and the transport cost is the Legendre transform of $\psi$ integrated against $g$.

---

## The Wasserstein Distance

The scalar `plan.distance` equals:

$$W_2^2(f, g) = \int_\Omega \|x - T(x)\|^2 \, g(x) \, dx = \sum_i \int_{C_i} \|x - y_i\|^2 \, g(x) \, dx$$

It is also the negative of the dual objective at the optimal weights:

$$W_2^2(f, g) = \sum_i m_i w_i - \int_\Omega \max_i \Bigl( y_i \cdot x - \tfrac{1}{2} w_i \Bigr) g(x) \, dx$$

For a shortcut when you only need the distance:

```python
from sdot import distance

d = distance( f, g )   # same as plan.distance, but doesn't store the full plan
```

---

## Barycenters and the Lloyd Algorithm

The **barycenters** `plan.barycenters[i]` are the centroids of each transport cell:

$$b_i = \frac{1}{m_i} \int_{C_i(w)} x \, g(x) \, dx$$

Moving each Dirac to its barycenter and repeating gives the **Lloyd algorithm**, which converges to an optimal quantization of $g$:

```python
positions = np.random.rand( 200, 2 )

for _ in range( 30 ):
    f    = SumOfDiracs( positions )
    plan = optimal_transport_plan( f, g )
    positions = plan.barycenters          # Lloyd step
```

<!-- INSERT IMAGE: animation of Lloyd convergence -->

---

## Gradients

`plan.distance` (and `distance(f, g)`) are differentiable with respect to the Dirac positions and masses. The gradient with respect to positions $y_i$ is:

$$\frac{\partial W_2^2}{\partial y_i} = m_i \cdot (y_i - b_i)$$

where $b_i$ is the barycenter of cell $i$. This has a clean geometric interpretation: the gradient points from the barycenter toward the Dirac, and vanishes exactly when the Dirac is at its barycenter (i.e., at the Lloyd fixed point).

In SDOT, this gradient is computed automatically through the JAX/PyTorch autodiff:

```python
import jax

grad_positions = jax.grad( lambda pos: distance( SumOfDiracs( pos ), g ) )( positions )
# grad_positions[i] ~ masses[i] * (positions[i] - barycenters[i])
```

---

## What's Next

- [Tutorial: Fitting a Gaussian mixture →](/tutorials/gaussian-mixture-fit)
- [Example: Lloyd quantization →](/examples/lloyd-quantization)
- [Example: Image transport →](/examples/image-transport)
- [API: `optimal_transport_plan` →](/api/ot-plan)

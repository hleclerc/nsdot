# API — OT Plan

## `distance( f, g, metric=None )`

Computes the Wasserstein-2 squared distance between source `f` and target `g`.

Returns a scalar tensor in the active framework (JAX or PyTorch). Differentiable with respect to the positions and weights of `f`, and with respect to the parameters of `g` if supported.

**Parameters**

| Name | Type | Description |
|---|---|---|
| `f` | source distribution | `SumOfDiracs`, `SumOfWeightedDiracs`, or their batch/1d variants |
| `g` | target distribution | `SplineGrid`, `PolynomialGrid`, `SumOfGaussians`, … |
| `metric` | metric object or `None` | Ground metric (default: squared Euclidean `Norm2`) |

**Returns** — scalar: $W_2^2$(f, g)

```python
from sdot import SumOfDiracs, PolynomialGrid, distance
import numpy as np

d = distance(
    SumOfDiracs( np.random.rand( 200, 2 ) ),
    PolynomialGrid( values = [[[1]]] ),
)
```

---

## `optimal_transport_plan( f, g, metric=None )`

Computes the full OT plan. Returns an `OtPlan` object with all transport quantities.

**Parameters** — same as `distance`.

**Returns** — `OtPlan`

---

## `OtPlan`

The result of `optimal_transport_plan`.

| Attribute | Shape | Description |
|---|---|---|
| `.distance` | scalar | $W_2^2$ transport cost |
| `.barycenters` | `(n, d)` | Centroid of each Laguerre cell |
| `.cell_masses` | `(n,)` | Integral of g over each cell |
| `.brenier_potential` | `(n,)` | Dual variable ψ (optimal weights) |
| `.power_diagram` | `PowerDiagram` | Underlying power diagram object |

All attributes are tensors in the active framework. `.distance` is differentiable.

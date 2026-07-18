# Distributions

SDOT handles a wide variety of distributions. To handle the large number of combination, the library generates and compiles the appropriate C++ kernels at runtime.

---

**Source** and **target** can be either a discrete measure (a sum of Dirac masses) or a a continuous density.

All the axes are named. The axes sizes can be dynamic and depend of other axis variables.

For instance,

```python
@aggregate
class Image( Distribution ):
    values      : Tensor( "shape( dim )" )
    frame       : Tensor( "dim + 1", "dim" )
    knots       : Tensor( "nb_knots[ dim ]" )

img = Image( values = [ [ 1, 2 ] ], knots = [ [ 1, 5 ], [ 6 ] ] )
print( img.dim )   # -> 2
print( img.shape ) # -> [ 1, 2 ]
print( img.nb_knots ) # -> [ 2, 1 ]
```

## Variants and aggregate

All the distributions (as well as all the classes with intermediate quantities like `Cell`, `PowerDiagram`, ...) have variants
- `BatchOf` prepends a `batch_size` axis on each Tensor and DynamicSize.
- `1d` specifically sets `dim` axes variables and remove the axes that reduce to value 1.

Typically, `BatchOf` is useful for GPU-parallel workloads. Besides, it can be used to get a list of objects in a compact way. For instance `PowerDiagram.cells` actually return a `BatchOfCell`.

Here is an example of `BatchOf` usage:

```python
# Compute 64 distances in parallel
f = sdot.BatchOfSumOfDiracs( positions ) # shape (64, n, d)
g = sdot.BatchOfSplineGrid( values )     # shape (64, ...)

d = sdot.distances( f, g )               # shape (64,)
```

`1d` is useful to simplify the codes, because of the lower ranks tensors. Here is an example :

```python
from sdot import BatchOfSumOfDiracs, BatchOfSplineGrid, distance

# Compute 64 distances in parallel
f = sdot.PiecewiseAffine1d( [ 0, 1 ], knots = [ 2, 3 ] ) # knots is a single tensor
g = sdot.SumOfDiracs1d( [ 1, 2, 3 ] ) # position : rank 1 tensor. Equivalent to SumOfDiracs( [[1],[2],[3]] )
f = SumOfDiracs1d( [ 0.1, 0.5, 0.9 ] ) # equivalent SumOfDiracs( [ [0.1], [0.5], [0.9] ] )

d = sdot.barycenters( f, g ) # shape (3,) (and not (3,1))
```

## Masses

By default, all the distributions are normalized so that the mass is 1 (excepted when integrals are not finite)
- if the user specifies `mass = None`, nothing is done, the distribution is used as-is
- it is possible to fix a specific value, like `mass = 0.5`

If the source and target have different total masses, SDOT automatically performs **partial transport** — only the overlapping mass fraction is transported (see the `partial` tag in the [Examples gallery →](/examples/))


### Discrete distributions

| Class | Description |
|---|---|
| `SumOfDiracs( positions )` | Equal-weight Dirac masses at given positions |
| `SumOfWeightedDiracs( positions, weights )` | Dirac masses with explicit weights |

```python
# 500 uniform diracs in 2D
f = sdot.SumOfDiracs( np.random.rand( 500, 2 ) )

# 50 diracs with explicit masses (must sum to 1 for full transport)
positions = np.random.rand( 50, 2 )
weights = np.ones( 50 ) / 50
f2 = sdot.SumOfWeightedDiracs( positions, weights )
```

### Continuous distributions

| Class | Description |
|---|---|
| `SplineGrid( values, continuity=1 )` | C0, C1 or C2 spline density on a regular grid |
| `PolynomialGrid( values )` | Piecewise polynomial density on a regular grid |
| `Grid( values )` | Shortcut for PolynomialGrid with 1 coefficient |
| `SumOfGaussians( centers, covariances, weights )` | Mixture of Gaussians _(coming soon)_ |
| `PolynomialCells( ... )` | Sum of polynomials defined on a batch of cells _(coming soon)_ |
| `Cells( ... )` | Shortcut for `PolynomialCells` with 1 coefficient per polynomial _(coming soon)_ |
| `Mesh( vertices, values, elements )` | Density on a mesh, following the vtk conventions  _(coming soon)_ |


### `PolynomialGrid`

Piecewise polynomial density on a regular axis-aligned grid. Efficient for images and simple analytic densities.

```python
from sdot import PolynomialGrid
import numpy as np

# Uniform density on [0,1]²  (1×1 cell, constant value 1)
uniform = PolynomialGrid( values = [[[1]]] )

# From a grayscale image
img = np.load( "cat.npy" ).astype( float )
g   = PolynomialGrid( values = img )
```

| Argument | Type | Description |
|---|---|---|
| `values` | array `(n₁, …, nₐ)` | Cell values — one scalar per cell (Q₀ basis) |
| `domain` | `[[min…], [max…]]` | Bounding box (default: `[0,1]ᵈ`) |
| `mass` | float | Total mass (default: 1 — normalized automatically) |

### `SplineGrid`

C¹ or higher continuity spline density on a grid. Can be seen as a specialization of `PolynomialGrid`.

```python
from sdot import SplineGrid
import numpy as np

# 1D natural cubic spline from sampled values
s = SplineGrid( [ 1, 4, 9, 4, 1 ], continuity = 1 )

# 2D spline from random values
g = SplineGrid( np.random.rand( 10, 10 ), continuity = 1 )
```

| Argument | Type | Description |
|---|---|---|
| `values` | array | Spline control values on the grid |
| `continuity` | int | Smoothness order (1 = C¹, 2 = C², …) |
| `domain` | `[[min…], [max…]]` | Bounding box (default: `[0,1]ᵈ`) |
| `mass` | float | Total mass (default: 1) |

### `SumOfGaussians` _(coming soon)_

::: tip Coming soon
Mixture of Gaussians target. Enables analytic integrals over Laguerre cells and is particularly useful for learning problems.

```python
from sdot import SumOfGaussians

g = SumOfGaussians(
    centers     = centers,      # (k, d)
    covariances = covs,         # (k, d, d)
    weights     = weights,      # (k,)
)
```
:::

### `Mesh` _(coming soon)_

::: tip Coming soon
Density defined on a surface or volume mesh (triangles, tetrahedra, …). Supports non-uniform density fields on unstructured grids.
:::

### `PolynomialCells` _(coming soon)_

::: tip Coming soon
Piecewise polynomial density on a polyhedral cell decomposition — the most general target type.
:::


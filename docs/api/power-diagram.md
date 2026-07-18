# API — PowerDiagram

## `PowerDiagram( positions, weights=None )`

Constructs a power diagram (Laguerre tessellation) from a set of weighted sites.

```python
from sdot import PowerDiagram
import numpy as np

pd = PowerDiagram( positions = np.random.rand( 100, 2 ) )
pd = PowerDiagram( positions = np.random.rand( 100, 2 ), weights = np.zeros( 100 ) )
```

| Argument | Type | Description |
|---|---|---|
| `positions` | array `(n, d)` | Site positions |
| `weights` | array `(n,)` or `None` | Laguerre weights (default: 0 — standard Voronoi) |

---

## `PowerDiagram.cells`

Returns the collection of Laguerre cells. Iterate or index to access individual cells.

```python
cells = pd.cells

for cell in cells:
    print( cell.measure )
    print( cell.barycenter )
    print( cell.facets )
```

---

## `Cell`

A single Laguerre cell (convex polytope).

| Attribute / Method | Description |
|---|---|
| `.measure` | Volume / area of the cell |
| `.barycenter` | Centroid |
| `.facets` | List of facets (each with normal and offset) |
| `.simplex_decomposition` | Decomposition into simplices |
| `.cut( normal, offset )` | Cut the cell with a half-space |
| `.plot()` | Display the cell (2D/3D) |

---

## `PowerDiagram.adjust_weights( dirac_masses, target_distribution )`

Runs the Newton solver to find weights such that cell masses match `dirac_masses`.

```python
import numpy as np
from sdot import PowerDiagram, PolynomialGrid

pd     = PowerDiagram( positions = np.random.rand( 200, 2 ) )
target = PolynomialGrid( values = [[[1]]] )

pd.adjust_weights(
    dirac_masses        = np.ones( 200 ) / 200,
    target_distribution = target,
)
```

After this call, `pd.weights` contains the optimal Laguerre weights.

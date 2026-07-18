# Computing Plans

## Wasserstein distance

```python
from sdot import SplineGrid, SumOfDiracs, distance
import numpy as np

f = SumOfDiracs( np.random.rand( 200, 2 ) )
g = SplineGrid( np.random.rand( 8, 8 ) )

d = distance( f, g )
print( d )   # scalar — the W₂² transport cost
```

## Full OT plan

```python
from sdot import SplineGrid, SumOfDiracs, optimal_transport_plan
import numpy as np

f = SumOfDiracs( np.random.rand( 200, 2 ) )
g = SplineGrid( np.random.rand( 8, 8 ) )

plan = optimal_transport_plan( f, g )

plan.distance          # W₂² transport cost (scalar)
plan.barycenters       # centroid of each Laguerre cell  — shape (n, d)
plan.cell_masses       # mass of each cell               — shape (n,)
plan.brenier_potential # dual potential ψ                — shape (n,)
```

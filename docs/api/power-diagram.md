# API — PowerDiagram

## `PowerDiagram( positions, weights=None, boundaries=None, accelerator=None, distribution=None, kernel_dtype=None, scratch_capacity=None )`

Constructs a power diagram (Laguerre tessellation) from a set of weighted sites.

`PowerDiagram` is a VIEW: it holds the seeds and the domain, and rebuilds whatever is asked for,
cell by cell, in a per-work-item scratch. How the seeds are stored is the job of a specialization
picked at construction (`accelerator`):

| Class | Storage | When |
|---|---|---|
| `PowerDiagram_Bsp` | `sorted_positions` / `sorted_weights` in the order of an `AaBsp` tree (`tree`), one leaf read contiguously; each cell is only cut by the seeds the tree cannot rule out | the default, as soon as the positions are concrete |
| `PowerDiagram_Plain` | `positions` / `weights` as given; every cell is cut by the `n - 1` bisectors | `accelerator = "plain"`, or traced positions (no tree can be built on them) |

Whatever the storage, `pd.positions` / `pd.weights` read and write the seeds in the USER's order,
and `measures` / `cells` come out in that order too. Setting `pd.weights = w` on a BSP storage
re-sorts the weights and refreshes the per-node weight majorant in one kernel, without rebuilding
the tree -- this is what `OtPlan` does at every step. Setting `pd.positions` rebuilds the tree.

```python
from sdot import PowerDiagram, box_half_spaces
import numpy as np

pd = PowerDiagram( positions = np.random.rand( 100, 2 ), boundaries = box_half_spaces( [ 0, 0 ], [ 1, 1 ] ) )
pd = PowerDiagram( positions = np.random.rand( 100, 2 ), weights = np.zeros( 100 ) )
```

| Argument | Type | Description |
|---|---|---|
| `positions` | array `(n, d)` | Site positions |
| `weights` | array `(n,)` or `None` | Laguerre weights (default: none — standard Voronoi) |
| `boundaries` | `( directions, offsets )` or `None` | The convex domain `direction . x <= offset`; absent, boundary cells stay unbounded |
| `accelerator` | `None` (an `AaBsp` built here), an `AaBsp` built on these positions, or `"plain"` | Which seeds are worth trying — changes the cost, never the result |
| `distribution` | `Image`, `SumOfGaussians`, ... | What the measures integrate against (default: Lebesgue) |
| `kernel_dtype` | `"FP32"` (default) or `"FP64"` | The float the geometry is cut in; measures and gradients come out in the positions' float |
| `scratch_capacity` | int | How many vertices per cell the per-work-item scratch is sized for at first (doubled on overflow) |

---

## `PowerDiagram.measures`

The measure (area, volume, or the integral of `distribution`) of every cell, `[ n ]`. Differentiable
with respect to `positions`, `weights` and the distribution's values.

---

## `PowerDiagram.moments`

`( masses, first, second )`: for every cell, `∫ρ`, `∫xρ` (`[ n, d ]`) and `∫|x|²ρ` -- what a
transport cost and the cell barycenters are made of. Same sweep as `measures`; closed forms on
piecewise-constant pieces (`Image`, Lebesgue), the adaptive quadrature otherwise. Not
differentiable: `OtPlan.cost_and_position_grad` derives the transport cost by the envelope
theorem instead.

---

## `PowerDiagram.hessian_rows()`

`( nb_nbrs, ids, vals )`: for every cell, its neighbours and `∫_facet ρ / ( 2 |pᵢ − pⱼ| )` -- the
sparse Jacobian of the measures with respect to the weights (`∂mᵢ/∂wⱼ = −vals`,
`∂mᵢ/∂wᵢ = Σ vals`), which is the Hessian of the dual transport functional. Piecewise-constant
distributions only.

---

## `PowerDiagram.cells`

Every cell, as one `Cell` batched over the seeds — what a display needs.

```python
cells = pd.cells
cells.measure                      # [ n ]
cells.vertex_positions[ i ]        # the vertices of cell i, [ nv, d ] ( cyclic order in 2D )
cells.faces[ i ]                   # cycles of vertex indices, one per face ( 2D / 3D )
cells.add_to_viz( Visualizer() )
```

`PowerDiagram.cell( i )` builds one cell on the Python side, a call per cut — the oracle of the
tests, not a display path.

---

## `Cell`

A convex polytope. `Cell( d, ... )` builds `Cell_1` (a segment), `Cell_2` (a polygon, vertices in
cyclic order) or `Cell_N` (a simple polytope, every vertex naming its `d` cuts and `d` neighbours in
`vertex_cuts` / `vertex_nbrs`). What is stored is the practical form -- `vertex_positions
[ nb_vertices, d ]` in the caller's float, `cut_ids [ nb_cuts ]` -- and it is converted to the
kernel's form (in `kernel_dtype`, `FP32` by default) on every call. Everything else is derived on the host.

| Attribute / Method | Description |
|---|---|
| `.measure` | Area / volume of the cell (`TF::max` when unbounded) |
| `.vertex_positions` | `[ nb_vertices, d ]` ( a list of them for a batched cell ) |
| `.edges`, `.faces` | Vertex index pairs / cycles |
| `.cut_planes` | `( directions, offsets )`, outward unit normals read off the geometry |
| `.cut_ids` | Which seed each cut faces, or a negative id ( domain, piece, infinite wall ) |
| `.is_bounded` | Whether no `INFINITE` wall carries a vertex any more |
| `.cut( direction, offset )` | Cut the cell with a half-space, in place |
| `.add_to_viz( viz )` | Draw it ( 2D / 3D faces, H-representation beyond ) |

`Cell.make_hypercube( d, origin, axes )` and `Cell.make_unbounded( d )` are the two starting points.

---

## Solving for the weights: `OtPlan`

The diagram itself does not solve anything -- it is a view. `OtPlan( src_dist, dst_dist, ... )`
holds ONE `PowerDiagram` built on the diracs of `src_dist`, and adjusts its weights until the
measure of every cell against `dst_dist` matches its dirac's mass, by setting `pd.weights = w` at
every evaluation (the tree is built once).

```python
import numpy as np
from sdot import OtPlan, SumOfDiracs, SumOfGaussians

plan = OtPlan( SumOfDiracs( np.random.rand( 200, 2 ) ), SumOfGaussians( ... ) )
plan.weights                 # the adjusted weights, in the order of the diracs
plan.power_diagram()         # the diagram at those weights
plan.cell_masses             # ~ the target masses
plan.cost                    # W_2^2 at the adjusted weights
plan.cost_and_position_grad()   # ... and its derivative w.r.t. the dirac positions (envelope theorem)
```

`objective = "least_squares"` (default) minimizes `0.5 |measures( w ) - masses|²` with a barrier;
`objective = "dual"` minimizes the Kantorovich dual functional itself by L-BFGS (gradient = the
residual, value and gradient from one `moments` sweep, no barrier) -- exact on an `Image`, limited
to the quadrature tolerance on a smooth density; `objective = "newton"` does the same with the
damped Newton of Kitagawa–Mérigot–Thibert (`hessian_rows`, a sparse solve per step): a step count
independent of the number of diracs, what a large cloud needs.

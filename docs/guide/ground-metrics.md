# Ground Metrics

The _ground metric_ defines the cost of moving a unit mass from one location to another. By default SDOT uses the squared Euclidean distance (`Norm2`), but many other metrics are supported.

---

## Default: Squared Euclidean (`Norm2`)

```python
from sdot import SumOfDiracs, SplineGrid, distance

# Uses Norm2 by default — no need to specify
d = distance( SumOfDiracs( positions ), SplineGrid( values ) )
```

The cost between point $x$ and Dirac $y_i$ is $\|x - y_i\|^2$. This gives the standard **Wasserstein-2 distance** $W_2$.

---

## Periodic Metrics _(coming soon)_

::: tip Coming soon
Useful for periodic domains (torus topology), e.g., textures or angle spaces.

```python
from sdot.metrics import PeriodicMetric, Norm2

metric = PeriodicMetric( Norm2(), period = [ 1.0, 1.0 ] )  # [0,1]² torus

plan = optimal_transport_plan( f, g, metric = metric )
```
:::

---

## Entropic Regularization _(coming soon)_

::: tip Coming soon
Adds an entropy penalty to the transport cost, smoothing the plan. Useful for high-dimensional problems or when a soft assignment is preferable.

```python
from sdot.metrics import Entropy

metric = Entropy( epsilon = 0.01 )   # regularization strength
```
:::

---

## Radial Kernel Combinations _(coming soon)_

::: tip Coming soon
Custom ground metrics built from radial functions:

```python
from sdot.metrics import Norm2, Polynomial, pos_part

# Cost = pos_part( w - r² )  where r = ‖x - y‖
metric = pos_part( "w - r**2" )

# Polynomial cost
metric = Polynomial( degree = 4 )
```
:::

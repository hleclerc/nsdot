# Fitting a Gaussian Mixture via Wasserstein Distance

This tutorial shows how to use the Wasserstein-2 distance as a loss function to fit the parameters of a Gaussian mixture model (GMM) to a target density.

The key insight: the gradient of $W_2^2$ with respect to the Dirac positions $y_i$ is:

$$\frac{\partial W_2^2}{\partial y_i} = m_i \cdot (y_i - b_i)$$

where $b_i$ is the centroid of the $i$-th transport cell. This is computed automatically through JAX/PyTorch autodiff — no need to implement it manually.

---

## Setup

```python
import numpy as np
import jax
import jax.numpy as jnp
from sdot import PolynomialGrid, SumOfDiracs, distance

# Target: a two-peak Gaussian mixture, represented as a 64×64 image
t = jnp.linspace( 0, 1, 64 )
x, y = jnp.meshgrid( t, t )
target_img = (
    jnp.exp( -50 * ( ( x - 0.3 )**2 + ( y - 0.3 )**2 ) ) +
    jnp.exp( -50 * ( ( x - 0.7 )**2 + ( y - 0.7 )**2 ) )
)

target = PolynomialGrid( values = target_img )
```

---

## Minimizing with JAX

```python
@jax.jit
def loss( positions ):
    return distance( SumOfDiracs( positions ), target )

# Start from random positions
positions = jnp.array( np.random.rand( 200, 2 ) )

# Gradient descent
lr = 5e-3
for step in range( 200 ):
    val, grad = jax.value_and_grad( loss )( positions )
    positions = positions - lr * grad
    if step % 20 == 0:
        print( f"step {step:4d}   W₂² = {val:.6f}" )
```

After convergence, the Diracs are clustered around the two Gaussian peaks — they have solved the quantization problem for this mixture.

<!-- INSERT IMAGE: before / after positions overlaid on the target density -->

---

## Minimizing with PyTorch

```python
import torch
import sdot
from sdot import PolynomialGrid, SumOfDiracs, distance

sdot.driver.framework = "torch"

target_t  = torch.tensor( np.array( target_img ), dtype = torch.float64 )
target    = PolynomialGrid( values = target_t )

positions = torch.rand( 200, 2, dtype = torch.float64, requires_grad = True )
optimizer = torch.optim.Adam( [ positions ], lr = 5e-3 )

for step in range( 200 ):
    optimizer.zero_grad()
    val = distance( SumOfDiracs( positions ), target )
    val.backward()
    optimizer.step()
    if step % 20 == 0:
        print( f"step {step:4d}   W₂² = {val.item():.6f}" )
```

---

## Fitting GMM Parameters Directly

When the target is a `SumOfGaussians`, you can fit both positions and covariances in the same loop:

::: tip Coming soon
`SumOfGaussians` is not yet implemented. The example below shows the planned API.

```python
from sdot import SumOfGaussians, SumOfDiracs, distance
import jax, jax.numpy as jnp

@jax.jit
def loss( means, log_covs, log_weights ):
    covs    = jnp.exp( log_covs )
    weights = jax.nn.softmax( log_weights )
    target  = SumOfGaussians( means, covs, weights )
    return distance( SumOfDiracs( observed_points ), target )

grad_fn = jax.grad( loss, argnums = ( 0, 1, 2 ) )
```
:::

---

## Connection to Lloyd's Algorithm

The gradient step `positions -= lr * grad` with `lr = 1 / masses` is exactly the **Lloyd step**: it moves each Dirac to the centroid of its transport cell. Full Lloyd convergence (lr = 1/m) is fast but can overshoot; gradient descent with smaller lr is more stable when the positions are far from equilibrium.

---

## What's Next

- [Example: Lloyd quantization →](/examples/lloyd-quantization)
- [Tutorial: OT plans explained →](/tutorials/ot-plan-intro)
- [API: `distance` and `optimal_transport_plan` →](/api/ot-plan)

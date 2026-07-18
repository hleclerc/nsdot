# Differentiability

Almost all outputs are differentiable with respect to all outputs.

For instance `distance`, `barycenters`, ... are differentiable with respect to the Dirac positions and weights, the density parameters (image values, frame position, knots, ...), and so on.

The [examples gallery →](/examples/) contains a lot of example that are using differentiation (see tag `differentiation`).

## With JAX

```python
import jax
import jax.numpy as jnp
from sdot import PolynomialGrid, SumOfDiracs, distance

target = PolynomialGrid( values = jnp.ones( ( 10, 10 ) ) )

@jax.jit
def loss( positions ):
    return distance( SumOfDiracs( positions ), target )

positions = jnp.array( np.random.rand( 100, 2 ) )
grad = jax.grad( loss )( positions )   # shape (100, 2)
```

## With PyTorch

```python
import torch
import sdot

sdot.driver.framework = "torch"

from sdot import PolynomialGrid, SumOfWeightedDiracs, distance

target    = PolynomialGrid( values = torch.ones( 10, 10 ) )
positions = torch.rand( 100, 2, requires_grad = True )
weights   = torch.ones( 100 ) / 100

d = distance( SumOfWeightedDiracs( positions, weights ), target )
d.backward()

print( positions.grad )   # shape (100, 2)
```


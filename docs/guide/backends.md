# Backends (JAX / PyTorch)

SDOT abstracts JAX and PyTorch behind a single `driver` object. All tensors SDOT returns live in the active framework, on the active device, with the active dtype — so they slot directly into your computation graph.

---

## Auto-detection

By default SDOT inspects already-imported modules and picks a backend in this order:

1. `jax` if already imported
2. `torch` if already imported
3. First available library on the system (`jax` before `torch`)

```python
import jax                  # imports jax first
import sdot                 # → sdot.driver.framework == "jax"
```

```python
import torch                # imports torch first
import sdot                 # → sdot.driver.framework == "torch"
```

---

## Manual configuration

```python
import sdot

sdot.driver.framework = "jax"      # "jax" | "torch"
sdot.driver.dtype     = "FP64"     # "FP32" | "FP64"
sdot.driver.device    = "cuda:0"   # "cpu" | "cuda:N" | "mps"
```

Or via environment variables (must be set before launching Python):

```bash
export SDOT_FRAMEWORK=torch
export SDOT_DTYPE=FP32
export SDOT_DEVICE=cuda:0
```

---

## JAX integration

All SDOT operations are compatible with `jax.jit`, `jax.grad`, `jax.vmap`, and `jax.lax.scan`.

```python
import jax
import jax.numpy as jnp
import sdot
from sdot import PolynomialGrid, SumOfDiracs, distance

sdot.driver.framework = "jax"
sdot.driver.dtype     = "FP64"

target = PolynomialGrid( values = jnp.ones( ( 10, 10 ) ) )

@jax.jit
def loss( positions ):
    return distance( SumOfDiracs( positions ), target )

positions = jnp.array( np.random.rand( 200, 2 ) )

value = loss( positions )
grad  = jax.grad( loss )( positions )   # differentiates through the OT solver
```

### Batching with `jax.vmap`

```python
# Equivalent to BatchOfSumOfDiracs, but using vmap
loss_batch = jax.vmap( loss )
grads = jax.grad( lambda ps: loss_batch( ps ).sum() )( positions_batch )
```

---

## PyTorch integration

```python
import torch
import sdot
from sdot import PolynomialGrid, SumOfWeightedDiracs, distance

sdot.driver.framework = "torch"
sdot.driver.dtype     = "FP64"

target    = PolynomialGrid( values = torch.ones( 10, 10 ) )
positions = torch.rand( 200, 2, requires_grad = True )
weights   = torch.ones( 200 ) / 200

d = distance( SumOfWeightedDiracs( positions, weights ), target )
d.backward()

print( positions.grad )   # shape (200, 2)
```

### Inside a `nn.Module`

```python
import torch.nn as nn
from sdot import PolynomialGrid, SumOfDiracs, distance

class QuantizationLoss( nn.Module ):
    def __init__( self, target_values ):
        super().__init__()
        self.target = PolynomialGrid( values = target_values )

    def forward( self, positions ):
        return distance( SumOfDiracs( positions ), self.target )
```

---

## dtype and precision

| `dtype` | JAX equivalent | PyTorch equivalent |
|---|---|---|
| `"FP32"` | `jnp.float32` | `torch.float32` |
| `"FP64"` | `jnp.float64` | `torch.float64` |

::: warning FP64 on GPU
NVIDIA GPUs support FP64 but at reduced throughput. On Apple Silicon (`mps`), FP64 is not supported — use `FP32`.

The Newton solver in SDOT typically converges in a handful of iterations and is numerically stable in FP32, but FP64 is recommended when computing gradients or for research-grade accuracy.
:::

---

## Dask (large-scale) _(coming soon)_

::: tip Coming soon
A Dask backend is planned for datasets that exceed GPU memory, distributing the power diagram construction across workers.
:::

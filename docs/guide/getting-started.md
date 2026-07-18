# Getting Started

## Installation using pip

```bash
pip install sdot
```

SDOT requires at least one backend. Install the one you prefer (or both):

```bash
pip install jax jaxlib          # JAX backend
pip install torch               # PyTorch backend
```

> **GPU support** — follow the standard JAX/PyTorch GPU installation instructions for your platform (CUDA, Metal, …). SDOT will automatically use the device your tensors are on.

Internally, SDOT uses C++, Cuda and Metal, through dynamic libraries. The packages come bundled with the libraries for the most common case

For more more exotic cases, SDOT may have to generate and compile some code. In this case, one will need a compiler toolchain (SDOT uses `xmake` as build system).

## Installation from the sources

Alternatively, to participate or get the last version, one can use the sources
```
git clone git@github.com:sdot-team/sdot.git
cd sdot
pip install -r requirements.txt
pip install -e
```

## Hello, World

```python
import numpy
from sdot

f = sdot.SumOfDiracs( np.random.rand( 1000, 2 ) )   # 1000 equal-weight diracs, 2D
g = sdot.SplineGrid( np.random.rand( 10, 10 ) )      # 10×10 spline density (auto-normalized)

print( sdot.distance( f, g ) )
```

SDOT infers the backend, dimension, dtype (also with the int type), and device automatically.


## The Driver

SDOT abstracts the underlying framework (JAX or PyTorch) behind a single `driver` object. You don't have to import JAX or PyTorch directly in your code — everything can go through `sdot.driver`.

### Choice of the backend

SDOT looks for backend in this order:

1. Modules already imported in the current session (`jax`, `torch`)
2. The `SDOT_FRAMEWORK` environment variable
3. First available library (`jax`, then `torch`)

### Configuration

It is also possible to "force" the parameters, notably during runtime

```python
import sdot

sdot.driver.framework = "jax"      # or "torch"
sdot.driver.dtype     = "FP64"     # FP32, FP64
sdot.driver.device    = "cuda:0"   # or "cpu", "mps", …
```

Or via environment variables:

```bash
export SDOT_DTYPE=FP32
export SDOT_DEVICE=cuda:0
```

All arrays returned by SDOT live in the same framework, so they slot directly into your JAX/PyTorch computation graph.


## What's Next

- [Distributions in depth →](/guide/distributions)
- [Differentiability →](/guide/differentiability)
- [Ground metrics →](/guide/ground-metrics)
- [OT plans in depth →](/guide/distributions)
- [Backend configuration →](/guide/backends)
- [Mathematical background →](/tutorials/ot-plan-intro)
- [Examples gallery →](/examples/)

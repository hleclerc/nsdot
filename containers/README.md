# Container images

Reproducible Apptainer/Singularity images lock the build toolchain version chain
(AdaptiveCpp ↔ LLVM ↔ CUDA ↔ compiler ↔ libstdc++), rather than depending on the host toolkit.
Each image pre-builds AdaptiveCpp and its pinned Boost into `/opt/sdot-cache`
(`SDOT_CACHE_DIR`), so runtime compilation never has to rebuild the toolchain.

The nsdot monorepo has three packages installed as **editable** inside each container
(in dependency order: loom → sdot → otrec). No `PYTHONPATH` is needed; `import loom`,
`import sdot`, `import otrec` work directly.

JAX and PyTorch intentionally use **separate images**. Their CUDA pip wheels pin independent
`nvidia-*-cu12` stacks; a shared Python environment can silently replace one framework's tested
CUDA/cuDNN combination with the other's. This is a packaging conflict, not an incompatibility
between the JAX and PyTorch Python APIs.

| image | AdaptiveCpp profile | backends | LLVM | CUDA at build | framework at runtime |
|---|---|---|---:|---|---|
| `cpu.def` | minimal | — | — | — | CPU wheels |
| `cuda-jax.def` | full | cuda | 20 | 12.8 | JAX CUDA 12 pip wheels |
| `cuda-torch.def` | full | cuda | 20 | 12.8 | official PyTorch CUDA image |

AdaptiveCpp v25.10.0 supports LLVM through 20; LLVM 21+ requires its experimental override.
LLVM 20 in turn constrains the AdaptiveCpp CUDA build to CUDA 12.8 here. Both final images omit
the system CUDA toolkit and retain only `libdevice.10.bc` plus a framework-owned `libcudart`
fallback for AdaptiveCpp's PTX JIT. This prevents host or build-toolkit CUDA libraries from
shadowing the framework's pinned wheels.

## Building

### Via `./run` (recommended)

Declare environments in `.envs.py` (copy from `.envs.py.example`) with an `Apptainer`
layer:

```python
from loom.cli.layers import env, Driver, Apptainer, Remote

env("cuda-jax", [Apptainer(image="containers/cuda-jax.sif")] + [Driver("jax")])
env("cuda-torch", [Apptainer(image="containers/cuda-torch.sif")] + [Driver("torch")])
env("cpu", [Apptainer(image="containers/cpu.sif")] + [Driver("jax")])
```

Then build:

```bash
./run build-sif --env cuda-jax               # build a specific image, locally
./run build-sif                               # build every env with an Apptainer layer
./run build-sif --fakeroot --force            # force rebuild with fakeroot
./run build-sif --env lmo-cuda-jax            # build remotely (env whose seq starts with Remote)
./run build-sif --scratch-dir /data/tmp       # set scratch dir for large builds
```

The `.def` file is derived automatically from the `image` path (`containers/cuda-jax.sif` →
`containers/cuda-jax.def`). To build on a remote machine, add a `Remote` layer in front —
that's what makes `build-sif` rsync the repo there first, then run `apptainer build` on
the host:

```python
LMO = [Remote(host="lmo", remote_dir="/home/leclerc/nsdot",
              python="/data/venvs/sdot/bin/python",
              apptainer_scratch="/data/singularity_tmp")]

env("lmo-cuda-jax", LMO + [Apptainer(image="containers/cuda-jax.sif")] + [Driver("jax")])
```

`Remote.apptainer_scratch` points to a filesystem with enough free space (~30 GB);
`build-sif` uses it automatically for `APPTAINER_TMPDIR` and `APPTAINER_CACHEDIR`,
and `--scratch-dir` overrides it per invocation.

### From the command line

Build from the repository root, so `%files` paths resolve:

```bash
apptainer build --fakeroot containers/cpu.sif        containers/cpu.def
apptainer build --fakeroot containers/cuda-jax.sif   containers/cuda-jax.def
apptainer build --fakeroot containers/cuda-torch.sif containers/cuda-torch.def
```

### Disk space (important on HPC)

The build needs substantial transient scratch (the CUDA `devel` base plus CUDA pip wheels).
Point Apptainer's scratch and layer cache at a filesystem with roughly 30 GB free before building:

```bash
export APPTAINER_TMPDIR=/path/scratch/atmp
export APPTAINER_CACHEDIR=/path/scratch/acache
mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"
```

For the older `singularity` executable, use `SINGULARITY_TMPDIR` and
`SINGULARITY_CACHEDIR` instead.

## Running

```bash
# CPU
apptainer exec containers/cpu.sif python -m loom.cli test

# CUDA: --nv exposes the NVIDIA driver from the host.
apptainer exec --nv containers/cuda-jax.sif   env SDOT_DEVICE=cuda python -m loom.cli test
apptainer exec --nv containers/cuda-torch.sif env SDOT_DEVICE=cuda python -m loom.cli test
```

`--nvccli` is an alternative where the site enables NVIDIA Container Toolkit. Apptainer's
standard `--nv` binds the host driver libraries and GPU devices; the host therefore needs a
CUDA-12-compatible NVIDIA driver. Newer drivers are backward compatible, but AdaptiveCpp
generates PTX at runtime, so an old driver can still reject PTX introduced by a newer toolchain.

Apptainer auto-mounts `$HOME` and the current directory. Kernel artifacts land in the project's
host `build/` directory; the in-image `/opt/sdot-cache` is intentionally read-only at runtime.

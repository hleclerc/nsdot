# Container images

Two Apptainer/Singularity images, one per execution context. Neither builds a toolchain any
more: kernels compile at run time with the image's `g++` (and, for CUDA, the CUDA compiler that
`jax[cuda13]` ships in `site-packages/nvidia/cu13/bin`), through the `ninja` from pip. The three
packages of the monorepo are installed **editable** (loom → sdot → otrec); `import loom`,
`import sdot`, `import otrec` work directly.

| image | framework | GPU |
|---|---|---|
| `cpu.def` | JAX + PyTorch, CPU wheels | — |
| `cuda-jax.def` | `jax[cuda13]` (CUDA libraries + compiler from pip) | host driver bound by `--nv` / `--nvccli` |

The CUDA image has no system CUDA toolkit: nothing on the host can shadow the pip-pinned
libraries, and the image does not depend on the driver version (which must merely support
CUDA 13). A PyTorch CUDA image would be the same recipe with `torch` instead of `jax[cuda13]`;
the two frameworks stay in separate images because their pip wheels pin independent
`nvidia-*` stacks.

## Building

### Via `./run` (recommended)

Declare environments in `.envs.py` (copy from `.envs.py.example`) with an `Apptainer`
layer:

```python
from loom.cli.layers import env, Driver, Apptainer, Remote

env("cuda-jax", [Apptainer(image="containers/cuda-jax.sif")] + [Driver("jax")])
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

`Remote.apptainer_scratch` points to a filesystem with enough free space (a few GB);
`build-sif` uses it automatically for `APPTAINER_TMPDIR` and `APPTAINER_CACHEDIR`,
and `--scratch-dir` overrides it per invocation.

### From the command line

Build from the repository root, so `%files` paths resolve:

```bash
apptainer build --fakeroot containers/cpu.sif        containers/cpu.def
apptainer build --fakeroot containers/cuda-jax.sif   containers/cuda-jax.def
```

### Disk space (important on HPC)

The CUDA build needs transient scratch for the CUDA pip wheels (a few GB).
Point Apptainer's scratch and layer cache at a filesystem with enough free space before building:

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
apptainer exec --nv containers/cuda-jax.sif python -m loom.cli test --device cuda
```

`--nvccli` is an alternative where the site enables NVIDIA Container Toolkit. Apptainer's
standard `--nv` binds the host driver libraries and GPU devices; the host therefore needs a
CUDA-13-compatible NVIDIA driver. Kernels are compiled for the architecture of the card that is
present (`-arch=sm_XX`, read off the driver), so a newer driver is never a problem.

Apptainer auto-mounts `$HOME` and the current directory. Kernel artifacts land in the project's
host `build/` directory (a checkout) or in `~/.cache/sdot` (an installed wheel).

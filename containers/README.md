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

### Via `errand` (recommended)

Declare the environment in `errandfile.py`, at the repository root, with an `Apptainer` layer.
Everything the build needs is part of the declaration -- the recipe it is built from, whether it
is built with `--fakeroot`, and the scratch directory it unpacks into:

```python
CUDA_JAX = Apptainer(
    image    = "containers/cuda-jax.sif",
    recipe   = "containers/cuda-jax.def",
    flags    = [ "--nvccli" ],          # how the image is ENTERED
    fakeroot = True,                    # how it is BUILT
    scratch  = "/data/singularity_tmp",
)

env( "cuda-jax", [ CUDA_JAX ], driver = "jax", cuda = True )
```

Then:

```bash
errand --envs                          # `not built` / `ok` / `stale`, per environment
errand --setup --env cuda-jax          # build it if it is missing or out of date
errand --setup force --env cuda-jax    # build it again, whatever it says
errand --setup --dry-run --env cuda-jax   # say what it would run, and run nothing
```

**It is built by itself, when it needs to be.** `errand --env cuda-jax <...>` checks the image
against what the declaration says before running anything, and brings it up to date; `--no-setup`
is how you say not to. An image that is already there and that errand has never seen is *adopted*,
not rebuilt -- what somebody else made is not errand's to overwrite.

To build on another machine, put an `Ssh` layer in front. The repository is rsynced there first
(the recipe is a file *here*, the build happens *there*), then `apptainer build` runs on the host:

```python
env( "lmo-cuda-jax", [ Ssh( host = "lmo", root = "/home/leclerc/nsdot" ), CUDA_JAX ],
     driver = "jax", cuda = True )
```

`scratch` points at a filesystem with a few GB free; it is used for both `APPTAINER_TMPDIR` and
`APPTAINER_CACHEDIR`, which is where a build dies halfway when `/tmp` is small. What differs from
machine to machine -- the host name, that scratch directory -- is read from `errand.local.py`
(untracked) with a default that works here:

```python
from errand.local import value
SCRATCH = value( "apptainer_scratch", "/data/singularity_tmp" )
```

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
errand --env cpu                    # or: apptainer exec containers/cpu.sif errand

# CUDA: --nv exposes the NVIDIA driver from the host.
errand --env cuda-jax               # --nv / --nvccli come from the layer's `flags`
```

`--nvccli` is an alternative where the site enables NVIDIA Container Toolkit. Apptainer's
standard `--nv` binds the host driver libraries and GPU devices; the host therefore needs a
CUDA-13-compatible NVIDIA driver. Kernels are compiled for the architecture of the card that is
present (`-arch=sm_XX`, read off the driver), so a newer driver is never a problem.

Apptainer auto-mounts `$HOME` and the current directory. Kernel artifacts land in the project's
host `build/` directory (a checkout) or in `~/.cache/sdot` (an installed wheel).

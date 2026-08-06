# Container images

Reproducible Apptainer/Singularity images lock the build toolchain version chain
(AdaptiveCpp ↔ LLVM ↔ CUDA ↔ compiler ↔ libstdc++), rather than depending on the host toolkit.
Each image pre-builds AdaptiveCpp and its pinned Boost into `/opt/sdot-cache`
(`SDOT_CACHE_DIR`), so runtime compilation never has to rebuild the toolchain.

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
apptainer exec containers/cpu.sif python scripts/run_cpp_tests.py

# CUDA: --nv exposes the NVIDIA driver from the host.
apptainer exec --nv containers/cuda-jax.sif   env SDOT_DEVICE=cuda python scripts/run_tests.py
apptainer exec --nv containers/cuda-torch.sif env SDOT_DEVICE=cuda python scripts/run_tests.py
```

`--nvccli` is an alternative where the site enables NVIDIA Container Toolkit. Apptainer's
standard `--nv` binds the host driver libraries and GPU devices; the host therefore needs a
CUDA-12-compatible NVIDIA driver. Newer drivers are backward compatible, but AdaptiveCpp
generates PTX at runtime, so an old driver can still reject PTX introduced by a newer toolchain.

Apptainer auto-mounts `$HOME` and the current directory. Kernel artifacts land in the project's
host `build/` directory; the in-image `/opt/sdot-cache` is intentionally read-only at runtime.

# Container images

Reproducible Apptainer/Singularity images that lock the whole toolchain version chain
(AdaptiveCpp ↔ LLVM ↔ CUDA ↔ compiler ↔ libstdc++) so builds don't depend on whatever a host
happens to have installed. Each image **pre-builds AdaptiveCpp + a pinned Boost** into an
in-image cache (`/opt/sdot-cache`, exported as `SDOT_CACHE_DIR`) and ships **JAX + PyTorch**, so
at runtime the provider finds `acpp` ready and only compiles kernels.

| image       | AdaptiveCpp profile | backends | LLVM | CUDA  | JAX / Torch |
|-------------|---------------------|----------|------|-------|-------------|
| `cpu.def`   | minimal             | —        | none | —     | CPU wheels  |
| `cuda.def`  | full                | cuda     | 20   | 12.8  | CUDA wheels |

Version chain note: `clang` in LLVM ≤ 20 only supports CUDA ≤ 12.8, so the CUDA image pins a
12.8 base. The two knobs likely to need bumping live at the top of each `%post` (base image tag,
LLVM version, torch wheel channel, `SDOT_ACPP_VERSION`).

## Building

Build **from the repo root** (so the `%files` paths resolve):

```bash
apptainer build --fakeroot containers/cpu.sif  containers/cpu.def
apptainer build --fakeroot containers/cuda.sif containers/cuda.def
```

### Disk space (important on HPC)

The build needs a lot of *transient* scratch (CUDA `devel` base ~6 GB + JAX's `nvidia-cu12`
wheels several GB). The default scratch is `/tmp`, which is tiny on most nodes → you'll hit
`No space left on device`. Point Apptainer's scratch **and** layer cache at a big filesystem
*before* building:

```bash
df -h                                       # find a FS with ~30 GB free (/scratch, $WORK, …)
export APPTAINER_TMPDIR=/path/scratch/atmp   # rootfs of the build sandbox lives here
export APPTAINER_CACHEDIR=/path/scratch/acache
mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"
```

(Using the older `singularity` binary? set `SINGULARITY_TMPDIR` / `SINGULARITY_CACHEDIR`.)
Setting `TMPDIR` for `pip` alone does **not** help — pip writes inside the sandbox, which lives
in `APPTAINER_TMPDIR`.

## Running

```bash
# CPU
apptainer exec containers/cpu.sif python scripts/run_cpp_tests.py

# CUDA — --nv exposes the host GPU driver
apptainer exec --nv containers/cuda.sif python scripts/run_cpp_tests.py
```

Apptainer auto-mounts `$HOME` and the current dir, so the repo is visible inside and kernel
build artifacts land in the project's `build/` on the host (the in-image `/opt/sdot-cache` is
read-only at runtime, which is fine — `acpp` is already built there).

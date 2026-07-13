"""Acquisition & local build of AdaptiveCpp (the SYCL compiler driver `acpp`).

The compiled kernels of this project are built with AdaptiveCpp. To keep the distributed
package `pip install`-able and self-contained, we do *not* rely on a system-wide install:
AdaptiveCpp is built on demand into a per-user cache directory and reused across runs.

Compilation always goes through the `acpp` driver — it injects its own include paths and
links its runtime, so callers only have to point it at the right `--acpp-targets`.

Feature profiles
-----------------
* "minimal" : CPU only (OpenMP, `omp.library-only`). Needs only CMake + a C++ compiler.
              No LLVM. The portable default that works on every platform Jax/Torch support.
* "full"    : enables the runtime-adaptive generic JIT (SSCP) flow plus CUDA/HIP/Intel
              backends *when their toolchains are present*. Requires an official LLVM >= 15.
              We never build LLVM ourselves — it must be provided by the system/CI.

AdaptiveCpp has **no Metal backend**: on macOS only the CPU backends exist, so Apple-GPU
work is handled by a separate (non-acpp) path.
"""
from pathlib import Path
import urllib.request
import subprocess
import platform
import tarfile
import shutil
import sys
import os

from sdot.util.info import infox

# Pinned AdaptiveCpp release. Overridable for testing / bumping. A git ref (tag or branch)
# is also accepted: if no release tarball matches, we fall back to `git clone`.
# v25.10.0 supports LLVM 15..20 (24.10 predates LLVM 20 and fails to compile against it).
ACPP_VERSION = os.getenv( "SDOT_ACPP_VERSION", "v25.10.0" )
ACPP_REPO    = "https://github.com/AdaptiveCpp/AdaptiveCpp"

# Boost is a hard build/link dependency of AdaptiveCpp (for the library-only nd_range
# support). We build a pinned Boost locally rather than relying on a system install, so the
# exact version is identical on every machine / CI runner.
BOOST_VERSION = os.getenv( "SDOT_BOOST_VERSION", "1.86.0" )

# AdaptiveCpp needs Boost.Fiber, which (per its CMake config) transitively requires the
# compiled libraries below. They must all be present in our prefix at the exact version,
# otherwise find_package falls back to a (wrong-version) system Boost and fails.
BOOST_LIBS = ( "context", "fiber", "atomic", "filesystem" )

VALID_PROFILES = ( "minimal", "full" )

# GPU backends are driven *explicitly* (ON for those a device asks for, OFF for the rest)
# instead of letting AdaptiveCpp auto-enable whatever it finds: a stray OpenCL/Level-Zero on
# the machine would otherwise drag in SPIRV-Tools + OpenCL headers we neither need nor ship.
# Maps our backend name -> AdaptiveCpp CMake option.
ACPP_BACKEND_FLAGS = {
    "cuda"      : "WITH_CUDA_BACKEND",
    "rocm"      : "WITH_ROCM_BACKEND",
    "opencl"    : "WITH_OPENCL_BACKEND",
    "level_zero": "WITH_LEVEL_ZERO_BACKEND",
}


def _backends_tag( backends ) -> str:
    return "+".join( sorted( backends ) ) if backends else "cpu"


# ─────────────────────────── cache / install layout ──────────────────────────


def _is_writable_dir( p: Path ) -> bool:
    """True if *p* already exists and is writable, or can be created."""
    try:
        p.mkdir( parents = True, exist_ok = True )
        return os.access( p, os.W_OK )
    except OSError:
        return False


def _cache_candidates() -> list:
    """Ordered list of candidate cache roots (no writability check).

    SDOT_CACHE_DIR is the first candidate when set, but is NOT exclusive — the list always
    continues with platform-conventional paths and /tmp fallbacks.  This matters at runtime
    inside containers where SDOT_CACHE_DIR points to a read-only pre-built cache
    (/opt/sdot-cache): is_available() finds the pre-built binary there, while ensure_acpp()
    / ensure_boost() fall through to the first writable candidate for any new builds.
    """
    candidates = []

    override = os.getenv( "SDOT_CACHE_DIR" )
    if override:
        candidates.append( Path( override ).expanduser() )

    if sys.platform == "darwin":
        candidates.append( Path.home() / "Library" / "Caches" / "sdot" )
    elif os.name == "nt":
        base = os.getenv( "LOCALAPPDATA" ) or str( Path.home() / "AppData" / "Local" )
        candidates.append( Path( base ) / "sdot" / "cache" )
    else:
        # Linux / other unix: XDG base dir spec.
        xdg = os.getenv( "XDG_CACHE_HOME" )
        base = Path( xdg ) if xdg else Path.home() / ".cache"
        candidates.append( base / "sdot" )

    # POSIX fallbacks: /tmp is almost always writable and is the last resort.
    if os.name != "nt":
        uid = os.getuid() if hasattr( os, "getuid" ) else "shared"
        candidates.append( Path( "/tmp" ) / f"sdot-cache-{ uid }" )
        candidates.append( Path( "/tmp" ) / "sdot-cache" )
    return candidates


def cache_root() -> Path:
    """First writable cache directory.

    Iterates _cache_candidates() and returns the first one that can be created and written
    to.  Use this when a *new* build is needed.  For locating an *existing* build that may
    live in a read-only candidate (e.g. a container's /opt/sdot-cache), use the search
    helpers acpp_prefix() / boost_prefix() directly — they scan all candidates.
    """
    for p in _cache_candidates():
        if _is_writable_dir( p ):
            return p
    raise RuntimeError(
        "sdot: could not find a writable cache directory. "
        "Set SDOT_CACHE_DIR to an explicit writable path."
    )


def _check_profile( profile: str ):
    if profile not in VALID_PROFILES:
        raise ValueError( f"unknown AdaptiveCpp profile { profile!r} (expected one of { VALID_PROFILES })" )


def acpp_prefix( profile: str = "minimal", backends = () ) -> Path:
    """Locate (or plan) the install prefix for a given acpp configuration.

    Scans all candidate cache roots for an existing build first — this lets a read-only
    pre-built cache (e.g. a container's /opt/sdot-cache) be found even when it is not
    writable.  Falls back to cache_root() (first writable candidate) when not found anywhere,
    so ensure_acpp() has a place to write the new build.

    Keyed by version + profile + backends + Boost + arch so several configurations can
    coexist without clashing.  Boost version matters because acpp bakes absolute Boost lib
    paths at build time.
    """
    _check_profile( profile )
    tag = f"{ ACPP_VERSION }-{ profile }-{ _backends_tag( backends ) }-boost{ BOOST_VERSION }-{ platform.machine() }"
    subdir = Path( "adaptivecpp" ) / tag
    for root in _cache_candidates():
        candidate = root / subdir
        if ( candidate / "bin" / "acpp" ).is_file():
            return candidate
    return cache_root() / subdir


def acpp_path( profile: str = "minimal", backends = () ) -> Path:
    """Path to the `acpp` driver for a given configuration (whether or not it exists yet)."""
    return acpp_prefix( profile, backends ) / "bin" / "acpp"


def is_available( profile: str = "minimal", backends = () ) -> bool:
    """True if `acpp` for this configuration is already built and runnable."""
    p = acpp_path( profile, backends )
    infox( p )
    return p.is_file() and os.access( p, os.X_OK )


# ──────────────────────────────── build tools ────────────────────────────────


def _cxx_compiler() -> str | None:
    """Resolve a C++ compiler: $CXX, then common names."""
    cxx = os.getenv( "CXX" )
    if cxx and ( shutil.which( cxx ) or Path( cxx ).is_file() ):
        return cxx
    for name in ( "c++", "clang++", "g++" ):
        if shutil.which( name ):
            return name
    return None


def _find_llvm_config() -> str | None:
    """Locate `llvm-config`, including the versioned names distros ship (e.g. `llvm-config-20`).

    Honours `LLVM_CONFIG`. Newest versions are preferred. AdaptiveCpp's 'full' profile needs
    LLVM >= 15, so we don't look below that.
    """
    override = os.getenv( "LLVM_CONFIG" )
    if override and ( shutil.which( override ) or Path( override ).is_file() ):
        return override
    names = [ "llvm-config" ] + [ f"llvm-config-{ v }" for v in range( 21, 14, -1 ) ]
    for name in names:
        p = shutil.which( name )
        if p:
            return p
    return None


def _llvm_has_clang_dev( llvm_config: str ) -> bool:
    """True if the Clang *development* headers sit next to this LLVM.

    AdaptiveCpp's 'full' profile builds a Clang plugin and needs `clang/AST/ASTContext.h` etc.
    On Debian these come from a separate package (`libclang-NN-dev`/`clang-NN`), so LLVM can be
    present while Clang dev is not.
    """
    try:
        r = subprocess.run( [ llvm_config, "--includedir" ], capture_output = True, text = True )
        if r.returncode == 0 and r.stdout.strip():
            return ( Path( r.stdout.strip() ) / "clang" / "AST" / "ASTContext.h" ).is_file()
    except Exception:
        pass
    return False


def _linux_distro_tags() -> str:
    """`ID` + `ID_LIKE` from /etc/os-release, lower-cased (e.g. 'ubuntu debian'). '' elsewhere."""
    try:
        data = Path( "/etc/os-release" ).read_text()
    except OSError:
        return ""
    fields = {}
    for line in data.splitlines():
        k, sep, v = line.partition( "=" )
        if sep:
            fields[ k ] = v.strip().strip( '"' )
    return ( fields.get( "ID", "" ) + " " + fields.get( "ID_LIKE", "" ) ).lower()


def _llvm_install_hint() -> str:
    """OS-aware, actionable suggestions for installing the LLVM+Clang the 'full' profile needs."""
    lines = [
        "  AdaptiveCpp's 'full' profile (CUDA/HIP/Intel + generic JIT) needs the LLVM & Clang",
        "  *development* packages (>= 15). We never build LLVM — install it, then re-run. Options:",
    ]
    tags = _linux_distro_tags() if sys.platform.startswith( "linux" ) else ""

    if sys.platform == "darwin":
        lines.append( "    • Homebrew:  brew install llvm   (then put its bin/ on PATH so llvm-config resolves)" )
    elif "debian" in tags or "ubuntu" in tags:
        lines += [
            "    • Debian/Ubuntu:  sudo apt install llvm-20-dev libclang-20-dev clang-20 libomp-20-dev",
            "      (adjust the version to what your distro ships; check `apt-cache search llvm-`)",
            "    • newer than the distro provides → official repo https://apt.llvm.org :",
            "        wget https://apt.llvm.org/llvm.sh && chmod +x llvm.sh && sudo ./llvm.sh 20 all",
        ]
    elif "fedora" in tags or "rhel" in tags or "centos" in tags:
        lines.append( "    • Fedora/RHEL:  sudo dnf install llvm-devel clang-devel libomp-devel" )
    elif "arch" in tags:
        lines.append( "    • Arch:  sudo pacman -S llvm clang openmp" )

    lines += [
        "    • Any platform (conda/mamba):  micromamba install -c conda-forge llvmdev clangdev",
        "    • Already installed elsewhere? set LLVM_CONFIG=/path/to/llvm-config (or add its bin/ to PATH).",
    ]
    return "\n".join( lines )


def _cuda_toolkit_root() -> str | None:
    """Locate a CUDA toolkit usable by AdaptiveCpp.

    AdaptiveCpp requires a *unified* toolkit root that contains `nvvm/` (libdevice) — the layout
    NVIDIA's installer / conda's `cuda-toolkit` produce. It explicitly rejects roots without
    `nvvm/` (see its cmake/FindCUDA.cmake). Distro packages like Ubuntu's `nvidia-cuda-toolkit`
    scatter files across /usr without such a root, so headers alone (`/usr/include/cuda_runtime.h`)
    are NOT enough — we must find a real root.
    """
    candidates = []
    for env in ( "CUDA_HOME", "CUDA_PATH", "CUDA_ROOT" ):
        v = os.getenv( env )
        if v:
            candidates.append( Path( v ) )
    nvcc = shutil.which( "nvcc" )
    if nvcc:
        candidates.append( Path( nvcc ).resolve().parents[ 1 ] )
    candidates += [ Path( "/usr/local/cuda" ), Path( "/usr/lib/cuda" ) ]

    for root in candidates:
        headers = ( root / "include" / "cuda_runtime.h" ).is_file() or ( root / "include" / "cuda.h" ).is_file()
        # `bin/nvcc` inside the root matters: CMake's FindCUDAToolkit roots itself on nvcc's
        # location, so a root without its own nvcc (e.g. Ubuntu's /usr/lib/cuda, whose nvcc lives
        # in /usr/bin) makes CMake fall back to /usr and miss nvvm — exactly the failure we guard.
        complete = headers and ( root / "nvvm" ).is_dir() and ( root / "bin" / "nvcc" ).is_file()
        if complete:
            return str( root )
    return None


def _cuda_install_hint() -> str:
    """Suggestions for getting a CUDA toolkit AdaptiveCpp can actually use."""
    lines = [
        "  AdaptiveCpp's CUDA backend needs a *unified* CUDA toolkit whose root contains `nvvm/`",
        "  (libdevice) — the layout NVIDIA's installer puts at /usr/local/cuda.",
        "  ⚠ Distro packages such as Ubuntu's `nvidia-cuda-toolkit` scatter files across /usr",
        "    without that root (no `<root>/nvvm`) and therefore CANNOT be used here. Options:",
    ]
    tags = _linux_distro_tags() if sys.platform.startswith( "linux" ) else ""
    if "debian" in tags or "ubuntu" in tags or "fedora" in tags or "rhel" in tags or "centos" in tags:
        lines.append( "    • Official toolkit (recommended): https://developer.nvidia.com/cuda-downloads" )
        lines.append( "      — the .run/.deb/.rpm installer creates /usr/local/cuda (with nvvm/)." )
    lines += [
        "    • Any platform (conda/mamba):  micromamba install -c nvidia 'cuda-toolkit=12.*'",
        "      then  export CUDA_HOME=$CONDA_PREFIX   (must contain include/ and nvvm/)",
        "    • Already have a real toolkit? set CUDA_HOME=/usr/local/cuda (or wherever nvvm/ lives).",
        "  ⚠ Version: clang in LLVM ≤ 20 only supports CUDA ≤ 12.8. CUDA 13 breaks at compile time",
        "    (`fatbinary: Unknown option '-image'`). Stick to a CUDA 12.x toolkit.",
    ]
    return "\n".join( lines )


def _check_build_tools( profile: str, backends = () ):
    """Raise a clear, actionable error if anything required to build AdaptiveCpp is missing.

    Build tools are only needed to *acquire* AdaptiveCpp (typically once, in CI or on a
    developer machine); the distributed package itself does not require them.
    """
    missing = []
    blocks = []

    if not shutil.which( "cmake" ):
        missing.append( "cmake (>=3.x)" )
    if _cxx_compiler() is None:
        missing.append( "a C++ compiler (set $CXX, or install clang++/g++)" )

    if profile == "full":
        llvm_config = _find_llvm_config()
        if llvm_config is None:
            missing.append( "LLVM/Clang dev packages (>= 15) — required by the 'full' profile" )
            blocks.append( _llvm_install_hint() )
        elif not _llvm_has_clang_dev( llvm_config ):
            missing.append( "Clang development headers (clang/AST/…) — LLVM found, but Clang dev is missing" )
            blocks.append( _llvm_install_hint() )

    if "cuda" in backends and _cuda_toolkit_root() is None:
        missing.append( "CUDA toolkit — required to build the 'cuda' backend" )
        blocks.append( _cuda_install_hint() )

    if missing:
        hint = ""
        if sys.platform == "darwin":
            hint = "\n  on macOS: `brew install cmake ninja libomp` (the OpenMP backend needs libomp)."
        raise RuntimeError(
            "cannot build AdaptiveCpp — missing build prerequisites:\n  - "
            + "\n  - ".join( missing )
            + hint
            + ( "\n\n" + "\n\n".join( blocks ) if blocks else "" )
        )


# ──────────────────────────────── source fetch ───────────────────────────────


def _fetch_source( version: str, work: Path ) -> Path:
    """Return a source tree for `version`, downloading a release tarball (preferred) or
    falling back to a shallow `git clone` for branches / when the tarball is unavailable.

    The extracted tree is cached under `work`; a `.ok` marker avoids re-fetching.
    """
    work.mkdir( parents = True, exist_ok = True )
    dest = work / f"src-{ version }"
    marker = dest.parent / f"src-{ version }.ok"
    if marker.is_file() and dest.is_dir():
        return dest

    if dest.exists():
        shutil.rmtree( dest )

    # GitHub strips a leading 'v' from the top-level directory inside the tarball.
    tar_name = version[ 1: ] if version.startswith( "v" ) else version
    url = f"{ ACPP_REPO }/archive/refs/tags/{ version }.tar.gz"
    fetched = False
    try:
        print( f"[acpp] downloading { url }", flush = True )
        tmp_tar = work / f"{ version }.tar.gz"
        with urllib.request.urlopen( url ) as resp, open( tmp_tar, "wb" ) as f:
            shutil.copyfileobj( resp, f )
        with tarfile.open( tmp_tar ) as tf:
            tf.extractall( work )
        tmp_tar.unlink()
        extracted = work / f"AdaptiveCpp-{ tar_name }"
        if extracted.is_dir():
            extracted.rename( dest )
            fetched = True
    except Exception as e:
        print( f"[acpp] tarball fetch failed ({ e }); falling back to git", flush = True )

    if not fetched:
        if not shutil.which( "git" ):
            raise RuntimeError( f"cannot fetch AdaptiveCpp { version }: tarball unavailable and git not found" )
        print( f"[acpp] git clone { ACPP_REPO } @ { version }", flush = True )
        subprocess.run(
            [ "git", "clone", "--depth", "1", "--branch", version, ACPP_REPO, str( dest ) ],
            check = True,
        )

    marker.write_text( "ok" )
    return dest


# ──────────────────────────────── boost (local) ──────────────────────────────


def boost_prefix() -> Path:
    """Locate (or plan) the install prefix for the locally-built Boost.

    Scans all candidate cache roots for an existing build first (same read-only-cache
    logic as acpp_prefix).  Falls back to cache_root() when not found.
    """
    subdir = Path( "boost" ) / f"{ BOOST_VERSION }-{ platform.machine() }"
    sentinel = Path( "include" ) / "boost" / "version.hpp"
    for root in _cache_candidates():
        candidate = root / subdir
        if ( candidate / sentinel ).is_file():
            return candidate
    return cache_root() / subdir


def boost_is_available() -> bool:
    p = boost_prefix()
    if not ( p / "include" / "boost" / "version.hpp" ).is_file():
        return False
    # every required component must have shipped its CMake config, or find_package fails
    return all( ( p / "lib" / "cmake" / f"boost_{ lib }-{ BOOST_VERSION }" ).is_dir() for lib in BOOST_LIBS )


def ensure_boost( *, force: bool = False ) -> Path:
    """Build a pinned Boost (context + fiber) into the cache and return its prefix.

    Uses Boost's own `b2` build (`bootstrap.sh`), producing a CMake package config so
    AdaptiveCpp's `find_package(Boost CONFIG)` picks it up via BOOST_ROOT. POSIX only for
    now (macOS / Linux — the platforms we target); Windows would need `bootstrap.bat`.
    """
    prefix = boost_prefix()
    if boost_is_available() and not force:
        return prefix

    if os.name == "nt":
        raise RuntimeError( "local Boost build is not implemented on Windows yet (use a system Boost)" )
    if _cxx_compiler() is None:
        raise RuntimeError( "a C++ compiler is required to build Boost (set $CXX or install clang++/g++)" )

    if force and prefix.exists():
        shutil.rmtree( prefix )

    work = cache_root() / "boost" / "_work"
    underscore = BOOST_VERSION.replace( ".", "_" )
    url = f"https://archives.boost.io/release/{ BOOST_VERSION }/source/boost_{ underscore }.tar.gz"

    work.mkdir( parents = True, exist_ok = True )
    src = work / f"boost_{ underscore }"
    if not ( src / "bootstrap.sh" ).is_file():
        if src.exists():
            shutil.rmtree( src )
        print( f"[boost] downloading { url }", flush = True )
        tmp_tar = work / f"boost_{ underscore }.tar.gz"
        with urllib.request.urlopen( url ) as resp, open( tmp_tar, "wb" ) as f:
            shutil.copyfileobj( resp, f )
        with tarfile.open( tmp_tar ) as tf:
            tf.extractall( work )
        tmp_tar.unlink()

    jobs = str( os.cpu_count() or 1 )
    with_flags = [ f"--with-{ lib }" for lib in BOOST_LIBS ]
    _run( [ "./bootstrap.sh", f"--prefix={ prefix }", f"--with-libraries={ ','.join( BOOST_LIBS ) }" ], cwd = src )
    _run( [ "./b2", "install",
            f"--prefix={ prefix }",
            *with_flags,
            "link=shared", "threading=multi", "variant=release",
            "cxxflags=-fPIC", f"-j{ jobs }" ], cwd = src )

    if not boost_is_available():
        raise RuntimeError( f"Boost build finished but { prefix } looks incomplete" )
    return prefix


# ─────────────────────────────────── build ───────────────────────────────────


def _env_flag( name: str ) -> bool:
    """Boolean env var, treating "0"/"false"/"no"/"off"/"" (any case) as False.

    Plain `os.getenv(name)` truthiness is wrong here: the string "0" is truthy in Python, so
    `SDOT_FORCE_BUILD=0` would still read as "force". This parses the value instead.
    """
    v = os.getenv( name )
    if v is None:
        return False
    return v.strip().lower() not in ( "", "0", "false", "no", "off" )


def _run( cmd, **kw ):
    print( "[acpp] $ " + " ".join( map( str, cmd ) ), flush = True )
    r = subprocess.run( list( map( str, cmd ) ), **kw )
    if r.returncode:
        raise RuntimeError( f"command failed ({ r.returncode }): { ' '.join( map( str, cmd ) ) }" )


def _cmake_build( src: Path, prefix: Path, profile: str, boost_root: Path, backends = () ):
    # Build dir lives *inside* the version-scoped source tree, so different AdaptiveCpp versions
    # never share a (poisoned) CMakeCache. Still keyed by profile/backends/boost/arch.
    build = src / f"build-{ profile }-{ _backends_tag( backends ) }-boost{ BOOST_VERSION }-{ platform.machine() }"
    build.mkdir( parents = True, exist_ok = True )

    generator = [ "-G", "Ninja" ] if shutil.which( "ninja" ) else []
    cxx = _cxx_compiler()

    config = [
        "cmake", "-S", src, "-B", build, *generator,
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DCMAKE_INSTALL_PREFIX={ prefix }",
        f"-DACPP_COMPILER_FEATURE_PROFILE={ profile }",
        # force our pinned Boost: CMAKE_PREFIX_PATH gives it precedence in CONFIG-mode search
        # (BOOST_ROOT is neutered by policy CMP0144 on recent CMake). The components are
        # installed at the exact version, so any system Boost is rejected as version-mismatched.
        f"-DCMAKE_PREFIX_PATH={ boost_root }",
        "-DBoost_NO_SYSTEM_PATHS=ON",
    ]

    # Drive backends explicitly (don't let AdaptiveCpp auto-enable whatever is installed):
    # ON for the ones requested, OFF for all others — this keeps an unrelated OpenCL/L0 on the
    # box from dragging in SPIRV-Tools + OpenCL headers.
    for name, flag in ACPP_BACKEND_FLAGS.items():
        config.append( f"-D{ flag }={ 'ON' if name in backends else 'OFF' }" )

    if cxx:
        config.append( f"-DCMAKE_CXX_COMPILER={ cxx }" )

    # point the 'full' profile at the installed LLVM (incl. versioned/non-default locations)
    if profile == "full":
        llvm_config = _find_llvm_config()
        if llvm_config:
            r = subprocess.run( [ llvm_config, "--cmakedir" ], capture_output = True, text = True )
            if r.returncode == 0 and r.stdout.strip():
                config.append( f"-DLLVM_DIR={ r.stdout.strip() }" )

    if "cuda" in backends:
        cuda_root = _cuda_toolkit_root()
        if cuda_root:
            config.append( f"-DCUDA_TOOLKIT_ROOT_DIR={ cuda_root }" )

    _run( config )
    _run( [ "cmake", "--build", build, "--parallel" ] )
    _run( [ "cmake", "--build", build, "--target", "install" ] )


def ensure_acpp( profile: str = "minimal", backends = (), *, force: bool = False ) -> Path:
    """Return the path to a usable `acpp` for (profile, backends), building it on demand.

    `backends` is the set of GPU backends to enable (e.g. ("cuda",)); empty means CPU-only.
    Idempotent: once built, subsequent calls just return the cached path. Pass
    `force=True` to rebuild from scratch.
    """
    _check_profile( profile )
    backends = tuple( backends )
    unknown = [ b for b in backends if b not in ACPP_BACKEND_FLAGS ]
    if unknown:
        raise ValueError( f"unknown AdaptiveCpp backend(s) { unknown } (known: { tuple( ACPP_BACKEND_FLAGS ) })" )

    target = acpp_path( profile, backends )
    if target.is_file() and not force:
        return target

    _check_build_tools( profile, backends )

    prefix = acpp_prefix( profile, backends )
    if force and prefix.exists():
        shutil.rmtree( prefix )

    boost_root = ensure_boost()

    work = cache_root() / "adaptivecpp" / "_work"
    src = _fetch_source( ACPP_VERSION, work )
    _cmake_build( src, prefix, profile, boost_root, backends )

    if not target.is_file():
        raise RuntimeError( f"AdaptiveCpp build finished but { target } is missing" )
    return target


# ──────────────────────────── compiling SYCL code ────────────────────────────


def _macos_omp_include_flags() -> list:
    """Extra `-I` flags so `<omp.h>` resolves when targeting the OpenMP backend on macOS.

    acpp already passes `-Xclang -fopenmp` and links the libomp it found at build time, but
    libomp is keg-only on Homebrew (and similar on MacPorts), so its *headers* are not on the
    default include path. We add them explicitly. Honours `SDOT_LIBOMP_DIR`.
    """
    if sys.platform != "darwin":
        return []

    candidates = []
    override = os.getenv( "SDOT_LIBOMP_DIR" )
    if override:
        candidates.append( Path( override ) )
    if shutil.which( "brew" ):
        try:
            r = subprocess.run( [ "brew", "--prefix", "libomp" ], capture_output = True, text = True )
            if r.returncode == 0 and r.stdout.strip():
                candidates.append( Path( r.stdout.strip() ) )
        except Exception:
            pass
    candidates += [ Path( "/opt/homebrew/opt/libomp" ), Path( "/usr/local/opt/libomp" ), Path( "/opt/local" ) ]

    for c in candidates:
        if ( c / "include" / "omp.h" ).is_file():
            return [ "-I", str( c / "include" ) ]

    raise RuntimeError(
        "libomp not found, but AdaptiveCpp's OpenMP backend needs <omp.h> on macOS.\n"
        "  install it: `brew install libomp` (or set SDOT_LIBOMP_DIR to its prefix)."
    )


def make_executable( exe_name, src_paths, device, *, profile = None, extra_flags = None ):
    """Compile & link `src_paths` into an executable using the `acpp` driver.

    SYCL counterpart of `make_executable`: the target device chooses the AdaptiveCpp
    target (`device.acpp_targets`) and the required feature profile. `acpp` handles its
    own include/runtime wiring, so we only add the project's C++ include dir. Returns the
    path to the built executable.
    """
    from . import build_dir, _src_root

    targets = device.acpp_targets
    if targets is None:
        raise RuntimeError(
            f"{ device } is not reachable through AdaptiveCpp (e.g. Apple GPU / Metal); "
            "use its dedicated backend instead."
        )

    profile = profile or device.acpp_profile
    acpp = ensure_acpp( profile, device.acpp_backends )

    out_dir = build_dir()
    out_dir.mkdir( parents = True, exist_ok = True )
    exe = out_dir / exe_name

    omp_flags = _macos_omp_include_flags() if targets.startswith( "omp" ) else []

    project_root = _src_root()
    cmd = [
        acpp,
        f"--acpp-targets={ targets }",
        "-std=c++20", "-O2",
        "-I", project_root / "src" / "cpp",
        *omp_flags,
        *( extra_flags or [] ),
        "-o", exe,
        *src_paths,
    ]
    _run( cmd )
    return exe

def make_library( lib_name, src_paths, device, *, profile = None, extra_flags = None ):
    """Compile & link `src_paths` into a shared library using the `acpp` driver.

    Same wiring as `make_executable`, but emits a relocatable shared object (`-shared
    -fPIC`) meant to be `dlopen`ed at runtime (e.g. to expose an XLA FFI handler symbol to
    Jax). The output file name is taken verbatim, so callers are expected to make it unique
    — typically a content hash of the sources + options (see `encode_base_62`).

    Disk cache: if the target already exists it is returned as-is, unless `SDOT_FORCE_BUILD`
    is set (the dev/test override, also used by `.private/Makefile`). Since the file name is
    a hash of the inputs, a changed source naturally produces a new name and a rebuild.
    Returns the path to the built library.
    """
    from . import build_dir, _src_root

    targets = device.acpp_targets
    if targets is None:
        raise RuntimeError(
            f"{ device } is not reachable through AdaptiveCpp (e.g. Apple GPU / Metal); "
            "use its dedicated backend instead."
        )

    out_dir = build_dir()
    out_dir.mkdir( parents = True, exist_ok = True )
    lib = out_dir / lib_name

    if lib.exists() and not _env_flag( "SDOT_FORCE_BUILD" ):
        return lib

    profile = profile or device.acpp_profile
    acpp = ensure_acpp( profile, device.acpp_backends )

    omp_flags = _macos_omp_include_flags() if targets.startswith( "omp" ) else []

    project_root = _src_root()
    cmd = [
        acpp,
        f"--acpp-targets={ targets }",
        "-std=c++20", "-O2",
        "-fPIC", "-shared",
        "-I", project_root / "src" / "cpp",
        *omp_flags,
        *( extra_flags or [] ),
        "-o", lib,
        *src_paths,
    ]
    _run( cmd )
    return lib

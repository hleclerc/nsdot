"""Compile C++ kernels into XLA-FFI handlers and expose them to Jax.

Pipeline (see `JaxDriver.call`): a C++ *body* is wrapped into a self-registering XLA FFI
handler, compiled to a shared library with AdaptiveCpp (`make_library`), `dlopen`ed, and
registered with `jax.ffi.register_ffi_target`. The returned target name feeds
`jax.ffi.ffi_call`, which inserts the call into the XLA program (works eager and under
`jax.jit`, on CPU and — later — CUDA).

Two caches, both keyed by a content hash of (source + device):
* disk : the compiled `.so`/`.dylib` (handled by `make_library`; a changed source yields a
         new hash, hence a new file and a rebuild).
* RAM  : `_loaded` keeps the `ctypes` handle mapped and marks the target as already
         registered, so we never `dlopen`/`register_ffi_target` the same handler twice in a
         process (a duplicate registration would raise).

STATUS — this is the minimal bootstrap: a no-argument handler that just prints and returns
a dummy int32 token (the token exists only to keep the call from being dead-code-eliminated
by XLA). Real argument/output binding — driven by `CallArgsAnalysis` / `IoCategory` — and
the separate backward handler for `custom_vjp` come next.
"""
from __future__ import annotations

import ctypes
import sys

import jax
import jax.numpy as jnp

from ..compilation.adaptive_cpp import make_library
from ..compilation import build_dir
from ..util.encode_base_62 import encode_base_62

# Fixed C symbol exported by every generated library. It is looked up per-`dlopen`ed handle,
# so the same name across distinct `.so` files never collides; uniqueness at the Jax level is
# carried by the *target name* (the content hash) instead.
_HANDLER_SYMBOL = "sdot_ffi_entry"

# Self-registering handler skeleton. `{body}` is the caller's C++ statements; the trailing
# int32 token write + `Ret<BufferR1<S32>>` binding give XLA a visible result so the call
# survives dead-code elimination.
_SOURCE_TEMPLATE = """\
#include "xla/ffi/api/ffi.h"
#include <cstdio>
#include <iostream>

namespace ffi = xla::ffi;

static ffi::Error sdot_ffi_impl( ffi::Result<ffi::BufferR1<ffi::S32>> out ) {{
{body}
    out->typed_data()[ 0 ] = 0;
    return ffi::Error::Success();
}}

XLA_FFI_DEFINE_HANDLER_SYMBOL( sdot_ffi_entry, sdot_ffi_impl,
    ffi::Ffi::Bind().Ret<ffi::BufferR1<ffi::S32>>() );
"""

# RAM cache: target name -> ctypes handle. Presence == "already registered with Jax".
_loaded: dict[ str, ctypes.CDLL ] = {}


def _lib_suffix() -> str:
    return ".dylib" if sys.platform == "darwin" else ".so"


def _ffi_include_flags() -> list:
    # jaxlib ships the header-only XLA FFI C++ API (xla/ffi/api/ffi.h) under this dir.
    return [ "-I", jax.ffi.include_dir() ]


def render_source( body: str ) -> str:
    """Wrap C++ *body* statements into a complete self-registering FFI handler source."""
    return _SOURCE_TEMPLATE.format( body = body )


def compile_and_register( source: str, device ) -> str:
    """Compile *source*, load and register it, and return its Jax FFI target name.

    Idempotent and cached: repeated calls with the same source + device reuse the compiled
    library and the existing registration.
    """
    name = "sdot_ffi_" + encode_base_62( source + "|" + str( device ) )
    if name in _loaded:
        return name

    src_path = build_dir() / f"{ name }.cpp"
    src_path.parent.mkdir( parents = True, exist_ok = True )
    src_path.write_text( source )

    lib_path = make_library(
        name + _lib_suffix(), [ src_path ], device,
        extra_flags = _ffi_include_flags(),
    )

    lib = ctypes.cdll.LoadLibrary( str( lib_path ) )
    handler = getattr( lib, _HANDLER_SYMBOL )
    jax.ffi.register_ffi_target(
        name, jax.ffi.pycapsule( handler ), platform = device.ffi_platform,
    )

    _loaded[ name ] = lib
    return name


def call_body( body: str, device ):
    """Compile a C++ *body* (no arguments yet) and return the result of its FFI call.

    Convenience for the current bootstrap step: renders the source, compiles/registers it,
    and invokes it. The int32 token array is returned as-is.
    """
    target = compile_and_register( render_source( body ), device )
    return jax.ffi.ffi_call( target, jax.ShapeDtypeStruct( ( 1, ), jnp.int32 ) )()


# Full handler skeleton (slice 3a): buffers are bound as FFI results, each aggregate arg is
# materialized as a small `struct` of views so the C++ body can write `cell.<field> = v`.
_CALL_TEMPLATE = """\
#include "xla/ffi/api/ffi.h"
#include <cstdint>
#include <iostream>

namespace ffi = xla::ffi;

// Minimal view over a ShapeVar's output buffer: `cell.<name> = v` writes the count.
template<class T>
struct ShapeVarView {{
    T *p;
    ShapeVarView &operator=( T v ) {{ *p = v; return *this; }}
    operator T() const {{ return *p; }}
}};

{struct_defs}

static ffi::Error sdot_ffi_impl( {params} ) {{
{struct_inits}
    {{
{body}
    }}
    return ffi::Error::Success();
}}

XLA_FFI_DEFINE_HANDLER_SYMBOL( sdot_ffi_entry, sdot_ffi_impl,
    ffi::Ffi::Bind(){binds} );
"""


def _struct_type_name( arg_name: str ) -> str:
    return "Sdot_" + arg_name


def call( code, ca, device ):
    """Compile `code.fwd_code` bound to the buffers described by `ca`, run it, return the
    reconstructed output object(s).

    Slice 3a: only `ShapeVar` buffers are bound (they carry `jax_*` emitters); tensors and
    `DEFINE_AXIS` come next. Buffer order is fixed by `ca.tensors` and shared between the FFI
    `Bind()`, the handler params and the `ffi_call` result specs.
    """
    buffers = [ t for t in ca.tensors if hasattr( t, "jax_cpp_member" ) ]

    struct_defs = []
    struct_inits = []
    for arg_name, arg_ca in ca.args.items():
        type_name = _struct_type_name( arg_name )
        struct_defs.append( arg_ca.jax_struct_def( type_name ) )
        struct_inits.append( arg_ca.jax_struct_init( arg_name, type_name ) )

    source = _CALL_TEMPLATE.format(
        struct_defs  = "\n\n".join( struct_defs ),
        params       = ", ".join( f"ffi::Result<{ b.jax_ffi_ret_type() }> { b.name }" for b in buffers ),
        struct_inits = "\n".join( struct_inits ),
        body         = code.fwd_code,
        binds        = "".join( f"\n        .Ret<{ b.jax_ffi_ret_type() }>()" for b in buffers ),
    )

    target = compile_and_register( source, device )

    results = jax.ffi.ffi_call( target, [ b.jax_out_spec() for b in buffers ] )()
    buffer_to_array = dict( zip( buffers, results ) )

    reconstructed = { name: arg_ca.jax_reconstruct( buffer_to_array ) for name, arg_ca in ca.args.items() }
    if len( reconstructed ) == 1:
        return next( iter( reconstructed.values() ) )
    return reconstructed

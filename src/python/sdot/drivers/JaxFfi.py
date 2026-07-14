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
import numpy

from ..compilation.adaptive_cpp import make_library
from ..compilation import build_dir
from ..util.encode_base_62 import encode_base_62
from .CallArg_Errors import ERRORS_VAR_NAME

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


def compile_and_register( source: str, device, prefix: str = "" ) -> str:
    """Compile *source*, load and register it, and return its Jax FFI target name.

    Idempotent and cached: repeated calls with the same source + device reuse the compiled
    library and the existing registration.
    """

    if not prefix:
        prefix = "sdot_ffi_"

    name = prefix + encode_base_62( source + "|" + str( device ) )
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


# Full handler skeleton: input buffers are bound as FFI args, output buffers as FFI results,
# and each aggregate arg is materialized as a small `struct` of views over them, so the C++
# body can read and write `cell.<field>`.
_CALL_TEMPLATE = """\
#include "xla/ffi/api/ffi.h"
#define SDOT_QUEUE {queue_type}
#include "sdot/Queue.h"
#include "sdot/support/algorithms/CartesianIndices.h"
#include "sdot/support/kernels/run_parallel.h"
#include "sdot/support/common_types.h"
#include "sdot/support/Ct.h"
#include "sdot/support/containers/TensorView.h"
#include "sdot/support/containers/ShapeVarView.h"
#include "sdot/support/containers/ErrorBuffer.h"
#include "sdot/support/containers/NoneTensor.h"
#include "sdot/support/containers/ZeroTensor.h"
#include <cstdint>
#include <iostream>

namespace ffi = xla::ffi;
using namespace sdot;

{axis_defs}

{struct_defs}

static ffi::Error sdot_ffi_impl( {params} ) {{
    // the one execution context of this call. A `sycl::queue` is expensive to create, and this
    // handler always runs on the same device (the device is in the TYPE of everything below), so
    // there is exactly one, made on first use -- and never destroyed: the SYCL runtime is torn
    // down before the statics of a dlopen'ed handler are, and a queue outliving it deadlocks.
    static Queue &queue = *new Queue();

    // what the body iterates over: the multi-indices of the batch axes. Unmapped, that is a single
    // item -- the EMPTY multi-index -- and a `vmap` is what gives it axes. Named ones: the body
    // applies `batch_index` to a value, which selects the axes it has and ignores the others.
{batch_indices}

{decls}
{seeds}
    {{
{body}
    }}
    return ffi::Error::Success();
}}

XLA_FFI_DEFINE_HANDLER_SYMBOL( sdot_ffi_entry, sdot_ffi_impl,
    ffi::Ffi::Bind(){binds} );
"""


def _axis_def( name: str ) -> str:
    # the same axis serves several tensors: guard each DEFINE_AXIS so it lands at most once.
    guard = f"SDOT_AXIS_{ name }"
    return ( f"#ifndef { guard }\n"
             f"#define { guard }\n"
             f"DEFINE_AXIS( { name } );\n"
             f"#endif" )


def _batch_indices_decl( ca ):
    """`global_batch_indices`: the multi-indices the body iterates over.

    Unmapped, `CartesianIndices<Tuple<>>` -- one item, the empty multi-index. A `vmap` gives it a
    NAMED axis, whose extent is read from a buffer at run time (a batch size is an extent like any
    other: making it a literal would recompile the kernel for every batch size)."""
    if not ca.batch_axes:
        return "    CartesianIndices<Tuple<>> global_batch_indices;"
    shape = ", ".join( "SI" for _ in ca.batch_axes )
    names = ", ".join( "_" + n for n in ca.batch_axes )
    sizes = ", ".join( ca.batch_dim_expr( n ) for n in ca.batch_axes )
    return ( f"    CartesianIndices<Tuple<{ shape }>,Tuple<{ names }>> "
             f"global_batch_indices{{ tuple( { sizes } ) }};" )


def _render_call( code, ca, device ):
    """The complete FFI handler source for this code bound to these buffers, and the attributes
    it expects.

    Inputs and outputs are disjoint buffers: an input is bound at the size its data actually
    has, an output is allocated at the capacity declared in Python. XLA FFI wants args before
    results, so the parameter list is inputs then outputs, `ca.tensors` fixing the order within
    each group.

    A root argument declares itself (`cpp_root_decl`): an aggregate as a `struct` of views over
    its buffers -- a *template*, hence the dedup of definitions by `type_name`, since the same
    class may appear twice in a call with different compile-time parameters -- and a bare
    tensor as the view itself, no wrapper needed.
    """
    inputs = [ t for t in ca.tensors if t.io_category.is_input ]
    outputs = [ t for t in ca.tensors if t.io_category.is_output ]

    # scalars that are neither data nor structure -- a capacity, typically. They cross as XLA FFI
    # ATTRIBUTES: baked into the call, not into the kernel, so a new capacity does not mean a new
    # compilation. (Extents need no attribute at all: XLA carries them next to the data.)
    attrs = [ a for b in ca.tensors if hasattr( b, "jax_attrs" ) for a in b.jax_attrs() ]

    struct_defs = {}
    for arg_ca in ca.args.values():
        struct_defs.update( arg_ca.cpp_struct_defs() )

    # XLA FFI binds in this order, and the handler's parameters must follow it: args, results,
    # then attributes.
    params = [ f"{ b.jax_ffi_type() } { b.ffi_name }" for b in inputs ]
    params += [ f"ffi::Result<{ b.jax_ffi_type() }> { b.ffi_name }" for b in outputs ]
    params += [ f"{ cpp_type } { name }" for name, cpp_type, _ in attrs ]

    binds = "".join( f"\n        .Arg<{ b.jax_ffi_type() }>()" for b in inputs )
    binds += "".join( f"\n        .Ret<{ b.jax_ffi_type() }>()" for b in outputs )
    binds += "".join( f'\n        .Attr<{ cpp_type }>( "{ name }" )' for name, cpp_type, _ in attrs )

    # the error buffer comes FIRST: the values that can fail are built holding a view on it.
    decls = [ ca.errors.cpp_root_decl( ERRORS_VAR_NAME ) ]
    decls += [ ca_.cpp_root_decl( n ) for n, ca_ in ca.args.items() ]

    seeds = [ ca.errors.cpp_seed_root( ERRORS_VAR_NAME ) ]
    seeds += [ ca_.cpp_seed_root( n ) for n, ca_ in ca.args.items() if hasattr( ca_, "cpp_seed_root" ) ]

    source = _CALL_TEMPLATE.format(
        queue_type    = device.cpp_queue_type,
        axis_defs     = "\n".join( _axis_def( n ) for n in ca.axis_names ),
        struct_defs   = "\n\n".join( struct_defs.values() ),
        params        = ", ".join( params ),
        batch_indices = _batch_indices_decl( ca ),
        decls         = "\n".join( decls ),
        seeds         = "\n".join( s for s in seeds if s ),
        body          = code.fwd_code,
        binds         = binds,
    )
    return source, inputs, outputs, attrs


def _make_op( code, ca, device, prefix ):
    """The call as a Jax operation -- with its own batching rule.

    A `vmap` cannot batch an FFI call by itself (it could only replay it item by item, or
    broadcast everything). Ours does the one thing that makes sense here: it RECOMPILES. The rule
    derives the code (one more batch axis) and the lowering (the buffers that gained a leading
    dimension), and calls the kernel that comes out -- one launch over N items, not N launches.

    The derived call is an op of the same kind, so a nested `vmap` just derives again.
    """
    @jax.custom_batching.custom_vmap
    def op( *arrays ):
        source, _, outputs, attrs = _render_call( code, ca, device )
        target = compile_and_register( source, device, prefix )
        results = jax.ffi.ffi_call( target, [ b.jax_out_spec() for b in outputs ] )(
            *arrays, **{ name: numpy.int64( value ) for name, _, value in attrs }
        )
        return list( results ) if isinstance( results, ( list, tuple ) ) else [ results ]

    @op.def_vmap
    def _( axis_size, in_batched, *arrays ):
        # `arrays` come with the mapped axis leading, and `in_batched` says which ones the vmap
        # actually mapped -- an unmapped input keeps its shape, and the kernel will let the batch
        # index pass through it rather than read a slice of it.
        inputs = [ t for t in ca.tensors if t.io_category.is_input ]
        batched_inputs = { t.ffi_name for t, mapped in zip( inputs, in_batched ) if mapped }

        axis_name, batched_code = code.with_batch_axis()
        batched_ca = ca.batched( axis_name, axis_size, batched_inputs )

        results = _make_op( batched_code, batched_ca, device, prefix )( *arrays )

        # ... and one output per item, save what belongs to the CALL rather than to an item: the
        # error buffer is one, and comes back unbatched.
        outputs = [ t for t in ca.tensors if t.io_category.is_output ]
        return results, [ t.takes_batch_axis() for t in outputs ]

    return op


def call( code, ca, device, prefix = "" ):
    """Run `code` on the buffers described by `ca`, and write the outputs back onto the objects
    the caller handed us."""
    inputs = [ t for t in ca.tensors if t.io_category.is_input ]
    outputs = [ t for t in ca.tensors if t.io_category.is_output ]

    # the kernel dereferences its buffers where IT runs, so an input has to be there: an array
    # built on the host would otherwise be read through a device pointer.
    arrays = [ jax.device_put( b.jax_input_array(), device.driver_version ) for b in inputs ]

    results = _make_op( code, ca, device, prefix )( *arrays )

    # an output attribute was EMPTY (that is what made it declarable as one), so filling it in
    # is not a mutation of anything the caller could already have observed. Under a `vmap` these
    # are the OUTER values (batch axis stripped by Jax), which is why the batched lowering had to
    # be a copy: this one still describes the tensors as the caller knows them.
    for buffer, array in zip( outputs, results ):
        buffer.jax_write_back( array )

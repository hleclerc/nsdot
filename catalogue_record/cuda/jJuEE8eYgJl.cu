#include "xla/ffi/api/ffi.h"
#include <loom/support/kernels/CudaQueue.h>
// the device is in the TYPE of everything below: the queue decides the memory space the kernel
// dereferences. `SDOT_QUEUE` is also what a hand-written header may read (`sdot/Queue.h`).
#define SDOT_QUEUE CudaQueue
namespace sdot { using Queue = SDOT_QUEUE; }
#include <loom/support/algorithms/CartesianIndices.h>
#include <loom/support/kernels/run_parallel.h>
#include <loom/support/common_types.h>
#include <loom/support/Ct.h>
#include <loom/support/containers/TensorView.h>
#include <loom/support/containers/ShapeVarView.h>
#include <loom/support/containers/ErrorBuffer.h>
#include <loom/support/containers/NoneTensor.h>
#include <loom/support/containers/ZeroTensor.h>
#include <loom/support/containers/FillTensor.h>
#include <loom/support/containers/ScalarValue.h>
#include <cstdint>
#include <iostream>

namespace ffi = xla::ffi;
using namespace sdot;

// the axes this call names, from their shared generated headers (`DEFINE_AXIS`), so anything
// below can spell them.
#include "sdot/generated/axes/dim_axis.h"
#include "sdot/generated/axes/num_axis.h"
#include "sdot/generated/axes/num_vertex.h"
#include "sdot/generated/axes/num_cut.h"
#include "sdot/generated/axes/num_thread.h"
#include "sdot/generated/axes/num_word.h"


// the headers the arguments and the body asked for: the manual struct of each aggregate (which
// pulls in its own generated macros), then whatever the body listed for itself.
#include "sdot/Cell_N.h"
#include "sdot/generated/aggregates/CellScratch_full.h"


// what the body runs per item: a NAMED functor at namespace scope (never a lambda -- a device
// compiler is happier with a plain struct), rendered by the code object from the call's arguments.
struct init_as_hypercube_fwd_kernel {
    template<class BatchIndex, class T_cut_id, class T_origin, class T_axes, class T_cell, class T_scratch>
    HD void operator()( BatchIndex batch_index, int thread_index, int nb_threads, T_cut_id cut_id, T_origin origin, T_axes axes, T_cell cell, T_scratch scratch ) const {
        cell( batch_index ).init_as_hypercube( scratch( batch_index ), origin, axes, cut_id );
    }
};


static ffi::Error sdot_ffi_impl( cudaStream_t xla_stream, ffi::BufferR1<ffi::F64> ffi_origin, ffi::BufferR2<ffi::F64> ffi_axes, ffi::Result<ffi::BufferR2<ffi::F64>> ffi_vertex_positions, ffi::Result<ffi::BufferR2<ffi::S32>> ffi_vertex_cuts, ffi::Result<ffi::BufferR2<ffi::S32>> ffi_vertex_nbrs, ffi::Result<ffi::BufferR1<ffi::S32>> ffi_cut_ids, ffi::Result<ffi::BufferR1<ffi::S32>> ffi_nb_vertices, ffi::Result<ffi::BufferR1<ffi::S32>> ffi_nb_cuts, ffi::Result<ffi::BufferR2<ffi::S32>> ffi_words, ffi::Result<ffi::BufferR1<ffi::S32>> ffi_nb_threads, ffi::Result<ffi::BufferR1<ffi::S32>> ffi_nb_words, ffi::Result<ffi::BufferR1<ffi::S32>> ffi_errors, int64_t attr_cut_id, int64_t max_ffi_nb_vertices, int64_t max_ffi_nb_cuts, int64_t max_ffi_nb_threads, int64_t max_ffi_nb_words ) {
    // the execution context of this call (the device is in the TYPE of everything below; how the
    // queue is obtained is the device's business, see `Device.cpp_queue_decl`).
    Queue queue( xla_stream );

    // what the body iterates over: the multi-indices of the batch axes. Unmapped, that is a single
    // item -- the EMPTY multi-index -- and a `vmap` is what gives it axes. Named ones: the body
    // applies `batch_index` to a value, which selects the axes it has and ignores the others.
    CartesianIndices<Tuple<>> global_batch_indices;

    auto errors = make_error_buffer( tensor_view<CudaGlobalMemorySpace>( ffi_errors->typed_data(), tuple( SI( ffi_errors->dimensions()[ 0 ] ) ) ), SI( 8 ) );
    auto cut_id = SI( attr_cut_id );
    auto origin = tensor_view<CudaGlobalMemorySpace>( ffi_origin.typed_data(), tuple( Ct<SI, 3>() ), tuple( dim_axis ) );
    auto axes = tensor_view<CudaGlobalMemorySpace>( ffi_axes.typed_data(), tuple( Ct<SI, 3>(), SI( ffi_axes.dimensions()[ 1 ] ) ), tuple( dim_axis, num_axis ) );
    auto cell = Cell_N{
        tensor_view<CudaGlobalMemorySpace>( ffi_vertex_positions->typed_data(), tuple( SI( ffi_vertex_positions->dimensions()[ 0 ] ), Ct<SI, 3>() ), tuple( num_vertex, dim_axis ) ),
        tensor_view<CudaGlobalMemorySpace>( ffi_vertex_cuts->typed_data(), tuple( SI( ffi_vertex_cuts->dimensions()[ 0 ] ), Ct<SI, 3>() ), tuple( num_vertex, dim_axis ) ),
        tensor_view<CudaGlobalMemorySpace>( ffi_vertex_nbrs->typed_data(), tuple( SI( ffi_vertex_nbrs->dimensions()[ 0 ] ), Ct<SI, 3>() ), tuple( num_vertex, dim_axis ) ),
        tensor_view<CudaGlobalMemorySpace>( ffi_cut_ids->typed_data(), tuple( SI( ffi_cut_ids->dimensions()[ 0 ] ) ), tuple( num_cut ) ),
        make_shape_var_view( tensor_view<CudaGlobalMemorySpace>( ffi_nb_vertices->typed_data(), tuple(  ), tuple(  ) ), SI( max_ffi_nb_vertices ), errors, SI( 0 ) ),
        make_shape_var_view( tensor_view<CudaGlobalMemorySpace>( ffi_nb_cuts->typed_data(), tuple(  ), tuple(  ) ), SI( max_ffi_nb_cuts ), errors, SI( 1 ) ),
        Ct<SI, 3>{}
    };
    auto cell_io = Cell_N_io{ OutList(), OutList(), OutList(), OutList(), OutList(), OutList(), InpList() };
    auto scratch = CellScratch{
        tensor_view<CudaGlobalMemorySpace>( ffi_words->typed_data(), tuple( SI( ffi_words->dimensions()[ 0 ] ), SI( ffi_words->dimensions()[ 1 ] ) ), tuple( num_thread, num_word ) ),
        make_shape_var_view( tensor_view<CudaGlobalMemorySpace>( ffi_nb_threads->typed_data(), tuple(  ), tuple(  ) ), SI( max_ffi_nb_threads ), errors, SI( 2 ) ),
        make_shape_var_view( tensor_view<CudaGlobalMemorySpace>( ffi_nb_words->typed_data(), tuple(  ), tuple(  ) ), SI( max_ffi_nb_words ), errors, SI( 3 ) ),
        Ct<SI, 64>{}
    };
    auto scratch_io = CellScratch_io{ OutList(), OutList(), OutList(), InpList() };
    errors.fill_with( queue, 0 );
    cell.vertex_positions.fill_with( queue, 0 );
    cell.vertex_cuts.fill_with( queue, 0 );
    cell.vertex_nbrs.fill_with( queue, 0 );
    cell.cut_ids.fill_with( queue, 0 );
    cell.nb_vertices.fill_with( queue, 0 );
    cell.nb_cuts.fill_with( queue, 0 );
    scratch.words.fill_with( queue, 0 );
    scratch.nb_threads.fill_with( queue, 0 );
    scratch.nb_words.fill_with( queue, 0 );
    {
run_parallel(
    queue,
    global_batch_indices,
    init_as_hypercube_fwd_kernel{},
    InpList(), cut_id, InpList(), origin, InpList(), axes, cell_io, cell, scratch_io, scratch
);
    }
    return ffi::Error::Success();
}

// the ONE exported symbol (everything else is hidden, see `HostCxx`): the declaration carries the
// visibility, the macro below defines it. Its NAME is a define, not part of the source (the source
// is hashed into the kernel's name): `sdot_ffi_entry` in a library of its own, a unique name when a
// catalogue links many kernels into one library (see `compilation/catalogue.py`).
#ifndef SDOT_FFI_ENTRY
#define SDOT_FFI_ENTRY sdot_ffi_entry
#endif
extern "C" LOOM_EXPORT XLA_FFI_Error *SDOT_FFI_ENTRY( XLA_FFI_CallFrame * );
XLA_FFI_DEFINE_HANDLER_SYMBOL( SDOT_FFI_ENTRY, sdot_ffi_impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::BufferR1<ffi::F64>>()
        .Arg<ffi::BufferR2<ffi::F64>>()
        .Ret<ffi::BufferR2<ffi::F64>>()
        .Ret<ffi::BufferR2<ffi::S32>>()
        .Ret<ffi::BufferR2<ffi::S32>>()
        .Ret<ffi::BufferR1<ffi::S32>>()
        .Ret<ffi::BufferR1<ffi::S32>>()
        .Ret<ffi::BufferR1<ffi::S32>>()
        .Ret<ffi::BufferR2<ffi::S32>>()
        .Ret<ffi::BufferR1<ffi::S32>>()
        .Ret<ffi::BufferR1<ffi::S32>>()
        .Ret<ffi::BufferR1<ffi::S32>>()
        .Attr<int64_t>( "attr_cut_id" )
        .Attr<int64_t>( "max_ffi_nb_vertices" )
        .Attr<int64_t>( "max_ffi_nb_cuts" )
        .Attr<int64_t>( "max_ffi_nb_threads" )
        .Attr<int64_t>( "max_ffi_nb_words" ) );

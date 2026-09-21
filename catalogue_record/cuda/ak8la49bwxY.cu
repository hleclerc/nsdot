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
#include "sdot/generated/axes/num_point.h"
#include "sdot/generated/axes/dim.h"
#include "sdot/generated/axes/bspnode_0.h"
#include "sdot/generated/axes/num_lohi.h"
#include "sdot/generated/axes/num_bsp_param.h"


// the headers the arguments and the body asked for: the manual struct of each aggregate (which
// pulls in its own generated macros), then whatever the body listed for itself.
#include "sdot/generated/aggregates/_BspCloud_full.h"
#include "sdot/generated/aggregates/_BspLevel_full.h"
#include "sdot/bsp_build_level.h"


// what the body runs per item: a NAMED functor at namespace scope (never a lambda -- a device
// compiler is happier with a plain struct), rendered by the code object from the call's arguments.
struct bsp_build_level_fwd_kernel {
    template<class BatchIndex, class T_src, class T_dst, class T_lvl, class T_perm, class T_leaf_size>
    HD void operator()( BatchIndex batch_index, int thread_index, int nb_threads, T_src src, T_dst dst, T_lvl lvl, T_perm perm, T_leaf_size leaf_size ) const {
        bsp_build_level( src, dst, perm, lvl.begin( batch_index ), lvl.end( batch_index ), lvl.box( batch_index ), lvl.wa( batch_index ), lvl.wb( batch_index ), lvl.mid( batch_index ), SI( leaf_size( 0 ) ) );
    }
};


static ffi::Error sdot_ffi_impl( cudaStream_t xla_stream, ffi::BufferR2<ffi::F64> ffi_positions, ffi::BufferR1<ffi::S64> ffi_order, ffi::BufferR1<ffi::S32> ffi_nb_points, ffi::BufferR1<ffi::S64> ffi_begin, ffi::BufferR1<ffi::S64> ffi_end, ffi::BufferR1<ffi::S64> ffi_leaf_size, ffi::Result<ffi::BufferR2<ffi::F64>> ffi_positions_, ffi::Result<ffi::BufferR1<ffi::S64>> ffi_order_, ffi::Result<ffi::BufferR1<ffi::S32>> ffi_nb_points_, ffi::Result<ffi::BufferR1<ffi::S64>> ffi_mid, ffi::Result<ffi::BufferR3<ffi::F64>> ffi_box, ffi::Result<ffi::BufferR1<ffi::S64>> ffi_perm, ffi::Result<ffi::BufferR1<ffi::S32>> ffi_errors, int64_t max_ffi_nb_points, int64_t max_ffi_nb_points_, int64_t nb_batch_bspnode_0 ) {
    // the execution context of this call (the device is in the TYPE of everything below; how the
    // queue is obtained is the device's business, see `Device.cpp_queue_decl`).
    Queue queue( xla_stream );

    // what the body iterates over: the multi-indices of the batch axes. Unmapped, that is a single
    // item -- the EMPTY multi-index -- and a `vmap` is what gives it axes. Named ones: the body
    // applies `batch_index` to a value, which selects the axes it has and ignores the others.
    CartesianIndices<Tuple<SI>,Tuple<_bspnode_0>> global_batch_indices{ tuple( SI( nb_batch_bspnode_0 ) ) };

    auto errors = make_error_buffer( tensor_view<CudaGlobalMemorySpace>( ffi_errors->typed_data(), tuple( SI( ffi_errors->dimensions()[ 0 ] ) ) ), SI( 8 ) );
    auto src = _BspCloud{
        tensor_view<CudaGlobalMemorySpace>( ffi_positions.typed_data(), tuple( SI( ffi_positions.dimensions()[ 0 ] ), Ct<SI, 2>() ), tuple( num_point, dim ) ),
        NoneTensor<double, Tuple<SI>, Tuple<_num_point>>{},
        tensor_view<CudaGlobalMemorySpace>( ffi_order.typed_data(), tuple( SI( ffi_order.dimensions()[ 0 ] ) ), tuple( num_point ) ),
        make_shape_var_view( tensor_view<CudaGlobalMemorySpace>( ffi_nb_points.typed_data(), tuple(  ), tuple(  ) ), SI( max_ffi_nb_points ), NoErrorBuffer{}, SI( -1 ) ),
        Ct<SI, 2>{}
    };
    auto src_io = _BspCloud_io{ InpList(), UndefList(), InpList(), InpList(), InpList() };
    auto dst = _BspCloud{
        tensor_view<CudaGlobalMemorySpace>( ffi_positions_->typed_data(), tuple( SI( ffi_positions_->dimensions()[ 0 ] ), Ct<SI, 2>() ), tuple( num_point, dim ) ),
        NoneTensor<double, Tuple<SI>, Tuple<_num_point>>{},
        tensor_view<CudaGlobalMemorySpace>( ffi_order_->typed_data(), tuple( SI( ffi_order_->dimensions()[ 0 ] ) ), tuple( num_point ) ),
        make_shape_var_view( tensor_view<CudaGlobalMemorySpace>( ffi_nb_points_->typed_data(), tuple(  ), tuple(  ) ), SI( max_ffi_nb_points_ ), NoErrorBuffer{}, SI( -1 ) ),
        Ct<SI, 2>{}
    };
    auto dst_io = _BspCloud_io{ OutList(), UndefList(), OutList(), OutList(), InpList() };
    auto lvl = _BspLevel{
        tensor_view<CudaGlobalMemorySpace>( ffi_begin.typed_data(), tuple( SI( ffi_begin.dimensions()[ 0 ] ) ), tuple( bspnode_0 ) ),
        tensor_view<CudaGlobalMemorySpace>( ffi_end.typed_data(), tuple( SI( ffi_end.dimensions()[ 0 ] ) ), tuple( bspnode_0 ) ),
        tensor_view<CudaGlobalMemorySpace>( ffi_mid->typed_data(), tuple( SI( ffi_mid->dimensions()[ 0 ] ) ), tuple( bspnode_0 ) ),
        tensor_view<CudaGlobalMemorySpace>( ffi_box->typed_data(), tuple( SI( ffi_box->dimensions()[ 0 ] ), Ct<SI, 2>(), Ct<SI, 2>() ), tuple( bspnode_0, num_lohi, dim ) ),
        NoneTensor<double, Tuple<SI, Ct<SI, 2>>, Tuple<_bspnode_0, _dim>>{},
        NoneTensor<double, Tuple<SI>, Tuple<_bspnode_0>>{},
        Ct<SI, 2>{},
        Ct<SI, 2>{}
    };
    auto lvl_io = _BspLevel_io{ InpList(), InpList(), OutList(), OutList(), UndefList(), UndefList(), InpList(), InpList() };
    auto perm = tensor_view<CudaGlobalMemorySpace>( ffi_perm->typed_data(), tuple( SI( ffi_perm->dimensions()[ 0 ] ) ), tuple( num_point ) );
    auto leaf_size = tensor_view<CudaGlobalMemorySpace>( ffi_leaf_size.typed_data(), tuple( SI( ffi_leaf_size.dimensions()[ 0 ] ) ), tuple( num_bsp_param ) );
    errors.fill_with( queue, 0 );
    dst.positions.fill_with( queue, 0 );
    dst.order.fill_with( queue, 0 );
    dst.nb_points.fill_with( queue, 0 );
    lvl.mid.fill_with( queue, 0 );
    lvl.box.fill_with( queue, 0 );
    {
run_parallel(
    queue,
    global_batch_indices,
    bsp_build_level_fwd_kernel{},
    src_io, src, dst_io, dst, lvl_io, lvl, OutList(), perm, InpList(), leaf_size
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
        .Arg<ffi::BufferR2<ffi::F64>>()
        .Arg<ffi::BufferR1<ffi::S64>>()
        .Arg<ffi::BufferR1<ffi::S32>>()
        .Arg<ffi::BufferR1<ffi::S64>>()
        .Arg<ffi::BufferR1<ffi::S64>>()
        .Arg<ffi::BufferR1<ffi::S64>>()
        .Ret<ffi::BufferR2<ffi::F64>>()
        .Ret<ffi::BufferR1<ffi::S64>>()
        .Ret<ffi::BufferR1<ffi::S32>>()
        .Ret<ffi::BufferR1<ffi::S64>>()
        .Ret<ffi::BufferR3<ffi::F64>>()
        .Ret<ffi::BufferR1<ffi::S64>>()
        .Ret<ffi::BufferR1<ffi::S32>>()
        .Attr<int64_t>( "max_ffi_nb_points" )
        .Attr<int64_t>( "max_ffi_nb_points_" )
        .Attr<int64_t>( "nb_batch_bspnode_0" ) );

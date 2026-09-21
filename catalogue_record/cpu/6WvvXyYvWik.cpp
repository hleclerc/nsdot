#include "xla/ffi/api/ffi.h"
#include <loom/support/kernels/CpuQueue.h>
// the device is in the TYPE of everything below: the queue decides the memory space the kernel
// dereferences. `SDOT_QUEUE` is also what a hand-written header may read (`sdot/Queue.h`).
#define SDOT_QUEUE CpuQueue
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
#include "sdot/generated/axes/img_pos_0.h"
#include "sdot/generated/axes/img_pos_1.h"
#include "sdot/generated/axes/img_pos_2.h"
#include "sdot/generated/axes/dim.h"
#include "sdot/generated/axes/dir.h"
#include "sdot/generated/axes/num_knot.h"
#include "sdot/generated/axes/num_cell_cum.h"


// the headers the arguments and the body asked for: the manual struct of each aggregate (which
// pulls in its own generated macros), then whatever the body listed for itself.
#include "sdot/Image.h"


// what the body runs per item: a NAMED functor at namespace scope (never a lambda -- a device
// compiler is happier with a plain struct), rendered by the code object from the call's arguments.
struct mass_fwd_kernel {
    template<class BatchIndex, class T_image>
    HD void operator()( BatchIndex batch_index, int thread_index, int nb_threads, T_image image ) const {
        image.current_mass( batch_index ) = image( batch_index ).measure();
    }
};


static ffi::Error sdot_ffi_impl( ffi::BufferR0<ffi::F64> ffi_target_mass, ffi::BufferR1<ffi::S32> ffi_shape, ffi::BufferR3<ffi::F64> ffi_values, ffi::Result<ffi::BufferR0<ffi::F64>> ffi_current_mass, ffi::Result<ffi::BufferR1<ffi::S32>> ffi_errors, int64_t max_ffi_shape ) {
    // the execution context of this call (the device is in the TYPE of everything below; how the
    // queue is obtained is the device's business, see `Device.cpp_queue_decl`).
    static Queue &queue = *new Queue();

    // what the body iterates over: the multi-indices of the batch axes. Unmapped, that is a single
    // item -- the EMPTY multi-index -- and a `vmap` is what gives it axes. Named ones: the body
    // applies `batch_index` to a value, which selects the axes it has and ignores the others.
    CartesianIndices<Tuple<>> global_batch_indices;

    auto errors = make_error_buffer( tensor_view<CpuHostMemorySpace>( ffi_errors->typed_data(), tuple( SI( ffi_errors->dimensions()[ 0 ] ) ) ), SI( 8 ) );
    auto image = Image{
        tensor_view<CpuHostMemorySpace>( ffi_target_mass.typed_data(), tuple(  ), tuple(  ) ),
        Ct<SI, 3>{},
        make_shape_var_view( tensor_view<CpuHostMemorySpace>( ffi_shape.typed_data(), tuple( SI( ffi_shape.dimensions()[ 0 ] ) ), tuple( UnnamedAxis{} ) ), SI( max_ffi_shape ), NoErrorBuffer{}, SI( -1 ) ),
        tensor_view<CpuHostMemorySpace>( ffi_values.typed_data(), tuple( SI( ffi_values.dimensions()[ 0 ] ), SI( ffi_values.dimensions()[ 1 ] ), SI( ffi_values.dimensions()[ 2 ] ) ), tuple( img_pos_0, img_pos_1, img_pos_2 ) ),
        NoneTensor<double, Tuple<Ct<SI, 3>>, Tuple<_dim>>{},
        NoneTensor<double, Tuple<Ct<SI, 3>, Ct<SI, 3>>, Tuple<_dir, _dim>>{},
        NoneTensor<double, Tuple<Ct<SI, 3>, SI>, Tuple<_dim, _num_knot>>{},
        tensor_view<CpuHostMemorySpace>( ffi_current_mass->typed_data(), tuple(  ), tuple(  ) ),
        ShapeVarView<NoneTensor<std::int32_t, Tuple<>, Tuple<>>, NoErrorBuffer>{ NoneTensor<std::int32_t, Tuple<>, Tuple<>>{}, SI( -1 ), NoErrorBuffer{}, SI( -1 ) },
        NoneTensor<double, Tuple<SI>, Tuple<_num_cell_cum>>{}
    };
    auto image_io = Image_io{ InpList(), InpList(), InpList(), InpList(), UndefList(), UndefList(), UndefList(), OutList(), UndefList(), UndefList() };
    errors.fill_with( queue, 0 );
    image.current_mass.fill_with( queue, 0 );
    {
run_parallel(
    queue,
    global_batch_indices,
    mass_fwd_kernel{},
    image_io, image
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
        .Arg<ffi::BufferR0<ffi::F64>>()
        .Arg<ffi::BufferR1<ffi::S32>>()
        .Arg<ffi::BufferR3<ffi::F64>>()
        .Ret<ffi::BufferR0<ffi::F64>>()
        .Ret<ffi::BufferR1<ffi::S32>>()
        .Attr<int64_t>( "max_ffi_shape" ) );

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
#include "sdot/generated/axes/dim.h"
#include "sdot/generated/axes/num_boundary.h"
#include "sdot/generated/axes/num_point.h"
#include "sdot/generated/axes/num_vertex.h"
#include "sdot/generated/axes/dim_axis.h"
#include "sdot/generated/axes/num_cut.h"
#include "sdot/generated/axes/thread_0.h"
#include "sdot/generated/axes/num_thread.h"
#include "sdot/generated/axes/num_word.h"


// the headers the arguments and the body asked for: the manual struct of each aggregate (which
// pulls in its own generated macros), then whatever the body listed for itself.
#include "sdot/PowerDiagram_Plain.h"
#include "sdot/Cell_2.h"
#include "sdot/generated/aggregates/CellScratch_full.h"


// what the body runs per item: a NAMED functor at namespace scope (never a lambda -- a device
// compiler is happier with a plain struct), rendered by the code object from the call's arguments.
struct power_diagram_measures_bwd_fwd_kernel {
    template<class BatchIndex, class T_power_diagram, class T_grad_for_power_diagram, class T_dom_cell, class T_grad_for_dom_cell, class T_res, class T_grad_for_res, class T_scratch>
    HD void operator()( BatchIndex batch_index, int thread_index, int nb_threads, T_power_diagram power_diagram, T_grad_for_power_diagram grad_for_power_diagram, T_dom_cell dom_cell, T_grad_for_dom_cell grad_for_dom_cell, T_res res, T_grad_for_res grad_for_res, T_scratch scratch ) const {
        power_diagram.measures_bwd( res, dom_cell, grad_for_res, grad_for_power_diagram.positions, grad_for_power_diagram.weights, scratch( batch_index ), power_diagram.unit_density(), 0, thread_index, nb_threads );
    }
};


static ffi::Error sdot_ffi_impl( ffi::BufferR1<ffi::F64> ffi_box_min, ffi::BufferR1<ffi::F64> ffi_box_max, ffi::BufferR2<ffi::F64> ffi_positions, ffi::BufferR1<ffi::F64> ffi_weights, ffi::BufferR2<ffi::F64> ffi_vertex_positions, ffi::BufferR1<ffi::S32> ffi_cut_ids, ffi::BufferR1<ffi::S32> ffi_nb_vertices, ffi::BufferR1<ffi::S32> ffi_nb_cuts, ffi::BufferR1<ffi::S32> ffi_nb_vertices_, ffi::BufferR1<ffi::S32> ffi_nb_cuts_, ffi::BufferR1<ffi::F64> ffi_res, ffi::BufferR1<ffi::F64> ffi_grad_for_res, ffi::Result<ffi::BufferR1<ffi::F64>> ffi_weights_, ffi::Result<ffi::BufferR3<ffi::S32>> ffi_words, ffi::Result<ffi::BufferR1<ffi::S32>> ffi_nb_threads, ffi::Result<ffi::BufferR1<ffi::S32>> ffi_nb_words, ffi::Result<ffi::BufferR1<ffi::S32>> ffi_errors, int64_t count_power_diagram_nb_points, int64_t count_grad_for_power_diagram_nb_points, int64_t max_ffi_nb_vertices, int64_t max_ffi_nb_cuts, int64_t max_ffi_nb_vertices_, int64_t max_ffi_nb_cuts_, int64_t max_ffi_nb_threads, int64_t max_ffi_nb_words, int64_t nb_batch_thread_0 ) {
    // the execution context of this call (the device is in the TYPE of everything below; how the
    // queue is obtained is the device's business, see `Device.cpp_queue_decl`).
    static Queue &queue = *new Queue();

    // what the body iterates over: the multi-indices of the batch axes. Unmapped, that is a single
    // item -- the EMPTY multi-index -- and a `vmap` is what gives it axes. Named ones: the body
    // applies `batch_index` to a value, which selects the axes it has and ignores the others.
    CartesianIndices<Tuple<SI>,Tuple<_thread_0>> global_batch_indices{ tuple( SI( nb_batch_thread_0 ) ) };

    auto errors = make_error_buffer( tensor_view<CpuHostMemorySpace>( ffi_errors->typed_data(), tuple( SI( ffi_errors->dimensions()[ 0 ] ) ) ), SI( 8 ) );
    auto power_diagram = PowerDiagram_Plain{
        tensor_view<CpuHostMemorySpace>( ffi_box_min.typed_data(), tuple( Ct<SI, 2>() ), tuple( dim ) ),
        tensor_view<CpuHostMemorySpace>( ffi_box_max.typed_data(), tuple( Ct<SI, 2>() ), tuple( dim ) ),
        NoneTensor<double, Tuple<SI, Ct<SI, 2>>, Tuple<_num_boundary, _dim>>{},
        NoneTensor<double, Tuple<SI>, Tuple<_num_boundary>>{},
        make_shape_var_view( ScalarValue<SI>{ SI( count_power_diagram_nb_points ) }, SI( -1 ), NoErrorBuffer{}, SI( -1 ) ),
        ShapeVarView<NoneTensor<std::int32_t, Tuple<>, Tuple<>>, NoErrorBuffer>{ NoneTensor<std::int32_t, Tuple<>, Tuple<>>{}, SI( -1 ), NoErrorBuffer{}, SI( -1 ) },
        Ct<SI, 2>{},
        tensor_view<CpuHostMemorySpace>( ffi_positions.typed_data(), tuple( SI( ffi_positions.dimensions()[ 0 ] ), Ct<SI, 2>() ), tuple( num_point, dim ) ),
        tensor_view<CpuHostMemorySpace>( ffi_weights.typed_data(), tuple( SI( ffi_weights.dimensions()[ 0 ] ) ), tuple( num_point ) )
    };
    auto power_diagram_io = PowerDiagram_Plain_io{ InpList(), InpList(), UndefList(), UndefList(), InpList(), UndefList(), InpList(), InpList(), InpList() };
    auto grad_for_power_diagram = PowerDiagram_Plain{
        NoneTensor<double, Tuple<Ct<SI, 2>>, Tuple<_dim>>{},
        NoneTensor<double, Tuple<Ct<SI, 2>>, Tuple<_dim>>{},
        NoneTensor<double, Tuple<SI, Ct<SI, 2>>, Tuple<_num_boundary, _dim>>{},
        NoneTensor<double, Tuple<SI>, Tuple<_num_boundary>>{},
        make_shape_var_view( ScalarValue<SI>{ SI( count_grad_for_power_diagram_nb_points ) }, SI( -1 ), NoErrorBuffer{}, SI( -1 ) ),
        ShapeVarView<NoneTensor<std::int32_t, Tuple<>, Tuple<>>, NoErrorBuffer>{ NoneTensor<std::int32_t, Tuple<>, Tuple<>>{}, SI( -1 ), NoErrorBuffer{}, SI( -1 ) },
        Ct<SI, 2>{},
        NoneTensor<double, Tuple<SI, Ct<SI, 2>>, Tuple<_num_point, _dim>>{},
        tensor_view<CpuHostMemorySpace>( ffi_weights_->typed_data(), tuple( SI( ffi_weights_->dimensions()[ 0 ] ) ), tuple( num_point ) )
    };
    auto grad_for_power_diagram_io = PowerDiagram_Plain_io{ UndefList(), UndefList(), UndefList(), UndefList(), InpList(), UndefList(), InpList(), UndefList(), OutList() };
    auto dom_cell = Cell_2{
        tensor_view<CpuHostMemorySpace>( ffi_vertex_positions.typed_data(), tuple( SI( ffi_vertex_positions.dimensions()[ 0 ] ), Ct<SI, 2>() ), tuple( num_vertex, dim_axis ) ),
        tensor_view<CpuHostMemorySpace>( ffi_cut_ids.typed_data(), tuple( SI( ffi_cut_ids.dimensions()[ 0 ] ) ), tuple( num_cut ) ),
        make_shape_var_view( tensor_view<CpuHostMemorySpace>( ffi_nb_vertices.typed_data(), tuple(  ), tuple(  ) ), SI( max_ffi_nb_vertices ), NoErrorBuffer{}, SI( -1 ) ),
        make_shape_var_view( tensor_view<CpuHostMemorySpace>( ffi_nb_cuts.typed_data(), tuple(  ), tuple(  ) ), SI( max_ffi_nb_cuts ), NoErrorBuffer{}, SI( -1 ) ),
        Ct<SI, 2>{}
    };
    auto dom_cell_io = Cell_2_io{ InpList(), InpList(), InpList(), InpList(), InpList() };
    auto grad_for_dom_cell = Cell_2{
        NoneTensor<double, Tuple<SI, Ct<SI, 2>>, Tuple<_num_vertex, _dim_axis>>{},
        NoneTensor<std::int32_t, Tuple<SI>, Tuple<_num_cut>>{},
        make_shape_var_view( tensor_view<CpuHostMemorySpace>( ffi_nb_vertices_.typed_data(), tuple(  ), tuple(  ) ), SI( max_ffi_nb_vertices_ ), NoErrorBuffer{}, SI( -1 ) ),
        make_shape_var_view( tensor_view<CpuHostMemorySpace>( ffi_nb_cuts_.typed_data(), tuple(  ), tuple(  ) ), SI( max_ffi_nb_cuts_ ), NoErrorBuffer{}, SI( -1 ) ),
        Ct<SI, 2>{}
    };
    auto grad_for_dom_cell_io = Cell_2_io{ UndefList(), UndefList(), InpList(), InpList(), InpList() };
    auto res = tensor_view<CpuHostMemorySpace>( ffi_res.typed_data(), tuple( SI( ffi_res.dimensions()[ 0 ] ) ), tuple( num_point ) );
    auto grad_for_res = tensor_view<CpuHostMemorySpace>( ffi_grad_for_res.typed_data(), tuple( SI( ffi_grad_for_res.dimensions()[ 0 ] ) ), tuple( num_point ) );
    auto scratch = CellScratch{
        tensor_view<CpuHostMemorySpace>( ffi_words->typed_data(), tuple( SI( ffi_words->dimensions()[ 0 ] ), SI( ffi_words->dimensions()[ 1 ] ), SI( ffi_words->dimensions()[ 2 ] ) ), tuple( thread_0, num_thread, num_word ) ),
        make_shape_var_view( tensor_view<CpuHostMemorySpace>( ffi_nb_threads->typed_data(), tuple( SI( ffi_nb_threads->dimensions()[ 0 ] ) ), tuple( thread_0 ) ), SI( max_ffi_nb_threads ), errors, SI( 0 ) ),
        make_shape_var_view( tensor_view<CpuHostMemorySpace>( ffi_nb_words->typed_data(), tuple( SI( ffi_nb_words->dimensions()[ 0 ] ) ), tuple( thread_0 ) ), SI( max_ffi_nb_words ), errors, SI( 1 ) ),
        Ct<SI, 64>{}
    };
    auto scratch_io = CellScratch_io{ OutList(), OutList(), OutList(), InpList() };
    errors.fill_with( queue, 0 );
    grad_for_power_diagram.weights.fill_with( queue, 0 );
    scratch.words.fill_with( queue, 0 );
    scratch.nb_threads.fill_with( queue, 0 );
    scratch.nb_words.fill_with( queue, 0 );
    {
run_parallel(
    queue,
    global_batch_indices,
    power_diagram_measures_bwd_fwd_kernel{},
    power_diagram_io, power_diagram, grad_for_power_diagram_io, grad_for_power_diagram, dom_cell_io, dom_cell, grad_for_dom_cell_io, grad_for_dom_cell, InpList(), res, InpList(), grad_for_res, scratch_io, scratch
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
        .Arg<ffi::BufferR1<ffi::F64>>()
        .Arg<ffi::BufferR1<ffi::F64>>()
        .Arg<ffi::BufferR2<ffi::F64>>()
        .Arg<ffi::BufferR1<ffi::F64>>()
        .Arg<ffi::BufferR2<ffi::F64>>()
        .Arg<ffi::BufferR1<ffi::S32>>()
        .Arg<ffi::BufferR1<ffi::S32>>()
        .Arg<ffi::BufferR1<ffi::S32>>()
        .Arg<ffi::BufferR1<ffi::S32>>()
        .Arg<ffi::BufferR1<ffi::S32>>()
        .Arg<ffi::BufferR1<ffi::F64>>()
        .Arg<ffi::BufferR1<ffi::F64>>()
        .Ret<ffi::BufferR1<ffi::F64>>()
        .Ret<ffi::BufferR3<ffi::S32>>()
        .Ret<ffi::BufferR1<ffi::S32>>()
        .Ret<ffi::BufferR1<ffi::S32>>()
        .Ret<ffi::BufferR1<ffi::S32>>()
        .Attr<int64_t>( "count_power_diagram_nb_points" )
        .Attr<int64_t>( "count_grad_for_power_diagram_nb_points" )
        .Attr<int64_t>( "max_ffi_nb_vertices" )
        .Attr<int64_t>( "max_ffi_nb_cuts" )
        .Attr<int64_t>( "max_ffi_nb_vertices_" )
        .Attr<int64_t>( "max_ffi_nb_cuts_" )
        .Attr<int64_t>( "max_ffi_nb_threads" )
        .Attr<int64_t>( "max_ffi_nb_words" )
        .Attr<int64_t>( "nb_batch_thread_0" ) );

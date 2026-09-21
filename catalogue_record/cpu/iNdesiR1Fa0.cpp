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
#include "sdot/generated/axes/num_bsp_seed.h"
#include "sdot/generated/axes/num_bsp_node.h"
#include "sdot/generated/axes/num_lohi.h"
#include "sdot/generated/axes/num_point.h"
#include "sdot/generated/axes/num_memo.h"
#include "sdot/generated/axes/num_vertex.h"
#include "sdot/generated/axes/dim_axis.h"
#include "sdot/generated/axes/num_cut.h"
#include "sdot/generated/axes/thread_0.h"
#include "sdot/generated/axes/num_thread.h"
#include "sdot/generated/axes/num_word.h"
#include "sdot/generated/axes/img_pos_0.h"
#include "sdot/generated/axes/img_pos_1.h"
#include "sdot/generated/axes/img_pos_2.h"
#include "sdot/generated/axes/dir.h"
#include "sdot/generated/axes/num_knot.h"
#include "sdot/generated/axes/num_cell_cum.h"


// the headers the arguments and the body asked for: the manual struct of each aggregate (which
// pulls in its own generated macros), then whatever the body listed for itself.
#include "sdot/PowerDiagram_Bsp.h"
#include "sdot/AaBsp.h"
#include "sdot/Cell_N.h"
#include "sdot/generated/aggregates/CellScratch_full.h"
#include "sdot/Image.h"


// what the body runs per item: a NAMED functor at namespace scope (never a lambda -- a device
// compiler is happier with a plain struct), rendered by the code object from the call's arguments.
struct power_diagram_measures_fwd_kernel {
    template<class BatchIndex, class T_power_diagram, class T_dom_cell, class T_res, class T_scratch, class T_distribution, class T_memo_nbrs_out, class T_memo_counts_out>
    HD void operator()( BatchIndex batch_index, int thread_index, int nb_threads, T_power_diagram power_diagram, T_dom_cell dom_cell, T_res res, T_scratch scratch, T_distribution distribution, T_memo_nbrs_out memo_nbrs_out, T_memo_counts_out memo_counts_out ) const {
        power_diagram.measures( res, dom_cell, scratch( batch_index ), distribution, memo_nbrs_out, memo_counts_out, thread_index, nb_threads );
    }
};


static ffi::Error sdot_ffi_impl( ffi::BufferR1<ffi::F64> ffi_box_min, ffi::BufferR1<ffi::F64> ffi_box_max, ffi::BufferR1<ffi::S64> ffi_seed_indices, ffi::BufferR1<ffi::S64> ffi_node_left, ffi::BufferR1<ffi::S64> ffi_node_right, ffi::BufferR1<ffi::S64> ffi_node_begin, ffi::BufferR1<ffi::S64> ffi_node_end, ffi::BufferR3<ffi::F64> ffi_node_box, ffi::BufferR2<ffi::F64> ffi_sorted_positions, ffi::BufferR2<ffi::S32> ffi_memo_nbrs, ffi::BufferR1<ffi::S32> ffi_memo_counts, ffi::BufferR2<ffi::F64> ffi_vertex_positions, ffi::BufferR2<ffi::S32> ffi_vertex_cuts, ffi::BufferR2<ffi::S32> ffi_vertex_nbrs, ffi::BufferR1<ffi::S32> ffi_cut_ids, ffi::BufferR1<ffi::S32> ffi_nb_vertices, ffi::BufferR1<ffi::S32> ffi_nb_cuts, ffi::BufferR0<ffi::F64> ffi_target_mass, ffi::BufferR1<ffi::S32> ffi_shape, ffi::BufferR3<ffi::F64> ffi_values, ffi::BufferR0<ffi::F64> ffi_current_mass, ffi::Result<ffi::BufferR1<ffi::F64>> ffi_res, ffi::Result<ffi::BufferR3<ffi::S32>> ffi_words, ffi::Result<ffi::BufferR1<ffi::S32>> ffi_nb_threads, ffi::Result<ffi::BufferR1<ffi::S32>> ffi_nb_words, ffi::Result<ffi::BufferR2<ffi::S32>> ffi_memo_nbrs_out, ffi::Result<ffi::BufferR1<ffi::S32>> ffi_memo_counts_out, ffi::Result<ffi::BufferR1<ffi::S32>> ffi_errors, int64_t count_power_diagram_nb_points, int64_t count_power_diagram_tree_nb_bsp_seeds, int64_t count_power_diagram_tree_nb_bsp_nodes, int64_t count_power_diagram_nb_memo, int64_t max_ffi_nb_vertices, int64_t max_ffi_nb_cuts, int64_t max_ffi_nb_threads, int64_t max_ffi_nb_words, int64_t max_ffi_shape, int64_t nb_batch_thread_0 ) {
    // the execution context of this call (the device is in the TYPE of everything below; how the
    // queue is obtained is the device's business, see `Device.cpp_queue_decl`).
    static Queue &queue = *new Queue();

    // what the body iterates over: the multi-indices of the batch axes. Unmapped, that is a single
    // item -- the EMPTY multi-index -- and a `vmap` is what gives it axes. Named ones: the body
    // applies `batch_index` to a value, which selects the axes it has and ignores the others.
    CartesianIndices<Tuple<SI>,Tuple<_thread_0>> global_batch_indices{ tuple( SI( nb_batch_thread_0 ) ) };

    auto errors = make_error_buffer( tensor_view<CpuHostMemorySpace>( ffi_errors->typed_data(), tuple( SI( ffi_errors->dimensions()[ 0 ] ) ) ), SI( 8 ) );
    auto power_diagram = PowerDiagram_Bsp{
        tensor_view<CpuHostMemorySpace>( ffi_box_min.typed_data(), tuple( Ct<SI, 3>() ), tuple( dim ) ),
        tensor_view<CpuHostMemorySpace>( ffi_box_max.typed_data(), tuple( Ct<SI, 3>() ), tuple( dim ) ),
        NoneTensor<double, Tuple<SI, Ct<SI, 3>>, Tuple<_num_boundary, _dim>>{},
        NoneTensor<double, Tuple<SI>, Tuple<_num_boundary>>{},
        make_shape_var_view( ScalarValue<SI>{ SI( count_power_diagram_nb_points ) }, SI( -1 ), NoErrorBuffer{}, SI( -1 ) ),
        ShapeVarView<NoneTensor<std::int32_t, Tuple<>, Tuple<>>, NoErrorBuffer>{ NoneTensor<std::int32_t, Tuple<>, Tuple<>>{}, SI( -1 ), NoErrorBuffer{}, SI( -1 ) },
        Ct<SI, 3>{},
        AaBsp{ tensor_view<CpuHostMemorySpace>( ffi_seed_indices.typed_data(), tuple( SI( ffi_seed_indices.dimensions()[ 0 ] ) ), tuple( num_bsp_seed ) ), tensor_view<CpuHostMemorySpace>( ffi_node_left.typed_data(), tuple( SI( ffi_node_left.dimensions()[ 0 ] ) ), tuple( num_bsp_node ) ), tensor_view<CpuHostMemorySpace>( ffi_node_right.typed_data(), tuple( SI( ffi_node_right.dimensions()[ 0 ] ) ), tuple( num_bsp_node ) ), tensor_view<CpuHostMemorySpace>( ffi_node_begin.typed_data(), tuple( SI( ffi_node_begin.dimensions()[ 0 ] ) ), tuple( num_bsp_node ) ), tensor_view<CpuHostMemorySpace>( ffi_node_end.typed_data(), tuple( SI( ffi_node_end.dimensions()[ 0 ] ) ), tuple( num_bsp_node ) ), tensor_view<CpuHostMemorySpace>( ffi_node_box.typed_data(), tuple( SI( ffi_node_box.dimensions()[ 0 ] ), Ct<SI, 2>(), Ct<SI, 3>() ), tuple( num_bsp_node, num_lohi, dim ) ), NoneTensor<double, Tuple<SI, Ct<SI, 3>>, Tuple<_num_bsp_node, _dim>>{}, NoneTensor<double, Tuple<SI>, Tuple<_num_bsp_node>>{}, make_shape_var_view( ScalarValue<SI>{ SI( count_power_diagram_tree_nb_bsp_seeds ) }, SI( -1 ), NoErrorBuffer{}, SI( -1 ) ), make_shape_var_view( ScalarValue<SI>{ SI( count_power_diagram_tree_nb_bsp_nodes ) }, SI( -1 ), NoErrorBuffer{}, SI( -1 ) ), Ct<SI, 2>{}, Ct<SI, 3>{} },
        tensor_view<CpuHostMemorySpace>( ffi_sorted_positions.typed_data(), tuple( SI( ffi_sorted_positions.dimensions()[ 0 ] ), Ct<SI, 3>() ), tuple( num_point, dim ) ),
        NoneTensor<double, Tuple<SI>, Tuple<_num_point>>{},
        tensor_view<CpuHostMemorySpace>( ffi_memo_nbrs.typed_data(), tuple( SI( ffi_memo_nbrs.dimensions()[ 0 ] ), SI( ffi_memo_nbrs.dimensions()[ 1 ] ) ), tuple( num_point, num_memo ) ),
        tensor_view<CpuHostMemorySpace>( ffi_memo_counts.typed_data(), tuple( SI( ffi_memo_counts.dimensions()[ 0 ] ) ), tuple( num_point ) ),
        make_shape_var_view( ScalarValue<SI>{ SI( count_power_diagram_nb_memo ) }, SI( -1 ), NoErrorBuffer{}, SI( -1 ) )
    };
    auto power_diagram_io = PowerDiagram_Bsp_io{ InpList(), InpList(), UndefList(), UndefList(), InpList(), UndefList(), InpList(), AaBsp_io{ InpList(), InpList(), InpList(), InpList(), InpList(), InpList(), UndefList(), UndefList(), InpList(), InpList(), InpList(), InpList() }, InpList(), UndefList(), InpList(), InpList(), InpList() };
    auto dom_cell = Cell_N{
        tensor_view<CpuHostMemorySpace>( ffi_vertex_positions.typed_data(), tuple( SI( ffi_vertex_positions.dimensions()[ 0 ] ), Ct<SI, 3>() ), tuple( num_vertex, dim_axis ) ),
        tensor_view<CpuHostMemorySpace>( ffi_vertex_cuts.typed_data(), tuple( SI( ffi_vertex_cuts.dimensions()[ 0 ] ), Ct<SI, 3>() ), tuple( num_vertex, dim_axis ) ),
        tensor_view<CpuHostMemorySpace>( ffi_vertex_nbrs.typed_data(), tuple( SI( ffi_vertex_nbrs.dimensions()[ 0 ] ), Ct<SI, 3>() ), tuple( num_vertex, dim_axis ) ),
        tensor_view<CpuHostMemorySpace>( ffi_cut_ids.typed_data(), tuple( SI( ffi_cut_ids.dimensions()[ 0 ] ) ), tuple( num_cut ) ),
        make_shape_var_view( tensor_view<CpuHostMemorySpace>( ffi_nb_vertices.typed_data(), tuple(  ), tuple(  ) ), SI( max_ffi_nb_vertices ), NoErrorBuffer{}, SI( -1 ) ),
        make_shape_var_view( tensor_view<CpuHostMemorySpace>( ffi_nb_cuts.typed_data(), tuple(  ), tuple(  ) ), SI( max_ffi_nb_cuts ), NoErrorBuffer{}, SI( -1 ) ),
        Ct<SI, 3>{}
    };
    auto dom_cell_io = Cell_N_io{ InpList(), InpList(), InpList(), InpList(), InpList(), InpList(), InpList() };
    auto res = tensor_view<CpuHostMemorySpace>( ffi_res->typed_data(), tuple( SI( ffi_res->dimensions()[ 0 ] ) ), tuple( num_point ) );
    auto scratch = CellScratch{
        tensor_view<CpuHostMemorySpace>( ffi_words->typed_data(), tuple( SI( ffi_words->dimensions()[ 0 ] ), SI( ffi_words->dimensions()[ 1 ] ), SI( ffi_words->dimensions()[ 2 ] ) ), tuple( thread_0, num_thread, num_word ) ),
        make_shape_var_view( tensor_view<CpuHostMemorySpace>( ffi_nb_threads->typed_data(), tuple( SI( ffi_nb_threads->dimensions()[ 0 ] ) ), tuple( thread_0 ) ), SI( max_ffi_nb_threads ), errors, SI( 0 ) ),
        make_shape_var_view( tensor_view<CpuHostMemorySpace>( ffi_nb_words->typed_data(), tuple( SI( ffi_nb_words->dimensions()[ 0 ] ) ), tuple( thread_0 ) ), SI( max_ffi_nb_words ), errors, SI( 1 ) ),
        Ct<SI, 32>{}
    };
    auto scratch_io = CellScratch_io{ OutList(), OutList(), OutList(), InpList() };
    auto distribution = Image{
        tensor_view<CpuHostMemorySpace>( ffi_target_mass.typed_data(), tuple(  ), tuple(  ) ),
        Ct<SI, 3>{},
        make_shape_var_view( tensor_view<CpuHostMemorySpace>( ffi_shape.typed_data(), tuple( SI( ffi_shape.dimensions()[ 0 ] ) ), tuple( UnnamedAxis{} ) ), SI( max_ffi_shape ), NoErrorBuffer{}, SI( -1 ) ),
        tensor_view<CpuHostMemorySpace>( ffi_values.typed_data(), tuple( SI( ffi_values.dimensions()[ 0 ] ), SI( ffi_values.dimensions()[ 1 ] ), SI( ffi_values.dimensions()[ 2 ] ) ), tuple( img_pos_0, img_pos_1, img_pos_2 ) ),
        NoneTensor<double, Tuple<Ct<SI, 3>>, Tuple<_dim>>{},
        NoneTensor<double, Tuple<Ct<SI, 3>, Ct<SI, 3>>, Tuple<_dir, _dim>>{},
        NoneTensor<double, Tuple<Ct<SI, 3>, SI>, Tuple<_dim, _num_knot>>{},
        tensor_view<CpuHostMemorySpace>( ffi_current_mass.typed_data(), tuple(  ), tuple(  ) ),
        ShapeVarView<NoneTensor<std::int32_t, Tuple<>, Tuple<>>, NoErrorBuffer>{ NoneTensor<std::int32_t, Tuple<>, Tuple<>>{}, SI( -1 ), NoErrorBuffer{}, SI( -1 ) },
        NoneTensor<double, Tuple<SI>, Tuple<_num_cell_cum>>{}
    };
    auto distribution_io = Image_io{ InpList(), InpList(), InpList(), InpList(), UndefList(), UndefList(), UndefList(), InpList(), UndefList(), UndefList() };
    auto memo_nbrs_out = tensor_view<CpuHostMemorySpace>( ffi_memo_nbrs_out->typed_data(), tuple( SI( ffi_memo_nbrs_out->dimensions()[ 0 ] ), SI( ffi_memo_nbrs_out->dimensions()[ 1 ] ) ), tuple( num_point, num_memo ) );
    auto memo_counts_out = tensor_view<CpuHostMemorySpace>( ffi_memo_counts_out->typed_data(), tuple( SI( ffi_memo_counts_out->dimensions()[ 0 ] ) ), tuple( num_point ) );
    errors.fill_with( queue, 0 );
    scratch.words.fill_with( queue, 0 );
    scratch.nb_threads.fill_with( queue, 0 );
    scratch.nb_words.fill_with( queue, 0 );
    {
run_parallel(
    queue,
    global_batch_indices,
    power_diagram_measures_fwd_kernel{},
    power_diagram_io, power_diagram, dom_cell_io, dom_cell, OutList(), res, scratch_io, scratch, distribution_io, distribution, OutList(), memo_nbrs_out, OutList(), memo_counts_out
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
        .Arg<ffi::BufferR1<ffi::S64>>()
        .Arg<ffi::BufferR1<ffi::S64>>()
        .Arg<ffi::BufferR1<ffi::S64>>()
        .Arg<ffi::BufferR1<ffi::S64>>()
        .Arg<ffi::BufferR1<ffi::S64>>()
        .Arg<ffi::BufferR3<ffi::F64>>()
        .Arg<ffi::BufferR2<ffi::F64>>()
        .Arg<ffi::BufferR2<ffi::S32>>()
        .Arg<ffi::BufferR1<ffi::S32>>()
        .Arg<ffi::BufferR2<ffi::F64>>()
        .Arg<ffi::BufferR2<ffi::S32>>()
        .Arg<ffi::BufferR2<ffi::S32>>()
        .Arg<ffi::BufferR1<ffi::S32>>()
        .Arg<ffi::BufferR1<ffi::S32>>()
        .Arg<ffi::BufferR1<ffi::S32>>()
        .Arg<ffi::BufferR0<ffi::F64>>()
        .Arg<ffi::BufferR1<ffi::S32>>()
        .Arg<ffi::BufferR3<ffi::F64>>()
        .Arg<ffi::BufferR0<ffi::F64>>()
        .Ret<ffi::BufferR1<ffi::F64>>()
        .Ret<ffi::BufferR3<ffi::S32>>()
        .Ret<ffi::BufferR1<ffi::S32>>()
        .Ret<ffi::BufferR1<ffi::S32>>()
        .Ret<ffi::BufferR2<ffi::S32>>()
        .Ret<ffi::BufferR1<ffi::S32>>()
        .Ret<ffi::BufferR1<ffi::S32>>()
        .Attr<int64_t>( "count_power_diagram_nb_points" )
        .Attr<int64_t>( "count_power_diagram_tree_nb_bsp_seeds" )
        .Attr<int64_t>( "count_power_diagram_tree_nb_bsp_nodes" )
        .Attr<int64_t>( "count_power_diagram_nb_memo" )
        .Attr<int64_t>( "max_ffi_nb_vertices" )
        .Attr<int64_t>( "max_ffi_nb_cuts" )
        .Attr<int64_t>( "max_ffi_nb_threads" )
        .Attr<int64_t>( "max_ffi_nb_words" )
        .Attr<int64_t>( "max_ffi_shape" )
        .Attr<int64_t>( "nb_batch_thread_0" ) );

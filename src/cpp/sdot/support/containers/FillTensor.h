#pragma once

#include "../common_types.h"
#include "../Ct.h"

namespace sdot {

/// A tensor whose every element reads as the SAME runtime value, with no per-element storage: only a
/// single scalar (a rank-0 buffer) backs it, and indexing ignores the index and yields that value.
///
/// Same spirit as `ZeroTensor` (a storageless, read-only VALUE known through its TYPE), but the value
/// is not baked into the type -- it is read through `data` at run time, so the SAME compiled kernel
/// serves any fill value (a `1/n` never recompiles). Lowered from a symbolic `Tensor.full( v )`. Read
/// only: there is no `operator=` (nowhere to write a broadcast). No `size()`: its logical extent is
/// not stored, so a kernel must take the count from a real buffer, never from a fill.
template<class _TF, class _Shape, class _AxisNames>
struct FillTensor {
    using            TF                     = _TF;
    using            Shape                  = _Shape;
    using            AxisNames              = _AxisNames;
    SCInt            ct_rank                = Shape::ct_size;

    const TF        *data;                  ///< points at the ONE scalar every element reads as

    constexpr auto   is_valid               () const { return Ct<bool,true>(); } ///< a real value, storageless

    // indexing yields a rank-0 fill over the same scalar; reading it yields the value.
    constexpr auto   operator()             ( auto &&... ) const { return FillTensor<TF, Tuple<>, Tuple<>>{ data }; }
    TF               value                  () const { return *data; }
    /* */            operator TF            () const { return *data; }

    // as a `run_parallel` argument: the scalar is already where the kernel runs (an FFI input XLA put
    // on the device), so it crosses unchanged, like `ZeroTensor`.
    constexpr auto   transfer_cost          ( const auto &/*queue*/, auto /*io_category*/ ) const { return Ct<double,0.0>(); }
    constexpr auto   make_available         ( auto &&/*queue*/, auto /*io_category*/, auto &&cont ) const { return cont( *this ); }

    void             display                ( auto &ds ) const { ds << "FillTensor(" << *data << ")"; }
};

} // namespace sdot

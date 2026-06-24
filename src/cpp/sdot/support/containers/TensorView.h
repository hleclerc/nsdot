#pragma once

// #include "../hardware/MemorySpace_GlobalCudaRam.h" // IWYU pragma: export
// #include "../hardware/MemorySpace_PinnedCpuRam.h" // IWYU pragma: export
// #include "../hardware/MemorySpace_CpuRam.h" // IWYU pragma: export
// #include "../hardware/Ptr.h"

#include "internal/contiguous_strides.h" // IWYU pragma: export
#include "../kernels/CpuHostMemorySpace.h"
#include "../kernels/Ptr.h" // IWYU pragma: export
#include "AxisNames.h" // IWYU pragma: export
#include "TupleRep.h" // IWYU pragma: export
#include <type_traits>
// #include "container_tags.h" // IWYU pragma: export
// #include "AxisNames.h" // IWYU pragma: export
// #include "Vector.h" // IWYU pragma: export

namespace sdot {


/// view on strided data (strides in bytes, handles non-contiguous arrays)
///
///
///   TF -> the scalar type
///   Shape ->  size for each axis
///   AxisNames -> name of each axis (UnnamedAxis if not named), see AxisNames.h / DEFINE_AXIS
template<
    class _TF,
    class _Shape,
    class _MemorySpace,
    class _AxisNames = DECAYED_TYPE_OF( unnamed_axes( _Shape{} ) ),
    class _Strides = DECAYED_TYPE_OF( contiguous_strides<_TF>( _Shape{} ) )
>
class TensorView {
public:
    using            MemorySpace            = _MemorySpace;
    using            AxisNames              = _AxisNames;
    using            Strides                = _Strides;
    using            Shape                  = _Shape;
    using            TF                     = _TF;
    using            TI                     = SI;

    using            value_type             = TF;

    using            RawByte                = std::conditional_t<std::is_const_v<TF>,const std::byte,std::byte>; ///< octet const ou non selon la constness de TF (les strides sont en octets)
    using            BytePtr                = Ptr<RawByte,MemorySpace>; ///< pointeur octet + zone mémoire (ce qu'on stocke)
    using            DataPtr                = Ptr<TF,MemorySpace>;      ///< pointeur typé renvoyé par data()
    SCInt            ct_rank                = Shape::ct_size;

    /* */            TensorView             ( DataPtr data, Shape shape, Strides strides ); ///< pointeur typé ; stocké en interne comme BytePtr (strides en octets)
    /* */            TensorView             ( const TensorView & ) = default; ///< Eigen-like view semantics: copy-construction shares the data (shallow), while operator= copies the elements (deep). The defaulted copy-ctor also silences -Wdeprecated-copy.
    /* */            TensorView             () = default;

    // generic info
    bool             not_surely_null        () const { return ! surely_null(); }
    bool             surely_null            () const; ///< is_invalid() || Zero tensor
    bool             is_invalid             () const; ///<
    bool             is_valid               () const; ///<

    MemorySpace      memory_space           () const { return _data.memory_space; }
    // (pas de membre display : le display() générique de display.h gère TensorView via shape()/value()/operator[])

    //
    Strides          strides                () const;
    auto             stride                 ( auto d ) const;
    auto             rank                   () const;

    // shape
    // for_each_index / for_each_item : à migrer (dépendent de cartesian_product/range)
    auto             indices_col_ordering   ( auto index ) const;
    auto             items_are_contiguous   () const; ///<
    void             for_each_scalar        ( auto &&func ) const; ///< appelle func( vue_rang_0 ) pour chaque élément (boucle simple récursive)
    auto             all_indices            () const;
    auto             nb_items               () const;
    auto             shape                  ( auto d ) const { return _shape[ d ]; }
    Shape            shape                  () const { return _shape; }
    auto             empty                  () const;
    auto             size                   () const;

    // content
    auto             data                   () const;

    auto             begin                  () const;
    auto             end                    () const;

    // operator() and operator[] produce a new tensor
    auto             operator()             ( const auto &index, auto ...rem ) const;
    auto             operator[]             ( const auto &index ) const { return operator()( index ); }
    auto             operator()             () const { return *this; }

    auto             offset                 ( const auto &index, auto ...rem ) const;
    auto             offset                 () const { return *this; }

    // scalar value/reference for a rank 1 tensor
    /* */            operator TF           () const { return value(); }
    TF               value                 () const;
    TF&              ref                   () const;

    // reassign
    void             copy_elements_from     ( const auto &that );
    void             operator-=             ( const auto &that );
    void             operator+=             ( const auto &that );
    void             operator*=             ( const auto &that );
    void             operator/=             ( const auto &that );
    void             operator=              ( const auto &that );
    void             operator=              ( const TensorView &that );
    void             spill_to               ( TensorView &that ); ///< copy data of *this to that, and use data from that

    // data copy / transfer
    auto             transfer_cost          ( const auto &queue, auto io_category ) const;

    // rend la vue accessible depuis le contexte d'exécution `queue` : si le coût de transfert
    // est nul on retype simplement le Ptr vers la zone kernel cible, sinon on transfère
    // (alloc + copy selon io_category). Appelle ensuite cont( vue_kernel ).
    auto             make_available         ( auto &&queue, auto io_category, auto &&cont ) const;

    void             fill_with              ( auto &&avaible_queues, TF value );
    void             fill_with              ( TF value );

    //
    auto             unsqueeze              ( auto axis ) const; ///< append a trailing dimension of size 1 (preserves strides)
    auto             squeeze                ( auto axis, auto index ) const; ///< axis = position (Ct) ou nom d'axe, + valeur
    auto             squeeze                ( auto axis_index ) const;       ///< axis_index = (nom = valeur)
    auto             row                    ( auto index ) const;

    Strides          _strides;              ///< strides en octets
    Shape            _shape;                ///<
    BytePtr          _data;                 ///< pointeur octet + zone mémoire, agrégés
};

auto tensor_view( auto *ptr, auto &&shape, auto &&axis_names, auto &&strides ) {
    using TF = DECAYED_TYPE_OF( *ptr );
    return TensorView<TF,DECAYED_TYPE_OF( shape ),CpuHostMemorySpace,DECAYED_TYPE_OF( axis_names ),DECAYED_TYPE_OF( strides )>(
        Ptr<TF,CpuHostMemorySpace>( ptr ), FORWARD( shape ), FORWARD( strides )
    );
}
auto tensor_view( auto *ptr, auto &&shape, auto &&axis_names ) { return tensor_view( ptr, shape, FORWARD( axis_names ), contiguous_strides<DECAYED_TYPE_OF( *ptr )>( shape ) ); }
auto tensor_view( auto *ptr, auto &&shape ) { return tensor_view( ptr, FORWARD( shape ), unnamed_axes( shape ) ); }



// #undef SDOT_DATA_ACCESSOR

} // namespace sdot

#include "TensorView.cxx" // IWYU pragma: export

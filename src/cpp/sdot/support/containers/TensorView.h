#pragma once

// #include "../hardware/MemorySpace_GlobalCudaRam.h" // IWYU pragma: export
// #include "../hardware/MemorySpace_PinnedCpuRam.h" // IWYU pragma: export
// #include "../hardware/MemorySpace_CpuRam.h" // IWYU pragma: export
// #include "../hardware/Ptr.h"

// #include "contiguous_strides.h" // IWYU pragma: export
// #include "container_tags.h" // IWYU pragma: export
// #include "AxisNames.h" // IWYU pragma: export
// #include "Vector.h" // IWYU pragma: export
// #include "Tuple.h" // IWYU pragma: export

namespace sdot {


/// view on strided data (strides in bytes, handles non-contiguous arrays)
///
///
///   TF -> the scalar type
///   Shape ->  size for each axis
///   AxisNames -> name of each axis (Void if not named)
template<
    class _TF,
    class _Shape,
    class _AxisNames,
    class _StrideFunc //DECAYED_TYPE_OF( contiguous_strides<_TF>( _Shape{} ) ),
>
class TensorView {
public:
//     using            MemorySpace          = _MemorySpace;
//     using            AxisNames            = _AxisNames;
//     using            Strides              = _Strides;
//     using            Shape                = _Shape;
//     using            TF                   = _TF;
//     using            TI                   = SI;

//     using            value_type             = TF;
//     SCInt            ct_rank                = Shape::ct_size;

//     HD               TensorView             ( TF *data, Shape shape, Strides strides );
//     HD               TensorView             ( const TensorView & ) = default; ///< Eigen-like view semantics: copy-construction shares the data (shallow), while operator= copies the elements (deep). The defaulted copy-ctor also silences -Wdeprecated-copy.
//     HD               TensorView             ();

//     // generic info
//     HD bool          not_surely_null        () const { return ! surely_null(); }
//     HD bool          surely_null            () const; ///< is_invalid() || Zero tensor
//     HD bool          is_invalid             () const; ///<
//     HD bool          is_valid               () const; ///<

//     // generic info
//     MemorySpace      memory_space           () const { return _memory_space; }
//     void             display                ( std::ostream &os ) const;
//     HD auto          rank                   () const;

//     //
//     HD Strides       strides                () const;
//     T_T  HD auto     stride                 ( T d ) const;

//     // shape
//     T_T  HD void     for_each_index         ( T &&func ) const;
//     T_T  HD void     for_each_item          ( T &&func ) const;
//     HD auto          is_contiguous          () const; ///< true iff strides match row-major contiguous layout
//     HD auto          all_indices            () const;
//     HD auto          nb_items               () const;
//     T_T  HD auto     shape                  ( T d ) const { return _shape[ d ]; }
//     HD Shape         shape                  () const { return _shape; }
//     HD auto          empty                  () const;
//     HD auto          size                   () const;

//     // content
//     HD auto          data                   () const;

//     HD auto          begin                  () const;
//     HD auto          end                    () const;

//     // operator() and operator[] produce a new tensor
//     T_Tv HD auto     operator()             ( const T &index, V ...rem ) const;
//     T_T  HD auto     operator[]             ( const T &index ) const { return operator()( index ); }
//     HD auto          operator()             () const { return *this; }

//     T_Tv HD auto     offset                 ( const T &index, V ...rem ) const;
//     HD auto          offset                 () const { return *this; }

//     // scalar value/reference for a rank 1 tensor
//     HD               operator TF            () const { return value(); }
//     HD TF            value                  () const;
//     HD TF&           ref                    () const;

//     // reassign
//     T_T  HD void     copy_elements_from     ( const T &that );
//     T_T  HD void     operator-=             ( const T &that );
//     T_T  HD void     operator+=             ( const T &that );
//     T_T  HD void     operator*=             ( const T &that );
//     T_T  HD void     operator/=             ( const T &that );
//     T_T  HD void     operator=              ( const T &that );
//     HD void          operator=              ( const TensorView &that );
//     HD void          spill_to               ( TensorView &that ); ///< copy data of *this to that, and use data from that

//     // data copy / transfer — arch-unaware (HD, valid in device code)
//     T_T  HD auto     transfer_cost          ( const T &execution_context ) const;

//     T_TA void        with_same_shape        ( const T &arch, A &&func ) const;
//     HD void          fill_with              ( TF value );

//     //
//     T_T  HD auto     unsqueeze              ( T axis ) const; ///< append a trailing dimension of size 1 (preserves strides)
//     T_T  HD auto     squeeze                ( T axis, PI index = 0 ) const;
//     HD auto          row                    ( PI index ) const;

//     // compile-time tags (see container_tags.h)
//     template<class... ExtraTags>
//     HD auto          with_tags              () const; ///< same view, with ExtraTags... added to the tag set (no-op for tags already present)

// private:
//     static HD RawPtr _sentinel              () { return RawPtr( nullptr ) + 1; }

//     MemorySpace      _memory_space;       ///<
//     RawPtr           _raw_ptr;            ///<
//     Strides          _strides;            ///< byte strides
//     Shape            _shape;              ///<
};

// #undef SDOT_DATA_ACCESSOR

} // namespace sdot

// #include "TensorView.cxx" // IWYU pragma: export

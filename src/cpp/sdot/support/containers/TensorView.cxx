#pragma once

// #include "../hardware/current_execution_context.h"
// #include "../hardware/Run.h"

// #include "CartesianProduct.h"
// #include "TensorView.h"
// #include "Range.h"

#define UTP template<class TF,class MemorySpace,class Shape,class Strides,class... Tags>
#define DTP TensorView<TF,MemorySpace,Shape,Strides,Tags...>

namespace sdot {

// template <typename T> struct Is_TensorView : std::false_type {};
// UTP struct Is_TensorView<DTP> : std::true_type {};

// namespace details::TensorView {
//     /// bind the leading TensorView params and expand a TagList into the trailing tag pack
//     template<class TF,class MemorySpace,class Shape,class Strides,class List> struct with_tag_list;
//     template<class TF,class MemorySpace,class Shape,class Strides,class... Tags>
//     struct with_tag_list<TF,MemorySpace,Shape,Strides,sdot::TagList<Tags...>> { using type = sdot::TensorView<TF,MemorySpace,Shape,Strides,Tags...>; };
// } // namespace details::TensorView

// UTP HD DTP DTP::make_invalid( Shape shape, Strides strides, MemorySpace memory_space ) {
//     return TensorView( _sentinel().template as<TF>(), shape, strides, memory_space );
// }

// UTP HD DTP::TensorView( TF *data, Shape shape, Strides strides, MemorySpace memory_space ) : _memory_space( memory_space ), _raw_ptr( reinterpret_cast<RawByte *>( data ) ), _strides( strides ), _shape( shape ) {
// }

// UTP HD DTP::TensorView() : _raw_ptr( nullptr ) {
// }

// UTP T_Tv HD auto DTP::operator()( const T &index, V ...rem ) const {
//     if constexpr ( HAS_CONSTEXPR_SIZE( index ) ) {
//         if constexpr ( DECAYED_TYPE_OF( index.size() )::value )
//             return operator()( index[ Ct<int,0>() ], index.without_index( Ct<int,0>() ), rem... );
//         else
//             return operator()( rem... );
//     } else
//         return row( index )( rem... );
// }

// UTP T_Tv HD auto DTP::offset( const T &index, V ...rem ) const {
//     TensorView res = *this;
//     if constexpr ( HAS_CONSTEXPR_SIZE( index ) ) {
//         if constexpr ( DECAYED_TYPE_OF( index.size() )::value )
//             return offset( index[ Ct<int,0>() ], index.without_index( Ct<int,0>() ), rem... );
//         else
//             return offset( rem... );
//     } else {
//         res._shape.set( 0_c, res._shape[ 0_c ] - index );
//         res._raw_ptr.raw += _strides[ 0_c ] * index;
//         return res.offset( rem... );
//     }
// }



// // `index` is a full multi-index, so a[index] / b[index] are rank-0 views.
// struct AddTensorItemElementwise {              ///< same-shape operand: a[i] += b[i]
//     T_TAB HD void operator()( T index, A a, B b ) const { a[ index ] += b[ index ]; }
// };
// struct AddTensorItemBroadcast {                ///< scalar / rank-0 operand: a[i] += b
//     T_TAB HD void operator()( T index, A a, B b ) const { a[ index ] += b; }
// };

// /// true iff `B` is a tensor operand with the same rank as the destination (-> element-wise);
// /// anything else (a scalar, or a lower-rank view) is broadcast.
// template<class B,int dst_rank>
// HD constexpr bool add_is_elementwise() {
//     if constexpr ( HAS_CT_RANK( B ) )
//         return int( B::ct_rank ) == dst_rank;
//     else
//         return false;
// }

// UTP T_T HD void DTP::operator+=( const T &that ) {
//     if constexpr ( ct_rank == 0 ) {
//         // base case: accumulate in place (also what stops the per-item recursion above)
//         if constexpr ( DECAYED_TYPE_OF( accessible_from( current_execution_context(), *this ) )::value ) {
//             ref() += TF( that );
//         } else {
//             TF cur = value();
//             cur += TF( that );
//             operator=( cur );
//         }
//     } else if constexpr ( add_is_elementwise<DECAYED_TYPE_OF( that ),ct_rank>() ) {
//         run_parallel( cartesian_product_ranges( _shape ), AddTensorItemElementwise(), inout( *this ), that );
//     } else {
//         run_parallel( cartesian_product_ranges( _shape ), AddTensorItemBroadcast(), inout( *this ), that );
//     }
// }

// UTP T_T HD void DTP::operator-=( const T &that ) {
//     if constexpr ( ct_rank == 0 ) {
//         if constexpr ( DECAYED_TYPE_OF( accessible_from( current_execution_context(), *this ) )::value ) {
//             ref() -= TF( that );
//         } else {
//             TF cur = value();
//             cur -= TF( that );
//             operator=( cur );
//         }
//     } else if constexpr ( add_is_elementwise<DECAYED_TYPE_OF( that ),ct_rank>() ) {
//         TODO; // run_parallel( cartesian_product_ranges( _shape ), AddTensorItemElementwise(), inout( *this ), that );
//     } else {
//         TODO; // run_parallel( cartesian_product_ranges( _shape ), AddTensorItemBroadcast(), inout( *this ), that );
//     }
// }

// UTP T_T HD void DTP::operator*=( const T & ) {
//     TODO;
// }

// UTP T_T HD void DTP::operator/=( const T & ) {
//     TODO;
// }

// UTP HD void DTP::operator=( const TensorView &that ) {
//     copy_elements_from( that );
// }

// UTP T_T HD void DTP::operator=( const T &that ) {
//     copy_elements_from( that );
// }

// UTP T_T HD auto DTP::squeeze( T axis, PI index ) const {
//     // `axis` may be addressed by name: if `T` is one of this view's named axes (an AxisNames
//     // tag), resolve it to its position and squeeze positionally. Otherwise `T` is a Ct<int,N>
//     // position and we fall through to the positional body.
//     if constexpr ( is_axis_name<T,Tags...> ) {
//         return squeeze( Ct<int,axis_position_of<T,Tags...>>(), index );
//     } else {
//     auto new_strides = _strides.without_index( axis );
//     auto new_shape = _shape.without_index( axis );

//     using NewStrides = DECAYED_TYPE_OF( new_strides );
//     using NewShape = DECAYED_TYPE_OF( new_shape );

//     ASSERT( index < _shape[ axis ] );

//     auto ptr = ( _raw_ptr + _strides[ axis ] * index ).template as<TF>();

//     // tags are rewritten through the squeeze derivation (identity by default)
//     using Op = container_ops::squeeze<DECAYED_TYPE_OF( axis )>;
//     using Result = typename details::TensorView::with_tag_list<TF,MemorySpace,NewShape,NewStrides,tags_after<Op,Tags...>>::type;
//     return Result( ptr, new_shape, new_strides, _memory_space );
//     }
// }

// UTP HD auto DTP::row( PI index ) const {
//     return squeeze( Ct<int,0>(), index );
// }

// UTP template<class... ExtraTags> HD auto DTP::with_tags() const {
//     // same data, tag pack extended with ExtraTags... (appended verbatim, no axis transform)
//     return TensorView<TF,MemorySpace,Shape,Strides,Tags...,ExtraTags...>( data().raw, _shape, _strides, _memory_space );
// }

// UTP HD auto DTP::data() const {
//     return Ptr<TF,MemorySpace>( _raw_ptr.template as<TF>(), _memory_space );
// }

// UTP HD TF DTP::value() const {
//     static_assert( ct_rank == 0 );
//     return data().value();
// }

// UTP TF& DTP::ref() const {
//     static_assert( ct_rank == 0 );
//     return *data();
// }

// UTP void DTP::display( std::ostream &os ) const {
//     if constexpr ( DECAYED_TYPE_OF( transfer_cost( ExecutionContext_Cpu{} ) )::value ) {
//         make_accessible( ExecutionContext_Cpu{}, *this, 1_b, 0_b, [&]( auto &&tensor ) {
//             tensor.display( os );
//         } );
//     } else if constexpr ( ct_rank == 0 ) {
//         os << value();
//     } else if constexpr ( ct_rank == 1 ) {
//         for( TI i = 0; i < shape( 0_c ); ++i )
//             sdot::display( os << ( i ? ", " : "" ), operator[]( i ) );
//     } else {
//         for( std::size_t i = 0; i < shape( 0_c ); ++i )
//             sdot::display( os << "\n  ", operator[]( i ) );
//     }
// }

// UTP HD auto DTP::nb_items() const {
//     return product( _shape );
// }

// UTP T_T HD void DTP::copy_elements_from( const T &that ) {
//     if constexpr ( ct_rank == 0 ) {
//         if constexpr( Is_TensorView<DECAYED_TYPE_OF( that )>::value )
//             ref() = that.value();
//         else
//             ref() = that;
//     } else {
//         if ( std::is_same_v<TF,typename DECAYED_TYPE_OF(that)::TF> && _strides == that.strides() && is_contiguous() ) {
//             copy( data(), that.data(), nb_items() );
//         } else  {
//             run_sequential( cartesian_product_ranges( _shape ), [&]( auto indices, auto &&a, auto &&b ) {
//                 a[ indices ] = b[ indices ];
//             }, out( *this ), that );
//         }
//     }
// }

// UTP HD auto DTP::is_contiguous() const {
//     return _strides == contiguous_strides<TF>( _shape );
// }

// UTP T_T HD void DTP::for_each_index( T &&func ) const {
//     cartesian_product( map( _shape, range<PI> ) ).for_each_item( FORWARD( func ) );
// }

// UTP T_T HD void DTP::for_each_item( T &&func ) const {
//     for_each_index( [&]( auto &index ) {
//         func( operator()( index ) );
//     } );
// }

// UTP HD auto DTP::size() const {
//     static_assert( ct_rank == 1, "..." );
//     return shape( Ct<int,0>() );
// }

// UTP HD auto DTP::empty() const {
//     if constexpr ( ct_rank == 0 )
//         return 0_b;
//     return _shape.apply_values( [&]( auto ...values ) {
//         return ( ( values == 0_c ) || ... || 0_b );
//     } );
// }

// namespace details::TensorView {
//     // Namespace-scope functors for arch-aware element-wise ops.
//     // Lambda bodies inside HD/GD template methods from .cxx files cause issues with
//     // some nvcc versions when the lambda references class-level template params (TF).
//     // Using concrete struct operator() avoids the problem.
//     template<class DstTV, class SrcTV, class BI>
//     struct TensorCopyFunctor {
//         DstTV dst;
//         SrcTV src;
//         GD void operator()( BI bi ) const {
//             dst( bi ).item() = src( bi ).item();
//         }
//     };

//     struct TensorFillFunctor {
//         T_TAB GD void operator()( T index, A dst, B value ) const {
//             dst( index ) = value;
//         }
//     };
// } // namespace details::TensorView

// UTP HD auto DTP::all_indices() const {
//     return cartesian_product_ranges( _shape );
// }

// UTP HD void DTP::fill_with( TF value ) {
//     run_parallel( all_indices(), details::TensorView::TensorFillFunctor(), Out(), *this, value );
// }

// // transfer_cost for TensorView: accessible without transfer → cost 0, else 1
// UTP T_T HD auto DTP::transfer_cost( const T &ec ) const {
//     return sdot::transfer_cost_per_byte( ec, _memory_space );
// }

// // UTP CPU_ONLY void DTP::get_data_from( const auto &that ) {
// //     // contiguous -> copy works for all the cases
// //     if ( is_contiguous() && that.is_contiguous() ) {
// //         copy( data(), that.data(), nb_items() );
// //         return;
// //     }

// //     // same memory space -> for each on indices
// //     run_sequential( _shape.all_indices(), details::TensorView::TensorCopyFunctor( *this, that ) );

// //     // else
// //     TODO;
// // }


// // UTP void DTP::with_same_shape( const auto &arch, auto &&func ) const {
// //     arch.template with_reservation<TF>( nb_items(), [&]( auto buf ) {
// //         auto new_strides = contiguous_strides<TF>( _shape );
// //         using NewStrides = DECAYED_TYPE_OF( new_strides );
// //         using BufMS      = typename DECAYED_TYPE_OF( buf )::MemorySpace;
// //         TensorView<TF,Shape,NewStrides,BufMS> res( buf.raw, _shape, new_strides, buf.memory_space );
// //         func( res );
// //     } );
// // }

// // UTP HD void DTP::spill_to( TensorView &that ) {
// //     that.get_data_from( *this );
// //     _raw_ptr = that._raw_ptr;
// // }


// UTP HD Strides DTP::strides() const {
//     return _strides;
// }

// UTP T_T HD auto DTP::stride( T d ) const {
//     return _strides[ d ];
// }

// // UTP HD PI DTP::nb_items() const {
// //     PI res = 1;
// //     for( PI d = 0; d < rank(); ++d )
// //         res *= shape( d );
// //     return res;
// // }


// UTP HD bool DTP::surely_null() const {
//     return false; // TensorView is always a real, non-null tensor in FFI bindings
//     // if ( is_invalid() )
//     //     return true;

//     // /* Version using lambdas and Ct<> (causes nvcc to crash in some cases)
//     // // empty tensor (any dimension == 0)
//     // if ( _shape.has_value( []( auto size ) -> bool { return size < Ct<int,1>(); } ) )
//     //     return true;
//     // // all strides zero (rank > 0) → surely-null by construction: all elements alias data()[0] == 0
//     // if ( rank() > 0 && ! _strides.has_value( []( auto size ) -> bool { return size != Ct<int,0>(); } ) )
//     //     return true;
//     // // single scalar: check value
//     // if ( ! _shape.has_value( []( auto size ) -> bool { return size > Ct<int,1>(); } ) )
//     //     return *data() == 0;
//     // */

//     // // empty tensor (any dimension == 0)
//     // for ( PI i = 0; i < ct_rank; ++i )
//     //     if ( _shape[ i ] == 0 )
//     //         return true;

//     // // all strides zero (rank > 0) → surely-null if *data()
//     // bool all_strides_zero = true;
//     // for ( PI i = 0; i < ct_rank; ++i ) {
//     //     if ( _strides[ i ] && _shape[ i ] > 1 ) {
//     //         all_strides_zero = false;
//     //         break;
//     //     }
//     // }
//     // if ( all_strides_zero )
//     //     return *data() == 0;

//     // return false;
// }

// UTP HD bool DTP::is_invalid() const {
//     return _raw_ptr == _sentinel();
// }

// UTP HD bool DTP::is_valid() const {
//     return _raw_ptr != _sentinel();
// }

// // UTP HD auto DTP::rank() const {
// //     return Ct<int,ct_rank>();
// // }


// // UTP HD auto DTP::begin() const {
// //     if constexpr ( ct_rank == 1 || ct_rank == 0 ) {
// //         return data();
// //     } else {
// //         TODO;
// //     }
// // }

// // UTP HD auto DTP::end() const {
// //     if constexpr ( ct_rank == 0 ) {
// //         return data() + 1;
// //     } else if constexpr ( ct_rank == 1 ) {
// //         return data() + size();
// //     } else {
// //         TODO;
// //     }
// // }


// // UTP HD auto DTP::unsqueeze( auto axis ) const {
// //     // Append a dimension of size 1.
// //     // The stride for the new axis is sizeof(T): matches contiguous layout when the source is contiguous.
// //     // For a size-1 axis the stride value is irrelevant for correctness, but we set it consistently.
// //     // constexpr int new_ct_rank = ct_rank >= 0 ? ct_rank + 1 : -1;
// //     // return TensorView<T,new_ct_rank,Arch>( data(), _shape.with_pushed_value( PI( 1 ) ), _strides.with_pushed_value( SI( sizeof( T ) ) ) );
// //     TODO;
// // }


// // UTP HD void DTP::for_each_index( auto &&func ) const {
// //     IndexRange<Shape>{ _shape }.for_each_item( FORWARD( func ) );
// // }

// // // #ifdef __CUDACC__
// // // // Functor at namespace scope so CUDA/cudafe can properly mangle the type in Thrust typedefs
// // // // (local structs with __device__ members cause "template argument N is invalid" in cudafe)
// // // template<class T, int ct_rank>
// // // struct TensorViewIndexer {
// // //     __device__ const T &operator()( sdot::PI i ) const {
// // //         sdot::SI off = 0;
// // //         for ( int d = ct_rank - 1; d >= 0; --d ) {
// // //             off += sdot::SI( i % ext[ d ] ) * str[ d ];
// // //             i /= ext[ d ];
// // //         }
// // //         return *reinterpret_cast<const T *>( src + off );
// // //     }

// // //     const std::byte *src;
// // //     sdot::PI ext[ ct_rank ];
// // //     sdot::SI str[ ct_rank ];
// // // };


// // NB: the cross-space transfer for TensorView used to live here as a make_accessible overload, but it
// // made make_accessible (a fully generic policy over scalars/aggregates/views) leak a TensorView-specific
// // overload. The accessible case (func(value)) is already handled generically in hardware/make_accessible.h.
// // When real strided cross-space transfer is implemented, expose it as a customization point on the data
// // type that the generic else-branch delegates to — not as a competing overload.

#undef UTP
#undef DTP

} // namespace sdot

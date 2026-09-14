#pragma once

#include "ASIMD_DEBUG_ON_OP.h" // IWYU pragma: export

#include "../support/prev_pow_2.h"
#include "../support/BitVec.h"
#include "../support/VecValues.h"
#include "../support/HaD.h"

namespace asimd {
namespace internal {

// SimdBoolImpl ---------------------------------------------------------
/// item_size = 1 to store bits
template<int nb_items,int item_size,class Arch>
struct SimdBoolImpl;

// int, splittable version
template<int nb_items,int item_size,class Arch> requires ( item_size >= 8 && nb_items >= 2 )
struct SimdBoolImpl<nb_items,item_size,Arch> {
    static constexpr int split_size_0 = prev_pow_2( nb_items );
    static constexpr int split_size_1 = nb_items - split_size_0;
    static constexpr int splittable = 1;
    struct Split {
        SimdBoolImpl<split_size_0,item_size,Arch> v0;
        SimdBoolImpl<split_size_1,item_size,Arch> v1;
    };
  
    union {
        typename PI_<item_size>::T values[ nb_items ];
        Split split;
    } data;
};

// int, atomic version
template<int nb_items,int item_size,class Arch> requires ( item_size >= 8 && nb_items < 2 )
struct SimdBoolImpl<nb_items,item_size,Arch> {
    static constexpr int splittable = 0;
    union {
        typename PI_<item_size>::T values[ nb_items ];
    } data;
};

// bool, splittable version
template<int nb_items,int item_size,class Arch> requires ( item_size == 1 && nb_items >= 16 )
struct SimdBoolImpl<nb_items,item_size,Arch> {
    static constexpr int split_size_0 = prev_pow_2( nb_items );
    static constexpr int split_size_1 = nb_items - split_size_0;
    static constexpr int splittable = 1;
    struct Split {
        SimdBoolImpl<split_size_0,item_size,Arch> v0;
        SimdBoolImpl<split_size_1,item_size,Arch> v1;
    };
  
    union {
        BitVec<nb_items> values;
        Split split;
    } data;
};

// bool, atomic version
template<int nb_items,int item_size,class Arch> requires ( item_size == 1 && nb_items < 16 )
struct SimdBoolImpl<nb_items,item_size,Arch> {
    static constexpr int splittable = 0;
    union {
        BitVec<nb_items> values;
    } data;
};


/// Helper to make a SimdBoolImpl with a register. Version where mask values are stored in integer
/// with size >= 8 bits.
///
/// SAME ABI RULE AS `SIMD_VEC_IMPL_REG`, and it was not applied here. This union used to hold
/// `PI32 values[ 8 ]` AND a `Split`: four eightbytes classified SSE,SSE,SSE,SSE, so every
/// lane-flavoured mask crossing a call went through MEMORY. Measured by `tests/abi_probe.cpp`, on
/// a `select` taking its mask by value: 9 instructions with 3 stack accesses, against 4 and 0
/// once the array became a vector type and `Split` left the union.
///
/// Dropping `Split` costs nothing: the only generic form that recursed into a mask split does so
/// on the BIT flavour, and is guarded by `HasSplit` anyway.
#define SIMD_BOOL_IMPL_REG_LARGE( COND, NB_ITEMS, ITEM_SIZE, TREG ) \
    template<class Arch> requires ( Arch::template Has<features::COND>::value ) \
    struct SimdBoolImpl<NB_ITEMS,ITEM_SIZE,Arch> { \
        static constexpr int split_size_0 = prev_pow_2( NB_ITEMS ); \
        static constexpr int split_size_1 = NB_ITEMS - split_size_0; \
        ASIMD_VALUES_TYPE( Values, PI##ITEM_SIZE, NB_ITEMS ); \
        union { \
            Values values; \
            TREG   reg; \
        } data; \
    };

#define SIMD_BOOL_IMPL_REG_BITS_UNSPLITABLE( COND, NB_ITEMS, TREG ) \
    template<class Arch> requires ( Arch::template Has<features::COND>::value ) \
    struct SimdBoolImpl<NB_ITEMS,1,Arch> { \
        union { \
            BitVec<NB_ITEMS> values; \
            TREG reg; \
        } data; \
    };

#define SIMD_BOOL_IMPL_REG_BITS_SPLITABLE( COND, NB_ITEMS, TREG ) \
    template<class Arch> requires ( Arch::template Has<features::COND>::value ) \
    struct SimdBoolImpl<NB_ITEMS,1,Arch> { \
        static constexpr int split_size_0 = prev_pow_2( NB_ITEMS ); \
        static constexpr int split_size_1 = NB_ITEMS - split_size_0; \
        struct Split { \
            SimdBoolImpl<NB_ITEMS/2,1,Arch> v0; \
            SimdBoolImpl<NB_ITEMS/2,1,Arch> v1; \
        }; \
        union { \
            BitVec<NB_ITEMS> values; \
            Split split; \
            TREG reg; \
        } data; \
    };

// init_mask -----------------------------------------------------------------
template<int nb_items,int item_size,class Arch> HaD
void init_mask( SimdBoolImpl<nb_items,item_size,Arch> &mask, bool a, bool b, bool c, bool d, bool e, bool f, bool g, bool h ) {
    if constexpr ( item_size == 1 )
        mask.data.values.set_values( a, b, c, d, e, f, g, h );
    else
        mask.data.values = { a, b, c, d, e, f, g, h };
}

template<int nb_items,int item_size,class Arch> HaD
void init_mask( SimdBoolImpl<nb_items,item_size,Arch> &mask, bool a, bool b, bool c, bool d ) {
    if constexpr ( item_size == 1 )
        mask.data.values.set_values( a, b, c, d );
    else
        mask.data.values = { a, b, c, d };
}

template<int nb_items,int item_size,class Arch> HaD
void init_mask( SimdBoolImpl<nb_items,item_size,Arch> &mask, bool a, bool b ) {
    if constexpr ( item_size == 1 )
        mask.data.values.set_values( a, b );
    else
        mask.data.values = { a, b };
}

template<int nb_items,int item_size,class Arch> HaD
void init_mask( SimdBoolImpl<nb_items,item_size,Arch> &mask, bool a ) {
    if constexpr ( item_size == 1 )
        mask.data.values.set_value( a );
    else
        mask.data.values = { a };
}

// at ------------------------------------------------------------------------
template<int nb_items,int item_size,class Arch> HaD
bool at( const SimdBoolImpl<nb_items,item_size,Arch> &mask, int i ) {
    return mask.data.values[ i ] != 0;
}

// any, all ------------------------------------------------------------------
//
// TWO FLAVOURS OF MASK, and the generic reductions have to serve both. The bit flavour stores a
// `BitVec`, which carries its own `any`/`all`; the LANE flavour stores a plain array of
// all-ones/all-zeros words, which does not. `all` used to call `values.all()` unconditionally --
// a hard compile error on every lane mask with no register reduction, e.g. `SimdBoolImpl<16,32>`
// -- and `any` was declared only for the bit flavour, so it did not resolve at all there.
template<int nb_items,int item_size,class Arch> HaD
bool any( const SimdBoolImpl<nb_items,item_size,Arch> &mask ) {
    if constexpr ( item_size == 1 ) {
        return mask.data.values.any();
    } else {
        for ( int i = 0; i < nb_items; ++i )
            if ( mask.data.values[ i ] ) return true;
        return false;
    }
}

template<int nb_items,int item_size,class Arch> HaD
bool all( const SimdBoolImpl<nb_items,item_size,Arch> &mask ) {
    if constexpr ( item_size == 1 ) {
        return mask.data.values.all();
    } else {
        for ( int i = 0; i < nb_items; ++i )
            if ( ! mask.data.values[ i ] ) return false;
        return true;
    }
}

#define SIMD_BOOL_IMPL_REG_REDUCTION( COND, NB_ITEMS, ITEM_SIZE, NAME, FUNC ) \
    template<class Arch> requires ( Arch::template Has<features::COND>::value ) HaD \
    bool NAME( const SimdBoolImpl<NB_ITEMS,ITEM_SIZE,Arch> &mask ) { \
        ASIMD_DEBUG_ON_OP(#NAME,#COND,#FUNC) return FUNC; \
    }


} // namespace internal
} // namespace asimd


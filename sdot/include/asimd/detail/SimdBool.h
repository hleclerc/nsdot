#pragma once

#include "impl/SimdBoolImpl_Generic.h"
#include "impl/SimdBoolImpl_X86.h"
#include "impl/SimdBoolImpl_Arm.h"

#include "architectures/NativeCpu.h"
#include "support/HaD.h"

namespace asimd {

/**
  One boolean per lane -- what a comparison returns, and what `select` consumes.

  Named after `svbool_t`, and for the same reason: it is a VECTOR OF TRUTH VALUES whose layout is
  the target's business. `item_size` (in bits) is that layout: 32 or 64 for a lane of all-ones,
  as SSE/AVX/NEON produce; 1 for a bit per lane in a mask register, as AVX-512 does. The two are
  the "flavours" the ops in `ops/` register against, and `to_bits` / `select` accept either.

  Not to be confused with a `LaneMask`, which is an INTEGER -- one bit per lane -- and masks
  operations rather than holding a comparison.
*/
template<int nb_items,int item_size=1,class Arch=NativeCpu>
struct SimdBool {
    using                     Impl      = internal::SimdBoolImpl<nb_items,item_size,Arch>;

    HaD                       SimdBool  ( bool a, bool b, bool c, bool d, bool e, bool f, bool g, bool h ) { internal::init_mask( impl, a, b, c, d, e, f, g, h ); }
    HaD                       SimdBool  ( bool a, bool b, bool c, bool d, bool e ) { internal::init_mask( impl, a, b, c, d, e ); }
    HaD                       SimdBool  ( bool a, bool b, bool c, bool d ) { internal::init_mask( impl, a, b, c, d ); }
    HaD                       SimdBool  ( bool a, bool b ) { internal::init_mask( impl, a, b ); }
    HaD                       SimdBool  ( bool a ) { internal::init_mask( impl, a ); }
    HaD                       SimdBool  ( Impl impl ) : impl( impl ) {}
    HaD                       SimdBool  () {}

    // individual items
    HaD bool                  operator[]( int i ) const { return internal::at( impl, i ); }
    static HaD constexpr int  size      () { return nb_items; }

    // arithmetic operators
    // HaD SimdBool           operator& ( const SimdBool &that ) const { return SimdVecinternal::anb( impl, that.impl ); }

    Impl                      impl;     ///<
};

template<int nb_items,int item_size,class Arch>
SimdBool<nb_items,item_size,Arch> simd_bool_from_simd_bool_impl( internal::SimdBoolImpl<nb_items,item_size,Arch> &&impl ) { return impl; }

template<int nb_items,int item_size,class Arch> HaD
bool any( const SimdBool<nb_items,item_size,Arch> &a ) { return internal::any( a.impl ); }

template<int nb_items,int item_size,class Arch> HaD
bool all( const SimdBool<nb_items,item_size,Arch> &a ) { return internal::all( a.impl ); }

} // namespace asimd


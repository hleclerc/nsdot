#pragma once

// =====================================================================================
// THE ONE HEADER TO INCLUDE. Everything under `detail/` is reachable from here; nothing under
// `detail/` is meant to be included on its own, and its layout may change.
//
//   #include <asimd/asimd.h>
//
//   using V = asimd::SimdVec<float,8>;                   // eight lanes, whatever the target
//   V   s = asimd::fma( a, b, c );
//   auto m = asimd::to_bits( s > V( 0.f ) );              // a mask, as an integer
//   V   t = asimd::permute( s, I::load_aligned( idx ) );  // variable-index permutation
//
// What comes in, by group:
//
//   architectures  `NativeCpu` (the compile-time target), `LargestCpu`, `X86Cpu<...>`,
//                  `ArmCpu<...>`, `ScalarCpu`, `CudaGpu<...>` and their feature tags
//   vectors        `SimdVec<T,size,Arch>`, `SimdBool<N,item_size,Arch>` (one boolean per lane),
//                  `SimdSize<T,Arch>`, `MaxSimdSize`, `NbSimdRegisters`, `SimdAlig`
//   lane sets      `LaneRange<beg,end>`, `LaneMask<beg,end>` -- "the lanes I will read", as a
//                  trailing argument: `add( a, b, LaneRange<0,3>() )` skips the registers that
//                  hold none of them; `V::load_partial( p, n )` / `store_partial` touch no byte
//                  outside them
//   operations     `fma`, `to_bits`, `mask_from_bits`, `select`, `permute`, `bcast_lane`,
//                  `rotate_lanes`, `ext_lanes`, the lazy comparisons (`a > b`, `eq`, `ge`) with
//                  `any`/`all`, `V::iota`, and the arithmetic operators
//   ranges         `SimdRange`, `SimdRangePtr` -- the vectorized loop over `[beg, end)`
//   pointers       `Ptr<T,alignment,offset>`, `Int<T,alignment,offset>` -- alignment carried
//                  in the type, so `load_aligned` versus `load` is decided at compile time
//   selection      `sel::rank`, `sel::require_at_least` -- which variant an operation picks
// =====================================================================================

#include "detail/architectures/NativeCpu.h"
#include "detail/architectures/LargestCpu.h"
#include "detail/architectures/CudaGpu.h"
#include "detail/architectures/CudaGpuFeatures.h"

#include "detail/support/common_types.h"
#include "detail/support/ceil.h"
#include "detail/support/gcd.h"
#include "detail/support/prev_pow_2.h"
#include "detail/support/BitVec.h"
#include "detail/support/Int.h"
#include "detail/support/N.h"
#include "detail/support/S.h"

#include "detail/Ptr.h"
#include "detail/SimdSize.h"
#include "detail/MaxSimdSize.h"
#include "detail/NbSimdRegisters.h"
#include "detail/SimdAlig.h"
#include "detail/SimdBool.h"
#include "detail/LaneSet.h"
#include "detail/SimdVec.h"
#include "detail/Selection.h"
#include "detail/SimdOps.h"
#include "detail/SimdRange.h"
#include "detail/SimdRangePtr.h"

#pragma once

// =============================================================================================
// WHAT AN OPERATION IS CALLED, AND WHAT A VARIANT IS CHOSEN ON.
//
// `ops::` holds one empty tag per operation; `Key` is the (element type, width, architecture,
// mask flavour) tuple a variant is registered against. Every file under `ops/` specializes
// `sel::Variant<tag, Key, rank>` -- this header is the vocabulary they share, and nothing else.
// =============================================================================================

#include "../Selection.h"
#include "../SimdBool.h"
#include "../SimdVec.h"

namespace asimd {

/// Operation tags.
namespace ops {
    struct fma {};
    struct permute {};
    struct select {};
    struct to_bits {};
    struct mask_from_bits {};
    struct cmp_gt {};
    struct cmp_lt {};
    struct cmp_eq {};
    struct cmp_ge {};
    template<int LANE> struct bcast_lane {};
    /// lanes `[0,n)` rotated by `K` (lane `K` lands in lane 0), lanes `[n,N)` untouched.
    /// The facade normalizes `K` into `[1,n)` before naming this tag, so a row may assume it.
    template<int K,int n> struct rotate_lanes {};
    /// `a[K..N) ++ b[0..K)` -- what ARM calls `EXT` and x86 `alignr`. `K` in `[1,N)` here too.
    template<int K> struct ext_lanes {};
    /// the lanes of a set from / to memory, touching no byte outside them. `run` is a template
    /// on the set.
    struct load_partial {};
    struct store_partial {};
}

/// Reads the item size back out of a mask type. Needed because the flavour a comparison returns
/// depends on the rank that was selected, and the caller has to key the next operation on it.
template<class M> struct mask_item_size;
template<int N,int IS,class Arch>
struct mask_item_size<internal::SimdBoolImpl<N,IS,Arch>> { static constexpr int value = IS; };

/// What a variant is chosen on. `MASK_BITS` is the mask item size for mask-taking operations
/// (1 = bits, 32 = full lanes) and stays 0 for the others.
template<class T,int N,class Arch,int MASK_BITS = 0> struct Key {};

} // namespace asimd

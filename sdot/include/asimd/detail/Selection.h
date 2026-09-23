#pragma once

// =====================================================================================
// VARIANT SELECTION.
//
// An operation usually has several possible implementations for a given (type, width,
// architecture): a generic one that walks lanes, one that splits across several registers, one
// that maps to a dedicated instruction, sometimes a better one still on targets with mask
// registers. Something has to pick.
//
// = WHY NOT PLAIN OVERLOADING WITH `requires`
//
//   CONSTRAINTS DO NOT ORDER EACH OTHER. Adding an AVX-512 form next to an AVX one makes the
//   call AMBIGUOUS as soon as both features are present. Writing the constraint as a conjunction
//   does not help: two `requires` clauses WRITTEN SEPARATELY yield distinct atomic constraints
//   even when textually identical, so there is no subsumption. Working around it means rebuilding
//   an ordering by hand, one operation at a time.
//
//   OVERLOAD RESOLUTION DEPENDS ON INCLUDE ORDER. A facade calling `internal::f(...)` through a
//   QUALIFIED name freezes resolution at its own definition: a register form declared later is
//   invisible. The code still compiles, still returns correct results, and silently loses all
//   vectorization. This happened here: a kernel built on `permute` ran at scalar speed with not
//   one `vpermps` in the binary.
//
//   AVAILABLE IS NOT BEST. On a Skylake-X, 512-bit exists and downclocks: it is slower than
//   256-bit. An instruction-set hierarchy therefore cannot serve as a preference order --
//   preference is a MEASUREMENT, not a generation.
//
// = THE THREE CHOICES HERE
//
//   AN EXPLICIT RANK, hence a TOTAL order: never ambiguous, and a preference can be corrected
//   when a benchmark disproves it, without touching any constraint.
//
//   A VARIANT IS A CLASS SPECIALIZATION, never a function overload: specializations are looked up
//   at the point of INSTANTIATION rather than of definition, so include order stops deciding.
//
//   THE CHOICE IS ASSERTABLE. This is the important one, and it comes from a scar: vectorization
//   was silently lost TWICE in that port -- once through include order, once because a union sent
//   every vector back to memory. A mechanism that picks without letting you check what it picked
//   only moves the problem. `require_at_least<...>` turns a silent fallback into a compile error.
//
// = WHEN TO USE A RANK, AND WHEN NOT TO
//
//   Use one when several implementations are viable FOR THE SAME ARGUMENT TYPES -- that is where
//   ambiguity lives. When the argument types already tell the implementations apart (a bit mask
//   versus a lane mask, say), ordinary overloading is unambiguous and a rank would only add noise.
//
// = KNOWN LIMITS
//
//   A specialization declared after an actual INSTANTIATION is ill-formed, no diagnostic required.
//   GCC does not report it and does what you would expect, but that is not guaranteed. So the
//   discipline "declare every backend before any use" still holds -- a single `backends.h`
//   included at a fixed point is enough -- and `require_at_least` is the safety net.
//
//   THE ORDER IS TOTAL BETWEEN RANKS, NOT WITHIN ONE. Two backends registering the same
//   (Op, Key, RANK) are as ambiguous as two overloads would be -- the rank buys nothing there.
//   Met in practice: a `permute` on four floats exists as `pshufb` (SSSE3, four instructions) and
//   as `vpermilps` (AVX, one), and a machine with AVX has both. Two ways out, and the choice
//   says something:
//     - give them different ranks, if one is simply preferable -- but REGISTER is one level, and
//       inventing 19 and 21 turns a readable scale into a pecking order;
//     - make the constraints MUTUALLY EXCLUSIVE, `Has<SSSE3> && ! Has<AVX>`, which states the
//       real relationship: the older form is the fallback, not a rival.
//   The second is what this codebase does. `ASIMD_OPS_REQ_EXCL` in `ops/Shapes.h` spells it.
// =====================================================================================

namespace asimd {
namespace sel {

/// Ranks. These are not instruction-set generations but PREFERENCES. A backend may declare itself
/// unavailable for one specific micro-architecture (512-bit on Skylake-X) without anything else
/// moving, since the key carries the `Arch`.
enum : int {
    GENERIC       =  0,   ///< lane by lane. Always available: this is the guaranteed fallback.
    SPLIT         = 10,   ///< several registers for one requested width -- what asimd is for
    REGISTER      = 20,   ///< a dedicated instruction
    MASK_REGISTER = 30,   ///< better still on targets with mask registers
    MAX_RANK      = 40
};

/// A variant. A backend registers by specializing this, with `available = true` and a `run`.
template<class Op, class Key, int RANK>
struct Variant { static constexpr bool available = false; };

/// THE SEARCH FOR THE BEST AVAILABLE RANK, AS A CLASS TEMPLATE.
///
/// It was a `constexpr` FUNCTION template, and gcc 13 rejected one instantiation of it outright:
///
///     Selection.h: error: 'constexpr int asimd::sel::search() [with ... int R = 40]'
///                          used before its definition
///
/// on three cells of `tests/test_arm_dispatch.cpp`, on x86, and not on gcc 15 or on clang.
/// "Used before its definition", for a specialization whose definition sits a few lines above the
/// use, is an INSTANTIATION-ORDERING complaint: the point of instantiation gcc picks for
/// `search<Op,Key,MAX_RANK>` -- which the variable template `rank` needs in order to be
/// initialized -- can land before the definition it requires, once the chain gets deep enough.
///
/// And the chain does get deep, by design: a rank's `available` may ask what rank the HALVES
/// reached (`ops/Split.h`), which re-enters `rank`, which re-enters the search, at a
/// smaller width. That recursion is the point of the split ranks; it is not going away.
///
/// A CLASS TEMPLATE HAS NO SUCH QUESTION. Its specializations are instantiated on demand, at the
/// point of use, with an ordering the standard pins down -- which is precisely the argument this
/// file already makes above for a variant being a class specialization rather than a function
/// overload. The one piece of the mechanism that was not following its own advice now does, and
/// the diagnostic gcc emitted is no longer expressible: there is no constexpr function left whose
/// definition could be used too early.
///
/// STILL LAZY, and that is load-bearing. The `bool` comes from `Variant<Op,Key,R>::available` in a
/// DEFAULT TEMPLATE ARGUMENT, so rank R-1 is looked at only when R is unavailable: a rank below
/// the one selected is never instantiated. That is what stops `SPLIT`'s `available` -- and the
/// recursion into the halves it carries -- from being evaluated at a width where `REGISTER`
/// already matched.
template<class Op, class Key, int R, bool = ( R >= 0 && Variant<Op,Key,R>::available )>
struct Search { static constexpr int value = R; };

template<class Op, class Key, int R>
struct Search<Op,Key,R,false> { static constexpr int value = Search<Op,Key,R-1>::value; };

/// the floor. Rank 0 is always available in practice -- the generic form -- so reaching this
/// means an operation with no variant at all, and -1 is what says so rather than a hang.
template<class Op, class Key>
struct Search<Op,Key,-1,false> { static constexpr int value = -1; };

/// The selected rank -- a constant, hence printable, comparable, assertable.
template<class Op, class Key>
inline constexpr int rank = Search<Op,Key,MAX_RANK>::value;

template<class Op, class Key, class... A>
decltype( auto ) call( A &&...a ) { return Variant<Op,Key,rank<Op,Key>>::run( static_cast<A&&>( a )... ); }

/// THE SAFETY NET. "On this target I want at least that much": an unexpected fallback becomes a
/// compile error instead of a 30 % loss you find out about on a benchmark, or never.
template<class Op, class Key, int MIN>
constexpr void require_at_least() {
    static_assert( rank<Op,Key> >= MIN, "no variant good enough for this target" );
}

} // namespace sel
} // namespace asimd

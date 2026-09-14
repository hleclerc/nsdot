#pragma once

#include <string>

namespace asimd {
namespace features {

/// A FUSED MULTIPLY-ADD EXISTS. Deliberately declared here rather than in `X86CpuFeatures.h`
/// or `ArmCpuFeatures.h`, because both need it and a feature is a TYPE: declared in both, the
/// two would collide the moment `NativeCpu.h` includes them together -- which it always does.
///
/// Sharing it is not a compromise, it is the accurate statement. `Has<FMA>` means "this target
/// can fuse", nothing more; WHICH instruction that is falls out of the width feature it is
/// conjoined with (`ASIMD_OPS_FMA( SSE2, FMA, ... )` is `vfmadd`, `ASIMD_OPS_FMA( NEON, FMA,
/// ... )` is `fmla`). It is its own feature on both architectures for the same reason: FMA is
/// orthogonal to the vector width on x86 -- there were AMD parts with one and not the other --
/// and on ARM it is architectural at A64 but needs VFPv4 on ARMv7.
struct FMA         { static std::string name() { return "FMA"        ; } };

struct Multithread { static std::string name() { return "Multithread"; } std::size_t nb_threads = 0; };
struct L1Cache     { static std::string name() { return "L1Cache"    ; } std::size_t amount = 0, ways = 0, line_size = 0; };
struct L2Cache     { static std::string name() { return "L2Cache"    ; } std::size_t amount = 0, ways = 0, line_size = 0; };
struct L3Cache     { static std::string name() { return "L3Cache"    ; } std::size_t amount = 0, ways = 0, line_size = 0; };
struct L4Cache     { static std::string name() { return "L4Cache"    ; } std::size_t amount = 0, ways = 0, line_size = 0; };

} // namespace features
} // namespace asimd

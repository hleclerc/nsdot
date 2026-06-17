#pragma once

#include "common_macros.h"
#include "ASSERT.h" // IWYU pragma: export
#include "TODO.h" // IWYU pragma: export
#include "info.h" // IWYU pragma: export

#include <cstdint>

namespace sdot {

using FP64 = double;
using FP32 = float;

using PI8  = std::uint8_t;
using PI32 = std::uint32_t;
using PI64 = std::uint64_t;

using SI8  = std::int8_t;
using SI32 = std::int32_t;
using SI64 = std::int64_t;

using SI = long long;
using PI = std::size_t;

// ctor args
struct SizeAndCtorArgs {};
struct Function {};
struct Reserved {};
struct FillWith {};
struct Values {};
struct Shape {};
struct Rank {};
struct Size {};
struct Out {};
struct Inp {};
struct Mut {};

// template<class TF> HD TF factorial( TF n ) {
//     TF res = 1;
//     for ( PI i = 2; i <= n; ++i )
//         res *= TF( i );
//     return res;
// }

// template<class TF> constexpr TF pow_rec( TF v, PI n ) {
//     return n ? pow_rec( v, n - 1 ) * v : 1;
// }

} // namespace sdot

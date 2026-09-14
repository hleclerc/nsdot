#pragma once

#include <type_traits>
#include <cstdint>

namespace asimd {

// number types
using PI64    = typename std::conditional< sizeof( unsigned long ) == 8, unsigned long, unsigned long long >::type;
using PI32    = std::uint32_t;
using PI16    = std::uint16_t;
using PI8     = std::uint8_t;

using SI64    = typename std::conditional< sizeof( signed long ) == 8, signed long, signed long long >::type;
using SI32    = std::int32_t;
using SI16    = std::int16_t;
using SI8     = std::int8_t;

using Bool    = bool;

using PI      = typename std::conditional< sizeof( void * ) == 8, PI64, PI32 >::type;
using SI      = typename std::conditional< sizeof( void * ) == 8, SI64, SI32 >::type;

using FP80    = long double;
using FP64    = double;
using FP32    = float;

template<int size> struct PI_ {};
template<> struct PI_< 8> { using T = PI8; };
template<> struct PI_<16> { using T = PI16; };
template<> struct PI_<32> { using T = PI32; };
template<> struct PI_<64> { using T = PI64; };

} // namespace asimd


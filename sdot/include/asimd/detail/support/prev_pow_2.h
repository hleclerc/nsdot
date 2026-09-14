#pragma once

#include <type_traits>
#include <bit>

namespace asimd {

/// The largest power of two STRICTLY BELOW `v` -- `prev_pow_2( 8 )` is 4, not 8. That is what a
/// split needs: `split_size_0` must leave something for `split_size_1`.
///
/// It used to be `1 << ( sizeof( I ) * 8 - __builtin_clz( v - 1 ) - 1 )`, which had two problems.
/// `__builtin_clz` DOES NOT EXIST ON MSVC, one of the three compilers this has to build with; and
/// `__builtin_clz( 0 )` is undefined, so `prev_pow_2( 1 )` was undefined behaviour in a function
/// evaluated at compile time for every vector width. `std::bit_floor` is C++20, `constexpr`, and
/// the same instruction on every compiler that has one.
template<class I> inline constexpr
I prev_pow_2( I v ) {
    using U = typename std::make_unsigned<I>::type;
    return v <= I( 2 ) ? I( 1 ) : I( std::bit_floor( U( v - 1 ) ) );
}

} // namespace asimd

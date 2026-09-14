#pragma once

namespace asimd {

template<class T>
constexpr T gcd( T i, T j ) {
    return j ? gcd( j, i % j ) : i;
}

} // namespace asimd

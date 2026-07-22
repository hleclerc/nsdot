#pragma once

#include "../kernels/Ptr.h"
#include <iterator>

namespace sdot {

/// Random-access iterator for strided data (stride in bytes).
/// Works with std algorithms, handles multi-dimensional tensors
/// by iterating over the last (innermost) axis.
template<class T, class MemorySpace>
class StridedIterator {
public:
    using difference_type = SI;
    using value_type = T;
    using pointer = Ptr<T, MemorySpace>;
    using reference = T&;
    using iterator_category = std::random_access_iterator_tag;

    StridedIterator() = default;
    explicit StridedIterator(Ptr<T, MemorySpace> data, SI stride = sizeof(T))
        : _data(data), _stride(stride) {}

    // dereferencing
    T& operator*() const { return *_data; }
    T* operator->() const { return _data.raw; }
    T& operator[](difference_type n) const { return *(_data + n * _stride); }

    // increment/decrement
    StridedIterator& operator++() {
        _data = _data + _stride;
        return *this;
    }
    StridedIterator operator++(int) {
        auto tmp = *this;
        ++(*this);
        return tmp;
    }
    StridedIterator& operator--() {
        _data = _data - _stride;
        return *this;
    }
    StridedIterator operator--(int) {
        auto tmp = *this;
        --(*this);
        return tmp;
    }

    // arithmetic
    StridedIterator& operator+=(difference_type n) {
        _data = _data + n * _stride;
        return *this;
    }
    StridedIterator& operator-=(difference_type n) {
        _data = _data - n * _stride;
        return *this;
    }
    StridedIterator operator+(difference_type n) const {
        auto tmp = *this;
        return tmp += n;
    }
    StridedIterator operator-(difference_type n) const {
        auto tmp = *this;
        return tmp -= n;
    }
    friend StridedIterator operator+(difference_type n, const StridedIterator& it) {
        return it + n;
    }

    // distance
    difference_type operator-(const StridedIterator& other) const {
        auto byte_diff = other._data.raw - _data.raw;
        return byte_diff / _stride;
    }

    // comparison
    bool operator==(const StridedIterator& other) const {
        return _data == other._data;
    }
    bool operator!=(const StridedIterator& other) const {
        return !(*this == other);
    }
    bool operator<(const StridedIterator& other) const {
        return _data.raw < other._data.raw;
    }
    bool operator<=(const StridedIterator& other) const {
        return !(*this > other);
    }
    bool operator>(const StridedIterator& other) const {
        return other < *this;
    }
    bool operator>=(const StridedIterator& other) const {
        return !(*this < other);
    }

private:
    Ptr<T, MemorySpace> _data;
    SI _stride = 0;
};

} // namespace sdot

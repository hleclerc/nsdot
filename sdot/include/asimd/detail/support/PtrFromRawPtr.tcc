#include "PtrFromRawPtr.h"

namespace asimd {

template<class Ptr,class T,int a,int o>
PtrFromRawPtr<Ptr,T,a,o>::PtrFromRawPtr( T *value ) : value( value ) {
}

template<class Ptr,class T,int a,int o>
T* PtrFromRawPtr<Ptr,T,a,o>::operator->() const {
    return value;
}

template<class Ptr,class T,int a,int o>
T& PtrFromRawPtr<Ptr,T,a,o>::operator*() const {
    return *value;
}

template<class Ptr,class T,int a,int o>
PtrFromRawPtr<Ptr,T,a,o>::operator T*() const {
    return value;
}

template<class Ptr,class T,int a,int o>
T* PtrFromRawPtr<Ptr,T,a,o>::get() const {
    return value;
}

template<class Ptr,class T,int a,int o> template<class I,int b,int p>
auto PtrFromRawPtr<Ptr,T,a,o>::operator+( Int<I,b,p> that ) const {
    constexpr int sp = b * 8 * sizeof( T );
    constexpr int na = _gcd( alignment, sp );
    constexpr int no = ( offset + p * 8 * sizeof( T ) ) % na;
    return typename Ptr::template Rebind<T,na,no>::type( value + that.get() );
}

template<class Ptr,class T,int a,int o> template<class I,int b,int p>
auto PtrFromRawPtr<Ptr,T,a,o>::operator-( Int<I,b,p> that ) const {
    constexpr int sp = b * 8 * sizeof( T );
    constexpr int na = _gcd( alignment, sp );
    constexpr int no = ( ( offset - p * 8 * sizeof( T ) ) % na + na ) % na;
    return typename Ptr::template Rebind<T,na,no>::type( value - that.get() );
}

template<class Ptr,class T,int a,int o> template<int d>
auto PtrFromRawPtr<Ptr,T,a,o>::operator+( N<d> ) const {
    constexpr int no = ( ( offset + d * 8 * sizeof( T ) ) % alignment + alignment ) % alignment;
    return typename Ptr::template Rebind<T,a,no>::type( value + d );
}

template<class Ptr,class T,int a,int o> template<int d>
auto PtrFromRawPtr<Ptr,T,a,o>::operator-( N<d> ) const {
    constexpr int no = ( ( offset - d * 8 * sizeof( T ) ) % alignment + alignment ) % alignment;
    return typename Ptr::template Rebind<T,a,no>::type( value - d );
}

template<class Ptr,class T,int a,int o> template<class D>
auto PtrFromRawPtr<Ptr,T,a,o>::operator+( D d ) const {
    return _wuo( value + d );
}

template<class Ptr,class T,int a,int o> template<class D>
auto PtrFromRawPtr<Ptr,T,a,o>::operator-( D d ) const {
    return _wuo( value - d );
}

template<class Ptr,class T,int a,int o>
template<class P> auto PtrFromRawPtr<Ptr,T,a,o>::_wuo( P *p ) const {
    constexpr int sp = 8 * sizeof( T );
    constexpr int na = _gcd( alignment, sp );
    constexpr int no = offset % na;
    return typename Ptr::template Rebind<T,na,no>::type( p );
}

template<class Ptr,class T,int a,int o>
constexpr int PtrFromRawPtr<Ptr,T,a,o>::_gcd( int i, int j ) {
    return j ? _gcd( j, i % j ) : i;
}

} // namespace asimd

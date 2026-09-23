#ifndef ASIMD_PtrFromRawPtr_H
#define ASIMD_PtrFromRawPtr_H

#include "Int.h"

namespace asimd {

/**
  Pointer with integer value guaranteed to be equal to
    `alignment * n + offset` with `n` an integer
*/
template<class Ptr,class T,int aligment_in_bits,int offset_in_bits>
struct PtrFromRawPtr {
    using                              pointed_type = T;
    static constexpr int               alignment    = aligment_in_bits;
    static constexpr int               offset       = offset_in_bits;

    /**/                               PtrFromRawPtr( T *value = nullptr );

    T*                                 operator->   () const;
    T&                                 operator*    () const;
    T*                                 get          () const;
    operator                           T*           () const;

    template<class I,int b,int p> auto operator+    ( Int<I,b,p> that ) const;
    template<class I,int b,int p> auto operator-    ( Int<I,b,p> that ) const;

    template<int d> auto               operator+    ( N<d> ) const;
    template<int d> auto               operator-    ( N<d> ) const;

    template<class D> auto             operator+    ( D d ) const;
    template<class D> auto             operator-    ( D d ) const;

protected:
    template<class P> auto             _wuo         ( P *p ) const;
    static constexpr int               _gcd         ( int a, int b );

    T*                                 value;
};

} // namespace asimd

#include "PtrFromRawPtr.tcc"

#endif // ASIMD_PtrFromRawPtr_H

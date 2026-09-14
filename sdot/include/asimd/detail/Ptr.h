#ifndef ASIMD_Ptr_H
#define ASIMD_Ptr_H

#include "support/PtrFromRawPtr.h"
#include "support/Int.h"

namespace asimd {

/**
  Pointer with integer value guaranteed to be writable as
    `alignment * n + offset` with `n` an integer
*/
template<class T,int aligment_in_bits,int offset_in_bits=0>
struct Ptr : PtrFromRawPtr<Ptr<T,aligment_in_bits,offset_in_bits>,T,aligment_in_bits,offset_in_bits> {
    template<class U,int a,int o> struct Rebind{ using type = Ptr<U,a,o>; };

    Ptr( T *value = nullptr ) { this->value = value; }
};

} // namespace asimd

#endif // ASIMD_Ptr_H

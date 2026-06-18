#pragma once

#include "../common_macros.h" // HD, FORWARD
#include <type_traits>

namespace sdot {

namespace detail {
    template<class T,class=void> struct has_for_each_item_method : std::false_type {};
    T_T struct has_for_each_item_method<T,void_t<decltype(std::declval<T>().for_each_item( AnyFunc<>() ))>> : std::true_type {};
}

T_TA HD auto for_each_item( T &&list, A &&func ) -> std::enable_if_t<detail::has_for_each_item_method<T>::value,void>  {
    list.for_each_item( FORWARD( func ) );
}

} // namespace sdot


#pragma once

#include "../common_macros.h"
#include "Tuple.h" // Tuple, tuple, map
#include <type_traits>

namespace sdot {

/// Base commune des types d'axes nommés générés par DEFINE_AXIS (sert à les détecter).
struct AxisBase {};

/// Axe sans nom (placeholder) : valeur par défaut des entrées d'AxisNames.
struct UnnamedAxis : AxisBase {
    void display( auto &os ) const { os << "_"; }
};

/// Indice attaché à un axe nommé (résultat de `axis = index`), consommé par
/// `TensorView::operator()` / `squeeze` pour retirer l'axe `axis_type` à la position `index`.
template<class AxisType, class Index>
struct AxisIndex {
    using axis_type = AxisType;
    Index index;
};

// --- traits -------------------------------------------------------------------------------
template<class A>          constexpr bool is_axis            = std::is_base_of_v<AxisBase, A>;
template<class>            struct IsAxisIndex                  : std::false_type {};
template<class A,class I>  struct IsAxisIndex<AxisIndex<A,I>>  : std::true_type  {};

/// Position (compile-time) d'un axe nommé `Name` dans un `Tuple` de noms ; -1 si absent.
template<class Name,class Tup> struct AxisPos;
template<class Name>                        struct AxisPos<Name,Tuple<>>            { static constexpr int value = -1; };
template<class Name,class Head,class... Tl> struct AxisPos<Name,Tuple<Head,Tl...>> {
    static constexpr int rest  = AxisPos<Name,Tuple<Tl...>>::value;
    static constexpr int value = std::is_same_v<Head,Name> ? 0 : ( rest < 0 ? -1 : rest + 1 );
};

/// `Tuple` d'axes « tous non nommés » dimensionné comme `shape` (on n'en prend que le type :
/// valeur par défaut d'`AxisNames`).
constexpr auto unnamed_axes( auto shape ) {
    return map( shape, []( auto ) { return UnnamedAxis{}; } );
}

/// Définit un type d'axe nommé + un objet utilisable comme `t( NAME = i )` / `squeeze( NAME, i )`.
#define DEFINE_AXIS( NAME )                                                                 \
    struct _##NAME : ::sdot::AxisBase {                                                     \
        constexpr auto operator=( auto index ) const {                                     \
            return ::sdot::AxisIndex<_##NAME,DECAYED_TYPE_OF( index )>{ FORWARD( index ) }; \
        }                                                                                  \
        void display( auto &os ) const { os << #NAME; }                                    \
    };                                                                                     \
    inline constexpr _##NAME NAME{}

} // namespace sdot

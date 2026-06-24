#pragma once

#include <type_traits>

namespace sdot {

struct UndefList { void display( auto &ds ) const { ds << "UndefList"; } };
struct OutList   { void display( auto &ds ) const { ds << "OutList"; } };
struct InpList   { void display( auto &ds ) const { ds << "InpList"; } };
struct MutList   { void display( auto &ds ) const { ds << "MutList"; } };

/// Catégorie « réduction » : porte l'opérateur SYCL (p.ex. `sycl::plus<double>()`). L'argument qui
/// suit dans `run_parallel` est la cible hôte : `run_parallel` alloue l'USM, l'initialise à l'identité,
/// construit la `sycl::reduction`, et recopie le résultat dans la cible une fois le kernel terminé.
template<class Op>
struct RedList {
    Op   op;
    void display( auto &ds ) const { ds << "RedList"; }
};
template<class Op> RedList( Op ) -> RedList<Op>;

template<class T>  constexpr bool is_red_list               = false;
template<class Op> constexpr bool is_red_list<RedList<Op>>  = true;

/// vrai pour tout tag de catégorie d'IO (y compris `RedList<...>`).
template<class T> constexpr bool is_io_category =
    std::is_same_v<T,UndefList> || std::is_same_v<T,OutList> ||
    std::is_same_v<T,MutList>   || std::is_same_v<T,InpList>  || is_red_list<T>;

/// Cible de réduction « mappée » par `run_parallel` : op SYCL + pointeur vers la variable hôte
/// résultat. Produite à la place de `make_available` quand la catégorie courante est un `RedList`.
template<class Op, class T>
struct ReductionTarget {
    Op  op;
    T  *host;
};

template<class T>          constexpr bool is_reduction_target                       = false;
template<class Op,class T> constexpr bool is_reduction_target<ReductionTarget<Op,T>> = true;

} // namespace sdot

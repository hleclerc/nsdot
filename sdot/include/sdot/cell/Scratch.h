#pragma once

// =====================================================================================
// LE SCRATCH : un seul tenseur de mots par work-item, decoupe par le C++ en ce dont il a besoin.
//
// Une cellule en memoire ( `Local1` / `Local2` / `LocalN` ) est faite de tableaux -- les sommets,
// les identifiants, les temporaires de la coupe -- dont la taille est UNE CAPACITE, decidee par
// l'hote : exactement ce qu'il faut pour une operation de `Cell_*.py` ( le nombre de sommets qu'une
// coupe peut produire se borne ), une supposition que loom fait grossir sur debordement pour un
// diagramme entier. Ces tableaux vivent tous dans UN tenseur d'entiers par work-item, que `Carver`
// decoupe : c'est le seul scratch qu'un kernel demande, et c'est lui qui porte l'axe de batch de
// l'appel ( `Cell.py::CellScratch` ).
//
// LES DEUX COTES DOIVENT S'ACCORDER sur la taille : `Local*::words_for( cap )` ici,
// `Cell_*.scratch_words( cap )` cote Python, avec la meme formule -- et `attach` verifie qu'il a la
// place, sans quoi il ne pose rien et le dit ( `ShapeVarView::set` ), plutot que d'ecrire a cote.
// =====================================================================================

#include <loom/support/common_types.h>
#include <cstdint>
#include <cstddef>

namespace sdot {

/// l'alignement de chaque tableau decoupe : de quoi charger huit `float` d'un coup
static constexpr SI scratch_align = 32;

/// arrondi de `n` elements de `T` a l'alignement, en MOTS de 32 bits
template<class T>
constexpr SI words_of( SI n ) {
    const SI bytes = n * SI( sizeof( T ) );
    return ( ( bytes + scratch_align - 1 ) / scratch_align ) * ( scratch_align / 4 );
}

/// decoupe une zone de `nb_words` mots en tableaux alignes, dans l'ordre des `take`
struct Carver {
    std::int32_t *base;
    SI            nb_words;
    SI            used = 0;
    bool          overflow = false;

    template<class T>
    T *take( SI n ) {
        T *res = reinterpret_cast<T *>( base + used );
        used += words_of<T>( n );
        if ( used > nb_words )
            overflow = true;
        return res;
    }
};

} // namespace sdot

#pragma once

// =====================================================================================
// LE CONTRAT 3D : la meme architecture qu'en 2D -- la CELLULE dirige, le fournisseur repond --
// et le meme `Local`. Pas une instruction machine ici.
//
// CE QUI CHANGE : en 2D les sommets en ordre cyclique sont toute la geometrie ; en 3D une coupe
// porte une FACE, et il faut dire l'adjacence explicitement. La cellule tient sa propre liste de
// coupes `cid[ 0 .. nc )` ( identifiants GLOBAUX ) et ses sommets ne manipulent que des INDICES
// dans cette liste -- le numero de face EST l'indice, aucune table a chercher.
//
// L'HYPOTHESE : polytope SIMPLE, trois plans par sommet. Vraie en position generale, et c'est elle
// qui rend la coupe purement combinatoire.
// =====================================================================================

#include "cell/Contrat2D.h"

namespace sf::d3 {

using d2::SI32;
using d2::RienDeLocal;
template<class F> using Local = d2::Local<F>;

/// LE DEMI-ESPACE, tel qu'un fournisseur le rend : `d . x <= off`.
template<class TK>
struct Plan3 {
    TK   dx, dy, dz, off;
    SI32 id;
};

/// CE QUE LE FOURNISSEUR VOIT : `nb` sommets dans trois tableaux ALIGNES, pour faire son propre
/// SIMD sur eux.
template<class TK>
struct EtatCell3 {
    int       nb;
    const TK *vx, *vy, *vz;
};

/// ce que `coupe` rend
enum : int {
    INCHANGEE = 0,   ///< aucun sommet dehors : la cellule n'a pas bouge
    COUPEE    = 1,
    VIDE      = 2,   ///< le demi-espace a tout emporte
    DEBORDE   = 3    ///< les tampons ne suffisent pas ; la cellule est restee INTACTE
};

} // namespace sf::d3

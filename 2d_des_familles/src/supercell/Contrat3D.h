#pragma once

// =====================================================================================
// LE CONTRAT 3D : ce que le fournisseur et le noyau se promettent.
//
// CE FICHIER NE CONTIENT PAS UNE INSTRUCTION MACHINE, et c'est la meme discipline qu'en 2D ou
// `Contrat2D.h` et `Noyau2D.h` sont separes : le plan, la vue que le fournisseur recoit et les
// traits d'introspection ne dependent d'aucune architecture. La cellule et sa coupe, elles, en
// dependent -- elles sont dans `Cellule3D.h`.
//
// Meme architecture qu'en 2D : la CELLULE dirige, le fournisseur repond. Le noyau ne connait
// aucune politique, ne calcule ni boule ni boite, et ne sait pas d'ou viennent les plans.
//
// = CE QUI CHANGE PAR RAPPORT A LA 2D, ET POURQUOI
//
// En 2D les sommets en ORDRE CYCLIQUE sont toute la geometrie : l'aire se lit par le lacet, et la
// coupe `i` porte l'arete `[ v_i, v_i+1 ]`. Rien de cela ne survit en 3D -- il n'y a plus d'ordre
// cyclique global, et une coupe porte une FACE, c'est-a-dire un cycle d'aretes. Il faut donc dire
// explicitement ce qui n'etait qu'implicite : les ARETES.
//
// = LA LISTE DE COUPES, ET POURQUOI ELLE EST LOCALE
//
// La cellule tient sa PROPRE liste de coupes, `cid[ 0 .. nc )`, qui porte les identifiants
// GLOBAUX. Tout le reste de la cellule -- les sommets -- ne manipule que des INDICES dans cette
// liste. Trois consequences, et la troisieme est la vraie raison :
//
//   1. les indices sont petits et bornes par la cellule, pas par le nuage. La ou un `cut_id` global
//      demande 32 bits, un indice local en demande 8 ou 16, ce qui divise d'autant le trafic quand
//      le balayage des sommets passera en SIMD. ( Ils sont encore en `int` ici : c'est le premier
//      levier a mesurer, pas a supposer. )
//
//   2. `cid` EST la liste des faces. En 2D la connectivite se lit sur les sommets ; en 3D il
//      fallait jusqu'ici une table de hachage pour retrouver le numero de face d'une coupe --
//      `Cell3T::gather_faces` en garde une de 256 entrees, et la recherche pesait un tiers du
//      temps. Ici le numero de face EST l'indice, donc la table disparait.
//
//   3. l'identite globale n'est lue qu'a la FIN, une fois par face, quand l'appelant veut savoir
//      qui est son voisin. Pendant la coupe elle ne sert a rien.
//
// La liste ne fait qu'AJOUTER pendant la vie de la cellule -- une coupe qui perd tous ses sommets
// y laisse une entree morte. `compacte()` les enleve, et le noyau ne l'appelle que lorsque la liste
// est pleine : compacter a chaque coupe couterait une passe sur les sommets pour un gain qui n'a
// pas encore ete mesure.
//
// = L'HYPOTHESE : POLYTOPE SIMPLE
//
// Chaque sommet est sur EXACTEMENT trois plans. C'est vrai en position generale, et c'est ce qui
// rend la coupe purement combinatoire : deux nouveaux sommets de la face creee sont voisins
// exactement quand ils partagent une ANCIENNE coupe.
// =====================================================================================

#include "supercell/Contrat2D.h"

#include <cmath>
#include <cstdint>

namespace noyau3d {

// LE CONTRAT EST LE MEME QU'EN 2D, et c'est voulu : `Local` et `veut_changement` ne parlent pas de
// dimension. On les reprend plutot que d'en ecrire des jumeaux qui divergeraient.
using noyau2d::RienDeLocal;
using noyau2d::veut_changement;
using noyau2d::veut_comptage;
template<class F> using Local = noyau2d::Local<F>;

/// LE DEMI-ESPACE, tel qu'un fournisseur le rend : `d . x <= off`. `id` est l'identifiant GLOBAL,
/// celui qui ira dans la liste de coupes de la cellule.
struct Plan3 {
    float dx, dy, dz, off;
    int   id;
};

/// CE QUE LE FOURNISSEUR VOIT DE LA CELLULE. Les memes champs qu'`EtatLarge` en 2D, avec un axe de
/// plus : des pointeurs sur des tableaux alignes, pour qu'un fournisseur puisse faire son propre
/// SIMD sur les sommets sans que le noyau ait a le prevoir.
struct EtatCell3 {
    int nb;
    const float *vx, *vy, *vz;
};

/// ce que `coupe` rend
enum : int {
    INCHANGEE = 0,   ///< aucun sommet dehors : la cellule n'a pas bouge
    COUPEE    = 1,
    VIDE      = 2,   ///< le demi-espace a tout emporte
    DEBORDE   = 3    ///< les tampons ne suffisent pas ; la cellule est restee INTACTE
};

} // namespace noyau3d

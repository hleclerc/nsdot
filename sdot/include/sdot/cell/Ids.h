#pragma once

#include <loom/support/common_macros.h> // HD

#include <climits>

namespace sdot {

// =====================================================================================
// L'IDENTITE D'UNE COUPE, telle qu'elle est rangee dans `cut_ids`.
//
// C'est la seule chose que la geometrie garde d'un plan une fois la coupe faite : le plan lui-meme
// se refabrique quand on en a besoin -- depuis les germes pour une bissectrice, depuis les sommets
// qui le portent pour tout le reste (voir `Local2::planes_from_vertices`).
//
//   id >= 0              la bissectrice avec le germe `id` (un `cut_id` de `PowerDiagram`)
//   -1 - k               le k-ieme plan du DOMAINE ; `BOUNDARY == domain_id( 0 )` est ce que porte
//                        une coupe qui ne fait face a aucun germe quand on n'a rien de plus precis
//                        a dire -- un `Cell.cut` fait depuis Python, une paroi de pave
//   PIECE                un plan de DECOUPE ajoute par une distribution (`Image::for_each_piece`)
//   INFINITE             une paroi du simplexe de remplacement d'une cellule NON BORNEE : elle
//                        n'existe pas, ses offsets sont inventes et repousses au fil des coupes
//                        (voir `Local2::grow_for`), et elle disparait le jour ou plus aucun sommet
//                        ne la porte
//
// Tout ce qui est `< 0` est « pas un germe » : l'adjoint de la mesure n'y envoie rien.
// =====================================================================================
namespace cell_ids {
    enum : int {
        INFINITE = INT_MIN,
        PIECE    = INT_MIN + 1,
        BOUNDARY = -1,
    };

    HD constexpr int  domain_id ( int k  ) { return -1 - k; }
    HD constexpr bool is_seed   ( int id ) { return id >= 0; }
    HD constexpr bool is_domain ( int id ) { return id < 0 && id > PIECE; }
    HD constexpr int  domain_num( int id ) { return -1 - id; }
}

/// ce qu'une coupe a fait de la cellule
namespace CutStatus {
    enum : int {
        UNCHANGED = 0,   ///< le demi-espace contenait deja toute la cellule : rien n'a bouge
        CUT       = 1,   ///< la cellule a ete coupee, en place
        EMPTY     = 2,   ///< le demi-espace a tout emporte : `nb == 0`
        OVERFLOW  = 3,   ///< la sortie ne tient pas dans la capacite ; la cellule est restee INTACTE
    };
}

} // namespace sdot

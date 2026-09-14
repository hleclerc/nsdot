#pragma once

#include <loom/support/common_macros.h>
#include "cell/Ids.h"

namespace sdot {

// De quoi DECOUPER une cellule en morceaux : UNE cellule de rechange, dans la forme locale, posee
// sur le scratch du work-item par l'appelant ( `PowerDiagram::measures` ). C'est le seul « scratch »
// que le contrat d'une distribution prevoit ( voir `distributions/Distribution.py` ).
//
// La cellule SOURCE n'est jamais touchee : `start` la recopie dans `piece` et coupe, les coupes
// suivantes coupent `piece` en place. C'est ce qui permet d'ouvrir un morceau apres l'autre a
// partir de la meme cellule.
//
// Les plans de decoupe portent `cell_ids::PIECE` : « pas un germe », et c'est exactement ce que
// l'adjoint lit pour savoir que leur part ne va nulle part ( `PowerDiagram::scatter_cell_grad` ).
template<class Local>
struct PieceWorkspace {
    using TK = typename Local::TKernel;
    static constexpr int D = Local::ct_dim;

    Local &piece;
    bool  overflow = false;   ///< une coupe n'a pas tenu : l'appelant le signale, le resultat sera jete

    /// ouvre un morceau : `src` coupe par `direction . x <= offset`. Rend `false` s'il n'a pas tenu.
    bool start( const Local &src, const auto &direction, auto offset ) {
        if ( ! piece.copy_from( src ) ) {
            overflow = true;
            return false;
        }
        return cut( direction, offset );
    }

    /// une coupe de plus sur le morceau en cours
    bool cut( const auto &direction, auto offset ) {
        typename Local::PlaneT p;
        for ( int d = 0; d < D; ++d )
            p.dir[ d ] = TK( direction[ d ] );
        p.off = TK( offset );
        p.id  = cell_ids::PIECE;
        if ( piece.cut( p ) == CutStatus::OVERFLOW ) {
            overflow = true;
            return false;
        }
        return true;
    }

    /// le morceau courant
    void with_current( auto &&func ) const { func( piece ); }

    /// 0 = le morceau est vide ( le pave ne rencontre pas la cellule ) -- pas une anomalie
    SI nb_vertices() const { return piece.nb_vertices(); }
};

} // namespace sdot

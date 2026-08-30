#pragma once

#include <loom/support/common_macros.h>

namespace sdot {

// De quoi DÉCOUPER une cellule en morceaux : deux cellules de rechange entre lesquelles faire la
// navette, et la table de compaction que leurs coupes demandent au-delà de 2D.
//
// C'est le seul « scratch » que le contrat d'une distribution prévoit (voir
// `distributions/Distribution.py`), et il est fourni par l'APPELANT -- `PowerDiagram::measures`,
// qui l'alloue par work-item comme tout le reste. Une distribution ne demande donc pas de la
// mémoire : elle dit seulement, DEPUIS PYTHON, combien de coupes de plus qu'une cellule un de ses
// morceaux peut porter (`extra_cuts_per_piece`), et ces deux cellules-là sont dimensionnées en
// conséquence.
//
// La cellule SOURCE n'est jamais touchée : `start` la lit et écrit dans `a`, les coupes suivantes
// font la navette `a <-> b`. C'est ce qui permet d'ouvrir un morceau après l'autre à partir de la
// même cellule sans jamais la copier.
//
// `a` et `b` ont chacun leur PROPRE paramètre de template, et non un seul pour les deux : deux
// instances Python distinctes d'un même agrégat peuvent très bien arriver ici avec des types C++
// distincts. C'est aussi pourquoi on n'expose pas de `current()` (il n'y aurait pas de type de
// retour commun) mais `with_current( func )`, qui branche sur la parité et appelle en place.
template<class TA,class TB,class TR>
struct PieceWorkspace {
    TA   a;
    TB   b;
    TR   corr;
    bool in_a = true;   ///< la parité de la navette ; `start` la remet à zéro

    /// Ouvre un morceau : `src` (qui ne bouge pas) coupé par `direction . x <= offset`, dans `a`.
    /// Rend `false` si le résultat n'a pas tenu (voir `Cell::cut` : rien n'est écrit, la capacité
    /// manquante est enregistrée, l'hôte relance avec le double).
    bool start( const auto &src, const auto &direction, auto offset, SI cut_id ) {
        in_a = true;
        return _cut( src, a, direction, offset, cut_id );
    }

    /// Une coupe de plus sur le morceau en cours.
    bool cut( const auto &direction, auto offset, SI cut_id ) {
        if ( in_a ) { in_a = false; return _cut( a, b, direction, offset, cut_id ); }
        in_a = true;  return _cut( b, a, direction, offset, cut_id );
    }

    /// Le morceau courant, passé à `func` (pas rendu : `a` et `b` n'ont pas le même type).
    void with_current( auto &&func ) const { if ( in_a ) func( a ); else func( b ); }

    /// 0 = le morceau est vide (le pavé ne rencontre pas la cellule) -- pas une anomalie.
    SI nb_vertices() const { return in_a ? SI( a.nb_vertices ) : SI( b.nb_vertices ); }

    bool _cut( const auto &src, auto &dst, const auto &direction, auto offset, SI cut_id ) {
        // les deux régimes de `Cell::cut` : au-delà de 2D le clip réécrit le treillis de faces et
        // le COMPACTE, d'où `corr` ; en deçà, un seul passage cyclique et rien à tabuler.
        if constexpr ( DECAYED_TYPE_OF( src )::ct_dim > 2 )
            return src.cut( dst, direction, offset, cut_id, corr );
        else
            return src.cut( dst, direction, offset, cut_id );
    }
};

} // namespace sdot

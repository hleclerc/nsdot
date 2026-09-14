#pragma once

// CE QUE LE FOURNISSEUR VOIT DE LA CELLULE, et ce qu'il peut y garder -- la partie du contrat du
// moteur ( voir `Moteur.h` ) qui ne depend d'aucune machine, separee pour que les cellules en
// memoire puissent la nommer sans tirer le moteur.

namespace sdot {

/// pour un fournisseur qui n'a rien a garder d'une coupe a l'autre
struct RienDeLocal {};

template<class F> struct local_of { using type = RienDeLocal; };
template<class F> requires requires { typename F::Local; }
struct local_of<F> { using type = typename F::Local; };
template<class F> using LocalOf = typename local_of<F>::type;

/// CE QUE LE FOURNISSEUR VOIT quand la cellule est en memoire : `nb` sommets, dans des tableaux.
template<class TK>
struct EtatMem {
    int       nb;
    const TK *vx, *vy;
    const int *cid;
    bool      bounded;   ///< faux : les sommets sont ceux d'un simplexe de remplacement, ne rien en deduire
};

/// pour les cellules de dimension > 2 : `nb` sommets, `D` tableaux de coordonnees
template<class TK,int D>
struct EtatMemN {
    int       nb;
    const TK *v[ D ];
    bool      bounded;
};

} // namespace sdot

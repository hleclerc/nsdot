#pragma once

// =====================================================================================
// LE MOTEUR : la cellule dirige, le fournisseur repond.
//
// Le noyau ne recoit ni liste, ni positions, ni germe : il recoit un FOURNISSEUR et lui demande,
// coupe apres coupe, « le prochain demi-espace ». Le fournisseur voit l'etat de la cellule -- ses
// sommets, ses `cid` -- et rend un `Plane`, ou `false` quand il n'a plus rien. Il decide donc
// APRES chaque coupe, en regardant la cellule qui retrecit, la ou une liste construite d'avance
// devait decider avant qu'elle existe. C'est la que vivent l'elagage et l'ordre des candidats ;
// ici il n'y a aucune politique.
//
//      template<class Etat> bool suivant( const Etat &e, Local &l, Plane<TK,D> &p );
//
// `Local` est a lui : s'il declare un type `Local`, le moteur en cree un par CELLULE dans sa
// propre frame et le lui repasse a chaque appel -- la pile d'une descente d'arbre, par exemple.
// Un fournisseur sans etat n'en declare pas et ne paie rien.
//
// `Etat` est un patron, pas un type : le meme fournisseur sert le chemin a registres ( 2D, CPU :
// `e.nb` est une constante de compilation et `e.vx` un vecteur SIMD ) et les chemins en memoire
// ( `e.nb` est un entier, `e.vx` un pointeur ). Une seule politique a ecrire.
//
// = DEUX CHEMINS
//
//   `run_memory`      la boucle nue : demander, couper en place, recommencer. Toute dimension,
//                     tout device ; c'est le chemin GPU, et celui des cellules non bornees.
//   `run_2d_registers` le noyau a registres ( `Moteur2Reg.h` ) : 2D, CPU, cellule BORNEE. Huit
//                     sommets dans trois vecteurs, une machine a etats sur leur nombre, et une
//                     EXCURSION en memoire quand la cellule deborde des huit voies -- d'ou elle
//                     revient des qu'elle redescend.
//
// `run` choisit, a la compilation sur la dimension et le device, a l'execution sur le bornage.
// =====================================================================================

#include "Plane.h"
#include "Etat.h"
#include "Ids.h"

namespace sdot {

/// LA BOUCLE NUE. Rend `0` quand la cellule est complete ( vide comprise ), `CutStatus::OVERFLOW`
/// si une coupe n'a pas tenu -- la cellule est alors restee au dernier etat valide, et l'appelant
/// doit le signaler plutot que de la mesurer.
template<class Cell,class Fourn>
int run_memory( Cell &c, Fourn &f, LocalOf<Fourn> &loc ) {
    typename Cell::PlaneT p;
    for ( ;; ) {
        if ( c.nb_vertices() == 0 )
            return 0;
        if ( ! f.suivant( c.etat(), loc, p ) )
            return 0;
        const int r = c.cut( p );
        if ( r == CutStatus::EMPTY )
            return 0;
        if ( r == CutStatus::OVERFLOW )
            return CutStatus::OVERFLOW;
    }
}

/// la meme boucle, qui S'ARRETE des que la cellule est bornee -- le relais est alors passe au
/// noyau a registres ( voir `run` ). Meme convention de retour que `run_memory`.
template<class Cell,class Fourn>
int run_memory_while_unbounded( Cell &c, Fourn &f, LocalOf<Fourn> &loc ) {
    typename Cell::PlaneT p;
    while ( ! c.bounded() ) {
        if ( c.nb_vertices() == 0 )
            return 0;
        if ( ! f.suivant( c.etat(), loc, p ) )
            return 0;
        const int r = c.cut( p );
        if ( r == CutStatus::EMPTY )
            return 0;
        if ( r == CutStatus::OVERFLOW )
            return CutStatus::OVERFLOW;
    }
    return 0;
}

} // namespace sdot

#include "Moteur2Reg.h"

namespace sdot {

/// LE POINT D'ENTREE. `ON_CPU` dit si le noyau a registres est disponible ( il est ecrit en
/// asimd, ce qu'un kernel GPU ne sait pas compiler en registres ).
template<bool ON_CPU,class Cell,class Fourn>
int run( Cell &c, Fourn &f ) {
    LocalOf<Fourn> loc{};
    if constexpr ( ON_CPU && Cell::ct_dim == 2 ) {
        if ( c.bounded() )
            return run_2d_registers( c, f, loc );
        // non bornee : en memoire tant qu'il reste des parois a repousser, puis en registres
        const int r = run_memory_while_unbounded( c, f, loc );
        if ( r || c.nb_vertices() == 0 || c.unbounded )
            return r;
        return run_2d_registers( c, f, loc );
    } else {
        return run_memory( c, f, loc );
    }
}

} // namespace sdot

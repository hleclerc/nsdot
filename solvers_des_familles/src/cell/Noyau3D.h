#pragma once

// =====================================================================================
// LE MOTEUR 3D. Une boucle, et c'est tout : la cellule demande un plan, coupe, redemande. Pas de
// machine a etats -- les sommets sont en memoire du premier au dernier coup -- et pas
// d'excursion : un debordement est un tampon trop petit, donc a signaler, pas a gerer.
// =====================================================================================

#include "cell/Cellule3D.h"

namespace sf::d3 {

/// Rend `0` si la cellule est complete ( vide comprise : `c->nv == 0` ), `DEBORDE` si un tampon a
/// manque -- la cellule est alors restee INTACTE, donc trop grande : a compter a part.
template<class Fourn, class Cel>
int moteur( Fourn *f, Cel *c ) {
    c->init_cube();
    Local<Fourn> loc{};
    for ( ;; ) {
        Plan3<typename Cel::TKernel> p;
        if ( ! f->suivant( c->etat(), loc, p ) )
            return 0;
        const int r = c->coupe( p );
        if ( r == VIDE )
            return 0;
        if ( r == DEBORDE )
            return DEBORDE;
    }
}

} // namespace sf::d3

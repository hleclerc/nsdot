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
int moteur( Fourn *f, Cel *c, Local<Fourn> *loc_out = nullptr ) {
    c->init_cube();
    Local<Fourn> loc{};
    int r = 0;
    for ( ;; ) {
        Plan3<typename Cel::TKernel> p;
        if ( ! f->suivant( c->etat(), loc, p ) ) { r = 0; break; }
        r = c->coupe( p );
        if constexpr ( requires { loc.nb_coupees; } ) loc.nb_coupees += r == COUPEE;
        if ( r == VIDE ) { r = 0; break; }
        if ( r == DEBORDE )
            break;
    }
    if ( loc_out ) *loc_out = loc;                       // les compteurs, pour qui les demande
    return r;
}

} // namespace sf::d3

#pragma once

// =====================================================================================
// LE MOTEUR 3D. Une boucle, et c'est tout.
//
// La cellule dirige : elle demande un plan, coupe, redemande. Le fournisseur n'a aucun moyen de
// pousser quoi que ce soit, et le moteur n'a aucune politique -- ni ordre, ni elagage, ni critere
// d'arret. Exactement le contrat 2D, sans la machine a etats : en 2D le nombre de sommets est une
// CONSTANTE DE COMPILATION qui pilote la forme des registres, et il faut redispatcher quand elle
// change. Ici les sommets sont en memoire du premier au dernier coup, donc il n'y a rien a
// redispatcher, et une boucle suffit.
//
// CE QUI N'Y EST PAS ENCORE, et qui est dit ici pour qu'on ne le croie pas oublie :
//
//   * pas de supplement `change` dans l'etat. Le fournisseur 2D a deux regimes s'en sert pour
//     compter les coupes sans effet ; aucun fournisseur 3D ne le demande encore, et l'ajouter
//     avant d'en avoir l'usage couterait sans rien rendre ( mesure 2D : +40 % quand il est
//     inconditionnel ).
//   * pas d'excursion. En 2D l'excursion existe parce que le regime NORMAL est en registres et que
//     le debordement est l'exception. Ici tout est deja en memoire : un debordement est un tampon
//     trop petit, donc une erreur a signaler, pas un regime a gerer.
// =====================================================================================

#include "supercell/Cellule3D.h"

namespace noyau3d {

/// Rend `0` si la cellule est complete, `DEBORDE` si un tampon a manque. Dans ce dernier cas la
/// cellule est restee INTACTE au dernier etat valide -- elle est donc TROP GRANDE, et l'appelant
/// doit la compter a part plutot que de la mesurer.
template<class Fourn, class Cel>
int moteur( Fourn *f, Cel *c ) {
    c->init_cube();
    Local<Fourn> loc{};
    for ( ;; ) {
        Plan3 p;
        if ( ! f->suivant( EtatCell3{ c->nv, c->vx, c->vy, c->vz }, loc, p ) )
            return 0;
        const int r = c->coupe( p );
        if ( r == VIDE )
            return 0;                                    // `nv == 0` : la cellule est vide, et juste
        if ( r == DEBORDE )
            return DEBORDE;
    }
}

/// LE MOTEUR EN REPARTANT D'UNE CELLULE DEJA FAITE, comme `moteur_depuis` en 2D. La phase 2 des
/// sur-cellules en aura besoin ; il n'y a rien de plus a ecrire que de ne pas initialiser le cube.
template<class Fourn, class Cel>
int moteur_depuis( Fourn *f, Cel *c ) {
    if ( c->nv <= 0 )
        return 0;
    Local<Fourn> loc{};
    for ( ;; ) {
        Plan3 p;
        if ( ! f->suivant( EtatCell3{ c->nv, c->vx, c->vy, c->vz }, loc, p ) )
            return 0;
        const int r = c->coupe( p );
        if ( r == VIDE )    return 0;
        if ( r == DEBORDE ) return DEBORDE;
    }
}

} // namespace noyau3d

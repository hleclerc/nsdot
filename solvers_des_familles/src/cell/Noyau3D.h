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
int moteur_depuis( Fourn *f, Cel *c, const TF lo[ 3 ], const TF hi[ 3 ], const SI32 fid[ 6 ],
                   Local<Fourn> *loc_out = nullptr ) {
    TF ox = 0, oy = 0, oz = 0;
    f->origine( ox, oy, oz );                            // le repere du germe ( `cell/Contrat2D.h` )
    c->init_boite( lo, hi, ox, oy, oz, fid );
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

/// LE MOTEUR. Meme filet qu'en 2D ( `cell/Boite.h` ) : on essaie la boite locale, et si la cellule
/// la touche on recommence depuis le domaine.
template<class Fourn, class Cel>
int moteur( Fourn *f, Cel *c, Local<Fourn> *loc_out = nullptr ) {
    static const TF dlo[ 3 ] = { 0, 0, 0 }, dhi[ 3 ] = { 1, 1, 1 };
    static const SI32 dfid[ 6 ] = { -1, -2, -3, -4, -5, -6 };

    if constexpr ( requires ( Fourn *g, TF *b ) { g->boite_depart( b, b, 1.0 ); } ) {
        double facteur = 1;                              // `CROISSANCE^essai`, sans appel a `pow`
        for ( int essai = 0; essai < MAX_REPRISES; ++essai, facteur *= CROISSANCE ) {
            TF lo[ 3 ], hi[ 3 ];
            if ( ! f->boite_depart( lo, hi, facteur ) )
                break;                                   // la boite couvre deja le domaine
            const SI32 fid[ 6 ] = { lo[ 0 ] > 0 ? SI32( FACE_ARTIF - 1 ) : SI32( -1 ),
                                    hi[ 0 ] < 1 ? SI32( FACE_ARTIF - 2 ) : SI32( -2 ),
                                    lo[ 1 ] > 0 ? SI32( FACE_ARTIF - 3 ) : SI32( -3 ),
                                    hi[ 1 ] < 1 ? SI32( FACE_ARTIF - 4 ) : SI32( -4 ),
                                    lo[ 2 ] > 0 ? SI32( FACE_ARTIF - 5 ) : SI32( -5 ),
                                    hi[ 2 ] < 1 ? SI32( FACE_ARTIF - 6 ) : SI32( -6 ) };
            const int r = moteur_depuis( f, c, lo, hi, fid, loc_out );
            // VIDE compte comme une sortie ( `cell/Boite.h`, cas ( a ) ) : en Laguerre une
            // cellule ne contient pas forcement son germe.
            if ( r == 0 && c->nv > 0 && ! c->touche_artificielle() )
                return r;
            nb_reprises.fetch_add( 1, std::memory_order_relaxed );
        }
    }
    return moteur_depuis( f, c, dlo, dhi, dfid, loc_out );
}

} // namespace sf::d3

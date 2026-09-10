#pragma once

// =====================================================================================
// LE CHEMIN SCALAIRE -- le meilleur qu'on sache ecrire, pas un miroir du noyau.
//
// Il n'est pas la pour ressembler au noyau SIMD mais pour etre DUR A BATTRE : ce qu'on cherche
// est le meilleur couple algorithme + implementation, et comparer le noyau a un scalaire bride
// ne repondrait a rien.
//
// IL Y EN A DEUX, ET C'EST LA MESURE QUI TRANCHE.
//
//   `en_place`     n'ecrit que les deux points d'intersection, plus le decalage qu'impose le
//                  changement de taille. C'est `coupe_large`, celle de l'excursion du noyau.
//   `tampon_neuf`  reconstruit le cycle dans un second tampon, comme le noyau SIMD y est force.
//
// En place est la meilleure forme dans `CellSoAT` -- en FP64, avec six tableaux a deplacer. Ici,
// en float et avec trois tableaux, ce n'est vrai que sur des cellules COURTES : le decalage se
// paie en branches ( `i1 <= j2`, `nb_out == 1`, `nb_out > 2`, `j3 >= 2` ) et en boucles a borne
// variable, la ou le tampon neuf n'a qu'une recopie droite que le vectoriseur prend. Mesure, ns
// par coupe : a n = 50, 13.2 en place contre 15.3 a neuf ; a n = 200, 9.6 contre 9.0 -- l'ordre
// s'inverse. On garde les deux et le banc les departage.
// =====================================================================================

#include "supercell/Contrat2D.h"

namespace noyau2d {

/// LA COUPE DANS UN TAMPON NEUF. Meme contrat que `coupe_large`.
template<int MaxNb>
[[gnu::always_inline]] inline int coupe_neuve( float *__restrict vx, float *__restrict vy, int *__restrict cid,
                 int nb, const Plan &p, float *__restrict s,
                 float *__restrict nvx, float *__restrict nvy, int *__restrict nid ) {
    int nb_out = 0;
    for ( int i = 0; i < nb; ++i ) {
        s[ i ] = p.dx * vx[ i ] + p.dy * vy[ i ] - p.off;
        nb_out += s[ i ] > 0;
    }
    if ( nb_out == 0 ) return nb;
    if ( nb_out == nb ) return 0;

    int i1 = 0;
    for ( int i = 0; i < nb; ++i ) {
        const int q = i ? i - 1 : nb - 1;
        if ( s[ i ] > 0 && ! ( s[ q ] > 0 ) ) { i1 = i; break; }
    }
    const int nb_in = nb - nb_out, new_nb = nb_in + 2;
    if ( new_nb > MaxNb ) return -1;

    const int j0 = ( i1 + nb - 1 ) % nb, j2 = ( i1 + nb_out - 1 ) % nb, j3 = ( j2 + 1 ) % nb;
    const float s0 = s[ j0 ], s1 = s[ i1 ], s2 = s[ j2 ], s3 = s[ j3 ];
    const float ta  = s0 / ( s0 - s1 );
    const float pax = vx[ j0 ] + ( vx[ i1 ] - vx[ j0 ] ) * ta;
    const float pay = vy[ j0 ] + ( vy[ i1 ] - vy[ j0 ] ) * ta;
    const float tb  = s3 / ( s3 - s2 );
    const float pbx = vx[ j3 ] + ( vx[ j2 ] - vx[ j3 ] ) * tb;
    const float pby = vy[ j3 ] + ( vy[ j2 ] - vy[ j3 ] ) * tb;

    for ( int q = 0; q < nb_in; ++q ) {                  // [ j3, j3+1, ... , j0 ]
        const int r = j3 + q < nb ? j3 + q : j3 + q - nb;
        nvx[ q ] = vx[ r ]; nvy[ q ] = vy[ r ]; nid[ q ] = cid[ r ];
    }
    nvx[ nb_in ] = pax; nvy[ nb_in ] = pay; nid[ nb_in ] = p.id;
    nvx[ nb_in + 1 ] = pbx; nvy[ nb_in + 1 ] = pby; nid[ nb_in + 1 ] = cid[ j2 ];
    for ( int q = 0; q < new_nb; ++q ) { vx[ q ] = nvx[ q ]; vy[ q ] = nvy[ q ]; cid[ q ] = nid[ q ]; }
    return new_nb;
}

/// le carre unite, cotes numerotes -1..-4 comme dans le noyau.
inline int carre_dans( float *vx, float *vy, int *cid ) {
    vx[ 0 ] = 0; vy[ 0 ] = 0;  vx[ 1 ] = 1; vy[ 1 ] = 0;
    vx[ 2 ] = 1; vy[ 2 ] = 1;  vx[ 3 ] = 0; vy[ 3 ] = 1;
    cid[ 0 ] = -1; cid[ 1 ] = -2; cid[ 2 ] = -3; cid[ 3 ] = -4;
    return 4;
}

/// LA CELLULE VIT EN LOCALES, PAS DANS L'ATELIER, et l'atelier ne recoit que le resultat.
///
/// Ce n'est pas un detail de style : c'est 20 % du chemin scalaire. Des tableaux locaux ont une
/// duree de vie et un alignement que le compilateur CONNAIT, et il en garde les premiers
/// elements en registres pour les cellules courtes -- qui sont la regle. Les memes tableaux
/// atteints par `a->vx` sont de la memoire quelconque : chaque coupe les relit. `__restrict`
/// rend l'absence de recouvrement, il ne rend pas ca.
///
/// L'excursion du noyau, elle, N'A PAS LE CHOIX : son etat doit survivre a un `musttail`, donc
/// il doit etre dans l'atelier. C'est le prix de l'aller-retour, et il ne se paie que sur les
/// 4 % de cellules qui debordent.
template<int MaxNb, class Fourn>
void en_place( Atelier<MaxNb> *a, Fourn *f ) {
    alignas( 32 ) float vx[ MaxNb ], vy[ MaxNb ], s[ MaxNb ];
    alignas( 32 ) int   cid[ MaxNb ];
    Local<Fourn> loc{};
    int nb = carre_dans( vx, vy, cid );
    for ( ;; ) {
        Plan p;
        if ( ! f->suivant( EtatLarge<float>{ nb, vx, vy, cid, true }, loc, p ) ) break;
        nb = coupe_large<MaxNb>( vx, vy, cid, nb, p, s );
        if ( nb <= 0 ) break;
    }
    a->nb = nb;
    for ( int i = 0; i < nb; ++i ) { a->vx[ i ] = vx[ i ]; a->vy[ i ] = vy[ i ]; a->cid[ i ] = cid[ i ]; }
}

template<int MaxNb, class Fourn>
void tampon_neuf( Atelier<MaxNb> *a, Fourn *f ) {
    alignas( 32 ) float vx[ MaxNb ], vy[ MaxNb ], s[ MaxNb ], nvx[ MaxNb ], nvy[ MaxNb ];
    alignas( 32 ) int   cid[ MaxNb ], nid[ MaxNb ];
    Local<Fourn> loc{};
    int nb = carre_dans( vx, vy, cid );
    for ( ;; ) {
        Plan p;
        if ( ! f->suivant( EtatLarge<float>{ nb, vx, vy, cid, true }, loc, p ) ) break;
        nb = coupe_neuve<MaxNb>( vx, vy, cid, nb, p, s, nvx, nvy, nid );
        if ( nb <= 0 ) break;
    }
    a->nb = nb;
    for ( int i = 0; i < nb; ++i ) { a->vx[ i ] = vx[ i ]; a->vy[ i ] = vy[ i ]; a->cid[ i ] = cid[ i ]; }
}

} // namespace noyau2d

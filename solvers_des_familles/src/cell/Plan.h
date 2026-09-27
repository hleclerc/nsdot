#pragma once

// =====================================================================================
// LE PLAN BISSECTEUR DE DEUX GERMES -- LE SEUL ENDROIT OU IL SE CONSTRUIT.
//
//     |x - p0|^2 - w0 <= |x - pj|^2 - wj   <=>   d . x <= off
//     d = pj - p0 ,   off = d . ( pj + p0 ) / 2 + ( w0 - wj ) / 2
//
// ET IL EST RENDU DANS LE REPERE DU GERME `p0`, parce que c'est la que vit la cellule
// ( `cell/Contrat2D.h` ). La translation ne coute rien -- elle SIMPLIFIE :
//
//     off_local = off - d . p0 = |d|^2 / 2 + ( w0 - wj ) / 2
//
// Les deux termes sont alors d'ordre `h^2`, l'ordre meme du resultat. En absolu `off` valait `h`
// et le noyau en retranchait `d . v ~ h` : c'etait l'annulation qu'on vient supprimer.
//
// Il y en avait QUATRE copies -- les deux fournisseurs BSP, le balayage temoin, le fournisseur
// en alpha -- et elles doivent etre identiques au bit pres, sans quoi le temoin ne temoigne plus
// de rien. Une seule fonction, donc.
//
// = POURQUOI LES ENTREES SONT EN `TS` ET LA SORTIE EN `TK`
//
// C'est LA reparation de la simple precision, et elle tient en une ligne : on calcule le plan
// dans le flottant des DONNEES ( `TS`, double -- l'arbre range les germes et les poids en double
// de toute facon ) et on n'arrondit qu'a la toute fin.
//
// Ce qui est en jeu, c'est le poids. `w0 - wj` arrondi D'ABORD porte l'erreur `eps |w|` ; le plan
// est alors deplace de `eps |w| / ( 2 |d| )` le long de sa normale, soit `eps |w| / h` pour des
// germes espaces de `h`. Rapporte a la taille de cellule : `eps |w| / h^2`, c'est-a-dire
// `eps |w| n^( 2/D )` -- et c'est EN `n`, donc ca empire quand on grossit. Sur le nuage de lignes
// resolu ( `|w|max = 0.13`, `n = 1e5`, 2D ) ca predit 8e-4, mille fois la tolerance de Newton ;
// mesure : ecart median de masse 2.0e-3, p99 22 %, 2978 aretes en desaccord ( README § 19 ).
//
// OR LES POIDS NE SONT GRANDS QUE DANS L'ABSOLU. Pour une paire qui partage vraiment une facette,
// le bissecteur tombe DANS les deux cellules, donc `|w0 - wj| <~ 2 |d| h ~ h^2` : la difference
// qui compte est du meme ordre que le terme geometrique. La calculer en double et n'arrondir
// qu'elle ramene l'erreur de `eps |w|` a `eps h^2` -- un facteur `|w| / h^2`, treize mille sur le
// cas ci-dessus.
//
// L'AUTRE MOITIE EST LE REPERE, et elle est faite aussi. Porter la difference des poids enleve le
// terme en `eps |w| n^( 2/D )` mais laisse celui de la POSITION : en absolu, `d . v` vaut `h` pour
// un `s` qui doit valoir `h^2`, donc `log( 1 / h )` chiffres partent a chaque coupe et il reste un
// plancher `eps n^( 1/D )` -- exactement ce qu'on mesure en Voronoi, ou il n'y a pas de poids du
// tout. Le repere du germe le supprime a son tour : `off_local` et `d . v` y sont tous deux
// d'ordre `h^2`, et il ne reste que `eps`. Voir `cell/Contrat2D.h`.
//
// =====================================================================================

#include "cell/Contrat2D.h"
#include "cell/Contrat3D.h"

namespace sf {

/// 2D. `TS` est le flottant des DONNEES ( `TF` sur tous les chemins de production ), `TK` celui
/// du noyau. Les passer egaux redonne l'ancien comportement, ce qui sert a mesurer l'ecart.
template<bool POIDS, class TK, class TS>
inline void bissect2( d2::Plan2<TK> &p, TS x0, TS y0, TS w0, TS xj, TS yj, TS wj, d2::SI32 id ) {
    const TS dx = xj - x0, dy = yj - y0;
    TS off = TS( 0.5 ) * ( dx * dx + dy * dy );          // = `off_absolu - d . p0`
    if constexpr ( POIDS ) off += TS( 0.5 ) * ( w0 - wj );
    p.dx = TK( dx );
    p.dy = TK( dy );
    p.off = TK( off );
    p.id = id;
}

/// 3D, le meme.
template<bool POIDS, class TK, class TS>
inline void bissect3( d3::Plan3<TK> &p, TS x0, TS y0, TS z0, TS w0,
                      TS xj, TS yj, TS zj, TS wj, d3::SI32 id ) {
    const TS dx = xj - x0, dy = yj - y0, dz = zj - z0;
    TS off = TS( 0.5 ) * ( dx * dx + dy * dy + dz * dz );   // = `off_absolu - d . p0`
    if constexpr ( POIDS ) off += TS( 0.5 ) * ( w0 - wj );
    p.dx = TK( dx );
    p.dy = TK( dy );
    p.dz = TK( dz );
    p.off = TK( off );
    p.id = id;
}

} // namespace sf

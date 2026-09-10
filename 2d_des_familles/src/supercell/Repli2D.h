#pragma once

// =====================================================================================
// LE REPLI -- ce que devient une cellule qui a echappe au noyau.
//
// `Noyau2D` sort avec `nb == -1` quand une coupe demanderait un neuvieme sommet. Ce fichier
// est ce qui se passe ensuite, et son interet n'est pas d'etre un filet de securite : c'est
// qu'il ne repart PAS de zero.
//
// CE QU'ON RECONSTRUIT, ET DEPUIS QUOI. Le noyau a legue trois choses -- ses huit `cid`, ses
// huit sommets en `float`, et `kesc`, l'indice de la coupe qui deborde et qui n'a pas encore
// ete appliquee. De ces trois, ON N'UTILISE QUE LES `cid`. C'est l'affirmation de l'entete
// de `Noyau2D.h` prise au mot : l'ordre porte la connectivite, donc la geometrie est
// redondante. Le plan de la coupe `i` se refabrique depuis `cid[ i ]` et les positions ; et
// l'invariant « la coupe `i` porte l'arete [ v_i, v_i+1 ] » dit que le sommet `i` est
// l'intersection des coupes `i-1` et `i`. Un systeme 2x2 par sommet, et la cellule renait EN
// DOUBLE, exacte -- les sommets `float` du noyau ne sont jamais relus.
//
// Ce qui compte pour le chronometre : on reprend a `kesc`, pas a zero. Les coupes deja faites
// par le noyau ne sont pas refaites. Les cellules qui echappent sont les grosses, donc celles
// qui ont le plus de coupes derriere elles -- c'est precisement la que l'economie porte.
//
// LE COUT DU 2x2. Huit divisions pour eviter les coupes deja faites. Mesure sur un nuage
// uniforme, K = 24 : le repli coute 6 % du temps du noyau par REPRISE contre 16 % par REJEU.
// Le legs divise donc par pres de trois le prix de l'echappement.
//
// CE QUE LA REPRISE NE PEUT PAS FAIRE, et il faut le savoir avant de la choisir. Elle HERITE
// de la suite de `cid` que le noyau a etablie EN FLOAT. Si le noyau a tranche une configuration
// limite du mauvais cote avant d'echapper, la reprise repart de cette decision : elle refait la
// geometrie en double, pas la COMBINATOIRE. Le rejeu, qui n'herite de rien, retombe sur le
// scalaire exact. Mesure : sur 7443 cellules replies a n = 200000, la reprise en rate 4 que le
// rejeu obtient -- 0.05 % des replies, 2e-5 du nuage. `main_noyau` chiffre les deux, et le
// choix depend de ce qu'on fait ensuite de la combinatoire.
// =====================================================================================

#include "geometry/Cell.h"
#include "supercell/Noyau2D.h"
#include "supercell/Noyau2DSca.h"

namespace noyau2d {

/// LE PLAN DE LA COUPE `id`, REFABRIQUE. `id >= 0` : la bissectrice de [ germe courant, id ].
/// `id < 0` : un cote du carre unite -- et c'est pour rendre ceci possible que `carre_unite`
/// numerote ses quatre cotes `-1..-4` au lieu de les marquer tous `-1`.
inline void plan( int id, const double *px, const double *py, double x0, double y0,
                  double &dx, double &dy, double &off ) {
    if ( id >= 0 ) {
        dx  = px[ id ] - x0;
        dy  = py[ id ] - y0;
        off = 0.5 * ( dx * ( px[ id ] + x0 ) + dy * ( py[ id ] + y0 ) );
        return;
    }
    //                     bas        droite     haut       gauche
    static const double cx[ 4 ] = {  0,  1,  0, -1 };
    static const double cy[ 4 ] = { -1,  0,  1,  0 };
    static const double cf[ 4 ] = {  0,  1,  1,  0 };
    const int s = -id - 1;
    dx = cx[ s ]; dy = cy[ s ]; off = cf[ s ];
}

/// LA REPRISE. Refait la cellule en double depuis les `cid` legues, applique la coupe qui avait
/// fait echapper, puis CONTINUE A TIRER DU MEME FOURNISSEUR -- celui que le noyau a laisse
/// exactement la ou il en etait. Le repli n'a donc pas a savoir d'ou viennent les diracs, et
/// marche avec n'importe quelle politique. Rend `false` si la cellule deborde a son tour.
template<int MaxNv, class Fourn>
bool reprise( pd::CellSoAT<MaxNv> &c, const Atelier<8> &at, Fourn *f, Local<Fourn> &loc,
              const double *px, const double *py, double x0, double y0 ) {
    static_assert( MaxNv >= 8, "le repli part des huit sommets du noyau" );

    c.nb = 8;
    for ( int i = 0; i < 8; ++i ) {
        c.cid[ i ] = at.cid[ i ];
        plan( at.cid[ i ], px, py, x0, y0, c.cdx[ i ], c.cdy[ i ], c.co[ i ] );
    }

    // le sommet `i` est l'intersection des coupes `i-1` et `i` -- c'est l'invariant, lu a
    // l'envers. Le determinant ne s'annule que pour deux coupes consecutives PARALLELES, ce
    // qu'un convexe non degenere n'a pas.
    for ( int i = 0, j = 7; i < 8; j = i++ ) {
        const double a1 = c.cdx[ j ], b1 = c.cdy[ j ], f1 = c.co[ j ];
        const double a2 = c.cdx[ i ], b2 = c.cdy[ i ], f2 = c.co[ i ];
        const double det = a1 * b2 - b1 * a2;
        c.vx[ i ] = ( f1 * b2 - b1 * f2 ) / det;
        c.vy[ i ] = ( a1 * f2 - f1 * a2 ) / det;
    }

    // LE PLAN EST REFABRIQUE EN DOUBLE, pas repris du fournisseur. Celui-ci rend des `Plan` en
    // float -- c'est ce dont le noyau a besoin -- et le repli n'en garde que l'IDENTITE : c'est
    // tout ce qui compte, puisque de `id` et des positions on refait la geometrie a la precision
    // qu'on veut. C'est la meme affirmation que celle de l'en-tete du noyau, appliquee ici.
    int j = at.attente.id;
    for ( ;; ) {
        double dx, dy, off;
        plan( j, px, py, x0, y0, dx, dy, off );
        if ( c.cut( dx, dy, off, j ) == pd::CutResult::overflow )
            return false;
        if ( c.nb == 0 )
            return true;
        Plan p;
        if ( ! f->suivant( EtatLarge<double>{ (int) c.nb, c.vx, c.vy, c.cid, true }, loc, p ) )
            return true;
        j = p.id;
    }
}

/// LE REJEU, pour comparaison : on jette ce que le noyau a fait et on refait tout en double.
/// C'est la variante honnete a battre -- si `reprise` ne la bat pas, le legs ne sert a rien.
/// Il veut un fournisseur NEUF -- celui du noyau est a moitie consomme.
template<int MaxNv, class Fourn>
bool rejeu( pd::CellSoAT<MaxNv> &c, Fourn *f, Local<Fourn> &loc,
            const double *px, const double *py, double x0, double y0 ) {
    c.init_as_unit_square();
    for ( ;; ) {
        Plan p;
        if ( ! f->suivant( EtatLarge<double>{ (int) c.nb, c.vx, c.vy, c.cid, true }, loc, p ) )
            return true;
        double dx, dy, off;
        plan( p.id, px, py, x0, y0, dx, dy, off );
        if ( c.cut( dx, dy, off, p.id ) == pd::CutResult::overflow )
            return false;
        if ( c.nb == 0 )
            return true;
    }
}

} // namespace noyau2d

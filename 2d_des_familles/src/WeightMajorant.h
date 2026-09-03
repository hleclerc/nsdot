#pragma once

#include "common.h"
#include <cmath>

namespace pd2d {

/// UN MAJORANT AFFINE DES POIDS d'une region : `w( y ) <= ax * y_x + ay * y_y + b` pour tout germe
/// `y` de la region.
///
/// = Pourquoi affine
///
/// La borne classique est un majorant CONSTANT -- le poids maximum de la region -- qui traite toute
/// la boite comme si le germe le plus lourd etait partout. Des poids qui varient regulierement dans
/// l'espace, ce qui est exactement le regime du transport optimal semi-discret ou les poids sont un
/// potentiel, y sont tres mal bornes.
///
/// = Pourquoi le degre 1 et pas plus
///
/// Parce qu'il ne coute RIEN de plus a tester. Le minimum de `|p - y|^2 - a . y` sur une boite est
/// SEPARABLE par axe, son minimum libre est en `y = p + a / 2`, et un `clamp` par axe donne la
/// reponse exacte. Le majorant constant fait le meme travail avec `a = 0`. Un majorant de degre 2
/// casserait cette separabilite, et le test cesserait d'etre en `O( d )`.
///
/// = Pourquoi les pentes sont en `float`
///
/// Parce que `a` est un CHOIX et non une mesure : n'importe quel `a` donne un majorant valide,
/// pourvu que `b` soit calcule AVEC ce `a`. On arrondit donc les pentes d'abord et on calcule `b`
/// ensuite -- exact, sans marge d'arrondi a prendre -- et le noeud garde sa taille.
struct WMaj {
    float ax = 0, ay = 0;
    TF    b  = 0;
};

/// Le majorant d'une tranche de germes, choisi entre l'AFFINE ajuste aux moindres carres et le
/// CONSTANT. Repris de `sdot` (`AaBsp.py::_weight_majorant`).
///
/// L'affine n'est retenu que s'il resserre franchement l'ETALEMENT des residus,
/// `max( w - a . y ) - min( w - a . y )` : c'est exactement le mou de la borne, puisqu'un majorant
/// vaut ce que vaut l'ecart entre lui et le poids reel du germe qui l'atteint. Les deux candidats
/// ne sont pas comparables dans l'absolu -- l'affine credite moins le cote « poids faible » de la
/// boite et plus le cote oppose -- et l'etalement est la facon honnete de trancher sans dependre
/// d'ou on regarde la boite.
///
/// Comparé A QUOI demande une precaution : un ajustement a trois parametres sur `m` points resserre
/// l'etalement meme quand il n'y a rien a ajuster, d'autant plus fort que `m` est petit -- et une
/// feuille est petite par construction. Le seuil est donc le resserrement que le HASARD donne deja,
/// `sqrt( 1 - d / ( m - 1 ) )`, et l'affine doit faire nettement mieux que lui.
///
/// `get( k, x, y, w )` livre le germe de rang `k`.
template<class Get>
WMaj weight_majorant( SI beg, SI end, Get &&get ) {
    const SI m = end - beg;
    WMaj r;
    if ( m <= 0 )
        return r;

    TF x, y, w;
    get( beg, x, y, w );
    TF wmin = w, wmax = w, sx = x, sy = y, sw = w;
    for ( SI k = beg + 1; k < end; ++k ) {
        get( k, x, y, w );
        wmin = w < wmin ? w : wmin;
        wmax = w > wmax ? w : wmax;
        sx += x; sy += y; sw += w;
    }

    const TF spread = wmax - wmin;
    if ( m >= 6 && spread > 0 ) {
        // CENTRE : les moindres carres sur les positions brutes seraient mal conditionnes des que
        // le noeud est loin de l'origine. La constante ne change pas l'etalement, `b` la reprend.
        const TF mx = sx / m, my = sy / m, mw = sw / m;
        TF sxx = 0, sxy = 0, syy = 0, sxw = 0, syw = 0;
        for ( SI k = beg; k < end; ++k ) {
            get( k, x, y, w );
            const TF qx = x - mx, qy = y - my, qw = w - mw;
            sxx += qx * qx; sxy += qx * qy; syy += qy * qy;
            sxw += qx * qw; syw += qy * qw;
        }
        const TF det = sxx * syy - sxy * sxy;
        if ( det > 0 ) {                                // `det == 0` : germes alignes, pas de pente
            const TF ax = ( syy * sxw - sxy * syw ) / det;
            const TF ay = ( sxx * syw - sxy * sxw ) / det;

            TF rmin = 0, rmax = 0;
            for ( SI k = beg; k < end; ++k ) {
                get( k, x, y, w );
                const TF v = w - ax * x - ay * y;
                if ( k == beg ) { rmin = rmax = v; }
                else { rmin = v < rmin ? v : rmin; rmax = v > rmax ? v : rmax; }
            }
            const TF by_chance = std::sqrt( std::max( TF( 0 ), TF( 1 ) - TF( 2 ) / ( m - 1 ) ) );
            if ( rmax - rmin < TF( 0.85 ) * by_chance * spread ) {
                r.ax = float( ax );
                r.ay = float( ay );
            }
        }
    }

    // `b` calcule avec les pentes STOCKEES, donc arrondies : le majorant est alors exact, et aucune
    // marge n'est a prendre.
    const TF ax = r.ax, ay = r.ay;
    TF b = 0;
    for ( SI k = beg; k < end; ++k ) {
        get( k, x, y, w );
        const TF v = w - ax * x - ay * y;
        b = ( k == beg || v > b ) ? v : b;
    }
    r.b = b;
    return r;
}

} // namespace pd2d

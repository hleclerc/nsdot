#pragma once

#include "util/common.h"
#include <cmath>

namespace pd {

/// UN MAJORANT AFFINE DES POIDS d'une region : `w( y ) <= a . y + b` pour tout germe `y` de la
/// region.
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
/// ensuite -- exact, sans marge d'arrondi a prendre -- et le noeud garde sa taille. En 3D ca compte
/// deux fois plus qu'en 2D : le noeud a deja six bornes de boite au lieu de quatre.
template<int D>
struct WMajT {
    float a[ D ] = {};
    TF    b      = 0;
};

using WMaj = WMajT<2>;

/// Les pentes, en `TF` et EN AGREGAT. Elles sont stockees en `float` (voir plus haut) et lues des
/// dizaines de fois par test d'eviction ; les convertir une fois, dans un objet que le compilateur
/// peut garder en registres, est ce que la version 2D faisait a la main avec deux `const TF`.
template<int D>
inline Vec<D> pentes( const WMajT<D> &wm ) {
    if constexpr ( D == 2 )
        return { TF( wm.a[ 0 ] ), TF( wm.a[ 1 ] ) };
    else if constexpr ( D == 3 )
        return { TF( wm.a[ 0 ] ), TF( wm.a[ 1 ] ), TF( wm.a[ 2 ] ) };
    else {
        Vec<D> r;
        for ( int d = 0; d < D; ++d ) r[ d ] = TF( wm.a[ d ] );
        return r;
    }
}

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
/// Compare A QUOI demande une precaution : un ajustement a `D + 1` parametres sur `m` points
/// resserre l'etalement meme quand il n'y a rien a ajuster, d'autant plus fort que `m` est petit --
/// et une feuille est petite par construction. Le seuil est donc le resserrement que le HASARD
/// donne deja, `sqrt( 1 - D / ( m - 1 ) )`, et l'affine doit faire nettement mieux que lui. Il
/// depend de `D`, et c'est la seule chose que la dimension change ici : en 3D on ajuste un
/// parametre de plus, donc le hasard resserre davantage et le seuil monte tout seul.
///
/// `get( k, y, w )` livre le germe de rang `k`.
template<int D, class Get>
WMajT<D> weight_majorant( SI beg, SI end, Get &&get ) {
    const SI m = end - beg;
    WMajT<D> r;
    if ( m <= 0 )
        return r;

    Vec<D> y;
    TF w;
    get( beg, y, w );
    TF wmin = w, wmax = w, sw = w;
    Vec<D> sy = y;
    for ( SI k = beg + 1; k < end; ++k ) {
        get( k, y, w );
        wmin = w < wmin ? w : wmin;
        wmax = w > wmax ? w : wmax;
        sw += w;
        for ( int d = 0; d < D; ++d ) sy[ d ] += y[ d ];
    }

    const TF spread = wmax - wmin;
    if ( m >= 3 * D && spread > 0 ) {
        // CENTRE : les moindres carres sur les positions brutes seraient mal conditionnes des que
        // le noeud est loin de l'origine. La constante ne change pas l'etalement, `b` la reprend.
        Vec<D> mu;
        for ( int d = 0; d < D; ++d ) mu[ d ] = sy[ d ] / m;
        const TF mw = sw / m;

        TF M[ D ][ D ] = {}, rhs[ D ] = {};
        for ( SI k = beg; k < end; ++k ) {
            get( k, y, w );
            Vec<D> q;
            for ( int d = 0; d < D; ++d ) q[ d ] = y[ d ] - mu[ d ];
            const TF qw = w - mw;
            for ( int d = 0; d < D; ++d ) {
                rhs[ d ] += q[ d ] * qw;
                for ( int e = 0; e < D; ++e ) M[ d ][ e ] += q[ d ] * q[ e ];
            }
        }

        TF a[ D ] = {};
        bool ok = false;
        if constexpr ( D == 2 ) {
            // CRAMER, et pas l'elimination generique : c'est la formule qui etait ici avant que la
            // dimension devienne un parametre, et la garder rend le majorant 2D BIT POUR BIT le
            // meme -- sans quoi une comparaison avec les mesures d'avant porterait sur un arbre
            // qui n'elague plus tout a fait pareil.
            const TF det = M[ 0 ][ 0 ] * M[ 1 ][ 1 ] - M[ 0 ][ 1 ] * M[ 0 ][ 1 ];
            if ( det > 0 ) {
                a[ 0 ] = ( M[ 1 ][ 1 ] * rhs[ 0 ] - M[ 0 ][ 1 ] * rhs[ 1 ] ) / det;
                a[ 1 ] = ( M[ 0 ][ 0 ] * rhs[ 1 ] - M[ 0 ][ 1 ] * rhs[ 0 ] ) / det;
                ok = true;
            }
        } else {
            // GAUSS avec pivot partiel sur la matrice normale. `D` vaut trois : le pivot coute
            // quelques comparaisons et evite d'avoir a distinguer les configurations degenerees
            // (germes coplanaires, alignes) autrement que par « le pivot est nul ».
            TF A[ D ][ D + 1 ];
            for ( int i = 0; i < D; ++i ) {
                for ( int j = 0; j < D; ++j ) A[ i ][ j ] = M[ i ][ j ];
                A[ i ][ D ] = rhs[ i ];
            }
            ok = true;
            for ( int c = 0; c < D && ok; ++c ) {
                int piv = c;
                for ( int i = c + 1; i < D; ++i )
                    if ( std::fabs( A[ i ][ c ] ) > std::fabs( A[ piv ][ c ] ) ) piv = i;
                if ( ! ( std::fabs( A[ piv ][ c ] ) > 0 ) ) { ok = false; break; }
                if ( piv != c )
                    for ( int j = c; j <= D; ++j ) { const TF t = A[ c ][ j ]; A[ c ][ j ] = A[ piv ][ j ]; A[ piv ][ j ] = t; }
                for ( int i = c + 1; i < D; ++i ) {
                    const TF f = A[ i ][ c ] / A[ c ][ c ];
                    for ( int j = c; j <= D; ++j ) A[ i ][ j ] -= f * A[ c ][ j ];
                }
            }
            if ( ok )
                for ( int i = D - 1; i >= 0; --i ) {
                    TF s = A[ i ][ D ];
                    for ( int j = i + 1; j < D; ++j ) s -= A[ i ][ j ] * a[ j ];
                    a[ i ] = s / A[ i ][ i ];
                }
        }

        if ( ok ) {
            TF rmin = 0, rmax = 0;
            for ( SI k = beg; k < end; ++k ) {
                get( k, y, w );
                TF v = w;
                for ( int d = 0; d < D; ++d ) v -= a[ d ] * y[ d ];
                if ( k == beg ) { rmin = rmax = v; }
                else { rmin = v < rmin ? v : rmin; rmax = v > rmax ? v : rmax; }
            }
            const TF by_chance = std::sqrt( std::max( TF( 0 ), TF( 1 ) - TF( D ) / ( m - 1 ) ) );
            if ( rmax - rmin < TF( 0.85 ) * by_chance * spread )
                for ( int d = 0; d < D; ++d ) r.a[ d ] = float( a[ d ] );
        }
    }

    // `b` calcule avec les pentes STOCKEES, donc arrondies : le majorant est alors exact, et aucune
    // marge n'est a prendre.
    TF b = 0;
    for ( SI k = beg; k < end; ++k ) {
        get( k, y, w );
        TF v = w;
        for ( int d = 0; d < D; ++d ) v -= TF( r.a[ d ] ) * y[ d ];
        b = ( k == beg || v > b ) ? v : b;
    }
    r.b = b;
    return r;
}

} // namespace pd

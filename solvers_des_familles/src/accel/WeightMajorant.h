#pragma once

// =====================================================================================
// LE MAJORANT AFFINE DES POIDS d'une region : `w( y ) <= a . y + b` pour tout germe `y`.
//
// C'est le seul endroit du banc ou une idee a rapporte un FACTEUR et pas des pour cent : sur le
// nuage a aires egales, 466 boites testees et 538 coupes tentees par cellule avec un majorant
// constant, 135 et 61 avec l'affine ( `2d_des_familles/README.md`, § 6 ).
//
// Pourquoi le degre 1 et pas plus : il ne coute RIEN a tester. `|v - y|^2 - a . y` est separable
// par axe, son minimum libre est en `y = v + a / 2`, et un `clamp` par axe donne le minimum sur la
// boite. Le constant est le cas `a = 0`. Un degre 2 casserait la separabilite.
//
// Pourquoi les pentes sont en `float` : `a` est un CHOIX, pas une mesure. N'importe quel `a` donne
// un majorant valide pourvu que `b` soit calcule AVEC ce `a` -- on arrondit donc les pentes, puis on
// calcule `b`, et le noeud garde sa taille.
// =====================================================================================

#include "util/common.h"
#include <algorithm>
#include <cmath>

namespace sf {

template<int D>
struct WMajT {
    float a[ D ] = {};
    TF    b      = 0;
};

/// Le majorant d'une tranche de germes : l'AFFINE ajuste aux moindres carres s'il resserre
/// franchement l'etalement des residus `max( w - a . y ) - min( w - a . y )`, le CONSTANT sinon.
/// Le seuil est le resserrement que le hasard donne deja sur `m` points avec `D + 1` parametres,
/// `sqrt( 1 - D / ( m - 1 ) )`, et l'affine doit faire nettement mieux que lui.
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
        // centre : les moindres carres sur les positions brutes seraient mal conditionnes des que
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

        // Gauss avec pivot partiel sur la matrice normale : `D` vaut deux ou trois, le pivot coute
        // quelques comparaisons et regle les configurations degenerees ( germes alignes,
        // coplanaires ) par « le pivot est nul ».
        TF A[ D ][ D + 1 ];
        for ( int i = 0; i < D; ++i ) {
            for ( int j = 0; j < D; ++j ) A[ i ][ j ] = M[ i ][ j ];
            A[ i ][ D ] = rhs[ i ];
        }
        bool ok = true;
        for ( int c = 0; c < D && ok; ++c ) {
            int piv = c;
            for ( int i = c + 1; i < D; ++i )
                if ( std::fabs( A[ i ][ c ] ) > std::fabs( A[ piv ][ c ] ) ) piv = i;
            if ( ! ( std::fabs( A[ piv ][ c ] ) > 0 ) ) { ok = false; break; }
            if ( piv != c )
                for ( int j = c; j <= D; ++j ) std::swap( A[ c ][ j ], A[ piv ][ j ] );
            for ( int i = c + 1; i < D; ++i ) {
                const TF f = A[ i ][ c ] / A[ c ][ c ];
                for ( int j = c; j <= D; ++j ) A[ i ][ j ] -= f * A[ c ][ j ];
            }
        }
        TF a[ D ] = {};
        if ( ok )
            for ( int i = D - 1; i >= 0; --i ) {
                TF s = A[ i ][ D ];
                for ( int j = i + 1; j < D; ++j ) s -= A[ i ][ j ] * a[ j ];
                a[ i ] = s / A[ i ][ i ];
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

    // `b` avec les pentes STOCKEES, donc arrondies : le majorant est exact, sans marge a prendre.
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

} // namespace sf

#pragma once

#include <loom/support/common_macros.h> // HD

#include <loom/support/common_types.h>

namespace sdot {

/// LE DEMI-ESPACE `dir . x <= off`, tel qu'un fournisseur le rend au noyau : la geometrie dans le
/// flottant du NOYAU (`TK`, `float` par defaut), et l'identite qui ira dans `cut_ids` -- c'est elle
/// qui porte la connectivite, et la geometrie du plan n'en depend pas (voir `cell/Ids.h`).
///
/// `dir` n'est pas normalisee : `off` est le produit scalaire auquel elle est comparee telle
/// quelle, donc `( 2n, 2o )` designe le meme demi-espace que `( n, o )`.
template<class TK,int D>
struct Plane {
    TK  dir[ D ];
    TK  off;
    int id;

    /// `dir . x - off` : positif DEHORS
    HD TK dist( const TK *x ) const {
        TK s = - off;
        for ( int d = 0; d < D; ++d )
            s += dir[ d ] * x[ d ];
        return s;
    }
};

/// LA BISSECTRICE DE PUISSANCE de `( p0, w0 )` et `( p1, w1 )`, cote `p0`. Calculee dans le
/// flottant des positions (`TF`) et rendue dans celui du noyau : le plan est arrondi UNE fois, sur
/// un resultat, et non terme a terme.
///
/// `|x - p0|^2 - w0 <= |x - p1|^2 - w1` perd son `|x|^2` des deux cotes et devient
/// `( p1 - p0 ) . x <= ( p1 - p0 ) . ( p0 + p1 ) / 2 + ( w0 - w1 ) / 2` : la mediatrice euclidienne
/// DECALEE le long de sa normale par l'ecart des poids. Ecrite non normalisee, ce qui est aussi
/// pourquoi le terme de poids est divise par deux et non par `|p1 - p0|`.
template<class TK,int D,class TF>
HD Plane<TK,D> bisector( const TF *p0, TF w0, const TF *p1, TF w1, int id ) {
    Plane<TK,D> res;
    TF off = ( w0 - w1 ) / 2;
    for ( int d = 0; d < D; ++d ) {
        const TF dd = p1[ d ] - p0[ d ];
        off += dd * ( p0[ d ] + p1[ d ] ) / 2;
        res.dir[ d ] = TK( dd );
    }
    res.off = TK( off );
    res.id  = id;
    return res;
}

} // namespace sdot

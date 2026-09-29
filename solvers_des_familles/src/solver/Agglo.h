#pragma once

// =====================================================================================
// L'AGGLOMERATION DES GERMES TROP PROCHES : resoudre en DEUX ETAPES.
//
// = Ce que la « degenerescence » est vraiment, et ce n'est pas une insolubilite
//
// Deux germes a distance `delta` sont separes par un plan dont le decalage le long de leur axe vaut
// `( w_i - w_j ) / ( 2 delta )`. Placer ce plan a une precision relative `eps` du diametre `L` de la
// cellule demande donc de connaitre `w_i - w_j` a `2 delta eps L` pres : L'ECART DE POIDS EST DIVISE
// PAR `delta`. A `delta = 1e-8`, `L = 1e-3` et `eps = 1e-6`, il faut `w_i - w_j` a 2e-17 pres, sous
// l'epsilon machine relatif des poids ( qui valent ~1e-1 ). Ce n'est donc pas un probleme insoluble,
// c'est UN PLANCHER DE PRECISION -- et il explique le `max|a-nu|/nu = 2.35e-6` exactement ou le banc
// s'arrete sur `lines5_s0.005`.
//
// D'ou la reparation, qui n'est ni un preconditionneur ni une suppression de germes :
//
//   1. AGGREGER : une grappe de germes a moins de `delta` devient UN germe, au barycentre pondere par
//      `nu` ( celui qui minimise le second moment de la grappe ), de cible `sum nu_i`. Le probleme
//      reduit n'a plus aucune paire proche, donc plus d'amplification.
//   2. RESOUDRE le probleme reduit -- le meme Newton, rien de special.
//   3. REDIVISER chaque cellule fusionnee entre les membres de sa grappe ( `--agrege-corr` ). Pour des
//      germes EXACTEMENT confondus c'est ARBITRAIRE : le cout `int |x - p_i|^2` ne depend pas du
//      membre auquel on attribue un morceau, donc toute partition aux bonnes masses est optimale. Pour
//      des germes a distance `delta`, le bon parametre est LA POSITION DU PLAN et non l'ecart de
//      poids : on la trouve par bissection sur l'aire, ce qui est parfaitement conditionne, et on en
//      DEDUIT `w_i - w_j = 2 delta * offset` a la fin. L'amplification devient une simple sortie.
//
// = La detection, sans aucune structure nouvelle et sans diagramme
//
// Les grappes du LIEN SIMPLE au seuil `delta` sont les composantes connexes du graphe des paires a
// moins de `delta`. Un hachage de grille au pas `delta` les donne : deux points a moins de `delta`
// sont dans la meme case ou dans deux cases voisines, donc comparer chaque point aux `3^D` cases
// autour de la sienne suffit. Aucun tri, aucun arbre, `O( n )` en esperance.
//
// ( `newton --agglo D` fait la meme chose depuis les aretes de Delaunay du diagramme de Voronoi --
//   Delaunay contient l'arbre couvrant minimal, donc les memes grappes. C'est la variante qui mesure
//   le cout quand le diagramme est de toute facon paye : 4 % d'un diagramme en 2D, 1 % en 3D. )
// =====================================================================================

#include "util/common.h"
#include <cmath>
#include <unordered_map>
#include <vector>

namespace sf {

/// LES GRAPPES de germes a moins de `delta` l'un de l'autre, par hachage de grille et union-find.
/// `rep[ i ]` devient le representant de la grappe de `i` ( le plus petit indice ). Rend le nombre de
/// germes qui DISPARAISSENT a la reduction, c'est-a-dire `n` moins le nombre de grappes.
template<int D>
SI grappes_proches( const TF *const *P, SI n, TF delta, std::vector<SI> &rep ) {
    rep.resize( n );
    for ( SI i = 0; i < n; ++i ) rep[ i ] = i;
    if ( ! ( delta > 0 ) ) return 0;

    auto trouve = [ & ]( SI i ) {
        while ( rep[ i ] != i ) { rep[ i ] = rep[ rep[ i ] ]; i = rep[ i ]; }
        return i;
    };
    auto unis = [ & ]( SI a, SI b ) {
        a = trouve( a ); b = trouve( b );
        if ( a != b ) rep[ a < b ? b : a ] = a < b ? a : b;
    };

    struct Cle {
        SI c[ D ];
        bool operator==( const Cle &o ) const {
            for ( int d = 0; d < D; ++d ) if ( c[ d ] != o.c[ d ] ) return false;
            return true;
        }
    };
    struct Hach {
        size_t operator()( const Cle &k ) const {   // FNV-1a sur les coordonnees de case
            size_t h = 1469598103934665603ull;
            for ( int d = 0; d < D; ++d ) { h ^= size_t( k.c[ d ] ); h *= 1099511628211ull; }
            return h;
        }
    };
    auto cle_de = [ & ]( SI i ) {
        Cle k;
        for ( int d = 0; d < D; ++d ) k.c[ d ] = SI( std::floor( P[ d ][ i ] / delta ) );
        return k;
    };

    std::unordered_map<Cle, std::vector<SI>, Hach> cases;
    cases.reserve( size_t( n ) );
    for ( SI i = 0; i < n; ++i ) cases[ cle_de( i ) ].push_back( i );

    const TF d2max = delta * delta;
    const int nv = D == 2 ? 9 : 27;
    for ( SI i = 0; i < n; ++i ) {
        const Cle k0 = cle_de( i );
        for ( int v = 0; v < nv; ++v ) {
            Cle k = k0;
            for ( int d = 0, t = v; d < D; ++d, t /= 3 ) k.c[ d ] += ( t % 3 ) - 1;
            const auto it = cases.find( k );
            if ( it == cases.end() ) continue;
            for ( SI j : it->second ) {
                if ( j <= i ) continue;
                TF s = 0;
                for ( int d = 0; d < D; ++d ) { const TF u = P[ d ][ i ] - P[ d ][ j ]; s += u * u; }
                if ( s < d2max ) unis( i, j );
            }
        }
    }

    SI perdus = 0;
    for ( SI i = 0; i < n; ++i ) { rep[ i ] = trouve( i ); perdus += rep[ i ] != i; }
    return perdus;
}

/// LE PROBLEME REDUIT depuis les grappes : un germe par grappe, au barycentre pondere par `nu`, de
/// cible `sum nu_i`. `vers[ i ]` est l'indice REDUIT du germe `i` ( donc plusieurs `i` partagent le
/// meme ), et `taille[ r ]` le nombre de membres de la grappe `r`.
template<int D>
void reduis( const TF *const *P, const std::vector<TF> &nu, const std::vector<SI> &rep,
             std::vector<TF> Q[ D ], std::vector<TF> &nur, std::vector<SI> &vers,
             std::vector<SI> &taille ) {
    const SI n = SI( rep.size() );
    vers.assign( n, -1 );
    SI m = 0;
    for ( SI i = 0; i < n; ++i )
        if ( rep[ i ] == i ) vers[ i ] = m++;
    for ( SI i = 0; i < n; ++i )
        if ( rep[ i ] != i ) vers[ i ] = vers[ rep[ i ] ];

    for ( int d = 0; d < D; ++d ) Q[ d ].assign( m, TF( 0 ) );
    nur.assign( m, TF( 0 ) );
    taille.assign( m, 0 );
    for ( SI i = 0; i < n; ++i ) {
        const SI r = vers[ i ];
        nur[ r ] += nu[ i ];
        ++taille[ r ];
        for ( int d = 0; d < D; ++d ) Q[ d ][ r ] += nu[ i ] * P[ d ][ i ];
    }
    for ( SI r = 0; r < m; ++r )
        for ( int d = 0; d < D; ++d )
            Q[ d ][ r ] /= nur[ r ] > 0 ? nur[ r ] : TF( 1 );
}

} // namespace sf

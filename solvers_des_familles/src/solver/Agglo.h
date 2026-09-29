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

#include "solver/Ecrasement.h"
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


// =====================================================================================
// LE REDECOUPAGE ( troisieme etape ) : rendre a chaque membre d'une grappe sa masse.
//
// Le probleme local est un transport semi-discret sur la cellule fusionnee, avec `k` germes. Ce qui le
// rend facile est le CHOIX DU PARAMETRE. Le plan qui separe deux membres a une normale FIXE
// ( `p_j - p_i` ) ; seul son decalage est libre. En le cherchant DIRECTEMENT -- une bissection sur
// l'aire du morceau, monotone -- on travaille sur une quantite de l'ordre de la cellule, donc
// parfaitement conditionnee. L'ecart de poids `w_i - w_j = 2 u . x_plan - ( |p_j|^2 - |p_i|^2 )` n'est
// calcule qu'a la FIN : l'amplification par `1 / delta` devient une SORTIE et jamais une inconnue.
//
// C'est toute la difference avec ce que Newton peut faire sur le nuage complet : lui n'a que les poids
// comme inconnues, donc il subit l'amplification et plafonne ( § 23.10 ).
//
// `k = 2` est traite EXACTEMENT, et c'est le seul cas qu'on ait mesure ( taille max 2 sur tous les
// nuages ). Pour `k > 2` on retire les membres un a un par des coupes successives : les masses sont
// bonnes -- donc le plan est optimal si les germes sont exactement confondus, le cout ne dependant
// alors pas du membre -- mais la partition n'est plus forcement un diagramme de puissance, donc les
// poids rendus ne sont qu'approches. C'est dit a l'appel.
// =====================================================================================

/// LES SOMMETS d'une cellule modelisee, dans l'ordre, aux poids de sa construction. Meme resolution de
/// Cramer que `ModeleCellule::aire`, dont ceci est l'extraction.
inline bool sommets_cellule( const ModeleCellule &m, std::vector<TF> &vx, std::vector<TF> &vy ) {
    const int nb = m.nb;
    vx.clear(); vy.clear();
    if ( nb < 3 ) return false;
    vx.resize( nb ); vy.resize( nb );
    for ( int j = 0; j < nb; ++j ) {
        const int a = j ? j - 1 : nb - 1;
        const TF det = m.nx[ a ] * m.ny[ j ] - m.ny[ a ] * m.nx[ j ];
        if ( ! ( std::fabs( det ) > 0 ) ) return false;
        vx[ j ] = ( m.c0[ a ] * m.ny[ j ] - m.c0[ j ] * m.ny[ a ] ) / det;
        vy[ j ] = ( m.nx[ a ] * m.c0[ j ] - m.nx[ j ] * m.c0[ a ] ) / det;
    }
    return true;
}

/// L'AIRE de la partie d'un polygone convexe ou `u . x <= s` ( Sutherland-Hodgman, une seule coupe ).
inline TF aire_coupee( const std::vector<TF> &vx, const std::vector<TF> &vy, TF ux, TF uy, TF s ) {
    const int nb = int( vx.size() );
    if ( nb < 3 ) return 0;
    std::vector<TF> ax, ay;
    ax.reserve( nb + 1 ); ay.reserve( nb + 1 );
    for ( int j = 0; j < nb; ++j ) {
        const int l = j + 1 < nb ? j + 1 : 0;
        const TF dj = ux * vx[ j ] + uy * vy[ j ] - s, dl = ux * vx[ l ] + uy * vy[ l ] - s;
        if ( dj <= 0 ) { ax.push_back( vx[ j ] ); ay.push_back( vy[ j ] ); }
        if ( ( dj < 0 ) != ( dl < 0 ) ) {
            const TF t = dj / ( dj - dl );
            ax.push_back( vx[ j ] + t * ( vx[ l ] - vx[ j ] ) );
            ay.push_back( vy[ j ] + t * ( vy[ l ] - vy[ j ] ) );
        }
    }
    TF a2 = 0;
    for ( size_t j = 0; j < ax.size(); ++j ) {
        const size_t l = j + 1 < ax.size() ? j + 1 : 0;
        a2 += ax[ j ] * ay[ l ] - ax[ l ] * ay[ j ];
    }
    return std::fabs( a2 ) * TF( 0.5 );
}

/// LE DECALAGE `s` du plan `u . x = s` qui donne l'aire `cible` au morceau `u . x <= s`, par
/// bissection. `u` n'est PAS normalise : c'est `p_j - p_i`, pour que `s` se convertisse directement en
/// ecart de poids. Rend l'aire obtenue.
inline TF coupe_a_l_aire( const std::vector<TF> &vx, const std::vector<TF> &vy, TF ux, TF uy,
                          TF cible, TF &s, int nb_bis = 80 ) {
    TF lo = INFINI, hi = -INFINI;
    for ( size_t j = 0; j < vx.size(); ++j ) {
        const TF d = ux * vx[ j ] + uy * vy[ j ];
        lo = std::min( lo, d ); hi = std::max( hi, d );
    }
    // l'aire est croissante en `s`, de 0 en `lo` a l'aire totale en `hi`
    for ( int k = 0; k < nb_bis; ++k ) {
        const TF mi = TF( 0.5 ) * ( lo + hi );
        if ( aire_coupee( vx, vy, ux, uy, mi ) < cible ) lo = mi; else hi = mi;
    }
    s = TF( 0.5 ) * ( lo + hi );
    return aire_coupee( vx, vy, ux, uy, s );
}

/// LE REDECOUPAGE D'UNE GRAPPE : `mem` sont ses membres ( positions `P`, cibles `nu` ), `m` la cellule
/// fusionnee, et `dw[ t ]` recoit le poids de `mem[ t ]` RELATIF a celui de la cellule fusionnee.
/// `aires[ t ]` recoit l'aire obtenue. Rend `false` si la cellule n'a pas pu etre lue.
///
/// Pour `k = 2` c'est exact. Au-dela, les coupes successives donnent les bonnes masses mais la
/// partition n'est plus forcement un diagramme de puissance ( voir en tete ).
inline bool redecoupe( const ModeleCellule &m, const TF *const *P, const std::vector<TF> &nu,
                       const std::vector<SI> &mem, std::vector<TF> &dw, std::vector<TF> &aires ) {
    const int k = int( mem.size() );
    dw.assign( k, TF( 0 ) );
    aires.assign( k, TF( 0 ) );
    std::vector<TF> vx, vy;
    if ( ! sommets_cellule( m, vx, vy ) ) return false;
    if ( k < 2 ) { aires[ 0 ] = aire_coupee( vx, vy, 1, 0, INFINI ); return true; }

    // on retire les membres un a un ; `reste` est le polygone encore a partager
    std::vector<TF> rx = vx, ry = vy;
    for ( int t = 0; t + 1 < k; ++t ) {
        const SI i = mem[ t ], j = mem[ t + 1 ];
        const TF ux = P[ 0 ][ j ] - P[ 0 ][ i ], uy = P[ 1 ][ j ] - P[ 1 ][ i ];
        if ( ! ( ux * ux + uy * uy > 0 ) ) return false;   // deux germes VRAIMENT au meme point
        TF s = 0;
        aires[ t ] = coupe_a_l_aire( rx, ry, ux, uy, nu[ i ], s );
        // `u . x = s` est le plan de puissance entre `i` et `j` :
        //     u . x = ( |p_j|^2 - |p_i|^2 + w_i - w_j ) / 2
        const TF ni = P[ 0 ][ i ] * P[ 0 ][ i ] + P[ 1 ][ i ] * P[ 1 ][ i ];
        const TF nj = P[ 0 ][ j ] * P[ 0 ][ j ] + P[ 1 ][ j ] * P[ 1 ][ j ];
        dw[ t + 1 ] = dw[ t ] - ( 2 * s - ( nj - ni ) );   // `w_j = w_i - ( 2 s - ( |p_j|^2 - |p_i|^2 ) )`
        // le reste, pour le tour suivant : la partie `u . x >= s`
        std::vector<TF> nx, ny;
        const int nb = int( rx.size() );
        for ( int q = 0; q < nb; ++q ) {
            const int l = q + 1 < nb ? q + 1 : 0;
            const TF dq = ux * rx[ q ] + uy * ry[ q ] - s, dl = ux * rx[ l ] + uy * ry[ l ] - s;
            if ( dq >= 0 ) { nx.push_back( rx[ q ] ); ny.push_back( ry[ q ] ); }
            if ( ( dq > 0 ) != ( dl > 0 ) ) {
                const TF tt = dq / ( dq - dl );
                nx.push_back( rx[ q ] + tt * ( rx[ l ] - rx[ q ] ) );
                ny.push_back( ry[ q ] + tt * ( ry[ l ] - ry[ q ] ) );
            }
        }
        rx.swap( nx ); ry.swap( ny );
    }
    aires[ k - 1 ] = rx.size() >= 3 ? aire_coupee( rx, ry, 1, 0, INFINI ) : TF( 0 );
    return true;
}

} // namespace sf

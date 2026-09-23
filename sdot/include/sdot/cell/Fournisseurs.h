#pragma once

#include <loom/support/common_macros.h> // HD

// =====================================================================================
// LES FOURNISSEURS D'UN DIAGRAMME DE PUISSANCE -- « quel demi-espace couper maintenant ? »
//
// Le moteur ( `Moteur.h` ) ne connait ni germe ni arbre : il tire des plans. Un fournisseur est un
// objet a une methode, `suivant( etat, local, plan )`, qui remplit le plan et rend `true`, ou
// rend `false` quand il n'a plus rien. Deux ici :
//
//   `FournisseurTous`   tous les autres germes, dans l'ordre. Aucune acceleration -- c'est ce que
//                       `make_cell` fait quand on ne lui donne pas d'accelerateur, et c'est le
//                       PLANCHER contre lequel les autres se mesurent.
//   `FournisseurBsp`    l'arbre BSP RETOURNE : une descente qui POUSSERAIT ses candidats devient
//                       un parcours SUSPENDU dont la pile explicite vit dans le `Local` que le
//                       moteur loge par cellule. Chaque `suivant` reprend la ou le precedent
//                       s'etait arrete : preordre, le fils le plus proche du germe d'abord, et un
//                       elagage qui voit la cellule TELLE QU'ELLE EST devenue, coupe apres coupe.
//
// LA BISSECTRICE est calculee dans le flottant des positions et rendue dans celui du noyau
// ( `bisector`, `Plane.h` ) : c'est le seul endroit ou les deux precisions se rencontrent.
// =====================================================================================

#include <loom/support/common_types.h>
#include "Elagage.h"
#include "Plane.h"
#include "Etat.h"

namespace sdot {

/// TOUS LES AUTRES GERMES, dans l'ordre du stockage.
template<class PD,class TK,int D>
struct FournisseurTous {
    using TF = typename PD::TF;

    const PD &pd;
    SI  n, k0, k = 0;
    TF  p0[ D ], w0;

    HD FournisseurTous( const PD &pd, SI k0 ) : pd( pd ), n( pd.nb_seeds() ), k0( k0 ) {
        const auto p = pd.point( k0 );
        for ( int d = 0; d < D; ++d )
            p0[ d ] = p[ d ];
        w0 = pd.weight( k0 );
    }

    template<class Etat>
    HD bool suivant( const Etat &, RienDeLocal &, Plane<TK,D> &p ) {
        if ( k == k0 ) ++k;                              // on ne se coupe pas soi-meme
        if ( k >= n ) return false;
        const SI j = k++;
        const auto pj = pd.point( j );
        TF p1[ D ];
        for ( int d = 0; d < D; ++d )
            p1[ d ] = pj[ d ];
        p = bisector<TK,D>( p0, w0, p1, pd.weight( j ), int( j ) );
        return true;
    }
};

/// L'ARBRE BSP, PARCOURU A LA DEMANDE. `pd.tree` est l'arbre ( `AaBsp.py` ), et les germes du
/// stockage sont DANS SON ORDRE : la feuille `[ begin, end )` se lit d'un seul tenant par
/// `pd.point( k )`, et l'identifiant d'une coupe est ce rang `k`. `POIDS` : diagramme de Laguerre
/// ( le majorant affine des poids entre dans l'elagage ) -- une constante de compilation, pour que
/// le cas euclidien ne paie ni les pentes ni les termes en `a . y`.
///
/// `MEMO` : LA MEMOIRE ( `PowerDiagram_Bsp.memo_nbrs / memo_counts` ). Le stockage se souvient,
/// par germe, des RANGS des voisins de sa cellule au dernier `measures` ( tries croissant ) ; ils
/// sont proposes EN PREMIER, avant toute descente, puis le parcours ordinaire les SAUTE ( aucun
/// plan deux fois : `cut` n'est pas idempotente ). Ce que ca epargne, ce ne sont pas les boites --
/// une boite qui contient un vrai voisin passe l'elagage quoi qu'il arrive -- ce sont les coupes
/// TRANSITOIRES, celles qu'un germe proche fait avant qu'un vrai voisin ne le supplante : la
/// moitie des coupes effectives en 3D, chacune une mise a jour du polytope. Mesure sur le banc
/// ( `solvers_des_familles`, README § 11 ) : -25 a -42 % du diagramme 3D, et les souvenirs
/// perimes -- ceux de Voronoi sur un diagramme de Laguerre -- rendent encore -18 % : un faux
/// souvenir ne coute qu'une premiere passe. Un germe sans souvenir ( `memo_counts == 0` ) prend
/// le chemin ordinaire, a la comparaison pres.
template<class PD,class TK,int D,bool POIDS,bool MEMO = false>
struct FournisseurBsp {
    using TF = typename PD::TF;

    /// L'ETAT DU PARCOURS, un par cellule, loge dans la frame du moteur. La pile est bornee par la
    /// PROFONDEUR de l'arbre, pas par sa taille : a chaque niveau on depile un noeud et on en
    /// empile deux. 64 niveaux valent 2^64 germes.
    struct Local {
        SI   pile[ 64 ];
        int  haut = 0;
        SI   k = 0, fin = 0;                             ///< la tranche de la feuille ouverte
        bool amorce = false;
        int  ipre = 0;                                   ///< MEMO : ou en est la pre-passe des souvenirs
        int  isaut = 0;                                  ///< MEMO : le curseur de saut dans la feuille ouverte
    };

    const PD &pd;
    SI  k0;
    SI  depth;                                           ///< `nb_nodes == 2^depth - 1`
    int npre = 0;                                        ///< MEMO : combien de souvenirs pour `k0`
    TF  p0[ D ], w0;
    TK  q0[ D ], v0;                                     ///< les memes, pour l'elagage

    HD FournisseurBsp( const PD &pd, SI k0 ) : pd( pd ), k0( k0 ) {
        const auto p = pd.point( k0 );
        for ( int d = 0; d < D; ++d ) {
            p0[ d ] = p[ d ];
            q0[ d ] = TK( p[ d ] );
        }
        w0 = pd.weight( k0 );
        v0 = TK( w0 );

        depth = 0;
        for ( SI m = SI( pd.tree.node_begin.shape( 0 ) ); m; m >>= 1 )
            ++depth;

        if constexpr ( MEMO )
            npre = int( pd.memo_counts( k0 ) );
    }

    /// MEMO : le `q`-ieme souvenir de `k0`, un rang
    HD SI souvenir( int q ) const {
        if constexpr ( MEMO ) return SI( pd.memo_nbrs( k0, q ) );
        else return 0;
    }

    HD void plan_du_rang( SI k, Plane<TK,D> &p ) const {
        const auto pj = pd.point( k );
        TF p1[ D ];
        for ( int d = 0; d < D; ++d )
            p1[ d ] = pj[ d ];
        p = bisector<TK,D>( p0, w0, p1, pd.weight( k ), int( k ) );
    }

    /// la boite du noeud, et le majorant de ses poids, dans le flottant du noyau
    HD Boite<TK,D> boite( SI n ) const {
        Boite<TK,D> B;
        for ( int d = 0; d < D; ++d ) {
            B.lo[ d ] = TK( pd.tree.node_box( n, 0, d ) );
            B.hi[ d ] = TK( pd.tree.node_box( n, 1, d ) );
            B.a[ d ] = 0;
        }
        B.b = 0;
        if constexpr ( POIDS ) {
            for ( int d = 0; d < D; ++d )
                B.a[ d ] = TK( pd.tree.node_wa( n, d ) );
            B.b = TK( pd.tree.node_wb( n ) );
        }
        return B;
    }

    /// le carre de la distance du germe a la boite du noeud `n` -- une CLEF D'ORDRE seulement, pour
    /// visiter le fils le plus proche en premier, donc les coupes qui mordent le plus en premier.
    HD TF proximite( SI n ) const {
        TF res = 0;
        for ( int d = 0; d < D; ++d ) {
            const TF lo = TF( pd.tree.node_box( n, 0, d ) ), hi = TF( pd.tree.node_box( n, 1, d ) );
            const TF e  = p0[ d ] < lo ? lo - p0[ d ] : ( p0[ d ] > hi ? p0[ d ] - hi : TF( 0 ) );
            res += e * e;
        }
        return res;
    }

    template<class Etat>
    HD bool suivant( const Etat &e, Local &l, Plane<TK,D> &p ) {
        if ( ! l.amorce ) {                              // la racine : indice 0, hauteur `depth`
            l.pile[ l.haut++ ] = depth;
            l.amorce = true;
        }

        if constexpr ( MEMO ) {                          // la pre-passe : les voisins d'hier, sans descendre
            if ( l.ipre < npre ) {
                plan_du_rang( souvenir( l.ipre++ ), p );
                return true;
            }
        }

        for ( ;; ) {
            // ---- une feuille est ouverte : on rend le germe suivant de sa tranche
            while ( l.k < l.fin ) {
                const SI k = l.k++;
                if ( k == k0 )
                    continue;
                if constexpr ( MEMO ) {                  // deja propose ? le curseur avance avec `k`
                    while ( l.isaut < npre && souvenir( l.isaut ) < k ) ++l.isaut;
                    if ( l.isaut < npre && souvenir( l.isaut ) == k ) { ++l.isaut; continue; }
                }
                plan_du_rang( k, p );
                return true;
            }

            if ( l.haut == 0 )
                return false;                            // l'arbre est epuise

            // `h` voyage sur la pile, empaquete avec l'indice ( six bits suffisent )
            const SI en = l.pile[ --l.haut ];
            const SI n  = en >> 6;
            const SI h  = en & 63;

            // un emplacement VIDE : le fils droit d'un noeud qui n'avait plus rien a partager
            const SI beg = SI( pd.tree.node_begin( n ) );
            const SI end = SI( pd.tree.node_end( n ) );
            if ( beg >= end )
                continue;

            // tout le sens de l'arbre : un sous-arbre qui ne peut plus atteindre la cellule n'est
            // pas descendu. Teste au DEPILAGE, donc contre la cellule telle qu'elle est maintenant.
            if ( ! peut_couper<POIDS,TK,D>( e, q0, v0, boite( n ) ) )
                continue;

            if ( h <= 1 ) {                              // une feuille
                l.k = beg;
                l.fin = end;
                if constexpr ( MEMO ) {                  // le curseur de saut : le premier souvenir >= beg
                    int lo = 0, hi = npre;
                    while ( lo < hi ) { const int m = ( lo + hi ) / 2; if ( souvenir( m ) < beg ) lo = m + 1; else hi = m; }
                    l.isaut = lo;
                }
                continue;
            }

            // les fils sont DEDUITS : l'arbre est binaire parfait en preordre, le gauche est juste
            // a cote et le droit a `2^( h - 1 )` noeuds. Le plus proche est empile en DERNIER.
            const SI g = n + 1, dr = n + ( SI( 1 ) << ( h - 1 ) ), hc = h - 1;
            if ( proximite( g ) <= proximite( dr ) ) {
                l.pile[ l.haut++ ] = ( dr << 6 ) | hc;
                l.pile[ l.haut++ ] = ( g  << 6 ) | hc;
            } else {
                l.pile[ l.haut++ ] = ( g  << 6 ) | hc;
                l.pile[ l.haut++ ] = ( dr << 6 ) | hc;
            }
        }
    }
};

} // namespace sdot

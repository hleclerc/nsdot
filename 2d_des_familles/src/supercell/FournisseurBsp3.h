#pragma once

// =====================================================================================
// `AaBsp3`, RETOURNE : le parcours 3D devient un fournisseur.
//
// Le pendant exact de `FournisseurBsp.h` en 2D, et il n'y avait rien a inventer : l'arbre
// `AaBspT<D>` etait DEJA generique, `AaBsp3 = AaBspT<3>` et son `build( X, Y, Z, W, ... )`
// existaient. Ce qui manquait etait le fournisseur -- c'est-a-dire l'elagage, et lui seul.
//
// = LE PARCOURS SUSPENDU
//
// `for_each_candidate` POUSSE : il descend l'arbre et appelle un foncteur. Le noyau, lui, TIRE.
// La pile explicite du parcours demenage donc dans le `Local` que le moteur loge pour le
// fournisseur, et chaque `suivant` reprend ou le precedent s'etait arrete. Meme preordre, meme
// ordre des fils, meme elagage.
//
// = CE QUE LA MESURE DIT
//
// `n` germes uniformes dans le cube, ns par GERME, arbre compris. CGAL est la triangulation
// reguliere 3D BORNEE AU CUBE ( points miroirs ) plus l'extraction des cellules et de leurs
// VOLUMES -- c'est-a-dire exactement notre travail, ni plus ni moins. Les deux rendent une somme
// de volumes de 1. Meme nuage, meme graine.
//
//                    nous ( leaf=4 )       CGAL         ecart
//   VORONOI
//   n=10000              12 217          13 794      nous x1.13
//   n=50000              12 926          10 052      CGAL x1.29
//   n=200000             13 663           8 961      CGAL x1.52
//
//   LAGUERRE, dw = 4 h^2
//   n=10000              10 903          10 270      CGAL x1.06
//   n=50000              11 835           6 827      CGAL x1.73
//   n=200000             12 684           5 914      CGAL x2.14
//
// LE TEMPS EST PLAT EN `n` -- 12,2 a 13,7 microsecondes de n=10000 a n=200000, et la lente montee
// est du cache --, donc la structure fait son travail : le cout par cellule ne depend plus du
// nuage. Le balayage complet, lui, est en `O( n )` par cellule ( 31,8 us a n=2000 deja ).
//
// CGAL PASSE DEVANT A PARTIR DE ~20 000 GERMES, et l'ecart croit avec `n` : sa triangulation
// s'amortit ( 10,8 puis 5,8 us par germe ) quand notre cout par cellule, lui, est constant.
//
// UNE PREMIERE VERSION DE CE BANC DONNAIT CGAL x3,2, ET C'ETAIT FAUX : elle ne lui demandait ni de
// borner au domaine ni de calculer les volumes. Le borner coute les points miroirs ( +20 a +30 %
// de points ) et l'extraction des volumes coute 3,1 us par germe en Voronoi -- soit, a n=200000,
// 4 872 ns qui deviennent 8 961. La lecon est generale : un temoin qui ne fait pas le meme travail
// ne mesure rien.
//
// = L'ELAGAGE EST AU MAXIMUM DE CE QU'IL PEUT RENDRE, ET CE N'EST PAS LA QUE CA COUTE
//
// C'etait l'hypothese naturelle, et elle est fausse. Le diagnostic ( `BSP3_COMPTE`, un binaire a
// part ) donne a n=50000 :
//
//   leaf = 1 :  121,7 noeuds depiles,  25,27 candidats proposes, dont 25,27 EFFECTIFS ( 100 % )
//   leaf = 8 :   59,6 noeuds depiles,  84,94 candidats proposes, dont 29,35 effectifs
//
// A `leaf = 1` TOUT CANDIDAT PROPOSE COUPE. On ne peut pas faire mieux que 15,05, qui est le
// nombre de faces ; les dix de trop sont des coupes qui ont eu un effet puis ont ete recouvertes,
// et c'est inherent a l'ordre, pas a l'elagage.
//
// ET POURTANT LE TEMPS NE BOUGE PAS : de `leaf = 16` ( 126 candidats ) a `leaf = 1` ( 25 ), soit
// CINQ FOIS MOINS de propositions, on passe de 15,4 a 15,1 us. Les candidats que l'elagage
// supprime sont ceux qui ne coutaient rien -- une coupe sans effet, c'est la premiere passe et
// rien d'autre, douze nanosecondes. `leaf = 4` est l'optimum, et il l'est de peu.
//
// LE PROFIL LE CONFIRME : 72 % du temps est dans `Cellule3::coupe`, 23 % dans le parcours et le
// volume. C'EST LA LE TRAVAIL RESTANT si l'on veut rattraper CGAL -- pas dans l'arbre, qui est
// fini. Le premier coup y a deja ete porte : supprimer la renumerotation des sommets gardes a
// rendu 15,5 % sur le diagramme complet ( voir `Cellule3D.h` ).
//
// = CE QUE LA 3D CHANGE PAR RAPPORT A LA 2D
//
//   1. LE TEST D'ELAGAGE COUTE UNE BOUCLE. En 2D les sommets sont en registres et le test est un
//      `vcmpps` ; ici ils sont en memoire et il y en a vingt-six. Mesure : 2558 sommets lus par
//      cellule a `leaf = 1`, ce qui reste petit -- l'inquietude etait legitime, elle n'a pas lieu.
//
//   2. LE NOEUD NE TIENT PLUS DANS UNE LIGNE DE CACHE. `2 D` bornes en `double` plus `D` pentes
//      font 60 octets a `D = 3` : deux lignes par visite au lieu d'une. C'est dit dans `AaBsp.h`,
//      et le `static_assert` y est conditionne a `D == 2` pour cette raison.
//
//   3. LE VOISINAGE EST PLUS GROS : ~15,2 faces contre ~6 en 2D, donc plus de coupes incompressibles.
//
// = L'ELAGAGE EST LEGITIME ICI, ET IL NE L'EST PAS COMME CRITERE D'ARRET
//
// Un critere d'ARRET doit trancher entre des candidats PROCHES, ou un rayon isotrope est trop
// lache des que les poids sont inegaux -- c'est la contrainte qu'on s'est donnee et elle tient.
// Ici on elimine des SOUS-ARBRES lointains, avec un test EXACT sur les sommets, pas une boule.
//
// = CE QUE LE `Local` PORTE
//
// La pile, et la position dans la tranche de la feuille courante. Rien d'autre : l'elagage etant
// exact et sans etat, il n'y a rien a mettre en cache donc rien a invalider -- ce fournisseur ne
// demande ni `Etat::change` ni `compte`.
//
// EXACTITUDE : somme des volumes a 1,3e-8 pres, et ZERO voisinage different du parcours sans
// elagage sur 2000 cellules en Laguerre et 3000 en Voronoi ( `pd_bspf3d`, temoin en `O( n^2 )` ).
// =====================================================================================

#include "spatial_accel/AaBsp.h"
#include "supercell/Elagage3D.h"
#include "supercell/Fournisseurs3D.h"

/// COMPTEURS DE DIAGNOSTIC, compiles a part : un compteur dans la boucle chaude fausserait ce
/// qu'il mesure. `visites` compte les noeuds depiles, `sommets` la somme des sommets lus par le
/// test d'elagage -- c'est ce produit qui dit si le parcours coute ce qu'on croit.
#ifndef BSP3_COMPTE
#define BSP3_COMPTE 0
#endif

namespace noyau3d {

#if BSP3_COMPTE
inline long long g_visites = 0, g_feuilles = 0, g_sommets = 0;
#endif


/// `POIDS` : diagramme de LAGUERRE au lieu de VORONOI. Constante de compilation, parce que le cas
/// euclidien ne doit payer ni les pentes du majorant ni les termes en `a . y`.
/// `W` : la largeur SIMD du test d'elagage, en voies de `float`.
template<class Arbre,bool POIDS = false,int W = 8>
struct FournisseurBsp3 {
    static constexpr int D = Arbre::dim;
    static_assert( D == 3, "le noyau de coupe est 3D" );

    /// L'ETAT DU PARCOURS, un par cellule, loge dans la frame du moteur. La pile est bornee par la
    /// PROFONDEUR de l'arbre, pas par sa taille : a chaque niveau on depile un noeud et on en
    /// empile deux. 48 niveaux valent 2^48 germes.
    struct Local {
        int  pile[ 48 ];
        int  haut = 0;
        int  k = 0, fin = 0;                             ///< la tranche de la feuille courante
        bool amorce = false;
    };

    const Arbre *arbre;
    float x0, y0, z0;                                    ///< le germe courant
    float w0 = 0;                                        ///< son poids ( ignore si `! POIDS` )
    int   i0;                                            ///< son identifiant, pour ne pas se couper

    FournisseurBsp3( const Arbre *arbre, float x0, float y0, float z0, int i0 )
        : arbre( arbre ), x0( x0 ), y0( y0 ), z0( z0 ), i0( i0 ) {}
    FournisseurBsp3( const Arbre *arbre, float x0, float y0, float z0, float w0, int i0 )
        : arbre( arbre ), x0( x0 ), y0( y0 ), z0( z0 ), w0( w0 ), i0( i0 ) {}

    template<class Etat>
    bool peut_couper( const typename Arbre::Node &nd, const Etat &e ) const {
        Boite3 B;
        for ( int d = 0; d < 3; ++d ) { B.lo[ d ] = (float) nd.lo[ d ]; B.hi[ d ] = (float) nd.hi[ d ]; }
        if constexpr ( POIDS ) {
            for ( int d = 0; d < 3; ++d ) B.a[ d ] = (float) nd.wm.a[ d ];
            B.b = (float) nd.wm.b;
        }
        return peut_couper_boite3<POIDS,W>( e, x0, y0, z0, w0, B );
    }

    /// le noeud le plus proche du GERME est visite en premier : les coupes qui mordent le plus
    /// arrivent tot, donc la cellule retrecit vite, donc l'elagage mord plus.
    float proximite( int n ) const {
        const auto &nd = arbre->nodes[ n ];
        const float x[ 3 ] = { x0, y0, z0 };
        float s = 0;
        for ( int d = 0; d < 3; ++d ) {
            const float lo = (float) nd.lo[ d ], hi = (float) nd.hi[ d ];
            const float e = x[ d ] < lo ? lo - x[ d ] : x[ d ] > hi ? x[ d ] - hi : 0.f;
            s += e * e;
        }
        return s;
    }

    template<class Etat>
    bool suivant( const Etat &e, Local &l, Plan3 &p ) {
        if ( ! l.amorce ) { l.pile[ l.haut++ ] = 0; l.amorce = true; }

        for ( ;; ) {
            // ---- une feuille est ouverte : on rend le germe suivant de sa tranche
            while ( l.k < l.fin ) {
                const int k = l.k++;
                const int id = (int) arbre->order[ k ];
                if ( id == i0 ) continue;
                bissect3<POIDS>( (float) arbre->seed_c( k, 0 ), (float) arbre->seed_c( k, 1 ),
                                 (float) arbre->seed_c( k, 2 ), POIDS ? (float) arbre->seed_w( k ) : 0.f,
                                 x0, y0, z0, w0, id, p );
                return true;
            }

            if ( l.haut == 0 )
                return false;                            // l'arbre est epuise

            const int h = l.pile[ --l.haut ];
            const auto &nd = arbre->nodes[ h ];

#if BSP3_COMPTE
            ++g_visites;
            g_sommets += e.nb;
#endif
            if ( ! peut_couper( nd, e ) )                // aucun germe de ce sous-arbre ne peut rien
                continue;

#if BSP3_COMPTE
            if ( nd.right < 0 ) ++g_feuilles;
#endif
            if ( nd.right < 0 ) { l.k = (int) nd.beg; l.fin = (int) nd.end; continue; }

            const int g = h + 1, dr = (int) nd.right;    // PREORDRE : le gauche est juste a cote
            if ( proximite( g ) <= proximite( dr ) ) { l.pile[ l.haut++ ] = dr; l.pile[ l.haut++ ] = g; }
            else                                       { l.pile[ l.haut++ ] = g;  l.pile[ l.haut++ ] = dr; }
        }
    }
};

} // namespace noyau3d

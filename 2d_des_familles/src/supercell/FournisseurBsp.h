#pragma once

// =====================================================================================
// `AaBsp`, RETOURNE : le parcours devient un fournisseur.
//
// = LE PROBLEME
//
// `AaBsp::for_each_candidate` POUSSE : il descend l'arbre et appelle `cut_with` pour chaque germe
// qu'il retient. Le noyau, lui, TIRE : il demande un demi-plan et coupe. Les deux ne peuvent pas
// se rencontrer sans que l'un des deux rende la main.
//
// La solution n'est pas de rendre le parcours recursif ni d'en faire une coroutine, mais de le
// SUSPENDRE : sa pile explicite -- qui existait deja, `SI stack[ 64 ]` sur la frame -- demenage
// dans le `Local` que le moteur loge pour le fournisseur. Chaque `suivant` reprend exactement ou
// le precedent s'etait arrete. Rien d'autre ne change : meme preordre, meme ordre des fils, meme
// elagage.
//
// = CE QUE LE FOURNISSEUR SAIT ET QUE LE NOYAU IGNORE
//
// Tout ce qui rend l'elagage possible : la geometrie des boites, le germe, et le fait que les
// candidats arrivent du plus proche au plus lointain. Le noyau ne voit que des demi-plans.
//
// = L'ELAGAGE
//
// Un germe `q` d'une boite `B` ne peut couper la cellule `C` que s'il existe `p` dans `C` avec
// `|p - q| < |p - p0|`. Donc si
//
//      dist^2( boite de C, B )  >=  max_{v sommet de C} |v - p0|^2
//
// aucun germe de `B` ne peut rien : on saute le sous-arbre entier. Les deux quantites sont
// separables par axe. C'est le meme critere que `PowerDiagram::may_be_cut` dans le cas euclidien ;
// le majorant affine des poids s'y ajoute de la meme facon, sans toucher a la structure.
//
// LE RAYON EST ICI LEGITIME, alors qu'il ne l'est pas comme critere d'ARRET. Un critere d'arret
// doit trancher entre des candidats PROCHES, ou un rayon isotrope est trop lache des que les
// poids sont inegaux. Ici il elimine des SOUS-ARBRES lointains, et il n'a pas de concurrent moins
// cher : la boite est deja lue, la comparaison est en O( D ).
//
// = CE QUE LE `Local` PORTE
//
// La pile, et la position dans la tranche de la feuille courante. Rien d'autre : l'elagage etant
// exact et vectoriel, il n'y a rien a mettre en cache donc rien a invalider -- ce fournisseur ne
// demande meme pas `Etat::change`, et l'`Etat` reste dans sa forme courte.
// =====================================================================================

#include "spatial_accel/AaBsp.h"
#include "supercell/Elagage.h"
#include "supercell/Fournisseurs.h"

#include <immintrin.h>

namespace noyau2d {

/// `POIDS` : diagramme de LAGUERRE au lieu de VORONOI. C'est une constante de compilation parce
/// que le cas euclidien ne doit rien payer -- ni les pentes du majorant, ni les termes en `a . y`.
template<class Arbre, bool POIDS = false>
struct FournisseurBsp {
    static constexpr int D = Arbre::dim;
    static_assert( D == 2, "le noyau de coupe est 2D" );

    /// L'ETAT DU PARCOURS, un par cellule, loge dans la frame du moteur.
    ///
    /// La pile est bornee par la PROFONDEUR de l'arbre, pas par sa taille : a chaque niveau on
    /// depile un noeud et on en empile deux. 48 niveaux valent 2^48 germes.
    struct Local {
        int   pile[ 48 ];
        int   haut = 0;
        int   k = 0, fin = 0;                            ///< la tranche de la feuille courante
        bool  amorce = false;
    };

    const Arbre *arbre;
    float x0, y0;                                        ///< le germe courant
    float w0 = 0;                                        ///< son poids ( ignore si `! POIDS` )
    int   i0;                                            ///< son identifiant, pour ne pas se couper

    FournisseurBsp( const Arbre *arbre, float x0, float y0, int i0 )
        : arbre( arbre ), x0( x0 ), y0( y0 ), i0( i0 ) {}
    FournisseurBsp( const Arbre *arbre, float x0, float y0, float w0, int i0 )
        : arbre( arbre ), x0( x0 ), y0( y0 ), w0( w0 ), i0( i0 ) {}

    /// L'ELAGAGE, EXACT ET VECTORIEL. Huit sommets, huit voies.
    ///
    /// Si un germe `q` de la boite `B` coupe la cellule `C`, il en retranche au moins un SOMMET --
    /// le demi-plan retire en contient un -- et ce sommet verifie `|v - q| < |v - p0|`. Donc :
    ///
    ///     rejeter B  <=>  pour TOUT sommet v :  dist^2( v, B ) >= |v - p0|^2
    ///
    /// et c'est EXACT, pas conservatif. C'est ce qui separe ce fournisseur d'une premiere version
    /// qui comparait la BOITE de la cellule a celle du noeud : elle proposait 280 candidats par
    /// cellule la ou il en faut 26, une boite majorant tres mal un polygone convexe.
    ///
    /// Le SIMD est a la charge du fournisseur, et c'est ici qu'il sert : `dist^2( v, B )` est
    /// separable par axe, donc deux `max`, deux `fmadd`, un `vcmpps`. Rien n'est mis en cache, donc
    /// ce fournisseur ne demande meme pas `Etat::change`.
    /// AVEC POIDS, `w( q ) <= a . q + b` ( le majorant AFFINE du sous-arbre ) entre dans le meme
    /// test. Tout est dans `Elagage.h` : les trois fournisseurs posent la meme question.
    template<class Etat>
    bool peut_couper( const typename Arbre::Node &nd, const Etat &e ) const {
        Boite B;
        for ( int d = 0; d < D; ++d ) { B.lo[ d ] = (float) nd.lo[ d ]; B.hi[ d ] = (float) nd.hi[ d ]; }
        if constexpr ( POIDS ) {
            B.a[ 0 ] = (float) nd.wm.a[ 0 ]; B.a[ 1 ] = (float) nd.wm.a[ 1 ]; B.b = (float) nd.wm.b;
        }
        return peut_couper_boite<POIDS>( e, x0, y0, w0, B );
    }

    /// le noeud le plus proche du GERME est visite en premier : les coupes qui mordent le plus
    /// arrivent tot, donc la cellule retrecit vite, donc l'elagage mord plus.
    float proximite( int n ) const {
        const auto &nd = arbre->nodes[ n ];
        float s = 0;
        for ( int d = 0; d < D; ++d ) {
            const float x = d ? y0 : x0;
            const float lo = (float) nd.lo[ d ], hi = (float) nd.hi[ d ];
            const float e = x < lo ? lo - x : x > hi ? x - hi : 0.f;
            s += e * e;
        }
        return s;
    }

    template<class Etat>
    bool suivant( const Etat &e, Local &l, Plan &p ) {
        if ( ! l.amorce ) { l.pile[ l.haut++ ] = 0; l.amorce = true; }

        for ( ;; ) {
            // ---- une feuille est ouverte : on rend le germe suivant de sa tranche
            while ( l.k < l.fin ) {
                const int k = l.k++;
                const int id = (int) arbre->order[ k ];
                if ( id == i0 ) continue;
                const float xj = (float) arbre->seed_x( k ), yj = (float) arbre->seed_y( k );
                p.dx = xj - x0;
                p.dy = yj - y0;
                p.off = 0.5f * ( p.dx * ( xj + x0 ) + p.dy * ( yj + y0 ) );
                if constexpr ( POIDS ) p.off += 0.5f * ( w0 - (float) arbre->seed_w( k ) );
                p.id = id;
                return true;
            }

            if ( l.haut == 0 )
                return false;                            // l'arbre est epuise

            const int h = l.pile[ --l.haut ];
            const auto &nd = arbre->nodes[ h ];

            if ( ! peut_couper( nd, e ) )                // aucun germe de ce sous-arbre ne peut rien
                continue;

            if ( nd.right < 0 ) { l.k = (int) nd.beg; l.fin = (int) nd.end; continue; }

            const int g = h + 1, dr = (int) nd.right;    // PREORDRE : le gauche est juste a cote
            if ( proximite( g ) <= proximite( dr ) ) { l.pile[ l.haut++ ] = dr; l.pile[ l.haut++ ] = g; }
            else                                       { l.pile[ l.haut++ ] = g;  l.pile[ l.haut++ ] = dr; }
        }
    }
};

} // namespace noyau2d

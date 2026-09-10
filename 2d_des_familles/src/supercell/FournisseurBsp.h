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
    /// AVEC POIDS, le meme critere devient : un germe `q` de `B`, de poids `w( q )`, coupe si
    ///
    ///     |v - q|^2 - w( q )  <  |v - p0|^2 - w0        pour un sommet v
    ///
    /// et `w( q ) <= a . q + b` ( le majorant AFFINE du sous-arbre ). Minorer le membre gauche
    /// demande donc de minimiser `|v - q|^2 - a . q` sur la boite : c'est separable par axe, le
    /// minimum libre est en `q = v + a / 2`, et un `clamp` par axe donne la reponse EXACTE.
    ///
    /// `<= 0` et non `< 0` : un plan qui passe exactement par un sommet n'enleve rien, donc
    /// l'admettre coute une coupe inutile la ou le refuser sur un arrondi perdrait une coupe VRAIE.
    template<class Etat>
    bool peut_couper( const typename Arbre::Node &nd, const Etat &e ) const {
        const float a0 = POIDS ? (float) nd.wm.a[ 0 ] : 0.f;
        const float a1 = POIDS ? (float) nd.wm.a[ 1 ] : 0.f;
        const float cb = POIDS ? float( w0 - nd.wm.b ) : 0.f;

        if constexpr ( requires { e.vx + e.vx; } ) {
            const __m256 lo0 = _mm256_set1_ps( (float) nd.lo[ 0 ] ), hi0 = _mm256_set1_ps( (float) nd.hi[ 0 ] );
            const __m256 lo1 = _mm256_set1_ps( (float) nd.lo[ 1 ] ), hi1 = _mm256_set1_ps( (float) nd.hi[ 1 ] );

            // le point de la boite le plus proche du sommet, DECALE d'une demi-pente
            __m256 y0v = e.vx, y1v = e.vy;
            if constexpr ( POIDS ) {
                y0v = _mm256_add_ps( y0v, _mm256_set1_ps( 0.5f * a0 ) );
                y1v = _mm256_add_ps( y1v, _mm256_set1_ps( 0.5f * a1 ) );
            }
            y0v = _mm256_min_ps( _mm256_max_ps( y0v, lo0 ), hi0 );
            y1v = _mm256_min_ps( _mm256_max_ps( y1v, lo1 ), hi1 );

            const __m256 e0 = _mm256_sub_ps( y0v, e.vx ), f0 = _mm256_sub_ps( e.vx, _mm256_set1_ps( x0 ) );
            const __m256 e1 = _mm256_sub_ps( y1v, e.vy ), f1 = _mm256_sub_ps( e.vy, _mm256_set1_ps( y0 ) );
            __m256 s = _mm256_fmadd_ps( e0, e0, _mm256_mul_ps( e1, e1 ) );
            s = _mm256_sub_ps( s, _mm256_fmadd_ps( f0, f0, _mm256_mul_ps( f1, f1 ) ) );
            if constexpr ( POIDS ) {
                s = _mm256_fnmadd_ps( _mm256_set1_ps( a0 ), y0v, s );
                s = _mm256_fnmadd_ps( _mm256_set1_ps( a1 ), y1v, s );
                s = _mm256_add_ps( s, _mm256_set1_ps( cb ) );
            }
            return ( _mm256_cmp_ps_mask( s, _mm256_setzero_ps(), _CMP_LE_OQ )
                     & ( ( 1u << Etat::nb ) - 1 ) ) != 0;
        } else {                                         // excursion : l'etat est en memoire
            const float a[ 2 ] = { a0, a1 };
            for ( int i = 0; i < e.nb; ++i ) {
                const float pv[ 2 ] = { e.vx[ i ], e.vy[ i ] };
                const float p0v[ 2 ] = { x0, y0 };
                float s = POIDS ? cb : 0.f;
                for ( int d = 0; d < D; ++d ) {
                    float y = pv[ d ] + ( POIDS ? 0.5f * a[ d ] : 0.f );
                    const float lo = (float) nd.lo[ d ], hi = (float) nd.hi[ d ];
                    y = y < lo ? lo : ( y > hi ? hi : y );
                    const float u = y - pv[ d ], f = pv[ d ] - p0v[ d ];
                    s += u * u - f * f;
                    if constexpr ( POIDS ) s -= a[ d ] * y;
                }
                if ( s <= 0 ) return true;
            }
            return false;
        }
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

#pragma once

// =====================================================================================
// L'ENCEINTE D'UN AGREGAT : `K` BANDES, EVENTUELLEMENT TOURNEES.
//
// L'enceinte est le convexe sur lequel la table pose sa question ( « l'agregat `B` peut-il encore
// couper une cellule de `A` ? » ). Elle doit contenir la reunion des cellules de phase 1 de `A`, et
// plus elle est serree, plus la ligne de table est courte.
//
// `K = 2` sans rotation, c'est la boite englobante -- ce qu'il y avait avant. `K = 2` avec rotation
// c'est la boite ORIENTEE ; `K = 4` ou `8`, le k-DOP.
//
// = LE TEST, ET POURQUOI IL PREND LES DIRECTIONS PAR PAIRES ORTHOGONALES
//
// La forme evidente -- « separes selon une direction, donc ecartes » -- est le theoreme des axes
// separateurs, et elle REGRESSE par rapport a la boite : le test de la boite ne compare pas un
// ecart mais la SOMME des deux ecarts au carre, ce qui est strictement plus fort pour une boite
// posee en diagonale. Passer de deux directions a quatre en ne gardant que le max ferait perdre
// plus que la forme ne fait gagner.
//
// On garde donc la somme, et elle reste VALIDE des que les deux directions sont ORTHOGONALES :
// si `a` et `b` realisent la distance et `d = b - a`, alors `u_k . d >= g_k` pour chaque direction
// ou les projections sont separees, et `|d|^2 = sum ( u_k . d )^2` sur une BASE orthonormee. Donc
//
//     dist^2( enceinte, boite )  >=  g_k^2 + g_{k + K/2}^2       pour chaque k < K / 2
//
// Les directions etant espacees de `pi / K`, `k` et `k + K/2` sont a angle droit : le k-DOP offre
// `K / 2` bases, on prend la meilleure, et pour `K = 2` on retrouve exactement l'ancien test.
//
// = CE QUE LA ROTATION COUTE
//
// Les noeuds de l'arbre BSP sont des boites ALIGNEES SUR LES AXES. Tourner l'enceinte resserre
// l'enceinte mais ELARGIT la projection du noeud -- une boite projetee sur une diagonale est plus
// large qu'elle. Les deux effets vont en sens contraire et rien ne dit a priori lequel gagne :
// c'est une mesure, pas un raisonnement.
//
// = CE QUE LA MESURE DIT ( n = 200000, rho = 8, uniforme, un fil )
//
//   Voronoi                agr/ligne   enceintes   table    TOTAL
//   2-DOP   ( la boite )      7.79      0.0136    0.0592   0.2334
//   4-DOP                     6.72      0.0153    0.0630   0.2377
//   8-DOP                     6.44      0.0169    0.0604   0.2400
//   2-DOP tourne              9.29      0.0158    0.0691   0.2478
//   8-DOP tourne              6.59      0.0184    0.0628   0.2411
//
// et la meme chose en Laguerre a `4 h^2` : 10.04 / 9.07 / 8.76 agregats par ligne, total 0.2583 /
// 0.2707 / 0.2739.
//
// DEUX VERDICTS, TOUS DEUX NEGATIFS, ET C'EST LE RESULTAT :
//
//   1. LA ROTATION PERD, et elle perd sur le nombre qu'elle etait censee ameliorer -- 9.29
//      agregats par ligne au lieu de 7.79. L'enceinte est pourtant plus serree ; c'est bien la
//      projection des boites du BSP qui se paie, et elle se paie plus cher que le gain de forme.
//      L'axe principal n'est pas en cause : le 8-DOP tourne, qui depend beaucoup moins de
//      l'orientation choisie, ne rattrape que jusqu'a 6.59, soit le 8-DOP droit.
//
//   2. LES DIRECTIONS SUPPLEMENTAIRES RACCOURCISSENT BIEN LES LIGNES -- de 17 % au 8-DOP -- mais
//      le test coute `K / 2` bases au lieu d'une et l'enceinte coute `K` projections par sommet au
//      lieu de deux. Le temps de table ne bouge pas et le total monte.
//
// La boite reste donc le defaut. Le reste est garde parce qu'il est ecrit, mesure, et qu'une
// distribution franchement anisotrope pourrait renverser le verdict -- pas parce qu'il sert ici.
// =====================================================================================

#include <cmath>

namespace noyau2d {

/// `K` directions a `pi / K` d'ecart, plus la portee. `ROT` : l'angle de base est choisi par
/// agregat ( axe principal des sommets ) au lieu d'etre zero.
template<int K, bool ROT>
struct Dop {
    static_assert( K >= 2 && K % 2 == 0, "les directions vont par paires orthogonales" );
    static constexpr int  nb_dirs = K;
    static constexpr bool tourne  = ROT;

    float ux[ K ], uy[ K ];                              ///< les directions, dans le repere du monde
    float lo[ K ], hi[ K ];
    float r2;

    /// `th` est l'angle de base. Sans rotation il vaut zero et `u_0` est l'axe des x.
    void oriente( float th ) {
        for ( int k = 0; k < K; ++k ) {
            const float a = th + float( k ) * ( 3.14159265358979f / K );
            ux[ k ] = std::cos( a ); uy[ k ] = std::sin( a );
        }
    }

    void vide() { for ( int k = 0; k < K; ++k ) { lo[ k ] = 1e30f; hi[ k ] = -1e30f; } r2 = -1e30f; }

    void ajoute( float x, float y ) {
        for ( int k = 0; k < K; ++k ) {
            const float p = ux[ k ] * x + uy[ k ] * y;
            lo[ k ] = p < lo[ k ] ? p : lo[ k ];
            hi[ k ] = p > hi[ k ] ? p : hi[ k ];
        }
    }

    /// `true` : aucun germe de la boite alignee `[ nlo, nhi ]` ne peut couper une cellule de `A`.
    bool ecarte( const float nlo[ 2 ], const float nhi[ 2 ] ) const {
        float g[ K ];
        for ( int k = 0; k < K; ++k ) {
            const float ax = ux[ k ], ay = uy[ k ];
            const float p0 = ax * ( ax > 0 ? nlo[ 0 ] : nhi[ 0 ] ) + ay * ( ay > 0 ? nlo[ 1 ] : nhi[ 1 ] );
            const float p1 = ax * ( ax > 0 ? nhi[ 0 ] : nlo[ 0 ] ) + ay * ( ay > 0 ? nhi[ 1 ] : nlo[ 1 ] );
            g[ k ] = p0 > hi[ k ] ? p0 - hi[ k ] : ( lo[ k ] > p1 ? lo[ k ] - p1 : 0.f );
        }
        float d2 = 0;
        for ( int k = 0; k < K / 2; ++k ) {               // la meilleure base orthonormee
            const float v = g[ k ] * g[ k ] + g[ k + K / 2 ] * g[ k + K / 2 ];
            d2 = v > d2 ? v : d2;
        }
        return d2 >= r2;
    }

    /// AVEC POIDS, `w( q ) <= a . q + b` sur la boite. On complete le carre :
    ///
    ///     |v - q|^2 - a . q  =  |q - ( v + a/2 )|^2 - |a|^2 / 4 - a . v
    ///
    /// donc il suffit d'ecarter la boite de l'enceinte DECALEE de `a/2`, avec la portee augmentee
    /// de `b + |a|^2/4 + h_E( a )`, ou `h_E` est la fonction d'appui de l'enceinte. Le decalage est
    /// une translation : il se lit directement sur les intervalles.
    ///
    /// `h_E` n'est pas connue exactement pour un k-DOP -- c'est une INTERSECTION de bandes, pas
    /// leur reunion -- mais la majorer suffit : le test devient un peu plus prudent, jamais faux.
    /// Sur chaque base orthonormee, `a` se decompose et l'appui de la BOITE de cette base majore
    /// celui du DOP ; on garde la meilleure des `K/2` bases.
    bool ecarte_p( const float nlo[ 2 ], const float nhi[ 2 ], float a0, float a1, float b ) const {
        const float dx = 0.5f * a0, dy = 0.5f * a1;
        float g[ K ], al[ K ];
        for ( int k = 0; k < K; ++k ) {
            const float ax = ux[ k ], ay = uy[ k ];
            const float sh = ax * dx + ay * dy;           // l'enceinte decalee, sur cette direction
            const float l = lo[ k ] + sh, h = hi[ k ] + sh;
            const float p0 = ax * ( ax > 0 ? nlo[ 0 ] : nhi[ 0 ] ) + ay * ( ay > 0 ? nlo[ 1 ] : nhi[ 1 ] );
            const float p1 = ax * ( ax > 0 ? nhi[ 0 ] : nlo[ 0 ] ) + ay * ( ay > 0 ? nhi[ 1 ] : nlo[ 1 ] );
            g[ k ] = p0 > h ? p0 - h : ( l > p1 ? l - p1 : 0.f );
            al[ k ] = ax * a0 + ay * a1;                  // la composante de `a` sur cette direction
        }
        // SANS ROTATION, LA PREMIERE BASE EST CELLE DES AXES, et la boite du noeud y est une boite :
        // la minimisation jointe `min ( |v - q|^2 - a . q )` y est donc SEPARABLE et EXACTE, la
        // fonction d'appui n'a pas a etre majoree. C'est strictement mieux que la forme completee
        // ci-dessous -- mesure : 10.04 agregats par ligne au lieu de 10.47 -- donc on la garde, et
        // les bases tournees ne font que s'y ajouter.
        if constexpr ( ! ROT ) {
            const float lo2[ 2 ] = { lo[ 0 ], lo[ K / 2 ] }, hi2[ 2 ] = { hi[ 0 ], hi[ K / 2 ] };
            const float a2[ 2 ] = { a0, a1 };
            float d2 = - b;
            for ( int d = 0; d < 2; ++d ) {
                float q = ( a2[ d ] > 0 ? hi2[ d ] : lo2[ d ] ) + 0.5f * a2[ d ];
                q = q < nlo[ d ] ? nlo[ d ] : ( q > nhi[ d ] ? nhi[ d ] : q );
                const float t = q < lo2[ d ] ? lo2[ d ] - q : ( q > hi2[ d ] ? q - hi2[ d ] : 0.f );
                d2 += t * t - a2[ d ] * q;
            }
            if ( d2 >= r2 ) return true;
        }

        const float cst = b + 0.25f * ( a0 * a0 + a1 * a1 );
        for ( int k = ROT ? 0 : 1; k < K / 2; ++k ) {     // la base 0 vient d'etre faite, exactement
            const int j = k + K / 2;
            const float d2 = g[ k ] * g[ k ] + g[ j ] * g[ j ];
            const float hE = al[ k ] * ( al[ k ] > 0 ? hi[ k ] : lo[ k ] )
                           + al[ j ] * ( al[ j ] > 0 ? hi[ j ] : lo[ j ] );
            if ( d2 >= r2 + cst + hE ) return true;
        }
        return false;
    }

    /// L'AXE PRINCIPAL des sommets, par les moments d'ordre deux. Ce n'est pas la boite orientee
    /// d'aire minimale -- celle-la demande l'enveloppe convexe et des calipers -- mais elle coute
    /// un `atan2` par agregat au lieu d'un tri.
    static float axe_principal( float sxx, float syy, float sxy ) {
        return 0.5f * std::atan2( 2.f * sxy, sxx - syy );
    }
};

} // namespace noyau2d

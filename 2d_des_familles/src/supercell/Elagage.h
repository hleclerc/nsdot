#pragma once

// =====================================================================================
// LE TEST D'ELAGAGE, UNE FOIS POUR TOUTES.
//
// Les trois fournisseurs posent la MEME question, a trois echelles differentes : un noeud de
// l'arbre BSP, un agregat de la table, un agregat de l'anneau 1. La question est :
//
//     un germe `q` de la boite `B`, de poids majore par `w( q ) <= a . q + b`, peut-il encore
//     retrancher quelque chose a la cellule de `p0` ?
//
// = LE CRITERE, ET POURQUOI IL EST EXACT
//
// Si `q` coupe la cellule, il en retranche au moins un SOMMET, et ce sommet verifie
//
//     |v - q|^2 - w( q )  <  |v - p0|^2 - w0
//
// La reciproque n'a pas besoin d'etre vraie : on cherche un test qui ne rejette JAMAIS a tort, pas
// un test qui accepte le moins possible. Donc on rejette `B` des que TOUS les sommets ont
//
//     min_{ q in B } ( |v - q|^2 - a . q - b )  >=  |v - p0|^2 - w0
//
// et c'est EXACT au sens ou le minimum est calcule exactement, pas majore : `|v - q|^2 - a . q` est
// separable par axe, son minimum libre est en `q = v + a / 2`, et un `clamp` par axe donne la
// reponse. Le majorant constant est le cas `a = 0` et ne coute pas moins cher.
//
// C'est ce qui separe ce test d'une premiere version qui comparait la BOITE de la cellule a celle
// du noeud : elle proposait 280 candidats par cellule la ou 26 suffisent, une boite majorant tres
// mal un polygone convexe.
//
// = POURQUOI LE SIMD EST ICI ET PAS DANS LE NOYAU
//
// Le noyau ne connait pas les criteres d'arret -- c'est le contrat. Les sommets lui appartiennent
// et arrivent en registres ; le fournisseur les lit tels quels. Deux `max`, deux `fmadd`, un
// `vcmpps` : le test coute une operation SIMD par boite, quel que soit le nombre de sommets.
//
// `<= 0` et non `< 0` : un plan qui passe exactement par un sommet n'enleve rien, donc l'admettre
// coute une coupe inutile la ou le refuser sur un arrondi perdrait une coupe VRAIE.
// =====================================================================================

#include <immintrin.h>
#include <vector>

namespace noyau2d {

/// Une boite de germes et le majorant affine de leurs poids. AoS et pas SoA : les sept champs sont
/// lus ensemble, et une ligne de cache en tient deux.
struct Boite {
    float lo[ 2 ], hi[ 2 ];
    float a[ 2 ] = { 0, 0 }, b = 0;                      ///< `w( q ) <= a . q + b`

    void vide() { lo[ 0 ] = lo[ 1 ] = 1e30f; hi[ 0 ] = hi[ 1 ] = -1e30f; }
    void ajoute( float x, float y ) {
        lo[ 0 ] = x < lo[ 0 ] ? x : lo[ 0 ]; hi[ 0 ] = x > hi[ 0 ] ? x : hi[ 0 ];
        lo[ 1 ] = y < lo[ 1 ] ? y : lo[ 1 ]; hi[ 1 ] = y > hi[ 1 ] ? y : hi[ 1 ];
    }
};

/// une boite de germes par agregat
using BoitesAgregats = std::vector<Boite>;

/// Le test lui-meme. `e` est l'etat du noyau : soit huit voies de registre ( `e.vx` est un vecteur
/// et `Etat::nb` une constante ), soit une excursion en memoire ( `e.vx` est un pointeur ).
template<bool POIDS, class Etat>
inline bool peut_couper_boite( const Etat &e, float x0, float y0, float w0, const Boite &B ) {
    const float a0 = POIDS ? B.a[ 0 ] : 0.f;
    const float a1 = POIDS ? B.a[ 1 ] : 0.f;
    const float cb = POIDS ? w0 - B.b : 0.f;

    if constexpr ( requires { e.vx + e.vx; } ) {
        const __m256 lo0 = _mm256_set1_ps( B.lo[ 0 ] ), hi0 = _mm256_set1_ps( B.hi[ 0 ] );
        const __m256 lo1 = _mm256_set1_ps( B.lo[ 1 ] ), hi1 = _mm256_set1_ps( B.hi[ 1 ] );

        // le point de la boite le plus proche du sommet, DECALE d'une demi-pente
        __m256 y0v = e.vx, y1v = e.vy;
        if constexpr ( POIDS ) {
            y0v = _mm256_add_ps( y0v, _mm256_set1_ps( 0.5f * a0 ) );
            y1v = _mm256_add_ps( y1v, _mm256_set1_ps( 0.5f * a1 ) );
        }
        y0v = _mm256_min_ps( _mm256_max_ps( y0v, lo0 ), hi0 );
        y1v = _mm256_min_ps( _mm256_max_ps( y1v, lo1 ), hi1 );

        const __m256 g0 = _mm256_sub_ps( y0v, e.vx ), f0 = _mm256_sub_ps( e.vx, _mm256_set1_ps( x0 ) );
        const __m256 g1 = _mm256_sub_ps( y1v, e.vy ), f1 = _mm256_sub_ps( e.vy, _mm256_set1_ps( y0 ) );
        __m256 s = _mm256_fmadd_ps( g0, g0, _mm256_mul_ps( g1, g1 ) );
        s = _mm256_sub_ps( s, _mm256_fmadd_ps( f0, f0, _mm256_mul_ps( f1, f1 ) ) );
        if constexpr ( POIDS ) {
            s = _mm256_fnmadd_ps( _mm256_set1_ps( a0 ), y0v, s );
            s = _mm256_fnmadd_ps( _mm256_set1_ps( a1 ), y1v, s );
            s = _mm256_add_ps( s, _mm256_set1_ps( cb ) );
        }
        return ( _mm256_cmp_ps_mask( s, _mm256_setzero_ps(), _CMP_LE_OQ )
                 & ( ( 1u << Etat::nb ) - 1 ) ) != 0;
    } else {                                             // excursion : l'etat est en memoire
        const float a[ 2 ] = { a0, a1 };
        const float p0[ 2 ] = { x0, y0 };
        for ( int i = 0; i < e.nb; ++i ) {
            const float v[ 2 ] = { e.vx[ i ], e.vy[ i ] };
            float s = POIDS ? cb : 0.f;
            for ( int d = 0; d < 2; ++d ) {
                float y = v[ d ] + ( POIDS ? 0.5f * a[ d ] : 0.f );
                y = y < B.lo[ d ] ? B.lo[ d ] : ( y > B.hi[ d ] ? B.hi[ d ] : y );
                const float u = y - v[ d ], f = v[ d ] - p0[ d ];
                s += u * u - f * f;
                if constexpr ( POIDS ) s -= a[ d ] * y;
            }
            if ( s <= 0 ) return true;
        }
        return false;
    }
}

} // namespace noyau2d

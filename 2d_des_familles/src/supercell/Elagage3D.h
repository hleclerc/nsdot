#pragma once

// =====================================================================================
// LE TEST D'ELAGAGE EN 3D. Meme critere qu'en 2D, meme exactitude, un axe de plus.
//
// La question, posee a un noeud de l'arbre BSP :
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
// On rejette donc `B` des que TOUS les sommets ont
//
//     min_{ q in B } ( |v - q|^2 - a . q - b )  >=  |v - p0|^2 - w0
//
// et c'est EXACT au sens ou le minimum est calcule, pas majore : `|v - q|^2 - a . q` est separable
// par axe, son minimum libre est en `q = v + a / 2`, et un `clamp` par axe donne la reponse. La
// reciproque n'a pas besoin d'etre vraie -- on veut un test qui ne rejette JAMAIS a tort, pas un
// test qui accepte le moins possible.
//
// `<= 0` et non `< 0` : un plan qui passe exactement par un sommet n'enleve rien, donc l'admettre
// coute une coupe inutile la ou le refuser sur un arrondi perdrait une coupe VRAIE.
//
// = CE QUE LA 3D CHANGE, ET C'EST LE POINT DELICAT
//
// En 2D les sommets sont en REGISTRES -- le noyau les y tient -- et le test coute UNE operation
// SIMD par boite, quel que soit leur nombre. En 3D ils sont en MEMOIRE, dans les tableaux alignes
// de la cellule, et il y en a vingt a vingt-cinq : le test coute une boucle. C'est le prix a
// l'entree de l'elagage 3D, et c'est lui qu'il faut regarder si le BSP rend moins que prevu.
//
// D'ou l'ecriture en asimd, par blocs de `W` voies, exactement comme la premiere passe de la
// coupe : chargements pleins tant qu'il reste un vecteur entier, puis une seule queue partielle.
// Les lanes hors de la queue sont indefinies apres `load_partial`, donc les bits correspondants
// sont masques -- sans quoi un `s` de hasard ferait croire a un sommet coupable et on renoncerait
// a elaguer ( ce serait conservatif, donc juste, mais lent ).
//
// LA SORTIE ANTICIPEE EST VOULUE : des qu'UN sommet peut etre coupe, la reponse est connue. Sur
// une cellule qui a encore beaucoup de sommets, c'est presque toujours le premier bloc qui tranche.
// =====================================================================================

#include <asimd/asimd.h>
#include <vector>

namespace noyau3d {

/// Une boite de germes et le majorant affine de leurs poids. AoS et pas SoA : les dix champs sont
/// lus ensemble, en un seul test.
struct Boite3 {
    float lo[ 3 ], hi[ 3 ];
    float a[ 3 ] = { 0, 0, 0 }, b = 0;                    ///< `w( q ) <= a . q + b`

    void vide() {
        for ( int d = 0; d < 3; ++d ) { lo[ d ] = 1e30f; hi[ d ] = -1e30f; }
    }
    void ajoute( float x, float y, float z ) {
        const float p[ 3 ] = { x, y, z };
        for ( int d = 0; d < 3; ++d ) {
            lo[ d ] = p[ d ] < lo[ d ] ? p[ d ] : lo[ d ];
            hi[ d ] = p[ d ] > hi[ d ] ? p[ d ] : hi[ d ];
        }
    }
};

using BoitesAgregats3 = std::vector<Boite3>;

/// Le test. `e` est ce que le fournisseur voit de la cellule : `nb` sommets dans trois tableaux
/// ALIGNES. Rend `true` s'il faut descendre dans la boite, `false` si le sous-arbre est mort.
template<bool POIDS,int W = 8,class Etat>
inline bool peut_couper_boite3( const Etat &e, float x0, float y0, float z0, float w0,
                                const Boite3 &B ) {
    using V = asimd::SimdVec<float,W>;

    const V l0( B.lo[0] ), h0( B.hi[0] ), l1( B.lo[1] ), h1( B.hi[1] ), l2( B.lo[2] ), h2( B.hi[2] );
    const V q0( x0 ), q1( y0 ), q2( z0 ), zero( 0.f );
    const V a0( POIDS ? B.a[0] : 0.f ), a1( POIDS ? B.a[1] : 0.f ), a2( POIDS ? B.a[2] : 0.f );
    const V m0( POIDS ? 0.5f * B.a[0] : 0.f ), m1( POIDS ? 0.5f * B.a[1] : 0.f ),
            m2( POIDS ? 0.5f * B.a[2] : 0.f );
    const V cb( POIDS ? w0 - B.b : 0.f );

    // `s = min_q ( |v - q|^2 - a . q ) - |v - p0|^2 - b + w0`, par axe, pour huit sommets.
    auto bilan = [ & ]( const V &vx, const V &vy, const V &vz ) {
        V y0v = vx, y1v = vy, y2v = vz;
        if constexpr ( POIDS ) { y0v = y0v + m0; y1v = y1v + m1; y2v = y2v + m2; }
        y0v = asimd::min( asimd::max( y0v, l0 ), h0 );   // le point de la boite le plus proche,
        y1v = asimd::min( asimd::max( y1v, l1 ), h1 );   // DECALE d'une demi-pente
        y2v = asimd::min( asimd::max( y2v, l2 ), h2 );

        const V g0 = y0v - vx, g1 = y1v - vy, g2 = y2v - vz;
        const V f0 = vx - q0,  f1 = vy - q1,  f2 = vz - q2;
        V s = asimd::fma( g0, g0, asimd::fma( g1, g1, g2 * g2 ) )
            - asimd::fma( f0, f0, asimd::fma( f1, f1, f2 * f2 ) );
        if constexpr ( POIDS )
            s = cb - asimd::fma( a0, y0v, asimd::fma( a1, y1v, a2 * y2v ) ) + s;
        return s;
    };

    const int plein = e.nb & ~( W - 1 );
    int i = 0;
    for ( ; i < plein; i += W ) {
        const V s = bilan( V::load_aligned( e.vx + i ),
                           V::load_aligned( e.vy + i ),
                           V::load_aligned( e.vz + i ) );
        if ( asimd::to_bits( asimd::ge( zero, s ) ) )
            return true;                                 // ce sommet-la peut etre coupe
    }
    if ( i < e.nb ) {
        const auto r = asimd::LaneRange<0>( e.nb - i );
        const V s = bilan( V::load_partial( e.vx + i, r ),
                           V::load_partial( e.vy + i, r ),
                           V::load_partial( e.vz + i, r ) );
        // LES LANES HORS QUEUE SONT INDEFINIES : on les masque, sinon un `s` de hasard nous ferait
        // renoncer a un elagage legitime.
        if ( asimd::to_bits( asimd::ge( zero, s ) ) & ( ( 1ull << ( e.nb - i ) ) - 1 ) )
            return true;
    }
    return false;
}

} // namespace noyau3d

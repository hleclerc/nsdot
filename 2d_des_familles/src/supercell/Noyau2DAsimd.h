#pragma once

// =====================================================================================
// LE MEME NOYAU, SUR ASIMD.
//
// Troisieme ecriture du meme algorithme -- apres les intrinseques et highway -- et le meme
// dictionnaire suffit : `fma`, `permute`, `select`, `to_bits`, plus `eq` pour designer une voie.
//
// CE QU'ASIMD A DE PLUS. La largeur est un parametre du TYPE, pas une consequence de la cible :
// `SimdVec<float,8>` compile partout, et quand le registre n'a pas huit voies l'implementation
// se DECOUPE recursivement ( `split_size_0 = prev_pow_2( size )`, le reste dans `v1` ). Highway
// refuse -- « Too many lanes » -- des que le descripteur depasse la cible, ce qui exclut NEON et
// tout x86 anterieur a AVX2 pour ce noyau-ci.
//
// CE QU'IL A FALLU AJOUTER ( `asimd/SimdOpsPlus.h`, `SimdOpsPlus_X86.h` ) :
//   `fma`, `select`, `to_bits`, `permute`, `eq`, `ge`, et les operateurs `-` `*` `/`.
// Rien d'exotique -- l'arithmetique interne existait deja, seules les facades manquaient -- sauf
// `permute` et `to_bits`, qui sont les deux primitives sur lesquelles tout ce noyau repose.
//
// LES COMPARAISONS D'ASIMD SONT PARESSEUSES : `a > b` ne calcule rien, il rend un objet qui
// garde ses deux operandes, et c'est `to_bits` ou `select` qui decide sous quelle forme le
// materialiser. C'est plus fin que ce que font highway et les intrinseques, ou la comparaison
// choisit sa representation avant de savoir a quoi elle servira.
// =====================================================================================

#include "supercell/Contrat2D.h"

#include <asimd/SimdOpsPlus.h>

namespace noyau2d {
namespace asimd2d {

using V  = asimd::SimdVec<float, 8>;
using VI = asimd::SimdVec<asimd::SI32, 8>;

/// « la voie i ». Par un motif de BITS, pas par une comparaison : `eq( iota, i )` demande un
/// `vpbroadcastd` et un `vpcmpeqd` la ou `mask_from_bits` descend a un `kmovb` sur une machine a
/// registres de masque. Trois voies sont designees a chaque coupe.
inline auto voie( int i ) { return asimd::mask_from_bits<8>( 1u << i ); }
/// diffuser une voie CONNUE A LA COMPILATION. Passer par `permute( v, VI( i ) )` marchait mais
/// coutait un `vpermps` a chaque fois ; `bcast_lane<0>` descend a un `vbroadcastss`.

/// ce que le fournisseur voit. Meme surface que partout ailleurs.
template<int NB>
struct EtatA {
    static constexpr int nb = NB;
    V vx, vy; VI cid;
    float x ( int i ) const { alignas( 32 ) float t[ 8 ]; vx.store_aligned( t ); return t[ i ]; }
    float y ( int i ) const { alignas( 32 ) float t[ 8 ]; vy.store_aligned( t ); return t[ i ]; }
    int   id( int i ) const { alignas( 32 ) asimd::SI32 t[ 8 ]; cid.store_aligned( t ); return t[ i ]; }
};

template<int NB, class Fourn, class Atl> void noyau  ( V vx, V vy, VI cid, Fourn *f, Atl *a, Local<Fourn> *loc );
template<        class Fourn, class Atl> void deborde( V vx, V vy, VI cid, Fourn *f, Atl *a, Local<Fourn> *loc );

template<int NB, class Fourn, class Atl>
void noyau( V vx, V vy, VI cid, Fourn *f, Atl *a, Local<Fourn> *loc ) {
    constexpr unsigned valid = ( 1u << NB ) - 1;
    const VI IOTA = VI::iota( 0 );

    for ( ;; ) {
        Plan p;
        if ( ! f->suivant( EtatA<NB>{ vx, vy, cid }, *loc, p ) )
            break;

        // ---- LE TEST, QUI EST DEJA LA COUPE.
        const V s = asimd::fma( V( p.dx ), vx, asimd::fma( V( p.dy ), vy, V( -p.off ) ) );
        const unsigned m = asimd::to_bits( s > V( 0.f ) ) & valid;

        if ( ! m )
            continue;
        if ( m == valid ) { a->nb = 0; return; }

        // ---- LES DEUX BOUTS DE LA PLAGE EXTERIEURE.
        const unsigned prev = ( ( m << 1 ) | ( m >> ( NB - 1 ) ) ) & valid;
        const unsigned next = ( ( m >> 1 ) | ( m << ( NB - 1 ) ) ) & valid;
        const int i1 = __builtin_ctz( m & ~prev );
        const int j2 = __builtin_ctz( m & ~next );
        const int j0 = i1 ? i1 - 1 : NB - 1;
        const int j3 = j2 + 1 < NB ? j2 + 1 : 0;
        const int nb_in = NB - __builtin_popcount( m );
        const int nn = nb_in + 2;

        // ---- LES DEUX INTERSECTIONS, EN UNE SEULE DIVISION. A en voie 0, B en voie 1 : les
        // deux ancres sont dans des voies DISTINCTES, donc pas de collision quand `j0 == j3`.
        const auto v1 = voie( 1 );
        const VI anc = asimd::select( v1, VI( j3 ), VI( j0 ) );
        const VI oth = asimd::select( v1, VI( j2 ), VI( i1 ) );
        const V vax = asimd::permute( vx, anc ), vox = asimd::permute( vx, oth );
        const V vay = asimd::permute( vy, anc ), voy = asimd::permute( vy, oth );
        const V sa  = asimd::permute( s,  anc ), so  = asimd::permute( s,  oth );
        const V t   = sa / ( sa - so );
        const V pcx = asimd::fma( vox - vax, t, vax );
        const V pcy = asimd::fma( voy - vay, t, vay );

        // ---- LE REMONTAGE. [ j3, j3+1, ... , j0, A, B ].
        VI og = VI( j3 ) + IOTA;
        og = asimd::select( asimd::ge( og, VI( NB ) ), og - VI( NB ), og );
        og = og & VI( 7 );                               // borner : cf. le portage highway

        const auto mA = voie( nb_in ), mB = voie( nb_in + 1 );

        V nvx = asimd::permute( vx, og );
        nvx = asimd::select( mA, asimd::bcast_lane<0>( pcx ), nvx );
        nvx = asimd::select( mB, asimd::bcast_lane<1>( pcx ), nvx );
        V nvy = asimd::permute( vy, og );
        nvy = asimd::select( mA, asimd::bcast_lane<0>( pcy ), nvy );
        nvy = asimd::select( mB, asimd::bcast_lane<1>( pcy ), nvy );
        VI nid = asimd::permute( cid, og );
        nid = asimd::select( mA, VI( p.id ), nid );
        nid = asimd::select( mB, asimd::permute( cid, VI( j2 ) ), nid );

        if ( nn == NB ) { vx = nvx; vy = nvy; cid = nid; continue; }
        if constexpr ( NB == 8 ) if ( nn > 8 ) {
            a->nb = 8;
            a->attente = p;
            vx.store_aligned( a->vx );
            vy.store_aligned( a->vy );
            cid.store_aligned( a->cid );
            [[gnu::musttail]] return deborde( vx, vy, cid, f, a, loc );
        }
        switch ( nn ) {
            case 3: [[gnu::musttail]] return noyau<3>( nvx, nvy, nid, f, a, loc );
            case 4: [[gnu::musttail]] return noyau<4>( nvx, nvy, nid, f, a, loc );
            case 5: [[gnu::musttail]] return noyau<5>( nvx, nvy, nid, f, a, loc );
            case 6: [[gnu::musttail]] return noyau<6>( nvx, nvy, nid, f, a, loc );
            case 7: [[gnu::musttail]] return noyau<7>( nvx, nvy, nid, f, a, loc );
            default: [[gnu::musttail]] return noyau<8>( nvx, nvy, nid, f, a, loc );
        }
    }

    a->nb = NB;
    vx.store_aligned( a->vx );
    vy.store_aligned( a->vy );
    cid.store_aligned( a->cid );
}

template<class Fourn, class Atl>
void deborde( V, V, VI, Fourn *f, Atl *a, Local<Fourn> *loc ) {
    constexpr int MaxNb = Atl::max_nb;
    if constexpr ( MaxNb <= 8 ) {
        a->nb = -1;
        return;
    } else {
        float s[ MaxNb ];
        int  nb = a->nb;
        Plan p  = a->attente;
        for ( ;; ) {
            nb = coupe_large<MaxNb>( a->vx, a->vy, a->cid, nb, p, s );
            if ( nb <= 0 ) { a->nb = nb; return; }
            if ( nb <= 8 ) {
                const V  rvx = V::load_aligned( a->vx );
                const V  rvy = V::load_aligned( a->vy );
                const VI rid = VI::load_aligned( a->cid );
                switch ( nb ) {
                    case 3: [[gnu::musttail]] return noyau<3>( rvx, rvy, rid, f, a, loc );
                    case 4: [[gnu::musttail]] return noyau<4>( rvx, rvy, rid, f, a, loc );
                    case 5: [[gnu::musttail]] return noyau<5>( rvx, rvy, rid, f, a, loc );
                    case 6: [[gnu::musttail]] return noyau<6>( rvx, rvy, rid, f, a, loc );
                    case 7: [[gnu::musttail]] return noyau<7>( rvx, rvy, rid, f, a, loc );
                    default: [[gnu::musttail]] return noyau<8>( rvx, rvy, rid, f, a, loc );
                }
            }
            if ( ! f->suivant( EtatLarge<float>{ nb, a->vx, a->vy, a->cid, true }, *loc, p ) ) {
                a->nb = nb;
                return;
            }
        }
    }
}

template<class Fourn, class Atl>
inline void carre_unite( Fourn *f, Atl *a ) {
    Local<Fourn> loc{};                                  // vit et meurt avec la cellule
    alignas( 32 ) static const float cx[ 8 ] = { 0, 1, 1, 0, 0, 0, 0, 0 };
    alignas( 32 ) static const float cy[ 8 ] = { 0, 0, 1, 1, 0, 0, 0, 0 };
    alignas( 32 ) static const asimd::SI32 ci[ 8 ] = { -1, -2, -3, -4, 0, 0, 0, 0 };
    noyau<4>( V::load_aligned( cx ), V::load_aligned( cy ), VI::load_aligned( ci ), f, a, &loc );
}

} // namespace asimd2d
} // namespace noyau2d

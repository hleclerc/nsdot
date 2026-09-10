// =====================================================================================
// LE MEME NOYAU, PORTABLE -- highway au lieu des intrinseques.
//
// Le portage tient en un dictionnaire ( `MulAdd`, `TableLookupLanes`, `IfThenElse`,
// `BitsFromMask` ) et ne change PAS un choix d'algorithme : meme plage exterieure contigue,
// memes deux `ctz`, meme unique division, meme remontage, meme chaine de `musttail`.
//
// POURQUOI HIGHWAY ET PAS XSIMD. Ce n'est pas une question de gout, c'est la LARGEUR. Sur ce
// Xeon, `xsimd::default_arch` avec `-march=native` est `avx512bw` : seize voies, 512 bits, donc
// la baisse de frequence qu'on a passe le fichier a eviter. xsimd n'a pas de cible « 256 bits
// avec registres de masque » -- sa liste x86 va de `sse2` a `avx512vbmi` sans `avx512vl` -- donc
// le choix se reduit a `avx2` ( 256 bits, mais masques en vecteurs ) ou 512 bits.
//
// Highway separe les deux : la CIBLE dit quelles instructions existent, le DESCRIPTEUR dit
// combien de voies on veut. `FixedTag<float, 8>` sur cible AVX3 donne du 256 bits AVX-512VL --
// `ymm` et registres `k`, aucun `zmm`, verifie au desassemblage. La largeur redevient un choix
// de l'auteur, et c'est exactement ce qui manquait.
//
// DEUX PIEGES, RENCONTRES EN PORTANT, ET QUI NE SE VOIENT PAS A LA LECTURE.
//
//   `Broadcast<i>` N'EST PAS UNE DIFFUSION. Sur x86 256 bits c'est un `vpshufd`, donc une
//   diffusion PAR BLOC DE 128 BITS : `Broadcast<0>` rend [ v0 v0 v0 v0 v4 v4 v4 v4 ], et le
//   `static_assert( kLane < 4 )` est le seul indice. La voie `nb_in` qu'on veut lire est
//   souvent au-dela de 3, donc on lisait `v4`. C'est `BroadcastLane<i>` qui diffuse sur tout
//   le vecteur. Mesure de la difference : aire 0.41 au lieu de 1.00, 31 cellules fausses sur
//   50 -- et un noyau qui paraissait 15 % PLUS RAPIDE, parce qu'une cellule fausse se vide tot.
//
//   LES INDICES DE PERMUTATION DOIVENT ETRE BORNES. `vpermps` ignore les bits hauts, donc les
//   intrinseques toleraient un index a 11 ; `TableLookupLanes` ne le promet pas.
//
// CE QUI RESTE A LA CHARGE DE L'ARCHITECTURE. Deux primitives ne sont pas gratuites partout :
//   `BitsFromMask`      -- un `kmovb` ici, un `vmovmskps` sur AVX2, et sur NEON une reduction
//                          en trois ou quatre instructions. C'est elle qui porte les `ctz`, le
//                          `popcount` et les rotations du masque, donc tout le sans-boucle.
//   `TableLookupLanes`  -- `vpermps` ici et sur AVX2, `tbl` sur NEON ( sur des octets, donc plus
//                          cher pour des voies de 32 bits ), `vrgather` sur RISC-V.
// Le fichier ne les cache pas : ce sont les deux endroits ou un portage ARM demandera une mesure
// avant d'etre cru.
//
// = LA FORME DU FICHIER, ET POURQUOI ELLE EST BIZARRE
//
// Pas de `#pragma once`, mais une GARDE A BASCULE. C'est ce qu'exige `HWY_DYNAMIC_DISPATCH` :
// pour mettre plusieurs jeux d'instructions dans un seul binaire, highway REINCLUT ce fichier
// une fois par cible, en basculant `HWY_TARGET_TOGGLE` entre deux inclusions. Une garde
// ordinaire ne laisserait passer que la premiere.
//
// `HWY_NAMESPACE` change lui aussi a chaque passage ( `N_AVX3`, `N_AVX2`, `N_SSE4`... ), donc
// chaque cible obtient son propre `noyau2d::N_xxx::noyau`, compile avec ses propres options via
// `HWY_BEFORE_NAMESPACE()`. Le contrat -- `Plan`, `Atelier`, `coupe_large` -- reste UNIQUE : il
// est dans `Contrat2D.h`, avec un `#pragma once`, parce qu'il ne contient pas une instruction
// machine.
// =====================================================================================

#include "supercell/Contrat2D.h"

#if defined( NOYAU2D_HWY_INL_H_ ) == defined( HWY_TARGET_TOGGLE )
#ifdef NOYAU2D_HWY_INL_H_
#undef NOYAU2D_HWY_INL_H_
#else
#define NOYAU2D_HWY_INL_H_
#endif

#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace noyau2d {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

using D  = hn::FixedTag<float, 8>;                       ///< HUIT voies, impose. Pas seize.
using DI = hn::FixedTag<int32_t, 8>;
using V  = hn::Vec<D>;
using VI = hn::Vec<DI>;

static constexpr D  d;
static constexpr DI di;

/// la voie `i`, connue a la compilation ou non -- `Eq( iota, i )` est la facon portable de
/// designer une voie, la ou les intrinseques prennent un masque immediat.
inline VI iota()             { return hn::Iota( di, 0 ); }
inline auto voie( int i )    { return hn::Eq( iota(), hn::Set( di, i ) ); }

/// CE QUE LE FOURNISSEUR VOIT, version portable. Meme surface que `Etat<NB>` -- `nb`, `x`,
/// `y`, `id` -- donc les fournisseurs, qui sont des patrons sur l'etat, marchent sans un
/// caractere de changement. C'est ce qui permet de comparer les deux noyaux sur exactement la
/// meme politique.
template<int NB>
struct EtatH {
    static constexpr int nb = NB;
    V vx, vy; VI cid;

    float x ( int i ) const { alignas( 32 ) float t[ 8 ]; hn::Store( vx, d, t ); return t[ i ]; }
    float y ( int i ) const { alignas( 32 ) float t[ 8 ]; hn::Store( vy, d, t ); return t[ i ]; }
    int   id( int i ) const { alignas( 32 ) int32_t t[ 8 ]; hn::Store( cid, di, t ); return t[ i ]; }
};

template<int NB, class Fourn, class Atl> void noyau  ( V vx, V vy, VI cid, Fourn *f, Atl *a, Local<Fourn> *loc );
template<        class Fourn, class Atl> void deborde( V vx, V vy, VI cid, Fourn *f, Atl *a, Local<Fourn> *loc );

template<int NB, class Fourn, class Atl>
void noyau( V vx, V vy, VI cid, Fourn *f, Atl *a, Local<Fourn> *loc ) {
    constexpr unsigned valid = ( 1u << NB ) - 1;
    const VI IOTA = iota();
    const V  ZERO = hn::Zero( d );

    for ( ;; ) {
        Plan p;
        if ( ! f->suivant( EtatH<NB>{ vx, vy, cid }, *loc, p ) )
            break;

        // ---- LE TEST, QUI EST DEJA LA COUPE.
        const V s = hn::MulAdd( hn::Set( d, p.dx ), vx,
                    hn::MulSub( hn::Set( d, p.dy ), vy, hn::Set( d, p.off ) ) );
        const unsigned m = (unsigned) hn::BitsFromMask( d, hn::Gt( s, ZERO ) ) & valid;

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

        // ---- LES DEUX INTERSECTIONS, EN UNE SEULE DIVISION. Les deux ancres sont rassemblees
        // dans des voies DISTINCTES -- A en voie 0, B en voie 1 -- ce qui rend impossible la
        // collision `j0 == j3` du cas `nb_in == 1`.
        const VI anc = hn::IfThenElse( voie( 1 ), hn::Set( di, j3 ), hn::Set( di, j0 ) );
        const VI oth = hn::IfThenElse( voie( 1 ), hn::Set( di, j2 ), hn::Set( di, i1 ) );
        const auto ianc = hn::IndicesFromVec( d, anc );
        const auto ioth = hn::IndicesFromVec( d, oth );
        const V vax = hn::TableLookupLanes( vx, ianc ), vox = hn::TableLookupLanes( vx, ioth );
        const V vay = hn::TableLookupLanes( vy, ianc ), voy = hn::TableLookupLanes( vy, ioth );
        const V sa  = hn::TableLookupLanes( s,  ianc ), so  = hn::TableLookupLanes( s,  ioth );
        const V t   = hn::Div( sa, hn::Sub( sa, so ) );
        const V pcx = hn::MulAdd( hn::Sub( vox, vax ), t, vax );
        const V pcy = hn::MulAdd( hn::Sub( voy, vay ), t, vay );

        // ---- LE REMONTAGE. [ j3, j3+1, ... , j0, A, B ].
        VI og = hn::Add( hn::Set( di, j3 ), IOTA );
        og = hn::IfThenElse( hn::Ge( og, hn::Set( di, NB ) ), hn::Sub( og, hn::Set( di, NB ) ), og );
        // BORNER LES INDICES, explicitement. Une seule soustraction ne ramene pas toutes les
        // voies sous `NB` ( `j3 + 7 - NB` peut valoir 11 ), et les voies au-dela de `nb_in` sont
        // du rebut. `vpermps` ignore les bits hauts, donc les intrinseques s'en tirent sans rien
        // dire ; `TableLookupLanes` ne le promet pas partout. Un `et` logique, et c'est defini.
        og = hn::And( og, hn::Set( di, 7 ) );
        const auto iog = hn::IndicesFromVec( d, og );

        const auto mA = voie( nb_in ), mB = voie( nb_in + 1 );

        V nvx = hn::TableLookupLanes( vx, iog );
        nvx = hn::IfThenElse( hn::RebindMask( d, mA ), hn::BroadcastLane<0>( pcx ), nvx );
        nvx = hn::IfThenElse( hn::RebindMask( d, mB ), hn::BroadcastLane<1>( pcx ), nvx );
        V nvy = hn::TableLookupLanes( vy, iog );
        nvy = hn::IfThenElse( hn::RebindMask( d, mA ), hn::BroadcastLane<0>( pcy ), nvy );
        nvy = hn::IfThenElse( hn::RebindMask( d, mB ), hn::BroadcastLane<1>( pcy ), nvy );
        VI nid = hn::TableLookupLanes( cid, hn::IndicesFromVec( di, og ) );
        nid = hn::IfThenElse( mA, hn::Set( di, p.id ), nid );
        nid = hn::IfThenElse( mB, hn::TableLookupLanes( cid, hn::IndicesFromVec( di, hn::Set( di, j2 ) ) ), nid );

        if ( nn == NB ) { vx = nvx; vy = nvy; cid = nid; continue; }
        if constexpr ( NB == 8 ) if ( nn > 8 ) {
            a->nb = 8;
            a->attente = p;
            hn::Store( vx, d, a->vx );
            hn::Store( vy, d, a->vy );
            hn::Store( cid, di, a->cid );
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
    hn::Store( vx, d, a->vx );
    hn::Store( vy, d, a->vy );
    hn::Store( cid, di, a->cid );
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
                const V  rvx = hn::Load( d, a->vx );
                const V  rvy = hn::Load( d, a->vy );
                const VI rid = hn::Load( di, a->cid );
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
    alignas( 32 ) static const int32_t ci[ 8 ] = { -1, -2, -3, -4, 0, 0, 0, 0 };
    noyau<4>( hn::Load( d, cx ), hn::Load( d, cy ), hn::Load( di, ci ), f, a, &loc );
}

} // namespace HWY_NAMESPACE
} // namespace noyau2d
HWY_AFTER_NAMESPACE();

#endif // garde a bascule

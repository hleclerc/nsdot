#pragma once

// =====================================================================================
// LE NOYAU 2D A REGISTRES : une machine a etats sur trois vecteurs de huit voies.
//
//      vx[ 0 .. 7 ]    les sommets, en ordre cyclique
//      vy[ 0 .. 7 ]
//      cid[ 0 .. 7 ]   la coupe qui porte l'arete [ v_i, v_i+1 ]
//
// `NB`, le nombre de sommets, est une CONSTANTE DE COMPILATION : le masque des voies valides, les
// rotations et l'ordre de sortie sont des immediats, et il n'y a aucune boucle. Huit voies parce
// que 98.4 % des etats intermediaires d'un diagramme ont huit sommets ou moins.
//
// LA COUPE SANS BOUCLE. L'exterieur d'un convexe coupe par un demi-plan est une plage CYCLIQUE
// contigue ; le masque de signe est un bloc de uns a rotation pres, et ses deux extremites se
// lisent en deux `ctz`. Deux permutations, une division, deux `select` : la cellule est recoupee.
//
// LA MACHINE A ETATS. `nb` change a presque chaque coupe effective, donc `etape<NB>` coupe tant que
// la taille ne bouge pas et REND la nouvelle taille sinon ; `moteur` redispatche par un `switch`.
// `etape` est `always_inline`, et ce n'est pas negociable : non inlinee, elle prendrait la cellule
// par reference a travers un vrai appel, donc par la MEMOIRE. ( La forme `musttail` du banc a ete
// mesuree equivalente ; celle-ci est portable. )
//
// LE DEBORDEMENT EST UNE EXCURSION. Depuis `NB == 8`, une coupe qui ne retranche qu'un sommet
// demanderait un neuvieme. La cellule se pose dans l'atelier, on continue en scalaire et en place,
// et DES QU'ELLE REDESCEND a huit sommets on la recharge dans les registres.
//
// ASIMD : `fma`, `permute`, `select`, `to_bits`, `mask_from_bits`, `bcast_lane`. La largeur est un
// parametre du type, donc `SimdVec<double,8>` est le meme code sur deux registres.
// =====================================================================================

#include "cell/Boite.h"
#include <cmath>
#include "cell/Contrat2D.h"
#include <asimd/asimd.h>
#include <cstdio>
#include <cstdlib>

namespace sf::d2 {

/// CE QUE LE FOURNISSEUR VOIT quand la cellule tient dans les registres. `nb` est une constante de
/// compilation. `x / y / id` sortent UNE voie par la pile : c'est cher, et voulu -- un fournisseur
/// qui ne regarde pas la cellule ne paie rien, un fournisseur qui la regarde paie ce que ca coute.
template<class TK, int NB>
struct EtatReg {
    using V  = asimd::SimdVec<TK,8>;
    using VI = asimd::SimdVec<SI32,8>;
    static constexpr int nb = NB;

    V  vx, vy;
    VI cid;

    TK   x ( int i ) const { alignas( 64 ) TK t[ 8 ]; vx.store_aligned( t ); return t[ i ]; }
    TK   y ( int i ) const { alignas( 64 ) TK t[ 8 ]; vy.store_aligned( t ); return t[ i ]; }
    SI32 id( int i ) const { alignas( 64 ) SI32 t[ 8 ]; cid.store_aligned( t ); return t[ i ]; }
};

namespace etats {

enum : int {
    FINI    = -2,   ///< le fournisseur n'a plus rien : la cellule est finie, a `NB` sommets
    DEBORDE = -1,   ///< au-dela de huit : l'atelier porte les huit sommets et `attente`
    VIDE    =  0    ///< un demi-plan a tout emporte
};

/// « la voie i », par un motif de BITS : un `kmovb` sur une machine a registres de masque.
inline auto voie( int i ) { return asimd::mask_from_bits<8>( 1u << i ); }

/// UNE ETAPE : coupe a `NB` fixe tant que la taille ne change pas.
template<int NB, class TK, class Fourn, class Atl>
[[gnu::always_inline]] inline int etape( asimd::SimdVec<TK,8> &vx, asimd::SimdVec<TK,8> &vy,
                                         asimd::SimdVec<SI32,8> &cid, Fourn *f, Atl *a,
                                         Local<Fourn> &loc ) {
    using V  = asimd::SimdVec<TK,8>;
    using VI = asimd::SimdVec<SI32,8>;
    constexpr unsigned valid = ( 1u << NB ) - 1;
    const VI IOTA = VI::iota( 0 );

    for ( ;; ) {
        Plan2<TK> p;
        if ( ! f->suivant( EtatReg<TK,NB>{ vx, vy, cid }, loc, p ) )
            return FINI;

        // ---- LE TEST, QUI EST DEJA LA COUPE : `s > 0` dehors.
        const V s = asimd::fma( V( p.dx ), vx, asimd::fma( V( p.dy ), vy, V( -p.off ) ) );
        const unsigned m = unsigned( asimd::to_bits( s > V( TK( 0 ) ) ) ) & valid;

        if ( ! m )
            continue;
        if ( m == valid )
            return VIDE;

        // ---- LES DEUX BOUTS DE LA PLAGE EXTERIEURE.
        const unsigned prev = ( ( m << 1 ) | ( m >> ( NB - 1 ) ) ) & valid;
        const unsigned next = ( ( m >> 1 ) | ( m << ( NB - 1 ) ) ) & valid;
        const int i1 = __builtin_ctz( m & ~prev );          // premier DEHORS
        const int j2 = __builtin_ctz( m & ~next );          // dernier DEHORS
        const int j0 = i1 ? i1 - 1 : NB - 1;                // dernier DEDANS avant
        const int j3 = j2 + 1 < NB ? j2 + 1 : 0;            // premier DEDANS apres
        const int nb_in = NB - __builtin_popcount( m );
        const int nn = nb_in + 2;

        // ---- LES DEUX INTERSECTIONS, EN UNE SEULE DIVISION. A en voie 0, B en voie 1 : les deux
        // ancres sont dans des voies distinctes, donc pas de collision quand `j0 == j3`.
        const auto v1 = voie( 1 );
        const VI anc = asimd::select( v1, VI( j3 ), VI( j0 ) );
        const VI oth = asimd::select( v1, VI( j2 ), VI( i1 ) );
        const V vax = asimd::permute( vx, anc ), vox = asimd::permute( vx, oth );
        const V vay = asimd::permute( vy, anc ), voy = asimd::permute( vy, oth );
        const V sa  = asimd::permute( s,  anc ), so  = asimd::permute( s,  oth );
        const V t   = sa / ( sa - so );
        const V pcx = asimd::fma( vox - vax, t, vax );
        const V pcy = asimd::fma( voy - vay, t, vay );

        // ---- LE REMONTAGE : `[ v_j3, ..., v_j0, A, B ]`, `A` sur la coupe neuve, `B` sur ce qui
        // reste de la coupe `j2`.
        VI og = VI( j3 ) + IOTA;
        og = asimd::select( asimd::ge( og, VI( NB ) ), og - VI( NB ), og );
        og = og & VI( 7 );

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

        // ---- L'ETAT NOUVEAU.
        if constexpr ( NB == 8 ) if ( nn > 8 ) {
            a->nb = 8;                                   // l'excursion partira de la
            a->attente = p;                              // ... et rejouera cette coupe-ci
            vx.store_aligned( a->lx );
            vy.store_aligned( a->ly );
            cid.store_aligned( a->cid );
            return DEBORDE;
        }
        vx = nvx; vy = nvy; cid = nid;
        if ( nn != NB )
            return nn;
    }
}

/// L'EXCURSION, en place dans l'atelier, jusqu'a ce que la cellule redescende a huit sommets.
/// Rend la nouvelle taille ( et recharge les registres ), `VIDE` ( `a->nb` renseigne : 0, ou -1
/// si l'atelier ne suffit pas ), ou `FINI` ( la cellule est dans l'atelier, a plus de huit cotes ).
template<class TK, class Fourn, class Atl>
inline int excursion( asimd::SimdVec<TK,8> &vx, asimd::SimdVec<TK,8> &vy,
                      asimd::SimdVec<SI32,8> &cid, Fourn *f, Atl *a, Local<Fourn> &loc ) {
    using V  = asimd::SimdVec<TK,8>;
    using VI = asimd::SimdVec<SI32,8>;
    constexpr int MaxNb = Atl::max_nb;
    TK s[ MaxNb ];
    int nb = a->nb;
    Plan2<TK> p = a->attente;
    for ( ;; ) {
        nb = coupe_large<MaxNb>( a->lx, a->ly, a->cid, nb, p, s );
        if ( nb <= 0 ) { a->nb = nb; return VIDE; }
        if ( nb <= 8 ) {
            vx  = V::load_aligned( a->lx );
            vy  = V::load_aligned( a->ly );
            cid = VI::load_aligned( a->cid );
            return nb;
        }
        if ( ! f->suivant( EtatLarge<TK>{ nb, a->lx, a->ly, a->cid }, loc, p ) ) {
            a->nb = nb;
            return FINI;
        }
    }
}

} // namespace etats

/// LE MOTEUR, DEPUIS UNE BOITE DONNEE ( en absolu ). `fid` porte l'identifiant de chacun des
/// quatre cotes : celui du domaine si le cote EST une face du domaine, un artificiel sinon --
/// voir `cell/Boite.h`. A la sortie l'atelier porte la cellule : `nb` sommets, ou 0, ou -1.
template<class TK, class Fourn, class Atl>
void moteur_depuis( Fourn *f, Atl *a, const TF lo[ 2 ], const TF hi[ 2 ], const SI32 fid[ 4 ] ) {
    using namespace etats;
    using V  = asimd::SimdVec<TK,8>;
    using VI = asimd::SimdVec<SI32,8>;

    // LA BOITE DANS LE REPERE DU GERME. Les quatre coins sont translates EN DOUBLE puis arrondis :
    // `lo - ox` porte alors l'erreur relative de sa propre valeur, pas celle du domaine.
    TF ox = 0, oy = 0;
    f->origine( ox, oy );
    const double ux[ 4 ] = { lo[ 0 ], hi[ 0 ], hi[ 0 ], lo[ 0 ] };
    const double uy[ 4 ] = { lo[ 1 ], lo[ 1 ], hi[ 1 ], hi[ 1 ] };
    alignas( 64 ) TK cx[ 8 ] = {}, cy[ 8 ] = {};
    alignas( 64 ) SI32 ci[ 8 ] = {};
    for ( int q = 0; q < 4; ++q ) {
        cx[ q ] = TK( ux[ q ] - ox );
        cy[ q ] = TK( uy[ q ] - oy );
        ci[ q ] = fid[ q ];
    }
    V  vx  = V::load_aligned( cx ), vy = V::load_aligned( cy );
    VI cid = VI::load_aligned( ci );
    a->ox = ox; a->oy = oy;
    int nb = 4;
    Local<Fourn> loc{};                                  // vit et meurt avec la cellule
    bool en_excursion = false;

    for ( ;; ) {
        int r;
        if ( en_excursion ) {
            r = excursion<TK>( vx, vy, cid, f, a, loc );
            en_excursion = false;
            if ( r == FINI || r == VIDE )                // l'atelier est deja renseigne
                return;
        } else {
            switch ( nb ) {
                case 3:  r = etape<3,TK>( vx, vy, cid, f, a, loc ); break;
                case 4:  r = etape<4,TK>( vx, vy, cid, f, a, loc ); break;
                case 5:  r = etape<5,TK>( vx, vy, cid, f, a, loc ); break;
                case 6:  r = etape<6,TK>( vx, vy, cid, f, a, loc ); break;
                case 7:  r = etape<7,TK>( vx, vy, cid, f, a, loc ); break;
                default: r = etape<8,TK>( vx, vy, cid, f, a, loc ); break;
            }
            if ( r == FINI ) {                           // finie en registres
                a->nb = nb;
                vx.store_aligned( a->lx );
                vy.store_aligned( a->ly );
                cid.store_aligned( a->cid );
                return;
            }
            if ( r == VIDE ) { a->nb = 0; return; }
            if ( r == DEBORDE ) { en_excursion = true; continue; }
        }
        nb = r;
    }
}

/// LE MOTEUR. Si le fournisseur propose une boite de depart a l'echelle locale, on l'essaie ; si
/// la cellule la TOUCHE, on recommence depuis le domaine. Voir `cell/Boite.h` : le test est
/// conservatif, donc le resultat est celui du domaine, toujours.
template<class TK, class Fourn, class Atl>
void moteur( Fourn *f, Atl *a ) {
    static const TF dlo[ 2 ] = { 0, 0 }, dhi[ 2 ] = { 1, 1 };
    static const SI32 dfid[ 4 ] = { -1, -2, -3, -4 };

    if constexpr ( requires ( Fourn *g, TF *b ) { g->boite_depart( b, b, 1.0 ); } ) {
        double facteur = 1;                              // `CROISSANCE^essai`, sans appel a `pow`
        for ( int essai = 0; essai < MAX_REPRISES; ++essai, facteur *= CROISSANCE ) {
            TF lo[ 2 ], hi[ 2 ];
            if ( ! f->boite_depart( lo, hi, facteur ) )
                break;                                   // la boite couvre deja le domaine
            // cote `q` = [ coin q, coin q+1 ] : bas ( y = lo1 ), droite ( x = hi0 ), haut, gauche
            const SI32 fid[ 4 ] = { lo[ 1 ] > 0 ? SI32( FACE_ARTIF - 1 ) : SI32( -1 ),   // bas
                                    hi[ 0 ] < 1 ? SI32( FACE_ARTIF - 2 ) : SI32( -2 ),   // droite
                                    hi[ 1 ] < 1 ? SI32( FACE_ARTIF - 3 ) : SI32( -3 ),   // haut
                                    lo[ 0 ] > 0 ? SI32( FACE_ARTIF - 4 ) : SI32( -4 ) }; // gauche
            moteur_depuis<TK>( f, a, lo, hi, fid );
            // VIDE OU DEBORDE COMPTE COMME UNE SORTIE : une cellule de Laguerre ne contient pas
            // forcement son germe, donc « rien dans la boite » ne veut pas dire « rien ». Voir
            // `cell/Boite.h`, cas ( a ) -- le piege que `check` a poids nuls ne montre pas.
            bool touche = a->nb <= 0;
            for ( int i = 0; i < a->nb; ++i )
                touche |= face_artificielle( a->cid[ i ] );
            if ( ! touche )
                return;
            nb_reprises.fetch_add( 1, std::memory_order_relaxed );
        }
    }
    moteur_depuis<TK>( f, a, dlo, dhi, dfid );
}

} // namespace sf::d2

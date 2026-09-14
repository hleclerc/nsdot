#pragma once

// =====================================================================================
// LE NOYAU A REGISTRES -- 2D, cellule bornee, CPU.
//
// La cellule y est trois vecteurs de huit voies :
//
//      vx[ 0 .. 7 ]    les sommets, en ordre cyclique
//      vy[ 0 .. 7 ]
//      cid[ 0 .. 7 ]   l'identite de la coupe qui porte l'arete `[ v_i, v_i+1 ]`
//
// et `NB`, le nombre de sommets, est une CONSTANTE DE COMPILATION : c'est elle qui rend le masque
// des voies valides immediat, les rotations immediates, et la boucle absente. Huit voies parce
// que c'est la fenetre que les statistiques designent : 98.4 % des etats intermediaires d'un
// diagramme de Voronoi ont huit sommets ou moins.
//
// CE QUI REND LA COUPE SANS BOUCLE. L'exterieur d'un convexe coupe par un demi-plan est une plage
// CYCLIQUE contigue. Le masque de signe est donc, a rotation pres, un bloc de uns : ses deux
// extremites se lisent en deux `ctz` sur des rotations du masque, et avec `NB` connu ces rotations
// sont des immediats. Il n'y a plus rien a parcourir.
//
// LA MACHINE A ETATS. `nb` change a presque chaque coupe effective ( `nb - nb_out + 2` ), donc
// chaque taille est une fonction `etape<NB>` qui coupe tant que la taille ne change pas et REND la
// nouvelle taille sinon ; `run_2d_registers` redispatche par un `switch`. `etape` est
// `always_inline`, et ce n'est pas negociable : non inlinee, elle prendrait la cellule par
// reference a travers un vrai appel, donc par la MEMOIRE, et les trois vecteurs cesseraient
// d'etre des registres. ( La forme `musttail` du banc a ete mesuree equivalente, et elle n'est
// pas portable. )
//
// LE DEBORDEMENT EST UNE EXCURSION. Depuis `NB == 8`, une coupe qui ne retranche qu'un sommet en
// demanderait un neuvieme. La cellule se pose alors dans `Local2` -- qui EST l'atelier -- et on
// continue a la couper en scalaire, en place, jusqu'a ce qu'elle redescende a huit sommets : elle
// est alors rechargee dans les registres et on repart. Le noyau rend donc toujours une cellule
// complete, et `OVERFLOW` ne subsiste que si la capacite du scratch est trop petite.
//
// LE DICTIONNAIRE ASIMD suffit : `fma`, `permute`, `select`, `to_bits`, `mask_from_bits`,
// `bcast_lane`. La largeur est un parametre du type, donc `SimdVec<double,8>` est le meme code
// sur deux registres -- ce qui rend le noyau `double` sans une ligne de plus.
// =====================================================================================

#include <asimd/asimd.h>
#include "Local2.h"
#include "Etat.h"

namespace sdot {

/// CE QUE LE FOURNISSEUR VOIT quand la cellule tient dans les registres. `nb` est une constante
/// de compilation ; `x / y / id` sortent UNE voie en passant par la pile -- c'est cher, et c'est
/// voulu : un fournisseur qui ne regarde pas la cellule ne paie rien ( le `store` est mort, le
/// compilateur l'enleve ), un fournisseur qui la regarde paie ce que ca coute vraiment.
template<class TK,int NB>
struct EtatReg {
    using V  = asimd::SimdVec<TK,8>;
    using VI = asimd::SimdVec<asimd::SI32,8>;
    static constexpr int nb = NB;

    V  vx, vy;
    VI cid;

    TK  x ( int i ) const { alignas( 64 ) TK t[ 8 ]; vx.store_aligned( t ); return t[ i ]; }
    TK  y ( int i ) const { alignas( 64 ) TK t[ 8 ]; vy.store_aligned( t ); return t[ i ]; }
    int id( int i ) const { alignas( 64 ) asimd::SI32 t[ 8 ]; cid.store_aligned( t ); return t[ i ]; }
};

namespace moteur2 {

enum : int {
    FINI    = -2,   ///< le fournisseur n'a plus rien : la cellule est finie, a `NB` sommets
    DEBORDE = -1,   ///< au-dela de huit : `Local2` porte les huit sommets, et `attente` la coupe
    VIDE    =  0,   ///< un demi-plan a tout emporte
};

/// « la voie i », par un motif de BITS : un `kmovb` sur une machine a registres de masque, la ou
/// `eq( iota, i )` demande un `vpbroadcastd` et un `vpcmpeqd`. Trois voies sont designees a
/// chaque coupe.
inline auto voie( int i ) { return asimd::mask_from_bits<8>( 1u << i ); }

/// UNE ETAPE : coupe a `NB` fixe tant que la taille ne change pas, rend la nouvelle taille sinon.
template<int NB,class TK,class Fourn>
[[gnu::always_inline]] inline int etape( asimd::SimdVec<TK,8> &vx, asimd::SimdVec<TK,8> &vy,
                                         asimd::SimdVec<asimd::SI32,8> &cid,
                                         Fourn &f, Local2<TK> &a, LocalOf<Fourn> &loc,
                                         Plane<TK,2> &attente ) {
    using V  = asimd::SimdVec<TK,8>;
    using VI = asimd::SimdVec<asimd::SI32,8>;
    constexpr unsigned valid = ( 1u << NB ) - 1;
    const VI IOTA = VI::iota( 0 );

    for ( ;; ) {
        Plane<TK,2> p;
        if ( ! f.suivant( EtatReg<TK,NB>{ vx, vy, cid }, loc, p ) )
            return FINI;

        // ---- LE TEST, QUI EST DEJA LA COUPE : `s > 0` dehors.
        const V s = asimd::fma( V( p.dir[ 0 ] ), vx, asimd::fma( V( p.dir[ 1 ] ), vy, V( - p.off ) ) );
        const unsigned m = unsigned( asimd::to_bits( s > V( TK( 0 ) ) ) ) & valid;

        if ( ! m )
            continue;
        if ( m == valid )
            return VIDE;

        // ---- LES DEUX BOUTS DE LA PLAGE EXTERIEURE.
        const unsigned prev = ( ( m << 1 ) | ( m >> ( NB - 1 ) ) ) & valid;
        const unsigned next = ( ( m >> 1 ) | ( m << ( NB - 1 ) ) ) & valid;
        const int i1 = __builtin_ctz( m & ~prev );          // premier DEHORS de la plage
        const int j2 = __builtin_ctz( m & ~next );          // dernier DEHORS
        const int j0 = i1 ? i1 - 1 : NB - 1;                // dernier DEDANS avant
        const int j3 = j2 + 1 < NB ? j2 + 1 : 0;            // premier DEDANS apres
        const int nb_in = NB - __builtin_popcount( m );
        const int nn = nb_in + 2;

        // ---- LES DEUX INTERSECTIONS, EN UNE SEULE DIVISION. A en voie 0, B en voie 1 : les deux
        // ancres ( les sommets DEDANS ) sont dans des voies distinctes, donc pas de collision quand
        // `j0 == j3`.
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
            a.nb = 8;                                    // l'excursion partira de la
            attente = p;                                 // ... et rejouera cette coupe-ci
            vx.store_unaligned( a.vx );
            vy.store_unaligned( a.vy );
            cid.store_unaligned( reinterpret_cast<asimd::SI32 *>( a.cid ) );
            return DEBORDE;
        }
        vx = nvx; vy = nvy; cid = nid;
        if ( nn != NB )
            return nn;
    }
}

/// L'EXCURSION, en place dans `Local2`, jusqu'a ce que la cellule redescende a huit sommets.
/// Rend la nouvelle taille ( et recharge les registres ), `VIDE`, `FINI` -- ou `OVERFLOW` de
/// `CutStatus` ( > 8 ) si la capacite ne suffit pas.
template<class TK,class Fourn>
inline int excursion( asimd::SimdVec<TK,8> &vx, asimd::SimdVec<TK,8> &vy,
                      asimd::SimdVec<asimd::SI32,8> &cid,
                      Fourn &f, Local2<TK> &a, LocalOf<Fourn> &loc, Plane<TK,2> p ) {
    using V  = asimd::SimdVec<TK,8>;
    using VI = asimd::SimdVec<asimd::SI32,8>;
    for ( ;; ) {
        const int r = a.cut( p );
        if ( r == CutStatus::EMPTY )
            return VIDE;
        if ( r == CutStatus::OVERFLOW )
            return CutStatus::OVERFLOW + 8;              // hors de `3..8`, et hors des codes < 0
        if ( a.nb <= 8 ) {
            vx  = V::load_unaligned( a.vx );
            vy  = V::load_unaligned( a.vy );
            cid = VI::load_unaligned( reinterpret_cast<const asimd::SI32 *>( a.cid ) );
            return a.nb;
        }
        if ( ! f.suivant( a.etat(), loc, p ) )
            return FINI;                                 // finie, avec plus de huit cotes
    }
}

} // namespace moteur2

/// LE NOYAU, en repartant de la cellule `c` telle qu'elle est. Rend `0` ( cellule complete, vide
/// comprise ) ou `CutStatus::OVERFLOW`.
template<class TK,class Fourn>
int run_2d_registers( Local2<TK> &c, Fourn &f, LocalOf<Fourn> &loc ) {
    using namespace moteur2;
    using V  = asimd::SimdVec<TK,8>;
    using VI = asimd::SimdVec<asimd::SI32,8>;

    int nb = c.nb;
    if ( nb <= 0 )
        return 0;
    if ( c.cap < 8 )
        return CutStatus::OVERFLOW;                      // l'atelier doit recevoir les huit voies

    V  vx = V( TK( 0 ) ), vy = V( TK( 0 ) );
    VI cid = VI( 0 );
    Plane<TK,2> attente;
    bool en_excursion = nb > 8;
    if ( ! en_excursion ) {
        // les huit voies sont chargees en entier : au-dela de `nb` ce sont les temporaires du
        // scratch, jamais lus par une voie valide
        vx  = V::load_unaligned( c.vx );
        vy  = V::load_unaligned( c.vy );
        cid = VI::load_unaligned( reinterpret_cast<const asimd::SI32 *>( c.cid ) );
    } else {
        // trop grosse pour les registres des le depart : on demande une premiere coupe en memoire
        if ( ! f.suivant( c.etat(), loc, attente ) )
            return 0;
    }

    for ( ;; ) {
        int r;
        if ( en_excursion ) {
            r = excursion( vx, vy, cid, f, c, loc, attente );
            en_excursion = false;
        } else {
            switch ( nb ) {
                case 3:  r = etape<3>( vx, vy, cid, f, c, loc, attente ); break;
                case 4:  r = etape<4>( vx, vy, cid, f, c, loc, attente ); break;
                case 5:  r = etape<5>( vx, vy, cid, f, c, loc, attente ); break;
                case 6:  r = etape<6>( vx, vy, cid, f, c, loc, attente ); break;
                case 7:  r = etape<7>( vx, vy, cid, f, c, loc, attente ); break;
                default: r = etape<8>( vx, vy, cid, f, c, loc, attente ); break;
            }
        }

        if ( r == FINI ) {
            if ( c.nb <= 8 || nb <= 8 ) {                // la cellule est dans les registres
                c.nb = nb;
                vx.store_unaligned( c.vx );
                vy.store_unaligned( c.vy );
                cid.store_unaligned( reinterpret_cast<asimd::SI32 *>( c.cid ) );
            }                                            // sinon elle est deja dans `c`
            return 0;
        }
        if ( r == VIDE ) {
            c.nb = 0;
            return 0;
        }
        if ( r == DEBORDE ) {
            en_excursion = true;
            nb = 9;
            continue;
        }
        if ( r > 8 )
            return CutStatus::OVERFLOW;
        nb = r;
    }
}

} // namespace sdot

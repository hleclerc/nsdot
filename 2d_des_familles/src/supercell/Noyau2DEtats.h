#pragma once

// =====================================================================================
// LE MEME NOYAU, SANS `musttail` -- une machine a etats.
//
// = CE QU'ON CHERCHE A ENLEVER
//
// `[[gnu::musttail]]` est la seule chose du noyau qui ne soit pas portable entre COMPILATEURS :
// gcc et clang l'ont, MSVC n'a rien d'equivalent -- ni attribut, ni garantie. Or toute la
// conception repose dessus : `NB` est une constante de compilation, donc changer de `NB` veut
// dire changer de fonction, et sans saut garanti chaque coupe qui change la taille ouvrirait une
// frame. Sur une cellule qui passe par trois ou quatre tailles, ce serait ruineux.
//
// = CE QU'IL FAUT GARDER, ET QUI EST TOUT LE PROBLEME
//
//   `NB` CONSTANTE DE COMPILATION. C'est elle qui rend `valid` immediat, les rotations du masque
//   immediates, et la boucle absente. Une machine a etats a `nb` variable perdrait exactement ce
//   qu'on est venu chercher.
//
//   LA CELLULE DANS SES REGISTRES. Trois `ymm` qui traversent la transition. C'est ce que le
//   `musttail` offrait gratuitement, l'ABI passant les `__m256` dans `ymm0..7`.
//
// = LA FORME RETENUE
//
// Une boucle unique dans `moteur`, un `switch` sur `nb`, et un `etape<NB>` par taille. La cellule
// vit dans des LOCALES du moteur, passees par reference ; `etape<NB>` coupe a `NB` fixe tant que
// la taille ne change pas, et rend la nouvelle taille des qu'elle change.
//
// `etape<NB>` est `always_inline`, et ce n'est pas negociable : non inlinee, elle prendrait la
// cellule par reference a travers un vrai appel, donc par la MEMOIRE, et on perdrait plus que ce
// que `musttail` faisait gagner. Inlinee, les references disparaissent et les trois `ymm` restent
// des registres d'un bout a l'autre de la boucle.
//
// Le prix : le corps des six tailles est inline dans une seule fonction, la ou la version
// `musttail` en faisait six fonctions distinctes. Meme volume de code au total, mais une seule
// allocation de registres a faire, sur un corps six fois plus gros.
//
// = CE QUE CA COUTE, ET LA SEULE SURPRISE
//
// Rien, ou un peu moins que rien -- MAIS LE SIGNE DEPEND D'UN DRAPEAU. Mediane de neuf lancers,
// nuage Morton, parcours en s'ecartant, `etats / musttail` :
//
//                            n = 200    n = 1000   n = 5000
//     regles d'aliasing par defaut       x0.957     x0.947     x0.948
//     `-fno-strict-aliasing`             x1.019     x1.040     x1.059
//
// Justesse identique dans les deux cas : memes aires au dernier chiffre, zero cellule differente,
// memes echappements ( 507 sur 2000 avec un atelier de huit ), meme excursion.
//
// POURQUOI LA MACHINE A ETATS GAGNE quand on la laisse gagner : `musttail` execute un EPILOGUE a
// chaque transition. Le desassemblage le montre -- `pop %r12 ; pop %r13 ; pop %r14 ; pop %rbp`
// juste avant chaque saut. Une cellule traverse trois ou quatre tailles, donc c'est trois ou
// quatre restaurations de registres sauvegardes. La boucle unique n'en a aucune.
//
// POURQUOI ELLE PERD SOUS `-fno-strict-aliasing`, en revanche, je ne le sais pas. L'explication
// evidente -- l'etat porte par REFERENCE devient suspect d'aliaser l'atelier -- est FAUSSE :
// ajouter `__restrict` sur les trois references ne change rien du tout ( x1.037 contre x1.040 ).
// Le drapeau vient de `AaBspPacked`, qui lit son arene par `reinterpret_cast` ; RIEN dans ce
// fichier n'en a besoin, donc compiler cette unite de traduction sans lui rendrait les 10 %
// d'ecart entre les deux lignes du tableau. C'est la piste, elle n'est pas verifiee.
// =====================================================================================

#include "supercell/Noyau2D.h"

namespace noyau2d {
namespace etats {

/// ce que `etape<NB>` rend, quand ce n'est pas une nouvelle taille.
enum : int {
    FINI    = -2,   ///< le fournisseur n'a plus rien : la cellule est finie, a `NB` sommets
    DEBORDE = -1,   ///< au-dela de huit : l'atelier porte les huit sommets et `attente`
    VIDE    =  0    ///< un demi-plan a tout emporte
};

/// UNE ETAPE : coupe a `NB` fixe tant que la taille ne change pas.
///
/// Le corps est celui de `noyau<NB>`, a une chose pres : la ou l'original saute vers `noyau<nn>`,
/// celui-ci REND `nn` et laisse le moteur redispatcher. Rien d'autre n'a bouge.
template<int NB, class Fourn, class Atl>
[[gnu::always_inline]] inline int etape( __m256 &vx, __m256 &vy, __m256i &cid, Fourn *f, Atl *a,
                                         Local<Fourn> &loc, bool &change ) {
    constexpr unsigned valid = ( 1u << NB ) - 1;
    const __m256i IOTA = Cst<NB>::iota();
    const __m256  ZERO = _mm256_setzero_ps();

    for ( ;; ) {
        Plan p;
        // LA FORME DE L'ETAT EST CHOISIE ICI, a la compilation. Sans supplement il tient dans
        // trois registres ; l'y ajouter inconditionnellement coutait 40 % a qui n'en veut pas
        // ( 5.63 -> 7.87 ns par plan ) -- l'`Etat` est construit a CHAQUE tentative.
        if constexpr ( veut_changement<Fourn>() ) {
            if ( ! f->suivant( Etat<NB,AvecSupplement>{ { change }, vx, vy, cid }, loc, p ) )
                return FINI;
            change = false;
        } else {
            if ( ! f->suivant( Etat<NB>{ {}, vx, vy, cid }, loc, p ) )
                return FINI;
        }

        const __m256 s = _mm256_fmadd_ps( _mm256_set1_ps( p.dx ), vx,
                         _mm256_fmsub_ps( _mm256_set1_ps( p.dy ), vy, _mm256_set1_ps( p.off ) ) );
        const unsigned m = _mm256_cmp_ps_mask( s, ZERO, _CMP_GT_OQ ) & valid;

        if ( ! m )
            continue;
        if ( m == valid )
            return VIDE;

        const unsigned prev = ( ( m << 1 ) | ( m >> ( NB - 1 ) ) ) & valid;
        const unsigned next = ( ( m >> 1 ) | ( m << ( NB - 1 ) ) ) & valid;
        const int i1 = __builtin_ctz( m & ~prev );
        const int j2 = __builtin_ctz( m & ~next );
        const int j0 = i1 ? i1 - 1 : NB - 1;
        const int j3 = j2 + 1 < NB ? j2 + 1 : 0;
        const int nb_in = NB - __builtin_popcount( m );
        const int nn = nb_in + 2;

        // les deux ancres dans des voies DISTINCTES -- A en voie 0, B en voie 1 -- donc pas de
        // collision quand `j0 == j3`, et une seule division.
        const __m256i anc = _mm256_mask_set1_epi32( _mm256_set1_epi32( j0 ), 0x2, j3 );
        const __m256i oth = _mm256_mask_set1_epi32( _mm256_set1_epi32( i1 ), 0x2, j2 );
        const __m256 vax = _mm256_permutexvar_ps( anc, vx ), vox = _mm256_permutexvar_ps( oth, vx );
        const __m256 vay = _mm256_permutexvar_ps( anc, vy ), voy = _mm256_permutexvar_ps( oth, vy );
        const __m256 sa  = _mm256_permutexvar_ps( anc, s  ), so  = _mm256_permutexvar_ps( oth, s  );
        const __m256 t   = _mm256_div_ps( sa, _mm256_sub_ps( sa, so ) );
        const __m256 pcx = _mm256_fmadd_ps( _mm256_sub_ps( vox, vax ), t, vax );
        const __m256 pcy = _mm256_fmadd_ps( _mm256_sub_ps( voy, vay ), t, vay );

        __m256i og = _mm256_add_epi32( _mm256_set1_epi32( j3 ), IOTA );
        og = _mm256_mask_sub_epi32( og, _mm256_cmpge_epi32_mask( og, _mm256_set1_epi32( NB ) ),
                                    og, _mm256_set1_epi32( NB ) );

        const __mmask8 mA = (__mmask8)( 1u << nb_in ), mB = (__mmask8)( 2u << nb_in );
        const __m256i  jA = _mm256_setzero_si256(), jB = _mm256_set1_epi32( 1 );

        __m256 nvx = _mm256_permutexvar_ps( og, vx );
        nvx = _mm256_mask_blend_ps( mA, nvx, _mm256_permutexvar_ps( jA, pcx ) );
        nvx = _mm256_mask_blend_ps( mB, nvx, _mm256_permutexvar_ps( jB, pcx ) );
        __m256 nvy = _mm256_permutexvar_ps( og, vy );
        nvy = _mm256_mask_blend_ps( mA, nvy, _mm256_permutexvar_ps( jA, pcy ) );
        nvy = _mm256_mask_blend_ps( mB, nvy, _mm256_permutexvar_ps( jB, pcy ) );
        __m256i nid = _mm256_permutexvar_epi32( og, cid );
        nid = _mm256_mask_set1_epi32( nid, mA, p.id );
        nid = _mm256_mask_blend_epi32( mB, nid, _mm256_permutexvar_epi32( _mm256_set1_epi32( j2 ), cid ) );

        // ---- L'ETAT NOUVEAU. Ecrit dans les references : c'est lui que le moteur redispatchera.
        if constexpr ( NB == 8 ) if ( nn > 8 ) {
            a->nb = 8;                                   // l'excursion partira de la
            a->attente = p;
            _mm256_store_ps( a->vx, vx );
            _mm256_store_ps( a->vy, vy );
            _mm256_store_si256( (__m256i *) a->cid, cid );
            return DEBORDE;
        }
        vx = nvx; vy = nvy; cid = nid;
        if constexpr ( veut_changement<Fourn>() ) change = true;
        if ( nn != NB )
            return nn;
    }
}

/// L'EXCURSION, en place dans l'atelier, jusqu'a ce que la cellule redescende a huit sommets.
/// Rend la nouvelle taille ( et recharge les registres ), ou `FINI`, ou `VIDE`.
template<class Fourn, class Atl>
inline int excursion( __m256 &vx, __m256 &vy, __m256i &cid, Fourn *f, Atl *a, Local<Fourn> &loc ) {
    constexpr int MaxNb = Atl::max_nb;
    if constexpr ( MaxNb <= 8 ) {
        a->nb = -1;
        return VIDE;                                     // pas de place : l'ancien echappement
    } else {
        float s[ MaxNb ];
        int  nb = a->nb;
        Plan p  = a->attente;
        for ( ;; ) {
            nb = coupe_large<MaxNb>( a->vx, a->vy, a->cid, nb, p, s );
            if ( nb <= 0 ) { a->nb = nb; return VIDE; }
            if ( nb <= 8 ) {
                vx  = _mm256_load_ps( a->vx );
                vy  = _mm256_load_ps( a->vy );
                cid = _mm256_load_si256( (const __m256i *) a->cid );
                return nb;
            }
            if ( ! f->suivant( EtatLarge<float>{ nb, a->vx, a->vy, a->cid, true }, loc, p ) ) {
                a->nb = nb;                              // finie, avec plus de huit cotes
                return VIDE;
            }
        }
    }
}

/// LE MOTEUR, EN REPARTANT D'UNE CELLULE DEJA FAITE au lieu du carre unite.
///
/// C'est ce que la phase 2 des sur-cellules demande : la phase 1 a coupe chaque cellule contre son
/// agregat et l'anneau 1, on la garde, et le second tour ne fait qu'ajouter les plans que la table
/// designe. Rien de neuf dans le moteur -- `excursion` recharge deja trois registres depuis
/// l'atelier et redispatche quand la cellule redescend sous huit sommets ; on ne fait qu'exposer
/// ce chemin.
///
/// L'atelier doit porter la cellule de depart : `a->nb` sommets dans `a->vx / vy / cid`.
template<class Fourn, class Atl>
void moteur_depuis( Fourn *f, Atl *a ) {
    int nb = a->nb;
    if ( nb <= 0 )
        return;                                          // deja vide : rien a reprendre

    __m256  vx = _mm256_setzero_ps(), vy = _mm256_setzero_ps();
    __m256i cid = _mm256_setzero_si256();
    if ( nb <= 8 ) {
        vx  = _mm256_load_ps( a->vx );
        vy  = _mm256_load_ps( a->vy );
        cid = _mm256_load_si256( (const __m256i *) a->cid );
    }

    Local<Fourn> loc{};
    bool change = true;

    for ( ;; ) {
        int r;
        switch ( nb ) {
            case 3:  r = etape<3>( vx, vy, cid, f, a, loc, change ); break;
            case 4:  r = etape<4>( vx, vy, cid, f, a, loc, change ); break;
            case 5:  r = etape<5>( vx, vy, cid, f, a, loc, change ); break;
            case 6:  r = etape<6>( vx, vy, cid, f, a, loc, change ); break;
            case 7:  r = etape<7>( vx, vy, cid, f, a, loc, change ); break;
            case 8:  r = etape<8>( vx, vy, cid, f, a, loc, change ); break;
            default: r = excursion( vx, vy, cid, f, a, loc ); break;
        }
        if ( r == FINI ) {
            a->nb = nb;
            _mm256_store_ps( a->vx, vx );
            _mm256_store_ps( a->vy, vy );
            _mm256_store_si256( (__m256i *) a->cid, cid );
            return;
        }
        if ( r == VIDE ) { if ( nb <= 8 ) a->nb = 0; return; }
        if ( r == DEBORDE ) { nb = 9; continue; }
        nb = r;
    }
}

/// LE MOTEUR. Une boucle, un `switch`, et la cellule en locales.
template<class Fourn, class Atl>
void moteur( Fourn *f, Atl *a ) {
    __m256  vx  = _mm256_setr_ps( 0, 1, 1, 0, 0, 0, 0, 0 );
    __m256  vy  = _mm256_setr_ps( 0, 0, 1, 1, 0, 0, 0, 0 );
    __m256i cid = _mm256_setr_epi32( -1, -2, -3, -4, 0, 0, 0, 0 );
    int nb = 4;
    // L'EMPLACEMENT DU FOURNISSEUR, dans la frame du moteur : il vit et meurt avec la cellule,
    // et s'il est vide il ne coute rien. Le noyau ne le lit jamais.
    Local<Fourn> loc{};
    bool change = true;

    for ( ;; ) {
        int r;
        switch ( nb ) {
            case 3:  r = etape<3>( vx, vy, cid, f, a, loc, change ); break;
            case 4:  r = etape<4>( vx, vy, cid, f, a, loc, change ); break;
            case 5:  r = etape<5>( vx, vy, cid, f, a, loc, change ); break;
            case 6:  r = etape<6>( vx, vy, cid, f, a, loc, change ); break;
            case 7:  r = etape<7>( vx, vy, cid, f, a, loc, change ); break;
            case 8:  r = etape<8>( vx, vy, cid, f, a, loc, change ); break;
            default: r = excursion( vx, vy, cid, f, a, loc ); break;
        }

        if ( r == FINI ) {                               // la taille n'a pas change
            a->nb = nb;
            _mm256_store_ps( a->vx, vx );
            _mm256_store_ps( a->vy, vy );
            _mm256_store_si256( (__m256i *) a->cid, cid );
            return;
        }
        if ( r == VIDE ) {                               // `a->nb` a deja ete renseigne
            if ( nb <= 8 ) a->nb = 0;
            return;
        }
        if ( r == DEBORDE ) { nb = 9; continue; }        // 9 : n'importe quoi hors de 3..8
        nb = r;
    }
}

} // namespace etats
} // namespace noyau2d

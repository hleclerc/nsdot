#pragma once

// =====================================================================================
// LE NOYAU A REGISTRES -- 2D, float, AVX-512VL sur 256 bits.
//
// Ce fichier ne contient PAS de classe cellule, et c'est tout son propos. La cellule y
// est trois registres :
//
//      ymm ps    vx[ 0 .. 7 ]        les sommets, en ordre cyclique
//      ymm ps    vy[ 0 .. 7 ]
//      ymm epi32 cid[ 0 .. 7 ]       le germe d'en face, < 0 pour le domaine
//
// `cdx / cdy / co` de `CellSoAT` ont disparu : le plan de la coupe `i` se REFABRIQUE
// depuis `cid[ i ]` et les positions des germes. C'est la meme observation que celle qui
// avait fait marcher `packe` -- l'ordre porte la connectivite, donc la geometrie est
// redondante. On passe de six tableaux a trois registres.
//
// POURQUOI 256 BITS ET NON 512. Le Xeon W-2145 (Skylake-X) a bien AVX-512, mais toute
// instruction 512 bits fait tomber la frequence. En restant a 256 bits avec l'encodage
// VL on garde ce qui compte vraiment ici -- les registres de MASQUE, `vpermps` a index
// variable, `vpbroadcastd` masque -- sans payer la licence. Et huit `float`, c'est la
// fenetre que les statistiques designent : 98.4 % des etats intermediaires ont huit
// sommets ou moins.
//
// CE QUI REND LA COUPE SANS BOUCLE. L'exterieur d'un convexe coupe par un demi-plan est
// une plage CYCLIQUE CONTIGUE. Le masque de signe est donc, a rotation pres, un bloc de
// uns : ses deux extremites se lisent en deux `ctz` sur des rotations du masque, et avec
// `NB` connu a la compilation ces rotations sont des immediats. Il n'y a plus rien a
// parcourir.
//
// LE SAUT. `nb` change a presque chaque coupe effective (`nb - nb_out + 2`, et `nb_out`
// vaut un ou deux). La boucle a `NB` fixe ne tourne donc que trois tours en moyenne : le
// cout de la transition compte. `[[gnu::musttail]]` la rend gratuite ET garantie -- l'ABI
// SysV passe les `__m256` dans `ymm0..ymm7`, donc la cellule TRAVERSE le saut dans ses
// registres, et le compilateur refuse de compiler s'il ne peut pas l'honorer.
//
// QUI MENE LA BOUCLE. Le noyau ne recoit ni liste, ni positions, ni germe : il recoit un
// FOURNISSEUR et lui demande, coupe apres coupe, « le prochain DEMI-PLAN ». Le fournisseur voit
// l'etat de la cellule -- ses sommets, ses `cid`, et `NB` COMME CONSTANTE DE COMPILATION -- et
// rend un `Plan` : une direction, un produit scalaire, et l'identite qui ira dans `cid`.
//
// LE NOYAU NE SAIT PLUS CE QU'EST UN DIRAC. La bissectrice de deux germes est une facon parmi
// d'autres de fabriquer un demi-plan ; elle vit desormais chez le fournisseur, avec les
// positions. Ce qui entre ici est de la geometrie pure, donc un diagramme de LAGUERRE ( des
// poids dans le produit scalaire ), une boite, un polygone de decoupe ou un plan de symetrie
// passent par la meme porte, sans une ligne de plus. C'est aussi ce qui fait disparaitre le
// `Ctx` : ses entrees sont parties chez le fournisseur, et ce qui reste est l'ATELIER, c'est a
// dire la sortie et l'endroit ou la cellule se pose quand elle ne tient plus dans les registres.
//
// Le renversement n'est pas cosmetique. Avec une liste, l'accelerateur decide TOUT avant que la
// geometrie existe. Avec un fournisseur il decide APRES chaque coupe, en regardant la cellule
// qui retrecit.
//
// LE DEBORDEMENT N'EST PLUS UN ABANDON, C'EST UNE EXCURSION. Depuis `NB == 8` une coupe a
// `nb_out == 1` demanderait un neuvieme sommet. La cellule se pose alors dans l'atelier, on
// continue a la couper EN SCALAIRE et en place -- et DES QU'ELLE REDESCEND a huit sommets on
// la recharge dans les registres et on repart en SIMD. Le va-et-vient ne coute pas une frame :
// `deborde` a exactement la signature de `noyau`, donc les deux sens sont des `musttail`.
//
// Consequence : le noyau rend TOUJOURS une cellule complete. `nb == -1` ne subsiste que si
// l'atelier lui-meme est trop petit -- et un atelier de huit sommets, qui ne peut pas heberger
// l'excursion, redonne exactement l'ancien comportement. Le choix est un parametre de type.
// =====================================================================================

#include "supercell/Contrat2D.h"

#include <immintrin.h>

namespace noyau2d {

/// CE QUE LE FOURNISSEUR VOIT DE LA CELLULE quand elle tient dans les registres. `nb` est une
/// constante de COMPILATION -- un fournisseur qui deroule sur les sommets le fait sans boucle a
/// borne variable.
///
/// Les accesseurs `x/y/id` passent par la pile : sortir UNE voie d'un `ymm` n'a pas d'autre
/// chemin. C'est cher, et c'est voulu -- un fournisseur qui ne regarde pas la cellule ne paie
/// rien du tout (le `store` est mort, le compilateur l'enleve), un fournisseur qui la regarde
/// paie ce que ca coute vraiment. Le cout n'est pas cache dans le noyau.
/// CE QUE LE MOTEUR AJOUTE A L'ETAT quand on le lui demande, et RIEN sinon.
///
/// Ce n'est pas de la coquetterie. Ajouter ces deux champs INCONDITIONNELLEMENT a fait perdre
/// 40 % au chemin qui ne s'en sert pas -- 5.63 -> 7.87 ns par plan a n = 200000, K = 48. L'`Etat`
/// est construit a CHAQUE tentative, des millions de fois, et douze octets de plus suffisent a le
/// sortir des registres. D'ou deux formes, choisies a la compilation par la politique du moteur.
struct SansSupplement {};

struct AvecSupplement {
    /// LA CELLULE A-T-ELLE CHANGE depuis l'appel precedent ? Le fournisseur s'en sert pour compter
    /// les rejets consecutifs -- donc pour decider d'une transition -- sans rien relire. Une fois
    /// sur sept, puisque 13.8 % des tentatives coupent.
    bool change;
};


/// CE QUE LE FOURNISSEUR VOIT DE LA CELLULE quand elle tient dans les registres. `nb` est une
/// constante de COMPILATION -- un fournisseur qui deroule sur les sommets le fait sans boucle a
/// borne variable.
///
/// Les accesseurs `x/y/id` passent par la pile : sortir UNE voie d'un `ymm` n'a pas d'autre
/// chemin. C'est cher, et c'est voulu -- un fournisseur qui ne regarde pas la cellule ne paie rien
/// du tout ( le `store` est mort, le compilateur l'enleve ), un fournisseur qui la regarde paie ce
/// que ca coute vraiment. Le cout n'est pas cache dans le noyau.
template<int NB, class SUP = SansSupplement>
struct Etat : SUP {
    static constexpr int nb = NB;
    __m256  vx, vy;
    __m256i cid;

    float x ( int i ) const { alignas( 32 ) float t[ 8 ]; _mm256_store_ps( t, vx ); return t[ i ]; }
    float y ( int i ) const { alignas( 32 ) float t[ 8 ]; _mm256_store_ps( t, vy ); return t[ i ]; }
    int   id( int i ) const { alignas( 32 ) int   t[ 8 ]; _mm256_store_si256( (__m256i *) t, cid ); return t[ i ]; }
};

/// la seule constante qui survive : `iota`, pour l'ordre de sortie. La rotation « voisin
/// suivant » a disparu avec la retouche en place -- on nomme desormais les quatre voies
/// ( j0, i1, j3, j2 ) explicitement, et un vecteur de rotation ne sert plus a rien.
template<int NB> struct Cst {
    static __m256i iota() { return _mm256_setr_epi32( 0, 1, 2, 3, 4, 5, 6, 7 ); }
};

template<int NB, class Fourn, class Atl> void noyau  ( __m256 vx, __m256 vy, __m256i cid, Fourn *f, Atl *a, Local<Fourn> *loc );
template<        class Fourn, class Atl> void deborde( __m256 vx, __m256 vy, __m256i cid, Fourn *f, Atl *a, Local<Fourn> *loc );

// PAS DE `[[gnu::flatten]]` ICI. `noyau<NB>` est MUTUELLEMENT RECURSIF -- `noyau<5>` cite
// `noyau<3..8>`, qui citent `noyau<5>`. `flatten` demande d'inliner tous les appels du corps,
// transitivement : sur un graphe cyclique gcc deroule jusqu'a epuiser la RAM ( mesure : 27 Go
// puis 29 Go de `cc1plus` sur une machine de 31 Go, tue deux fois par l'OOM killer, avec le
// terminal autour ). Il est de toute facon en contradiction avec `musttail`, qui dit l'inverse
// -- ne m'inline pas, saute. Sans lui : 2.2 s, 350 Mo, et les sauts restent des `jmp`.
template<int NB, class Fourn, class Atl>
void noyau( __m256 vx, __m256 vy, __m256i cid, Fourn *f, Atl *a, Local<Fourn> *loc ) {
    constexpr unsigned valid = ( 1u << NB ) - 1;
    const __m256i IOTA = Cst<NB>::iota();
    const __m256  ZERO = _mm256_setzero_ps();

    for ( ;; ) {
        // ---- LE DEMI-PLAN SUIVANT, DEMANDE au fournisseur. C'est le seul endroit ou le noyau
        // apprend sur quoi travailler, et il n'apprend rien d'autre : ni d'ou vient ce plan, ni
        // ce qu'il represente.
        Plan p;
        if ( ! f->suivant( Etat<NB>{ {}, vx, vy, cid }, *loc, p ) )
            break;

        // ---- LE TEST, QUI EST DEJA LA COUPE. Deux `vfmadd`, un `vcmpps` vers un masque.
        const __m256 s = _mm256_fmadd_ps( _mm256_set1_ps( p.dx ), vx,
                         _mm256_fmsub_ps( _mm256_set1_ps( p.dy ), vy, _mm256_set1_ps( p.off ) ) );
        const unsigned m = _mm256_cmp_ps_mask( s, ZERO, _CMP_GT_OQ ) & valid;

        if ( ! m )                                       // le cas frequent : ~70 % des tentatives
            continue;
        if ( m == valid ) { a->nb = 0; return; }

        // ---- LES DEUX BOUTS DE LA PLAGE EXTERIEURE. Rotations a decalage IMMEDIAT.
        const unsigned prev = ( ( m << 1 ) | ( m >> ( NB - 1 ) ) ) & valid;   // prev[ i ] = m[ i-1 ]
        const unsigned next = ( ( m >> 1 ) | ( m << ( NB - 1 ) ) ) & valid;   // next[ i ] = m[ i+1 ]
        const int i1 = __builtin_ctz( m & ~prev );       // premier DEHORS   ( unique )
        const int j2 = __builtin_ctz( m & ~next );       // dernier  DEHORS  ( unique )
        const int j0 = i1 ? i1 - 1 : NB - 1;             // dernier DEDANS avant
        const int j3 = j2 + 1 < NB ? j2 + 1 : 0;         // premier DEDANS apres
        const int nb_in = NB - __builtin_popcount( m );
        const int nn = nb_in + 2;

        // ---- LES DEUX INTERSECTIONS, EN UNE SEULE DIVISION.
        //
        // On veut A sur l'arete ( j0 -> i1 ) ANCREE en j0, et B sur ( j2 -> j3 ) ancree en j3 :
        // l'ancre est toujours le sommet DEDANS, faute de quoi `t == 0` ne rend pas exactement
        // le sommet et la plage cesse d'etre contigue.
        //
        // ON NE MODIFIE PAS UNE VOIE EN PLACE. La version qui gardait les points dans la voie de
        // leur ancre -- `rotf` partout sauf en `j3` qui pointe sur `j2` -- a l'air de marcher et
        // ne marche pas : quand UN SEUL sommet reste dedans, `j0 == j3`, et cette unique retouche
        // ecrase la voie dont A avait besoin. A se calculait alors sur ( j0 -> j2 ) et sortait
        // CONFONDU avec B. `nb_in == 1` est 13.7 % des coupes effectives et touche 55.4 % des
        // cellules : la combinatoire etait fausse a 56 %.
        //
        // On RASSEMBLE donc les deux ancres dans des voies DISTINCTES -- A en voie 0, B en voie
        // 1 -- avec leur vis-a-vis en face. Plus de collision possible par construction, quel que
        // soit `nb_in`, et les deux points sortent toujours du meme `vdivps`.
        const __m256i anc = _mm256_mask_set1_epi32( _mm256_set1_epi32( j0 ), 0x2, j3 );
        const __m256i oth = _mm256_mask_set1_epi32( _mm256_set1_epi32( i1 ), 0x2, j2 );
        const __m256 vax = _mm256_permutexvar_ps( anc, vx ), vox = _mm256_permutexvar_ps( oth, vx );
        const __m256 vay = _mm256_permutexvar_ps( anc, vy ), voy = _mm256_permutexvar_ps( oth, vy );
        const __m256 sa  = _mm256_permutexvar_ps( anc, s  ), so  = _mm256_permutexvar_ps( oth, s  );
        const __m256 t  = _mm256_div_ps( sa, _mm256_sub_ps( sa, so ) );
        const __m256 pcx = _mm256_fmadd_ps( _mm256_sub_ps( vox, vax ), t, vax );
        const __m256 pcy = _mm256_fmadd_ps( _mm256_sub_ps( voy, vay ), t, vay );

        // ---- LE REMONTAGE. Le cycle sortant est [ j3, j3+1, ... , j0, A, B ] :
        //   arete j0 -> A  : morceau de l'ancienne coupe j0, deja portee par la voie j0
        //   arete A  -> B  : la NOUVELLE coupe
        //   arete B  -> j3 : morceau de l'ancienne coupe j2
        __m256i og = _mm256_add_epi32( _mm256_set1_epi32( j3 ), IOTA );
        og = _mm256_mask_sub_epi32( og, _mm256_cmpge_epi32_mask( og, _mm256_set1_epi32( NB ) ),
                                    og, _mm256_set1_epi32( NB ) );

        const __mmask8 mA = (__mmask8)( 1u << nb_in ), mB = (__mmask8)( 2u << nb_in );
        const __m256i  jA = _mm256_setzero_si256(), jB = _mm256_set1_epi32( 1 );  // A en voie 0, B en voie 1

        __m256 nvx = _mm256_permutexvar_ps( og, vx );
        nvx = _mm256_mask_blend_ps( mA, nvx, _mm256_permutexvar_ps( jA, pcx ) );
        nvx = _mm256_mask_blend_ps( mB, nvx, _mm256_permutexvar_ps( jB, pcx ) );
        __m256 nvy = _mm256_permutexvar_ps( og, vy );
        nvy = _mm256_mask_blend_ps( mA, nvy, _mm256_permutexvar_ps( jA, pcy ) );
        nvy = _mm256_mask_blend_ps( mB, nvy, _mm256_permutexvar_ps( jB, pcy ) );
        __m256i nid = _mm256_permutexvar_epi32( og, cid );
        nid = _mm256_mask_set1_epi32( nid, mA, p.id );
        nid = _mm256_mask_blend_epi32( mB, nid, _mm256_permutexvar_epi32( _mm256_set1_epi32( j2 ), cid ) );

        // ---- LE SAUT. `nn` va de 3 a NB+1 ; seul NB==8 peut deborder.
        if ( nn == NB ) { vx = nvx; vy = nvy; cid = nid; continue; }
        if constexpr ( NB == 8 ) if ( nn > 8 ) {
            // ---- L'EXCURSION. On pose les HUIT sommets d'AVANT la coupe et le plan en attente :
            // `deborde` appliquera lui-meme `p`, dans une representation qui a la place.
            a->nb = 8;
            a->attente = p;
            _mm256_store_ps( a->vx, vx );
            _mm256_store_ps( a->vy, vy );
            _mm256_store_si256( (__m256i *) a->cid, cid );
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
    _mm256_store_ps( a->vx, vx );
    _mm256_store_ps( a->vy, vy );
    _mm256_store_si256( (__m256i *) a->cid, cid );
}

/// L'EXCURSION -- la cellule ne tient plus dans les registres, on la coupe en place jusqu'a ce
/// qu'elle y revienne.
///
/// MEME SIGNATURE QUE `noyau`, et ce n'est pas un hasard : c'est ce qui rend les deux sens du
/// va-et-vient `musttail`. Les trois registres passes ne servent a rien ici -- l'etat est dans
/// l'atelier -- mais les garder dans la signature evite d'ouvrir une frame a chaque passage.
template<class Fourn, class Atl>
void deborde( __m256, __m256, __m256i, Fourn *f, Atl *a, Local<Fourn> *loc ) {
    constexpr int MaxNb = Atl::max_nb;

    if constexpr ( MaxNb <= 8 ) {
        // pas de place : l'ancien echappement. `attente` reste renseigne pour qui veut reprendre.
        a->nb = -1;
        return;
    } else {
        float s[ MaxNb ];
        int  nb = a->nb;                                 // huit, et `attente` pas encore appliquee
        Plan p  = a->attente;

        for ( ;; ) {
            nb = coupe_large<MaxNb>( a->vx, a->vy, a->cid, nb, p, s );
            if ( nb <= 0 ) { a->nb = nb; return; }        // vide, ou atelier trop petit

            if ( nb <= 8 ) {
                // ---- LE RETOUR. La cellule tient de nouveau dans trois registres : on recharge
                // et on repart en SIMD. Les voies au-dela de `nb` sont du rebut, et le masque
                // `valid` du noyau les ignore.
                const __m256  rvx = _mm256_load_ps( a->vx );
                const __m256  rvy = _mm256_load_ps( a->vy );
                const __m256i rid = _mm256_load_si256( (const __m256i *) a->cid );
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
                a->nb = nb;                              // fini, et la cellule a plus de huit cotes
                return;
            }
        }
    }
}

/// L'ENTREE : le carre unite, quatre sommets, quatre cotes de domaine. Ils sont numerotes
/// `-1..-4` et non tous `-1` : c'est ce qui permet de refabriquer le plan d'un cote du domaine
/// depuis son seul `cid`.
template<class Fourn, class Atl>
inline void carre_unite( Fourn *f, Atl *a ) {
    Local<Fourn> loc{};                                  // vit et meurt avec la cellule
    const __m256  vx  = _mm256_setr_ps( 0, 1, 1, 0, 0, 0, 0, 0 );
    const __m256  vy  = _mm256_setr_ps( 0, 0, 1, 1, 0, 0, 0, 0 );
    const __m256i cid = _mm256_setr_epi32( -1, -2, -3, -4, 0, 0, 0, 0 );
    noyau<4>( vx, vy, cid, f, a, &loc );
}

} // namespace noyau2d

#pragma once

// =====================================================================================
// LA BOITE DE DEPART DE LA CELLULE, ET LE FILET QUI LA REND SURE.
//
// = POURQUOI
//
// Le repere du germe ( `cell/Contrat2D.h` ) rend exactes les coupes TARDIVES, pas les premieres :
// un sommet cree quand la cellule mesure encore `L` porte l'erreur `eps L`, et la cellule part a
// la taille du DOMAINE. D'ou le plancher `eps L / h = eps n^( 1/D )` qui survivait ( README
// § 19.6 ), et que le diagnostic `SF_R` a identifie : demarrer d'un carre vingt fois plus petit
// divisait deja l'erreur par 7.3 a `n = 1e6`.
//
// La vraie reparation est donc de partir d'une boite qui a DEJA L'ECHELLE DE LA CELLULE. Le BSP
// en offre une gratuitement : la FEUILLE qui contient le germe. Elle tient une poignee de germes,
// donc sa taille est celle de l'espacement local -- y compris la ou les germes sont serres, ce
// qu'un `h` global ( `n^( -1/D )` ) ne saurait pas faire.
//
// = LE FILET, ET POURQUOI IL EST EXACT
//
// Rien ne garantit qu'une cellule tienne dans sa boite : en Laguerre un poids eleve l'emmene loin
// de son germe. On ne devine donc pas, ON CONSTATE. Les faces de la boite qui ne sont pas des
// faces du DOMAINE recoivent un identifiant ARTIFICIEL ( `<= FACE_ARTIF` ) ; si l'une d'elles
// survit dans la cellule finie, c'est que la cellule touche la boite, donc qu'elle en sort
// peut-etre -- et on RECOMMENCE depuis le domaine.
//
// C'est exact, et l'argument tient en deux cas. Soit `C` la cellule vraie, convexe, et `B` la
// boite. Si `C` n'est pas incluse dans `B`, alors ou bien
//   ( a ) `C inter B` est VIDE -- on ne voit aucune face, et c'est le piege ;
//   ( b ) `C inter B` n'est pas vide, donc `C` a des points des deux cotes de `dB`, donc
//         `C inter B` a une face portee par `dB` : on la voit.
// D'ou le test complet : **une cellule VIDE compte comme une sortie**. On serait tente de
// raisonner « la cellule contient son germe, qui est dans la boite, donc ( a ) n'arrive pas » --
// C'EST FAUX EN LAGUERRE : un poids assez bas met la cellule ailleurs que sur son germe, ou la
// supprime. Le banc l'a paye : `check` passait ( ses poids valent zero par defaut, `--weights 0`,
// donc chaque cellule contenait bien son germe ) et Newton divergeait des la premiere iteration
// a poids non nuls.
// La reciproque reste fausse -- une cellule peut effleurer `dB` sans en sortir, une cellule peut
// etre vraiment vide -- donc on recommence parfois pour rien. Le test est CONSERVATIF, jamais
// permissif, et c'est le seul sens qui compte.
//
// = LA REPRISE AGRANDIT LA BOITE, ELLE NE SAUTE PAS AU DOMAINE
//
// Ce n'est pas une economie de temps, c'est une question de JUSTESSE : une cellule reprise au
// domaine retrouve l'erreur `eps x taille du domaine` qu'on vient justement d'enlever. Comme le
// plancher de Newton suit le MAXIMUM de l'erreur et non sa mediane, quelques pour cent de
// cellules reprises suffisent a le fixer. On agrandit donc d'un facteur `CROISSANCE` a chaque
// essai ; le domaine n'est que le dernier recours -- et il finit toujours par etre atteint, donc
// le resultat est celui du domaine, toujours.
//
// LA FEUILLE SOUS-ESTIME BEAUCOUP SUR UN NUAGE GROUPE, et c'est pourquoi la croissance est a
// quatre et non a deux. Sur le nuage de LIGNES, dix germes colineaires font une feuille en
// lamelle, alors que leurs cellules s'etendent PERPENDICULAIREMENT jusqu'a la ligne voisine : il
// faut plusieurs ordres de grandeur. `MAX_REPRISES` est dimensionne pour que la boucle s'arrete
// toujours parce que la boite a couvert le domaine, jamais parce qu'on a epuise le compteur.
//
// `nb_reprises` compte les agrandissements : c'est l'instrument qui dit si la boite est bien
// choisie -- au-dela de quelques pour cent, elle ne l'est pas.
// =====================================================================================

#include "util/common.h"
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>

namespace sf {

/// Les cotes du domaine portent `-1 .. -2D`. Une face de la boite de depart qui n'est PAS une
/// face du domaine porte `FACE_ARTIF - d` : la voir survivre veut dire « recommencer ».
inline constexpr std::int32_t FACE_ARTIF = -101;
inline constexpr bool face_artificielle( std::int32_t id ) { return id <= FACE_ARTIF; }

/// LE COMPTEUR DE REPRISES, global et atomique. C'est un INSTRUMENT d'etude, pas un rouage : il
/// n'est touche que lorsqu'une cellule sort de sa boite, donc jamais dans le cas courant.
inline std::atomic<long long> nb_reprises{ 0 };

/// le facteur d'agrandissement a chaque essai, et combien d'essais : `4^12 = 1.7e7`, de quoi
/// passer de la plus fine lamelle au domaine entier sans jamais buter sur le compteur.
inline constexpr double CROISSANCE  = 4;
inline constexpr int    MAX_REPRISES = 12;

/// LA DILATATION de la boite de feuille, en nombre de fois sa plus grande arete. `0` eteint tout
/// et rend le domaine -- le comportement d'avant, garde pour pouvoir remesurer ( `SF_DIL=0` ).
///
/// UN, ET C'EST MESURE. Depuis que la reprise DOUBLE au lieu de sauter au domaine, l'erreur est
/// monotone en dilatation -- plus la boite est petite, mieux c'est -- et le seul frein est le
/// nombre de reprises. A `n = 1e5`, 2D, poids `h^2`, ecart float / double sur la masse :
///
///      dilatation      mediane     p99        max
///      0 ( domaine )   6.78e-07   2.07e-05   6.69e-05
///      1               3.44e-08   4.59e-07   1.52e-06
///      2               5.23e-08   7.35e-07   2.81e-06
///      4               8.45e-08   1.39e-06   4.57e-06
///      8               1.40e-07   2.71e-06   7.94e-06
///
/// `3.44e-08` vaut 0.57 fois l'epsilon du `float` : la cellule mediane est exacte a l'arrondi
/// pres. En dessous de 1 on ne gagne plus rien -- la feuille EST l'echelle locale -- et on paie
/// des reprises.
inline double dilatation() {
    static const double d = []{
        if ( const char *e = std::getenv( "SF_DIL" ) ) return std::atof( e );
        return 1.0;
    }();
    return d;
}

/// La boite de depart du germe de rang `k` : sa feuille, DILATEE, centree sur le germe, et
/// intersectee avec le domaine. Rend `false` s'il faut partir du domaine.
template<int D, class Arbre>
inline bool boite_de_feuille( const Arbre &arbre, SI k, const TF p0[ D ], TF lo[ D ], TF hi[ D ],
                              double facteur = 1 ) {
    const double dil = dilatation() * facteur;
    if ( dil <= 0 || k < 0 || SI( arbre.feuille_de.size() ) <= k )
        return false;
    const auto &nd = arbre.nodes[ arbre.feuille_de[ k ] ];
    TF r = 0;                                            // le rayon : la plus grande arete, dilatee
    for ( int d = 0; d < D; ++d )
        r = std::max( r, nd.hi[ d ] - nd.lo[ d ] );
    r *= TF( dil );
    if ( ! ( r > 0 ) )
        return false;
    bool utile = false;
    for ( int d = 0; d < D; ++d ) {
        lo[ d ] = p0[ d ] - r;
        hi[ d ] = p0[ d ] + r;
        if ( lo[ d ] > 0 ) utile = true; else lo[ d ] = 0;
        if ( hi[ d ] < 1 ) utile = true; else hi[ d ] = 1;
    }
    return utile;                                        // sinon c'est le domaine, autant le dire
}

} // namespace sf

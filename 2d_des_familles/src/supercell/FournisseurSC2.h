#pragma once

// =====================================================================================
// LA PHASE 2 DES SUR-CELLULES : ce qui garantit qu'aucune coupe n'est ratee.
//
// La phase 1 a coupe chaque cellule contre son agregat et l'anneau 1. Ces cellules sont PLUS
// GRANDES que les vraies -- moins de plans les coupent -- et leur reunion est la SUR-CELLULE de
// l'agregat. Avec des poids, une cellule lointaine peut encore en manger une : il faut donc
// repasser, et c'est ce que fait la phase 2.
//
// = L'ENCEINTE, ET LE TEST QUI VA AVEC
//
// De l'agregat `A` on retient deux choses, calculees une fois :
//
//     E_A   la boite de tous les sommets des cellules de phase 1 de ses membres
//     R_A   le plus grand `|v - p_i|^2 - w_i` sur ces memes sommets, `p_i` etant le germe du membre
//
// Un germe `q` d'un noeud `B`, de poids majore par `w( q ) <= a . q + b`, ne peut couper aucune
// cellule de `A` si
//
//     min_{ v in E_A, q in B } ( |v - q|^2 - a . q - b )  >=  R_A
//
// -- c'est le critere du fournisseur BSP, transpose de la cellule a son enceinte. Il se teste une
// fois par agregat et sert a ses ~8 membres : c'est TOUTE l'amortisation de l'affaire.
//
// LE TERME DE POIDS N'EST PAS FACULTATIF. La premiere version comparait `dist^2( E_A, B )` a
// `max |v - p_i|^2` sans rien de plus : en Laguerre avec `w ~ U[ 0, 4 h^2 ]` le majorant des poids
// est du MEME ORDRE que le rayon carre, et 18 cellules sur 200000 perdaient une coupe. La somme
// des aires ne le disait pas -- elle valait 1.0000018 -- il a fallu comparer cellule par cellule
// au fournisseur BSP pour le voir.
//
// La minimisation reste SEPARABLE PAR AXE, et par axe elle est convexe : `t( q )^2` est la
// distance carree a un intervalle, donc convexe, et `- a q` est lineaire. Son minimum libre est
// en `q = ( a > 0 ? eh : el ) + a / 2`, et un `clamp` sur la boite de `B` donne la reponse EXACTE.
//
// = POURQUOI LA TABLE NE GARDE QUE L'ANNEAU 2 ET AU-DELA
//
// L'anneau 1 est deja dans la cellule de phase 1. Le repasser serait un no-op paye plein tarif.
//
// = LE GARDE-FOU, ET POURQUOI IL EST INDISPENSABLE
//
// L'amortissement suppose que l'enceinte enveloppe QUELQUE CHOSE. Mesure : en Voronoi, 0.1 % des
// agregats ont une cellule de phase 1 qui couvre presque tout le carre -- l'agregat et son anneau 1
// ne la bornent pas -- et ces 25 agregats sur 25000 font A EUX SEULS la moitie de la table. En
// Laguerre a `4 h^2`, c'est 1 % des agregats dont la ligne contient TOUS les autres.
//
// Aucune forme d'enceinte ne rattrape cela. On les SORT : des que la ligne depasse un plafond
// ( 48, le plat d'une courbe tres plate entre 24 et 96 ), elle est abandonnee et les membres de
// l'agregat sont calcules par le fournisseur BSP, depuis le carre. Le pire cas est ainsi borne par
// une RESSOURCE et non par un seuil geometrique, ce qui vaut mieux sur des nuages anisotropes.
//
// Effet mesure, Laguerre `4 h^2`, n = 200000 : total 1.406 s -> 0.256 s, et 4 % des cellules
// passent par le BSP.
//
// = CE QUE LE FOURNISSEUR FAIT
//
// Il deroule la liste d'agregats que la table donne pour `A`, et propose leurs membres. Le noyau
// repart de la cellule de phase 1 ( `moteur_depuis` ), pas du carre unite.
//
// L'ORDRE EST MORTON LOCAL : les membres d'un agregat sont tries en Morton a l'interieur de
// l'agregat, et les agregats de la table par distance. En phase 1 l'ordre ne changeait rien --
// tout etait propose, sans elagage ni arret, donc seul le transitoire pouvait bouger et il ne
// bougeait pas ( 1.9 % des coupes hors registres, mesure ). Ici c'est different : on elague, donc
// l'ordre decide de ce qu'on n'essaie pas.
// =====================================================================================

#include "supercell/Agregats.h"
#include "supercell/Contrat2D.h"
#include "supercell/Elagage.h"

namespace noyau2d {

/// L'enceinte d'un agregat : la boite de ses cellules de phase 1, et la PORTEE.
///
/// `r2` est `max_{i, v} ( |v - p_i|^2 - w_i )` : le poids du membre en fait partie, sans quoi le
/// critere de la table ne dit rien en Laguerre. Sans poids c'est le rayon carre habituel.
struct Enceinte { float lo[ 2 ], hi[ 2 ], r2; };

/// LA TABLE : pour chaque agregat, les agregats de l'anneau 2 et au-dela qui peuvent encore le
/// manger. En CSR, comme tout le reste ici.
///
/// Elle porte aussi la BOITE DES GERMES de chaque agregat et le majorant AFFINE de leurs poids. La
/// table est partagee par les ~8 membres, donc grossiere par construction : c'est le filtre par
/// CELLULE ci-dessous qui la resserre, et il coute une operation SIMD par agregat de la ligne.
///
/// LE MAJORANT EST AFFINE, `w( q ) <= a . q + b`, comme dans l'arbre BSP et pour la meme raison :
/// un majorant constant traite l'agregat comme si son germe le plus lourd etait partout, et en
/// transport semi-discret les poids sont un potentiel, donc ils varient regulierement dans
/// l'espace. Il ne coute rien de plus a tester -- le minimum reste separable par axe.
struct TableSC {
    std::vector<int> deb, ag;
    const BoitesAgregats *bo = nullptr;                  ///< boite + majorant, par agregat
};

template<int D, bool POIDS = false>
struct FournisseurSC2 {
    static_assert( D == 2, "le noyau de coupe est 2D" );

    struct Local {
        bool amorce = false;
        int  q = 0;                                      ///< ou on en est dans la ligne de table
        int  k = 0, fin = 0;                             ///< la plage de l'agregat courant
    };

    const pd::supercell::Gros<D> *G;
    const TableSC *T;
    int   i0, a0;
    float x0, y0, w0 = 0;

    FournisseurSC2( const pd::supercell::Gros<D> *G, const TableSC *T, int a0, int i0 )
        : G( G ), T( T ), i0( i0 ), a0( a0 ),
          x0( (float) G->Ppp[ 0 ][ i0 ] ), y0( (float) G->Ppp[ 1 ][ i0 ] ),
          w0( POIDS ? (float) G->Wp[ i0 ] : 0.f ) {}

    /// LE FILTRE PAR CELLULE. La table est PARTAGEE par les ~8 membres de l'agregat, donc lache
    /// par construction ; ce test la resserre contre CETTE cellule pour une operation SIMD par
    /// agregat de la ligne. Le critere et son exactitude sont dans `Elagage.h`.
    template<class Etat>
    bool peut_couper( int b, const Etat &e ) const {
        return peut_couper_boite<POIDS>( e, x0, y0, w0, ( *T->bo )[ b ] );
    }

    template<class Etat>
    bool suivant( const Etat &e, Local &l, Plan &p ) {
        if ( ! l.amorce ) { l.amorce = true; l.q = T->deb[ a0 ]; l.k = 0; l.fin = 0; }
        for ( ;; ) {
            if ( l.k < l.fin ) {
                const int j = l.k++;
                const float xj = (float) G->Ppp[ 0 ][ j ], yj = (float) G->Ppp[ 1 ][ j ];
                p.dx = xj - x0;
                p.dy = yj - y0;
                p.off = 0.5f * ( p.dx * ( xj + x0 ) + p.dy * ( yj + y0 ) );
                if constexpr ( POIDS ) p.off += 0.5f * ( w0 - (float) G->Wp[ j ] );
                p.id = j;
                return true;
            }
            if ( l.q >= T->deb[ a0 + 1 ] ) return false;
            const int b = T->ag[ l.q++ ];
            if ( ! peut_couper( b, e ) ) continue;        // cet agregat ne peut plus rien : suivant
            l.k = (int) G->mdeb[ b ]; l.fin = (int) G->mdeb[ b + 1 ];
        }
    }
};

} // namespace noyau2d

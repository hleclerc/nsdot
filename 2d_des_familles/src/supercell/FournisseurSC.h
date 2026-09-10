#pragma once

// =====================================================================================
// LA PHASE 1 DES SUR-CELLULES, EN FOURNISSEUR.
//
// = A QUOI SERT LA PHASE 1
//
// Pas a amortir : a GARANTIR. Avec des poids, une cellule peut en ecraser d'autres, donc on ne
// peut pas se contenter des proches voisins et esperer. On calcule d'abord chaque cellule contre
// son agregat et l'anneau 1 ; les cellules ainsi obtenues sont PLUS GRANDES que les vraies --
// moins de plans les coupent -- et leur reunion est la SUR-CELLULE de l'agregat. La phase 2
// regarde ensuite, table en main, quels agregats non explores pourraient encore les manger.
//
// La sur-cellule n'est donc connue qu'une fois TOUS les membres de l'agregat faits. C'est une
// barriere, et elle est incontournable : deux passes sur les cellules, pas un flux unique.
//
// = L'ORDRE DES CANDIDATS, ET POURQUOI IL DECIDE DE TOUT ICI
//
// L'etape 4 d'origine coupe PAR AGREGAT : les plans arrivent groupes par voisin et non par
// distance, donc la cellule traverse des etats transitoires bien plus larges que sa forme finale.
// C'est ce qui force `CellSC` a 256 sommets en 3D.
//
// Notre noyau tient HUIT sommets dans des registres. Un transitoire large ne le rend pas faux --
// l'excursion rend toujours une cellule complete -- mais il lui retire tout son interet. L'ordre
// n'est donc plus une optimisation, c'est la condition.
//
// D'ou : dans l'agregat, on part de `i` et on s'ECARTE ( i+1, i-1, i+2, i-2... ) ; sur un nuage
// trie en Morton c'est approximativement du plus proche au plus lointain. Puis les agregats de
// l'anneau 1, tries par distance de leur MEDIAN au germe. Mesure faite plus tot sur ce meme
// principe : le pire etat intermediaire reste a 9 sommets au lieu de 27.
//
// DEUX CURSEURS ET PAS UNE BOUCLE DE REJET pour le parcours sortant : la forme evidente monte
// l'ecart jusqu'a `2 |A|` et rejette la moitie des indices, ce qui coutait +135 % mesure.
// =====================================================================================

#include "supercell/Agregats.h"
#include "supercell/Contrat2D.h"

#include <algorithm>

namespace noyau2d {

/// `POIDS` a la compilation : le cas euclidien ne doit pas payer les termes de poids.
template<int D, bool POIDS = false>
struct FournisseurSC1 {
    static_assert( D == 2, "le noyau de coupe est 2D" );
    static constexpr int MAX_ANNEAU1 = 24;               ///< degre grossier mesure : 5.96

    /// L'ETAT DU PARCOURS, un par cellule, loge dans la frame du moteur.
    struct Local {
        bool amorce = false;
        int  lo = 0, hi = 0;                             ///< les deux curseurs dans l'agregat
        bool cote = true;
        int  na = 0, ia = 0;                             ///< anneau 1 : combien, ou on en est
        int  k = 0, fin = 0;                             ///< la plage de l'agregat voisin courant
        int  ann[ MAX_ANNEAU1 ];                         ///< les agregats de l'anneau 1, tries
    };

    const pd::supercell::Gros<D> *G;
    int   i0;                                            ///< le germe, indice PERMUTE
    int   a0;                                            ///< son agregat
    float x0, y0, w0 = 0;

    /// L'AGREGAT EST DONNE, PAS RETROUVE. `G.lab` indexe les germes D'ORIGINE ; apres la
    /// permutation, `i0` est un indice PERMUTE et `lab[ i0 ]` designe autre chose. L'appelant
    /// parcourt de toute facon agregat par agregat -- c'est ce que fait l'etape 4 -- donc il l'a
    /// sous la main.
    FournisseurSC1( const pd::supercell::Gros<D> *G, int a0, int i0 )
        : G( G ), i0( i0 ), a0( a0 ),
          x0( (float) G->Ppp[ 0 ][ i0 ] ), y0( (float) G->Ppp[ 1 ][ i0 ] ),
          w0( POIDS ? (float) G->Wp[ i0 ] : 0.f ) {}

    void plan_vers( int j, Plan &p ) const {
        const float xj = (float) G->Ppp[ 0 ][ j ], yj = (float) G->Ppp[ 1 ][ j ];
        p.dx = xj - x0;
        p.dy = yj - y0;
        p.off = 0.5f * ( p.dx * ( xj + x0 ) + p.dy * ( yj + y0 ) );
        if constexpr ( POIDS ) p.off += 0.5f * ( w0 - (float) G->Wp[ j ] );
        p.id = j;
    }

    /// l'anneau 1, trie par distance du MEDIAN au germe. Peu d'elements ( ~6 ), donc un tri par
    /// insertion : une table de tri ferait plus de mal que de bien a cette taille.
    void prepare( Local &l ) const {
        const int b0 = (int) G->adeb[ a0 ], b1 = (int) G->adeb[ a0 + 1 ];
        float d2[ MAX_ANNEAU1 ];
        l.na = 0;
        for ( int q = b0; q < b1 && l.na < MAX_ANNEAU1; ++q ) {
            const int b = (int) G->adj[ q ];
            if ( b == a0 ) continue;
            const int m = (int) G->median[ b ];
            const float ex = (float) G->Ppp[ 0 ][ m ] - x0, ey = (float) G->Ppp[ 1 ][ m ] - y0;
            float dd = ex * ex + ey * ey;
            int r = l.na++;
            while ( r > 0 && d2[ r - 1 ] > dd ) { d2[ r ] = d2[ r - 1 ]; l.ann[ r ] = l.ann[ r - 1 ]; --r; }
            d2[ r ] = dd; l.ann[ r ] = b;
        }
        l.lo = i0 - 1; l.hi = i0 + 1; l.cote = true;
        l.ia = 0; l.k = 0; l.fin = 0;
        l.amorce = true;
    }

    template<class Etat>
    bool suivant( const Etat &, Local &l, Plan &p ) {
        if ( ! l.amorce ) prepare( l );

        // ---- l'agregat, en S'ECARTANT de `i0` : approximativement du plus proche au plus loin
        const int m0 = (int) G->mdeb[ a0 ], m1 = (int) G->mdeb[ a0 + 1 ];
        for ( ;; ) {
            int j;
            if      ( l.cote && l.hi < m1 ) { l.cote = false; j = l.hi++; }
            else if ( l.lo >= m0 )          { l.cote = true;  j = l.lo--; }
            else if ( l.hi < m1 )           {                 j = l.hi++; }
            else break;
            plan_vers( j, p );
            return true;
        }

        // ---- puis l'anneau 1, agregat par agregat, du plus proche au plus lointain
        for ( ;; ) {
            if ( l.k < l.fin ) { plan_vers( l.k++, p ); return true; }
            if ( l.ia >= l.na ) return false;
            const int b = l.ann[ l.ia++ ];
            l.k = (int) G->mdeb[ b ]; l.fin = (int) G->mdeb[ b + 1 ];
        }
    }
};

} // namespace noyau2d

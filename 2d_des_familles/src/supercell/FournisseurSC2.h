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
//     R_A^2 le plus grand `|v - p_i|^2` sur ces memes sommets, `p_i` etant le germe du membre
//
// Un germe `q` d'une boite `B` ne peut couper aucune cellule de `A` si
//
//     dist^2( E_A, B )  >=  R_A^2
//
// -- c'est le critere du fournisseur BSP, transpose de la cellule a son enceinte. Il se teste une
// fois par agregat et sert a ses ~8 membres : c'est TOUTE l'amortisation de l'affaire.
//
// = POURQUOI LA TABLE NE GARDE QUE L'ANNEAU 2 ET AU-DELA
//
// L'anneau 1 est deja dans la cellule de phase 1. Le repasser serait un no-op paye plein tarif.
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

#include <immintrin.h>

namespace noyau2d {

/// L'enceinte d'un agregat : la boite de ses cellules de phase 1, et le rayon.
struct Enceinte { float lo[ 2 ], hi[ 2 ], r2; };

/// LA TABLE : pour chaque agregat, les agregats de l'anneau 2 et au-dela qui peuvent encore le
/// manger. En CSR, comme tout le reste ici.
///
/// Elle porte aussi la BOITE DES GERMES de chaque agregat et le majorant de leurs poids. La table
/// est partagee par les ~8 membres, donc grossiere par construction : c'est le filtre par CELLULE
/// ci-dessous qui la resserre, et il coute une operation SIMD par agregat de la ligne.
struct TableSC {
    std::vector<int>   deb, ag;
    std::vector<float> lo0, lo1, hi0, hi1;               ///< boite des germes, par agregat
    std::vector<float> wmax;                             ///< majorant CONSTANT des poids
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

    /// LE FILTRE PAR CELLULE, exact et vectoriel -- le meme argument que pour le BSP : si un germe
    /// `q` de la boite `B` coupe la cellule, il en retranche au moins un SOMMET, et ce sommet
    /// verifie `|v - q|^2 - w_q < |v - p0|^2 - w0`. Donc rejeter `B` des que TOUS les sommets ont
    /// `dist^2( v, B ) >= |v - p0|^2 - w0 + wmax_B`.
    ///
    /// C'est ce qui manquait a la premiere version : la table donne 37 agregats par ligne, soit
    /// ~300 germes proposes par cellule, la ou 26 suffisent. La table est partagee, donc lache ;
    /// le filtre la resserre contre CETTE cellule pour une operation SIMD par agregat.
    template<class Etat>
    bool peut_couper( int b, const Etat &e ) const {
        const float marge = POIDS ? T->wmax[ b ] - w0 : 0.f;
        if constexpr ( requires { e.vx + e.vx; } ) {
            const __m256 z = _mm256_setzero_ps();
            const __m256 l0 = _mm256_set1_ps( T->lo0[ b ] ), h0 = _mm256_set1_ps( T->hi0[ b ] );
            const __m256 l1 = _mm256_set1_ps( T->lo1[ b ] ), h1 = _mm256_set1_ps( T->hi1[ b ] );
            const __m256 ex = _mm256_max_ps( _mm256_max_ps( _mm256_sub_ps( l0, e.vx ),
                                                            _mm256_sub_ps( e.vx, h0 ) ), z );
            const __m256 ey = _mm256_max_ps( _mm256_max_ps( _mm256_sub_ps( l1, e.vy ),
                                                            _mm256_sub_ps( e.vy, h1 ) ), z );
            const __m256 d2 = _mm256_fmadd_ps( ex, ex, _mm256_mul_ps( ey, ey ) );
            const __m256 rx = _mm256_sub_ps( e.vx, _mm256_set1_ps( x0 ) );
            const __m256 ry = _mm256_sub_ps( e.vy, _mm256_set1_ps( y0 ) );
            __m256 r2 = _mm256_fmadd_ps( rx, rx, _mm256_mul_ps( ry, ry ) );
            if constexpr ( POIDS ) r2 = _mm256_add_ps( r2, _mm256_set1_ps( marge ) );
            return ( _mm256_cmp_ps_mask( d2, r2, _CMP_LT_OQ )
                     & ( ( 1u << Etat::nb ) - 1 ) ) != 0;
        } else {
            for ( int i = 0; i < e.nb; ++i ) {
                const float vx = e.vx[ i ], vy = e.vy[ i ];
                const float tx = vx < T->lo0[b] ? T->lo0[b] - vx : ( vx > T->hi0[b] ? vx - T->hi0[b] : 0.f );
                const float ty = vy < T->lo1[b] ? T->lo1[b] - vy : ( vy > T->hi1[b] ? vy - T->hi1[b] : 0.f );
                const float rx = vx - x0, ry = vy - y0;
                if ( tx * tx + ty * ty < rx * rx + ry * ry + marge ) return true;
            }
            return false;
        }
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

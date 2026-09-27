#pragma once

// =====================================================================================
// LE TEMOIN : tous les autres germes, dans l'ordre, sans aucun elagage. En `O( n )` par cellule,
// donc `O( n^2 )` en tout -- il ne sert pas a aller vite, il sert a VERIFIER l'elagage : meme
// noyau, memes plans, seul le fournisseur change. Si le majorant affine se trompe d'un signe, les
// cellules divergent, et c'est la seule facon de le voir.
//
// « MEMES PLANS » EST A PRENDRE AU PIED DE LA LETTRE : le temoin passe par `bissect2` / `bissect3`
// comme le fournisseur BSP, donc ses entrees sont en `TF` et non en `TK`. Sinon, en `--kernel
// float`, le temoin serait QUATRE CHIFFRES moins precis que ce qu'il teste et `check` accuserait
// l'elagage de ses propres arrondis.
// =====================================================================================

#include "cell/Contrat2D.h"
#include "cell/Contrat3D.h"
#include "cell/Plan.h"

namespace sf {

/// 2D. `px / py / pw` dans l'ordre de l'arbre, `ids` les identifiants ; `sens = -1` deroule a
/// l'envers -- meme resultat mathematique, autre ordre de coupes, ce qui calibre le desaccord du
/// au flottant.
template<class TK, bool POIDS>
struct Balayage2 {
    const TF *px, *py, *pw;
    const d2::SI32 *ids;
    int n, k = 0, sens = 1;
    TF  x0, y0, w0;
    d2::SI32 i0;

    void origine( TF &x, TF &y ) const { x = x0; y = y0; }

    template<class Etat>
    bool suivant( const Etat &, d2::RienDeLocal &, d2::Plan2<TK> &p ) {
        for ( ;; ) {
            if ( k >= n ) return false;
            const int j = sens > 0 ? k++ : n - 1 - k++;
            if ( ids[ j ] == i0 ) continue;
            bissect2<POIDS>( p, x0, y0, w0, px[ j ], py[ j ], POIDS ? pw[ j ] : TF( 0 ), ids[ j ] );
            return true;
        }
    }
};

/// 3D, le meme.
template<class TK, bool POIDS>
struct Balayage3 {
    const TF *px, *py, *pz, *pw;
    const d3::SI32 *ids;
    int n, k = 0, sens = 1;
    TF  x0, y0, z0, w0;
    d3::SI32 i0;

    void origine( TF &x, TF &y, TF &z ) const { x = x0; y = y0; z = z0; }

    template<class Etat>
    bool suivant( const Etat &, d3::RienDeLocal &, d3::Plan3<TK> &p ) {
        for ( ;; ) {
            if ( k >= n ) return false;
            const int j = sens > 0 ? k++ : n - 1 - k++;
            if ( ids[ j ] == i0 ) continue;
            bissect3<POIDS>( p, x0, y0, z0, w0, px[ j ], py[ j ], pz[ j ],
                             POIDS ? pw[ j ] : TF( 0 ), ids[ j ] );
            return true;
        }
    }
};

} // namespace sf

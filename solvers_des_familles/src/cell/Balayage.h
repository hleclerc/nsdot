#pragma once

// =====================================================================================
// LE TEMOIN : tous les autres germes, dans l'ordre, sans aucun elagage. En `O( n )` par cellule,
// donc `O( n^2 )` en tout -- il ne sert pas a aller vite, il sert a VERIFIER l'elagage : meme
// noyau, memes plans, seul le fournisseur change. Si le majorant affine se trompe d'un signe, les
// cellules divergent, et c'est la seule facon de le voir.
// =====================================================================================

#include "cell/Contrat2D.h"
#include "cell/Contrat3D.h"

namespace sf {

/// 2D. `px / py / pw` dans l'ordre de l'arbre, `ids` les identifiants ; `sens = -1` deroule a
/// l'envers -- meme resultat mathematique, autre ordre de coupes, ce qui calibre le desaccord du
/// au flottant.
template<class TK, bool POIDS>
struct Balayage2 {
    const TK *px, *py, *pw;
    const d2::SI32 *ids;
    int n, k = 0, sens = 1;
    TK  x0, y0, w0;
    d2::SI32 i0;

    template<class Etat>
    bool suivant( const Etat &, d2::RienDeLocal &, d2::Plan2<TK> &p ) {
        for ( ;; ) {
            if ( k >= n ) return false;
            const int j = sens > 0 ? k++ : n - 1 - k++;
            if ( ids[ j ] == i0 ) continue;
            const TK xj = px[ j ], yj = py[ j ];
            p.dx = xj - x0; p.dy = yj - y0;
            p.off = TK( 0.5 ) * ( p.dx * ( xj + x0 ) + p.dy * ( yj + y0 ) );
            if constexpr ( POIDS ) p.off += TK( 0.5 ) * ( w0 - pw[ j ] );
            p.id = ids[ j ];
            return true;
        }
    }
};

/// 3D, le meme.
template<class TK, bool POIDS>
struct Balayage3 {
    const TK *px, *py, *pz, *pw;
    const d3::SI32 *ids;
    int n, k = 0, sens = 1;
    TK  x0, y0, z0, w0;
    d3::SI32 i0;

    template<class Etat>
    bool suivant( const Etat &, d3::RienDeLocal &, d3::Plan3<TK> &p ) {
        for ( ;; ) {
            if ( k >= n ) return false;
            const int j = sens > 0 ? k++ : n - 1 - k++;
            if ( ids[ j ] == i0 ) continue;
            const TK xj = px[ j ], yj = py[ j ], zj = pz[ j ];
            p.dx = xj - x0; p.dy = yj - y0; p.dz = zj - z0;
            p.off = TK( 0.5 ) * ( p.dx * ( xj + x0 ) + p.dy * ( yj + y0 ) + p.dz * ( zj + z0 ) );
            if constexpr ( POIDS ) p.off += TK( 0.5 ) * ( w0 - pw[ j ] );
            p.id = ids[ j ];
            return true;
        }
    }
};

} // namespace sf

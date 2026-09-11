// « QUEL SITE EST LE PLUS PROCHE ? » -- une grille reguliere sur les sites, et rien d'autre.
//
// C'est le seul service dont l'affectation des diracs a besoin, et il ne merite pas un arbre : les
// sites sont `n / rho` points a peu pres uniformes, donc une grille a ~1 site par case repond en
// une coque de rayon 1 presque toujours. Le critere d'arret est EXACT (voir `plus_proche`), ce
// n'est pas une heuristique de voisinage.

#pragma once

#include "util/common.h"
#include <algorithm>
#include <cmath>
#include <vector>

namespace pd::supercell {

// ------------------------------------------------------------- LE PLUS PROCHE SITE

/// Une grille reguliere sur les SITES, pour l'affectation. On cherche le plus proche au sens
/// euclidien (voir l'en-tete : l'agregation est libre), par coques de rayon croissant.
///
/// LE CRITERE D'ARRET est celui qui rend la reponse EXACTE et non approchee : une fois toutes les
/// cellules de rayon de Tchebychev `<= r` examinees, tout point non examine est a distance L-infini
/// au moins `r * h` du point interroge, ou `h` est la plus petite arete de cellule. On s'arrete
/// donc des que la meilleure distance trouvee descend sous `r * h`, et pas avant.
template<int D>
struct GrilleSites {
    Vec<D>  lo, inv;
    int     res[ D ] = {};
    TF      hmin = 0;
    std::vector<SI> deb, cont;                          ///< tri par comptage : debuts et contenu
    const TF *const *P = nullptr;
    const SI *sites = nullptr;

    SI cellule( Vec<D> x, int *c ) const {
        SI k = 0;
        for ( int d = 0; d < D; ++d ) {
            int i = int( ( x[ d ] - lo[ d ] ) * inv[ d ] );
            c[ d ] = i < 0 ? 0 : ( i >= res[ d ] ? res[ d ] - 1 : i );
            k = k * res[ d ] + c[ d ];
        }
        return k;
    }

    void build( const TF *const *Pp, const SI *st, SI ns ) {
        P = Pp; sites = st;
        for ( int d = 0; d < D; ++d ) { lo[ d ] = P[ d ][ st[ 0 ] ]; }
        Vec<D> hi = lo;
        for ( SI i = 1; i < ns; ++i )
            for ( int d = 0; d < D; ++d ) {
                lo[ d ] = std::min( lo[ d ], P[ d ][ st[ i ] ] );
                hi[ d ] = std::max( hi[ d ], P[ d ][ st[ i ] ] );
            }
        // ~1 site par cellule : c'est le reglage qui minimise le travail total, une coque de
        // rayon 1 suffisant alors presque toujours.
        double vol = 1;
        for ( int d = 0; d < D; ++d ) vol *= std::max( 1e-12, double( hi[ d ] - lo[ d ] ) );
        const double h = std::pow( vol / std::max( SI( 1 ), ns ), 1.0 / D );
        SI nc = 1;
        hmin = TF( 1e30 );
        for ( int d = 0; d < D; ++d ) {
            const double e = std::max( 1e-12, double( hi[ d ] - lo[ d ] ) );
            res[ d ] = std::max( 1, std::min( 1024, int( e / h ) ) );
            inv[ d ] = TF( res[ d ] / ( e * ( 1 + 1e-12 ) ) );
            hmin = std::min( hmin, TF( e / res[ d ] ) );
            nc *= res[ d ];
        }

        std::vector<SI> cnt( nc + 1, 0 );
        std::vector<SI> ci( ns );
        int c[ D ];
        for ( SI i = 0; i < ns; ++i ) {
            Vec<D> x;
            for ( int d = 0; d < D; ++d ) x[ d ] = P[ d ][ st[ i ] ];
            ci[ i ] = cellule( x, c );
            ++cnt[ ci[ i ] + 1 ];
        }
        for ( SI k = 0; k < nc; ++k ) cnt[ k + 1 ] += cnt[ k ];
        deb = cnt;
        cont.resize( ns );
        std::vector<SI> pos( cnt.begin(), cnt.end() - 1 );
        for ( SI i = 0; i < ns; ++i ) cont[ pos[ ci[ i ] ]++ ] = i;
    }

    /// rend l'indice du site le plus proche DANS `sites` (et non l'indice du germe).
    SI plus_proche( Vec<D> x ) const {
        int c[ D ];
        cellule( x, c );
        SI best = -1;
        TF bd = TF( 1e300 );
        int rmax = 0;
        for ( int d = 0; d < D; ++d ) rmax = std::max( rmax, res[ d ] );
        for ( int r = 0; r <= rmax; ++r ) {
            // la COQUE de rayon `r` : le pave `[ c - r, c + r ]` prive de son interieur.
            int n_side = 2 * r + 1;
            SI tot = 1;
            for ( int d = 0; d < D; ++d ) tot *= n_side;
            for ( SI t = 0; t < tot; ++t ) {
                SI q = t;
                int idx[ D ];
                bool bord = false, dehors = false;
                for ( int d = D - 1; d >= 0; --d ) {
                    const int o = int( q % n_side ) - r;
                    q /= n_side;
                    if ( o == -r || o == r ) bord = true;
                    idx[ d ] = c[ d ] + o;
                    if ( idx[ d ] < 0 || idx[ d ] >= res[ d ] ) dehors = true;
                }
                if ( dehors || ( r > 0 && ! bord ) )
                    continue;
                SI k = 0;
                for ( int d = 0; d < D; ++d ) k = k * res[ d ] + idx[ d ];
                for ( SI u = deb[ k ]; u < deb[ k + 1 ]; ++u ) {
                    const SI i = cont[ u ];
                    Vec<D> y;
                    for ( int d = 0; d < D; ++d ) y[ d ] = P[ d ][ sites[ i ] ];
                    const TF dd = dist2( x, y );
                    if ( dd < bd ) { bd = dd; best = i; }
                }
            }
            // le critere d'arret exact : voir l'en-tete de la structure.
            if ( best >= 0 && bd <= TF( r ) * hmin * TF( r ) * hmin )
                break;
        }
        return best;
    }
};

} // namespace pd::supercell

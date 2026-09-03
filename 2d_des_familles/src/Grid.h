#pragma once

#include "WeightMajorant.h"
#include "common.h"
#include <cmath>
#include <vector>

namespace pd2d {

/// UNE GRILLE REGULIERE sur le carre unite, avec les germes RANGES PAR CASE.
///
/// Ce qu'elle est ici : un moyen d'amorcer la cellule sans DESCENDRE. Le germe est dans une case
/// qu'on trouve par deux divisions ; ses voisins immediats sont les huit cases autour, en acces
/// direct. Aucune boite a tester, aucun chemin racine -> feuille a parcourir -- et c'est ce chemin
/// qui, dans un arbre, coute une quinzaine de noeuds par cellule.
///
/// Ce qu'elle n'est PAS : une structure pour un nuage quelconque. Une grille suppose une densite a
/// peu pres uniforme. MESURE sur les nuages de `cases/` : elle prend 4.4x sur un nuage regroupe
/// autour de lignes, et 96x quand les poids eloignent les cellules de leurs germes, la ou l'arbre
/// ne bouge pas d'un pour cent sur le premier.
struct Grid {
    struct Seed {
        TF x, y;
        TF w;                   ///< le poids
        SI id, _pad;
    };

    SI g = 1;                   ///< resolution : `g x g` cases
    TF h = 1;                   ///< le pas, `1 / g`
    std::vector<SI> beg;        ///< `g*g + 1` offsets (CSR) dans `pts`
    std::vector<Seed> pts;      ///< les germes, GROUPES par case et contigus
    std::vector<WMaj> cwm;      ///< par case : le MAJORANT AFFINE de ses poids
    TF wmax_all = 0;            ///< le majorant CONSTANT global, pour le critere d'arret des anneaux

    static constexpr const char *name = "grid";

    // la boucle EXTERNE veut un acces sequentiel : elle lit `pts` dans l'ordre, ce qui la fait
    // avancer case par case, donc en voisinage spatial.
    SI nb_seeds() const { return SI( pts.size() ); }
    TF seed_x( SI k ) const { return pts[ k ].x; }
    TF seed_y( SI k ) const { return pts[ k ].y; }
    TF seed_w( SI k ) const { return pts[ k ].w; }
    SI seed_id( SI k ) const { return pts[ k ].id; }

    /// LES CANDIDATS : amorcer, puis completer.
    ///
    /// 1. Les 3x3 cases autour du germe, en acces direct et SANS test d'eviction -- une case
    ///    voisine est presque toujours utile, et la tester couterait plus que de la couper.
    /// 2. Puis les anneaux `r = 2, 3, ...`, chacun teste case par case CONTRE LA CELLULE DEJA
    ///    PETITE.
    ///
    /// = Ce qui rend la seconde passe exacte, et courte
    ///
    /// Un germe `y` ne peut couper que s'il existe `p` dans la cellule avec
    /// `|p - y|^2 - w_y <= |p - p0|^2 - w0`. Avec `d = |y - p0|` et `R = max |p - p0|` sur la
    /// cellule, `|p - y| >= d - R`, donc la condition impose `d^2 - 2 d R + w0 - w_y <= 0` :
    /// **des que `d >= R` et `d^2 - 2 d R + w0 - wm > 0`, plus rien ne peut couper**. Sans poids
    /// c'est le `d > 2R` habituel ; avec eux, oublier le terme de poids rend des cellules trop
    /// grandes SANS QUE RIEN NE LE SIGNALE (mesure : 3.9e-05 d'erreur a `--weights 100`).
    ///
    /// Et `R` RETRECIT a chaque coupe : la premiere passe fait exactement ce qu'il faut pour que la
    /// seconde s'arrete tout de suite.
    template<class MayCut, class CutWith, class Reach2>
    void for_each_candidate( SI k0, MayCut &&may_cut, CutWith &&cut_with, Reach2 &&reach2 ) const {
        const TF p0x = pts[ k0 ].x, p0y = pts[ k0 ].y, w0 = pts[ k0 ].w;
        const SI i0 = pts[ k0 ].id;
        SI cx = SI( p0x * g ), cy = SI( p0y * g );
        cx = cx < 0 ? 0 : ( cx >= g ? g - 1 : cx );
        cy = cy < 0 ? 0 : ( cy >= g ? g - 1 : cy );

        // ---- 1. le voisinage immediat, sans rien tester
        for ( SI dy = -1; dy <= 1; ++dy ) {
            const SI y = cy + dy;
            if ( y < 0 || y >= g )
                continue;
            for ( SI dx = -1; dx <= 1; ++dx ) {
                const SI x = cx + dx;
                if ( x < 0 || x >= g )
                    continue;
                if ( ! emit_cell( x, y, i0, cut_with ) )
                    return;
            }
        }

        // ---- 2. les anneaux, contre une cellule deja petite
        for ( SI r = 2; ; ++r ) {
            if ( cx - r < 0 && cx + r >= g && cy - r < 0 && cy + r >= g )
                return;                                 // l'anneau est entierement hors grille

            // `d^2 - 2 d R + w0 - wm > 0` mais SANS la racine : on l'ecrit `A > 2 d R` avec
            // `A = d^2 + w0 - wm`, et comme les deux membres sont positifs quand `A > 0`, on eleve
            // au carre. Une racine par anneau, ce n'est pas rien quand l'anneau 2 est le seul
            // qu'on teste.
            //
            // `wmax_all` est un majorant GLOBAL, donc lache : un seul germe tres lourd force a
            // elargir partout. Le resserrer demanderait une hierarchie de maxima -- ce que l'arbre
            // a par construction, et c'est une limite de plus au passif de la grille.
            const TF d = ( r - 1 ) * h;                 // distance MINIMALE du germe a l'anneau
            const TF d2 = d * d, r2 = reach2();
            const TF A = d2 + w0 - wmax_all;
            if ( d2 >= r2 && A > 0 && A * A > 4 * d2 * r2 )
                return;                                 // plus rien ne peut couper, POIDS COMPRIS

            for ( SI y = cy - r; y <= cy + r; ++y ) {
                if ( y < 0 || y >= g )
                    continue;
                const bool edge_row = ( y == cy - r || y == cy + r );
                for ( SI x = cx - r; x <= cx + r; ++x ) {
                    if ( x < 0 || x >= g )
                        continue;
                    if ( ! edge_row && x != cx - r && x != cx + r )
                        continue;                       // l'interieur a deja ete vu
                    const TF lox = x * h, loy = y * h;
                    if ( ! may_cut( lox, loy, lox + h, loy + h, cwm[ y * g + x ] ) )
                        continue;
                    if ( ! emit_cell( x, y, i0, cut_with ) )
                        return;
                }
            }
        }
    }

    /// `leaf` est ici le nombre VISE de germes par case : la resolution en decoule.
    void build( const TF *X, const TF *Y, const TF *W, SI n, SI leaf ) {
        g = SI( std::sqrt( double( n ) / double( leaf > 0 ? leaf : 1 ) ) );
        g = g < 1 ? 1 : g;
        h = TF( 1 ) / g;

        // un tri par COMPTAGE : deux passes, pas de comparaison, et les germes finissent groupes
        // par case ET contigus -- c'est la meme idee que les feuilles empaquetees de `AaBspPacked`.
        beg.assign( size_t( g ) * g + 1, 0 );
        for ( SI i = 0; i < n; ++i )
            ++beg[ cell_of( X[ i ], Y[ i ] ) + 1 ];
        for ( size_t c = 1; c < beg.size(); ++c )
            beg[ c ] += beg[ c - 1 ];

        pts.resize( n );
        std::vector<SI> at( beg.begin(), beg.end() - 1 );
        for ( SI i = 0; i < n; ++i )
            pts[ at[ cell_of( X[ i ], Y[ i ] ) ]++ ] = Seed{ X[ i ], Y[ i ], W ? W[ i ] : TF( 0 ), i, 0 };

        cwm.assign( size_t( g ) * g, WMaj{} );
        wmax_all = 0;
        if ( W )
            for ( size_t c = 0; c + 1 < beg.size(); ++c ) {
                cwm[ c ] = weight_majorant( beg[ c ], beg[ c + 1 ], [ & ]( SI k, TF &x, TF &y, TF &w ) {
                    x = pts[ k ].x; y = pts[ k ].y; w = pts[ k ].w;
                } );
                for ( SI k = beg[ c ]; k < beg[ c + 1 ]; ++k )
                    wmax_all = pts[ k ].w > wmax_all ? pts[ k ].w : wmax_all;
            }
    }

private:
    SI cell_of( TF x, TF y ) const {
        SI cx = SI( x * g ), cy = SI( y * g );
        cx = cx < 0 ? 0 : ( cx >= g ? g - 1 : cx );
        cy = cy < 0 ? 0 : ( cy >= g ? g - 1 : cy );
        return cy * g + cx;
    }

    /// Les germes d'une case, remis a l'appelant. `false` = il veut s'arreter.
    template<class F>
    bool emit_cell( SI cx, SI cy, SI i0, F &&cut_with ) const {
        const SI c = cy * g + cx;
        for ( SI k = beg[ c ]; k < beg[ c + 1 ]; ++k )
            if ( pts[ k ].id != i0 && ! cut_with( pts[ k ].x, pts[ k ].y, pts[ k ].w, pts[ k ].id ) )
                return false;
        return true;
    }
};

} // namespace pd2d

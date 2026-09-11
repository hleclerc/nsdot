#pragma once

#include "geometry/WeightMajorant.h"
#include "util/common.h"
#include <cmath>
#include <vector>

namespace pd {

/// UNE GRILLE REGULIERE sur le cube unite, avec les germes RANGES PAR CASE.
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
///
/// = Ce que la dimension change, et c'est le point interessant
///
/// Rien dans le CRITERE d'arret des anneaux -- il ne parle que de distances. Mais un anneau de
/// rayon `r` compte `( 2r+1 )^D - ( 2r-1 )^D` cases, soit `8 r` en 2D et `24 r^2 + 2` en 3D : le
/// prix d'un anneau de trop est cubique la ou il etait quadratique. Une grille est donc STRICTEMENT
/// plus fragile en 3D, et pour une raison qui n'a rien a voir avec l'implementation.
template<int D>
struct GridT {
    static constexpr int dim = D;

    struct Seed {
        TF c[ D ];
        TF w;                   ///< le poids
        SI id;
    };

    SI g = 1;                   ///< resolution : `g^D` cases
    TF h = 1;                   ///< le pas, `1 / g`
    std::vector<SI> beg;        ///< `g^D + 1` offsets (CSR) dans `pts`
    std::vector<Seed> pts;      ///< les germes, GROUPES par case et contigus
    std::vector<WMajT<D>> cwm;  ///< par case : le MAJORANT AFFINE de ses poids
    TF wmax_all = 0;            ///< le majorant CONSTANT global, pour le critere d'arret des anneaux

    static constexpr const char *name = "grid";

    // la boucle EXTERNE veut un acces sequentiel : elle lit `pts` dans l'ordre, ce qui la fait
    // avancer case par case, donc en voisinage spatial.
    SI nb_seeds() const { return SI( pts.size() ); }
    Vec<D> seed( SI k ) const {
        if constexpr ( D == 2 )
            return { pts[ k ].c[ 0 ], pts[ k ].c[ 1 ] };
        else if constexpr ( D == 3 )
            return { pts[ k ].c[ 0 ], pts[ k ].c[ 1 ], pts[ k ].c[ 2 ] };
        else {
            Vec<D> r;
            for ( int d = 0; d < D; ++d ) r[ d ] = pts[ k ].c[ d ];
            return r;
        }
    }
    TF seed_c( SI k, int d ) const { return pts[ k ].c[ d ]; }
    TF seed_x( SI k ) const { return pts[ k ].c[ 0 ]; }
    TF seed_y( SI k ) const { return pts[ k ].c[ 1 ]; }
    TF seed_w( SI k ) const { return pts[ k ].w; }
    SI seed_id( SI k ) const { return pts[ k ].id; }

    /// LES CANDIDATS : amorcer, puis completer.
    ///
    /// 1. Les `3^D` cases autour du germe, en acces direct et SANS test d'eviction -- une case
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
        const Vec<D> p0 = seed( k0 );
        const TF w0 = pts[ k0 ].w;
        const SI i0 = pts[ k0 ].id;
        SI ctr[ D ];
        for ( int d = 0; d < D; ++d ) {
            SI c = SI( p0[ d ] * g );
            ctr[ d ] = c < 0 ? 0 : ( c >= g ? g - 1 : c );
        }

        // ---- 1. le voisinage immediat, sans rien tester
        SI at[ D ];
        if ( ! shell( ctr, 1, true, at, 0, i0, cut_with, may_cut, false ) )
            return;

        // ---- 2. les anneaux, contre une cellule deja petite
        for ( SI r = 2; ; ++r ) {
            bool dehors = true;
            for ( int d = 0; d < D; ++d )
                dehors &= ctr[ d ] - r < 0 && ctr[ d ] + r >= g;
            if ( dehors )
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

            if ( ! shell( ctr, r, false, at, 0, i0, cut_with, may_cut, true ) )
                return;
        }
    }

    /// `leaf` est ici le nombre VISE de germes par case : la resolution en decoule.
    void build( const TF *const *P, const TF *W, SI n, SI leaf ) {
        g = SI( std::pow( double( n ) / double( leaf > 0 ? leaf : 1 ), 1.0 / D ) );
        g = g < 1 ? 1 : g;
        h = TF( 1 ) / g;

        size_t nc = 1;
        for ( int d = 0; d < D; ++d ) nc *= size_t( g );

        // un tri par COMPTAGE : deux passes, pas de comparaison, et les germes finissent groupes
        // par case ET contigus -- c'est la meme idee que les feuilles empaquetees de `AaBspPacked`.
        beg.assign( nc + 1, 0 );
        for ( SI i = 0; i < n; ++i )
            ++beg[ cell_of( P, i ) + 1 ];
        for ( size_t c = 1; c < beg.size(); ++c )
            beg[ c ] += beg[ c - 1 ];

        pts.resize( n );
        std::vector<SI> at( beg.begin(), beg.end() - 1 );
        for ( SI i = 0; i < n; ++i ) {
            Seed s;
            for ( int d = 0; d < D; ++d ) s.c[ d ] = P[ d ][ i ];
            s.w = W ? W[ i ] : TF( 0 );
            s.id = i;
            pts[ at[ cell_of( P, i ) ]++ ] = s;
        }

        cwm.assign( nc, WMajT<D>{} );
        wmax_all = 0;
        if ( W )
            for ( size_t c = 0; c + 1 < beg.size(); ++c ) {
                cwm[ c ] = weight_majorant<D>( beg[ c ], beg[ c + 1 ], [ & ]( SI k, Vec<D> &y, TF &w ) {
                    for ( int d = 0; d < D; ++d ) y[ d ] = pts[ k ].c[ d ];
                    w = pts[ k ].w;
                } );
                for ( SI k = beg[ c ]; k < beg[ c + 1 ]; ++k )
                    wmax_all = pts[ k ].w > wmax_all ? pts[ k ].w : wmax_all;
            }
    }
    void build( const TF *X, const TF *Y, const TF *W, SI n, SI l ) requires ( D == 2 ) {
        const TF *P[ 2 ] = { X, Y };
        build( P, W, n, l );
    }
    void build( const TF *X, const TF *Y, const TF *Z, const TF *W, SI n, SI l ) requires ( D == 3 ) {
        const TF *P[ 3 ] = { X, Y, Z };
        build( P, W, n, l );
    }

private:
    SI cell_of( const TF *const *P, SI i ) const {
        SI r = 0;
        for ( int d = D - 1; d >= 0; --d ) {
            SI c = SI( P[ d ][ i ] * g );
            c = c < 0 ? 0 : ( c >= g ? g - 1 : c );
            r = r * g + c;
        }
        return r;
    }

    /// L'ANNEAU de rayon `r` autour de `ctr`, parcouru par un compteur odometrique sur les `D`
    /// axes. `plein` prend tout le bloc (c'est le voisinage immediat) ; sinon on ne garde que la
    /// COQUILLE, `max_d |x_d - ctr_d| == r`, l'interieur ayant deja ete vu.
    template<class CutWith, class MayCut>
    bool shell( const SI *ctr, SI r, bool plein, SI *at, int d, SI i0, CutWith &&cut_with,
                MayCut &&may_cut, bool teste ) const {
        if ( d == D ) {
            if ( ! plein ) {
                bool bord = false;
                for ( int e = 0; e < D; ++e )
                    bord |= at[ e ] == ctr[ e ] - r || at[ e ] == ctr[ e ] + r;
                if ( ! bord )
                    return true;
            }
            SI c = 0;
            for ( int e = D - 1; e >= 0; --e )
                c = c * g + at[ e ];
            if ( teste ) {
                Vec<D> lo, hi;
                for ( int e = 0; e < D; ++e ) { lo[ e ] = at[ e ] * h; hi[ e ] = lo[ e ] + h; }
                if ( ! may_cut( lo, hi, cwm[ c ] ) )
                    return true;
            }
            for ( SI k = beg[ c ]; k < beg[ c + 1 ]; ++k )
                if ( pts[ k ].id != i0 && ! cut_with( seed( k ), pts[ k ].w, pts[ k ].id ) )
                    return false;
            return true;
        }
        for ( SI v = ctr[ d ] - r; v <= ctr[ d ] + r; ++v ) {
            if ( v < 0 || v >= g )
                continue;
            at[ d ] = v;
            if ( ! shell( ctr, r, plein, at, d + 1, i0, cut_with, may_cut, teste ) )
                return false;
        }
        return true;
    }
};

using Grid  = GridT<2>;
using Grid3 = GridT<3>;

} // namespace pd

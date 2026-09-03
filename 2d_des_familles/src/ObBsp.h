#pragma once

#include "AaBsp.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace pd2d {

/// UN BSP NON ALIGNE SUR LES AXES : la mediane est prise le long d'une direction CHOISIE, pas
/// forcement `x` ou `y`. Les regions deviennent des polygones convexes quelconques.
///
/// = Ce qui reste aligne, et pourquoi
///
/// LE VOLUME ENGLOBANT. Le test d'eviction demande `min |p - y|^2` et `max_y( a . y )` sur la
/// region : sur une boite alignee les deux sont separables par axe (deux `clamp`, deux coins), sur
/// un polytope quelconque le point le plus proche est un probleme quadratique. On garde donc la
/// boite englobante des germes du sous-arbre -- exactement ce que `AaBsp` stocke -- et SEUL LE
/// PARTAGE change. Tout l'aval (`for_each_candidate`, `may_be_cut`, `nearness`) est inchange, donc
/// la mesure isole vraiment l'effet de l'orientation.
///
/// = Le piege, qui est la vraie question
///
/// Une coupe alignee laisse deux boites DISJOINTES le long de l'axe coupe. Une coupe oblique laisse
/// deux boites qui SE RECOUVRENT : couper un carre par sa diagonale donne deux triangles dont les
/// boites font chacune presque tout le carre. L'oblique ne peut donc gagner que si le nuage local
/// est lui-meme allonge dans une direction oblique -- et c'est mesurable.
///
/// = Comment la direction est choisie
///
/// Par un critere SAH sur SIX candidates -- `x`, `y`, les deux diagonales, l'axe principal du nuage
/// local et sa perpendiculaire -- notees par la somme des demi-perimetres des deux boites filles.
/// `x` et `y` etant dans le jeu, le critere ne peut pas etre battu par la regle « l'axe le plus
/// long » sur son propre terrain : si l'oblique perd, ce n'est pas faute d'avoir eu le choix.
///
/// L'evaluation se fait sur un ECHANTILLON de 256 germes -- six medianes et douze boites par noeud
/// sur la tranche entiere multiplieraient le cout de construction par six, pour departager des
/// candidates que 256 points separent deja tres bien. La coupe RETENUE, elle, est exacte.
struct ObBsp : AaBsp {
    static constexpr const char *name = "obsp";

    void build( const TF *X, const TF *Y, const TF *W, SI n, SI leaf ) {
        leaf_size = leaf;
        order.resize( n );
        for ( SI i = 0; i < n; ++i )
            order[ i ] = i;
        px.resize( n ); py.resize( n );
        if ( W )
            pw.resize( n );

        nodes.clear();
        nodes.reserve( 2 * ( n / std::max<SI>( leaf, 1 ) + 1 ) );
        for ( int c = 0; c < 6; ++c ) hist[ c ] = 0;
        _build_ob( X, Y, W, 0, n );
        // QUELLE direction a ete retenue, et combien de fois : sans ca on ne sait pas si un
        // resultat identique a `bsp` veut dire « l oblique ne sert a rien » ou « l oblique n a
        // jamais ete choisi ».
        if ( std::getenv( "PD2D_OBSP_DIRS" ) ) {
            SI t = 0;
            for ( int c = 0; c < 6; ++c ) t += hist[ c ];
            std::printf( "  obsp: coupes x %.0f%% y %.0f%% diag %.0f%%/%.0f%% propre %.0f%%/%.0f%%  (%d noeuds)\n",
                         100.0 * hist[ 0 ] / t, 100.0 * hist[ 1 ] / t, 100.0 * hist[ 2 ] / t,
                         100.0 * hist[ 3 ] / t, 100.0 * hist[ 4 ] / t, 100.0 * hist[ 5 ] / t,
                         int( t ) );
        }

        for ( SI k = 0; k < n; ++k ) {
            px[ k ] = X[ order[ k ] ];
            py[ k ] = Y[ order[ k ] ];
            if ( W )
                pw[ k ] = W[ order[ k ] ];
        }
    }

private:
    mutable SI hist[ 6 ] = { 0, 0, 0, 0, 0, 0 };   ///< la direction retenue, par candidate
    static constexpr SI ech = 256;          ///< la taille de l'echantillon de notation

    /// La direction de coupe. Rend `false` si aucune candidate ne separe quoi que ce soit.
    bool choose( const TF *X, const TF *Y, SI beg, SI end, TF &ux, TF &uy ) const {
        TF bx[ ech ], by[ ech ];
        const SI m = end - beg;
        const SI step = m > ech ? m / ech : 1;
        SI ns = 0;
        for ( SI k = beg; k < end && ns < ech; k += step ) {
            bx[ ns ] = X[ order[ k ] ];
            by[ ns ] = Y[ order[ k ] ];
            ++ns;
        }
        if ( ns < 4 )
            return false;

        // L'AXE PRINCIPAL de l'echantillon : la covariance est 2x2, donc son vecteur propre
        // dominant s'ecrit sans iteration, `theta = atan2( 2 sxy, sxx - syy ) / 2`.
        TF mx = 0, my = 0;
        for ( SI i = 0; i < ns; ++i ) { mx += bx[ i ]; my += by[ i ]; }
        mx /= ns; my /= ns;
        TF sxx = 0, sxy = 0, syy = 0;
        for ( SI i = 0; i < ns; ++i ) {
            const TF qx = bx[ i ] - mx, qy = by[ i ] - my;
            sxx += qx * qx; sxy += qx * qy; syy += qy * qy;
        }
        const TF th = TF( 0.5 ) * std::atan2( 2 * sxy, sxx - syy );
        const TF cx = std::cos( th ), cy = std::sin( th );
        const TF r2 = TF( 0.7071067811865476 );
        const TF dx[ 6 ] = { 1,  0,  r2,  r2,  cx, -cy };
        const TF dy[ 6 ] = { 0,  1,  r2, -r2,  cy,  cx };

        TF best = 0;
        int bc = 0;
        bool got = false;
        // `PD2D_OBSP_AA` : les deux candidates ALIGNEES seulement. L arbre redevient alors
        // celui de `bsp` (au choix du critere pres), ce qui mesure ce que coute le fait
        // d etre une AUTRE instanciation du meme code -- disposition, alignement -- et non
        // l obliquite.
        static const int nc_dir = std::getenv( "PD2D_OBSP_AA" ) ? 2 : 6;
        for ( int c = 0; c < nc_dir; ++c ) {
            TF t[ ech ], s[ ech ];
            for ( SI i = 0; i < ns; ++i )
                t[ i ] = s[ i ] = dx[ c ] * bx[ i ] + dy[ c ] * by[ i ];
            const SI h = ns / 2;
            std::nth_element( s, s + h, s + ns );
            const TF med = s[ h ];

            // les deux boites, notees par leur DEMI-PERIMETRE : l'aire tomberait a zero des que
            // les germes sont alignes, et departagerait alors n'importe quoi.
            TF lo[ 2 ][ 2 ], hi[ 2 ][ 2 ];
            SI cnt[ 2 ] = { 0, 0 };
            for ( SI i = 0; i < ns; ++i ) {
                const int e = t[ i ] < med ? 0 : 1;
                if ( cnt[ e ]++ == 0 ) {
                    lo[ e ][ 0 ] = hi[ e ][ 0 ] = bx[ i ];
                    lo[ e ][ 1 ] = hi[ e ][ 1 ] = by[ i ];
                } else {
                    lo[ e ][ 0 ] = std::min( lo[ e ][ 0 ], bx[ i ] );
                    hi[ e ][ 0 ] = std::max( hi[ e ][ 0 ], bx[ i ] );
                    lo[ e ][ 1 ] = std::min( lo[ e ][ 1 ], by[ i ] );
                    hi[ e ][ 1 ] = std::max( hi[ e ][ 1 ], by[ i ] );
                }
            }
            if ( ! cnt[ 0 ] || ! cnt[ 1 ] )         // la direction ne separe rien
                continue;
            const TF sc = ( hi[ 0 ][ 0 ] - lo[ 0 ][ 0 ] ) + ( hi[ 0 ][ 1 ] - lo[ 0 ][ 1 ] )
                        + ( hi[ 1 ][ 0 ] - lo[ 1 ][ 0 ] ) + ( hi[ 1 ][ 1 ] - lo[ 1 ][ 1 ] );
            if ( ! got || sc < best ) { best = sc; ux = dx[ c ]; uy = dy[ c ]; bc = c; got = true; }
        }
        if ( got ) ++hist[ bc ];
        return got;
    }

    /// Le meme squelette que `AaBsp::_build` -- preordre, boite, majorant, mediane -- avec la
    /// direction en plus.
    SI _build_ob( const TF *X, const TF *Y, const TF *W, SI beg, SI end ) {
        const SI me = SI( nodes.size() );
        nodes.push_back( Node{} );

        TF lox = X[ order[ beg ] ], hix = lox, loy = Y[ order[ beg ] ], hiy = loy;
        for ( SI k = beg + 1; k < end; ++k ) {
            const TF x = X[ order[ k ] ], y = Y[ order[ k ] ];
            lox = x < lox ? x : lox;  hix = x > hix ? x : hix;
            loy = y < loy ? y : loy;  hiy = y > hiy ? y : hiy;
        }
        nodes[ me ].lo[ 0 ] = lox; nodes[ me ].lo[ 1 ] = loy;
        nodes[ me ].hi[ 0 ] = hix; nodes[ me ].hi[ 1 ] = hiy;
        nodes[ me ].beg = beg;
        nodes[ me ].end = end;

        if ( W )
            nodes[ me ].wm = weight_majorant( beg, end, [ & ]( SI k, TF &x, TF &y, TF &w ) {
                const SI i = order[ k ];
                x = X[ i ]; y = Y[ i ]; w = W[ i ];
            } );

        TF ux = 1, uy = 0;
        if ( end - beg <= leaf_size || ! ( hix - lox > 0 || hiy - loy > 0 )
             || ! choose( X, Y, beg, end, ux, uy ) ) {
            nodes[ me ].right = -1;
            return me;
        }

        const SI mid = beg + ( end - beg ) / 2;
        std::nth_element( order.begin() + beg, order.begin() + mid, order.begin() + end,
                          [ & ]( SI p, SI q ) {
                              return ux * X[ p ] + uy * Y[ p ] < ux * X[ q ] + uy * Y[ q ];
                          } );

        _build_ob( X, Y, W, beg, mid );                 // == me + 1
        nodes[ me ].right = _build_ob( X, Y, W, mid, end );
        return me;
    }
};

} // namespace pd2d

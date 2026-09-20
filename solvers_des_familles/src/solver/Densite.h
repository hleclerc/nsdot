#pragma once

// =====================================================================================
// UNE DENSITE SUR LE CARRE : un plancher uniforme plus une somme de gaussiennes isotropes,
//
//     rho( x ) = plancher + sum_k m_k G_k( x ),   G_k( x ) = exp( -|x - c_k|^2 / 2 s_k^2 ) / ( 2 pi s_k^2 )
//
// et la CONVOLUTION par une gaussienne de largeur `s` : les gaussiennes s'elargissent en
// `s_k' = sqrt( s_k^2 + s^2 )`, rien d'autre ne change ( le plancher est invariant ). Le support est
// le carre, avant comme apres : les cellules sont deja coupees par lui.
//
// = La masse d'une cellule SANS quadrature de surface
//
// Le flux `F = f( r ) ( x - c )` avec `f( r ) = m ( 1 - exp( -r^2 / 2 s^2 ) ) / ( 2 pi r^2 )` verifie
// `div F = m G` : la masse d'un polygone est la circulation de `F . n` sur son bord, et le long
// d'une arete `( x - c ) . n = d` est CONSTANT ( la distance signee du centre a la droite ). Reste
// une integrale 1D par arete et par gaussienne,
//
//     masse = sum_aretes  d  int_t0^t1  m ( 1 - exp( -( d^2 + t^2 ) / 2 s^2 ) ) / ( 2 pi ( d^2 + t^2 ) ) dt
//
// dont l'integrande est LISSE ( c'est `-expm1`, pas `1 - exp` ) : a l'echelle `max( |d|, s )` pres
// du pied de la perpendiculaire, puis `1 / t^2`. Des morceaux geometriques depuis le pied, huit
// points de Gauss chacun, et la ou l'exponentielle est negligeable ( `exp( -seuil )` : c'est
// "l'amplitude en deca de laquelle on ne calcule pas" ) la forme close `atan`. Une cellule loin
// de toute gaussienne ne coute que des `atan`, et sa masse sort a l'arrondi pres de zero.
//
// La FACETTE est en forme close ( `erf` ), et la DERIVEE de la masse par rapport a `s` aussi :
//
//     d masse / d s_k = -( m / ( 2 s_k^2 sqrt( 2 pi ) ) ) sum_aretes  d  exp( -d^2 / 2 s_k^2 ) [ erf ]
//
// puisque `d f / d s = -m exp( -r^2 / 2 s^2 ) / ( 2 pi s^3 )` -- et `d s_k' / d s = s / s_k'`.
//
// 2D seulement.
// =====================================================================================

#include "util/common.h"
#include <algorithm>
#include <cmath>
#include <vector>

namespace sf {

struct Gaussienne {
    TF cx, cy;        ///< le centre
    TF sigma;         ///< la largeur ( avant convolution )
    TF masse;         ///< la masse sur le plan entier
};

struct Densite {
    std::vector<Gaussienne> g;
    TF plancher = 0;          ///< la densite uniforme ajoutee
    TF s        = 0;          ///< la largeur de la convolution ( 0 : la densite elle-meme )
    TF seuil    = 40;         ///< `exp( -seuil )` : en deca, on ne calcule pas

    TF sigma_eff( SI k ) const { return std::sqrt( g[ k ].sigma * g[ k ].sigma + s * s ); }

    /// la densite en un point
    TF rho( TF x, TF y ) const {
        TF r = plancher;
        for ( SI k = 0; k < SI( g.size() ); ++k ) {
            const TF sig = sigma_eff( k ), ex = x - g[ k ].cx, ey = y - g[ k ].cy;
            const TF q = ( ex * ex + ey * ey ) / ( 2 * sig * sig );
            if ( q < seuil ) r += g[ k ].masse * std::exp( -q ) / ( 2 * M_PI * sig * sig );
        }
        return r;
    }

    TF max_rho() const {
        TF m = plancher;
        for ( SI k = 0; k < SI( g.size() ); ++k ) m = std::max( m, rho( g[ k ].cx, g[ k ].cy ) );
        return m;
    }

    /// LA MASSE d'une cellule 2D ( `cel.nb`, `cel.vx`, `cel.vy`, `cel.cid` ), ses facettes contre
    /// les autres germes ( `facette( j, masse_de_la_facette )` ), et si `dds` n'est pas nul la derivee
    /// de la masse par rapport a `s`.
    template<class Cel, class Facette>
    TF mesure( const Cel &cel, Facette &&facette, TF *dds = nullptr ) const {
        const int nb = cel.nb;
        if ( nb <= 0 ) { if ( dds ) *dds = 0; return 0; }

        TF a2 = 0;                                       // l'aire signee, deux fois : l'orientation
        for ( int i = 0, j = nb - 1; i < nb; j = i++ )
            a2 += TF( cel.vx[ j ] ) * TF( cel.vy[ i ] ) - TF( cel.vx[ i ] ) * TF( cel.vy[ j ] );
        const TF sgn = a2 >= 0 ? 1 : -1;

        TF masse = plancher * TF( 0.5 ) * std::fabs( a2 ), dm = 0;
        for ( int i = 0, j = nb - 1; i < nb; j = i++ ) {  // l'arete [ v_j, v_i ], portee par `cid[ j ]`
            const TF xj = TF( cel.vx[ j ] ), yj = TF( cel.vy[ j ] );
            const TF ex = TF( cel.vx[ i ] ) - xj, ey = TF( cel.vy[ i ] ) - yj;
            const TF L = std::sqrt( ex * ex + ey * ey );
            if ( ! ( L > 0 ) ) continue;
            const TF ux = ex / L, uy = ey / L;           // la tangente, et la normale SORTANTE
            const TF nx = sgn * uy, ny = -sgn * ux;
            TF fac = plancher * L;
            for ( SI k = 0; k < SI( g.size() ); ++k ) {
                const TF sig = sigma_eff( k ), m = g[ k ].masse;
                const TF px = xj - g[ k ].cx, py = yj - g[ k ].cy;
                const TF d = px * nx + py * ny;          // > 0 : le centre du cote interieur
                const TF t0 = px * ux + py * uy, t1 = t0 + L;
                // la facette, et la derivee : la meme exponentielle, le meme erf
                const TF qd = d * d / ( 2 * sig * sig );
                if ( qd < seuil ) {
                    const TF e = std::exp( -qd );
                    const TF E = std::erf( t1 / ( sig * M_SQRT2 ) ) - std::erf( t0 / ( sig * M_SQRT2 ) );
                    fac += m * e * E / ( 2 * sig * std::sqrt( 2 * M_PI ) );
                    if ( dds && s > 0 )
                        dm += ( s / sig ) * ( -m / ( 2 * sig * sig * std::sqrt( 2 * M_PI ) ) ) * d * e * E;
                }
                masse += m * circulation( d, t0, t1, sig );
            }
            if ( cel.cid[ j ] >= 0 ) facette( cel.cid[ j ], fac );
        }
        if ( dds ) *dds = dm;
        return masse;
    }

    /// `d int_t0^t1 ( 1 - exp( -( d^2 + t^2 ) / 2 s^2 ) ) / ( 2 pi ( d^2 + t^2 ) ) dt`, le `d` compris
    /// ( la forme close `atan` le porte deja ).
    TF circulation( TF d, TF t0, TF t1, TF sig ) const {
        const TF d2 = d * d, is2 = 1 / ( 2 * sig * sig );
        const TF lam = std::max( std::fabs( d ), sig );
        TF r = 0;
        auto morceau = [ & ]( TF a, TF b ) {         // `a < b`
            const TF tmin2 = ( a <= 0 && b >= 0 ) ? 0 : std::min( a * a, b * b );
            if ( ( d2 + tmin2 ) * is2 > seuil ) {    // l'exponentielle est negligeable
                if ( d != 0 ) r += ( std::atan( b / d ) - std::atan( a / d ) ) / ( 2 * M_PI );
                return;
            }
            if ( d == 0 ) return;                    // `d * ( borne )` : rien
            const TF c = ( a + b ) / 2, h = ( b - a ) / 2;
            TF sm = 0;
            for ( int q = 0; q < NG; ++q ) {
                const TF t = c + h * gx[ q ], r2 = d2 + t * t;
                sm += gw[ q ] * ( -std::expm1( -r2 * is2 ) ) / r2;
            }
            r += d * h * sm / ( 2 * M_PI );
        };
        // les bornes : 0, +-lam, +-2 lam, ... rognees a [ t0, t1 ]
        auto cote = [ & ]( TF sgn ) {               // le cote `sgn * t >= 0`
            const TF lo = std::max( TF( 0 ), sgn > 0 ? t0 : -t1 ), hi = std::max( TF( 0 ), sgn > 0 ? t1 : -t0 );
            if ( hi <= lo ) return;
            TF a = lo, b = lam;
            while ( b <= a ) b *= 2;
            for ( ; ; b *= 2 ) {
                const TF bb = std::min( b, hi );
                if ( sgn > 0 ) morceau( a, bb ); else morceau( -bb, -a );
                if ( bb >= hi ) break;
                a = bb;
            }
        };
        cote( +1 );
        cote( -1 );
        return r;
    }

#ifndef SF_DENSITE_NG
#define SF_DENSITE_NG 8
#endif
    static constexpr int NG = SF_DENSITE_NG;
    static constexpr TF gx8[ 8 ] = { -0.9602898564975363, -0.7966664774136267, -0.5255324099163290, -0.1834346424956498,
                                      0.1834346424956498,  0.5255324099163290,  0.7966664774136267,  0.9602898564975363 };
    static constexpr TF gw8[ 8 ] = {  0.1012285362903763,  0.2223810344533745,  0.3137066458778873,  0.3626837833783620,
                                      0.3626837833783620,  0.3137066458778873,  0.2223810344533745,  0.1012285362903763 };
    static constexpr TF gx5[ 5 ] = { -0.9061798459386640, -0.5384693101056831, 0.0, 0.5384693101056831, 0.9061798459386640 };
    static constexpr TF gw5[ 5 ] = {  0.2369268850561891,  0.4786286704993665, 0.5688888888888889, 0.4786286704993665, 0.2369268850561891 };
    static constexpr TF gx4[ 4 ] = { -0.8611363115940526, -0.3399810435848563, 0.3399810435848563, 0.8611363115940526 };
    static constexpr TF gw4[ 4 ] = {  0.3478548451374538,  0.6521451548625461, 0.6521451548625461, 0.3478548451374538 };
    static constexpr const TF *gx = NG == 8 ? gx8 : NG == 5 ? gx5 : gx4;
    static constexpr const TF *gw = NG == 8 ? gw8 : NG == 5 ? gw5 : gw4;
};

} // namespace sf

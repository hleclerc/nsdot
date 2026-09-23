#pragma once

// =====================================================================================
// LA CONTINUATION EN LARGEUR : la densite convolee par une gaussienne de largeur `s`, de large a nulle.
//
// Ce que le banc a conclu sur les densites qui se concentrent ( `solvers_des_familles` README § 9 ) :
// Newton direct casse des que des cellules n'ont plus de masse ( des germes loin de toute bosse :
// pas de plancher, des lignes de hessienne nulles ), et ce ne sont pas les methodes du premier
// ordre qui s'en sortent mieux. Ce qui marche est de RESOUDRE D'ABORD POUR LA DENSITE ETALEE --
// convolee par une gaussienne large, donc positive partout -- puis de resserrer, `s / sqrt( 2 )`
// a chaque etape, les poids d'une etape servant de depart a la suivante, jusqu'a la densite
// elle-meme. Le ratio `sqrt( 2 )` est le meilleur mesure ( 2 et 2^( 1/4 ) font plus de diagrammes ),
// et c'est le pas par les limites en masse ( `Limites.h` ) qui fait tomber les reculs dans la zone
// dure, ou les aiguilles se forment. La tangente `dw / ds` n'apporte plus rien avec lui : elle
// n'est pas reprise.
//
// Ce que « convoler » veut dire, distribution par distribution :
//   * une somme de gaussiennes : les largeurs deviennent `sqrt( sigma_i^2 + s^2 )`, rien d'autre
//     ( `SumOfGaussians::with_convolution` ) ;
//   * une image : ses valeurs floutees sur sa grille ( un filtre gaussien separable, tronque a
//     quatre ecarts-types, renormalise au bord : positif, la masse conservee ), le support inchange ;
//   * une densite constante : rien a faire, elle est deja positive partout.
// =====================================================================================

#include <loom/support/algorithms/CartesianIndices.h>
#include "../SumOfGaussians.h"
#include "../UnitDensity.h"
#include "../Image.h"
#include <optional>
#include <vector>
#include <cmath>

namespace sdot {
namespace otplan {

/// la distribution `dist` a la largeur `s` : `at( s )` rend une reference valable jusqu'au prochain `at`
template<class Dist>
struct Convolee {
    const Dist &dist;
    Convolee( const Dist &dist ) : dist( dist ) {}
    static constexpr bool possible = false;              ///< rien a convoler
    const Dist &at( double ) { return dist; }
    double echelle_min( double ) const { return 0; }     ///< en dessous de quoi la convolution ne change plus rien
};

/// les gaussiennes : une copie a `conv_s = s`
template<class... T>
struct Convolee<SumOfGaussians<T...>> {
    using Dist = SumOfGaussians<T...>;
    const Dist &dist;
    std::optional<Dist> courant;
    static constexpr bool possible = true;
    Convolee( const Dist &dist ) : dist( dist ) {}
    const Dist &at( double s ) {
        if ( s <= 0 ) return dist;
        courant.emplace( dist.with_convolution( typename Dist::TF( s ) ) );
        return *courant;
    }
    double echelle_min( double ) const { return double( dist.smallest_sigma() ) / 4; }
};

/// une image : ses valeurs floutees, dans un tampon a nous
template<class... T>
struct Convolee<Image<T...>> {
    using Dist = Image<T...>;
    using TF = typename Dist::TF;
    static constexpr int D = Dist::ct_dim;
    const Dist &dist;
    std::vector<double> src, buf, tmp;                   ///< les valeurs, en ligne ( ordre C )
    std::vector<TF> out;
    std::optional<Dist> courant;
    static constexpr bool possible = true;

    Convolee( const Dist &dist ) : dist( dist ) {
        auto shape = dist.values.shape();
        CartesianIndices<DECAYED_TYPE_OF( shape )> cells{ shape };
        src.resize( PI( cells.size() ) );
        for ( PI flat = 0; flat < PI( cells.size() ); ++flat )
            src[ flat ] = cells[ flat ].apply_values( [&]( auto ...i ) { return double( dist.values( i... ) ); } );
    }

    /// le pas de la grille sur l'axe `a` ( la longueur de `frame( a )` fois l'ecart moyen des noeuds ),
    /// et le plus petit d'entre eux
    double pas( int a ) const {
        return dist.with_defaults( [&]( auto &&img ) {       // `frame` / `knots` peuvent etre absents : leurs defauts
            double l2 = 0;
            for ( int c = 0; c < D; ++c ) { const double f = double( img.frame( a, c ) ); l2 += f * f; }
            const SI nb = SI( img.values.shape( a ) );
            const double etendue = double( img.knots( a, nb ) ) - double( img.knots( a, 0 ) );
            return std::sqrt( l2 ) * etendue / std::max<SI>( nb, 1 );
        } );
    }
    double echelle_min( double ) const {
        double r = pas( 0 );
        for ( int a = 1; a < D; ++a ) r = std::min( r, pas( a ) );
        return r / 4;
    }

    const Dist &at( double s ) {
        if ( s <= 0 ) return dist;
        // le filtre separable : sur chaque axe, une gaussienne d'ecart-type `s / pas` en pixels,
        // tronquee a quatre ecarts-types et renormalisee cellule par cellule ( le bord )
        buf = src;
        SI shape[ D ], stride[ D ];
        SI total = 1;
        for ( int a = D - 1; a >= 0; --a ) { shape[ a ] = SI( dist.values.shape( a ) ); stride[ a ] = total; total *= shape[ a ]; }
        for ( int a = 0; a < D; ++a ) {
            const double sp = s / pas( a );
            if ( ! ( sp > 1e-3 ) ) continue;
            const SI r = std::min<SI>( SI( std::ceil( 4 * sp ) ), shape[ a ] );
            std::vector<double> ker( 2 * r + 1 );
            for ( SI k = -r; k <= r; ++k ) ker[ k + r ] = std::exp( -0.5 * double( k * k ) / ( sp * sp ) );
            tmp.assign( PI( total ), 0.0 );
            for ( SI flat = 0; flat < total; ++flat ) {
                const SI i = ( flat / stride[ a ] ) % shape[ a ];
                double sum = 0, wsum = 0;
                const SI lo = std::max<SI>( -r, -i ), hi = std::min<SI>( r, shape[ a ] - 1 - i );
                for ( SI k = lo; k <= hi; ++k ) { sum += ker[ k + r ] * buf[ flat + k * stride[ a ] ]; wsum += ker[ k + r ]; }
                tmp[ flat ] = sum / wsum;
            }
            buf.swap( tmp );
        }
        out.assign( buf.begin(), buf.end() );
        using V = DECAYED_TYPE_OF( dist.values );
        auto vue = tensor_view<typename V::MemorySpace>( out.data(), dist.values.shape(), typename V::AxisNames{} );
        courant.emplace( Dist{ dist.target_mass, dist.nb_dims, dist.shape, vue, dist.origin, dist.frame, dist.knots,
                               dist.current_mass, dist.nb_cells_cum, dist.cell_cum_mass } );
        return *courant;
    }
};

/// LES ETAPES : `s0, s0 / r, ...` tant que `s >= s_min`, puis `0`
inline std::vector<double> etapes( double s0, double ratio, double s_min ) {
    std::vector<double> res;
    if ( s0 > 0 && ratio > 1 )
        for ( double s = s0; s >= s_min * ( 1 - 1e-12 ) && s > 0; s /= ratio )
            res.push_back( s );
    res.push_back( 0 );
    return res;
}

} // namespace otplan
} // namespace sdot

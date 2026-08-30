#pragma once

// the members + the axes this body names, written to the build include tree by `CallArg_Aggregate`.
#include <sdot/generated/aggregates/SumOfGaussians.h>
#include <loom/support/common_macros.h>
#include <loom/support/containers/Vector.h>
#include <loom/support/atomic_add.h>
#include "PointwiseDensity.h"

namespace sdot {

// Une somme de gaussiennes ISOTROPES, vue comme densité à intégrer sur une cellule.
//
//   rho( x ) = somme_i  w_i * exp( - r_i^2 / ( 2 s_i^2 ) ) / ( 2 pi s_i^2 ) ^ ( d / 2 ),
//   r_i = | x - c_i |
//
// `w_i` est la MASSE de la gaussienne `i`, pas sa hauteur -- la masse totale est donc la somme des
// poids, et normaliser est une division (voir `distributions/SumOfGaussians.py`).
//
// C'est la première densité qui n'est pas constante par morceaux, et tout ce qu'elle a à dire tient
// en trois réponses PONCTUELLES : la valeur, le gradient, et où ranger `d rho / d paramètres`.
// Elle ne DÉCOUPE rien (son morceau est la cellule entière) et ne sait rien des cellules ; c'est
// `PowerDiagram::integrate_into` qui, voyant `is_constant == false`, les découpe en simplices et y
// fait sa quadrature. Le partage est là, et il n'a pas de troisième côté.
SDOT_TEMPLATE_DECL_FOR_SumOfGaussians
struct SumOfGaussians {
    SDOT_ATTRIBUTES_OF_SumOfGaussians

    static constexpr int ct_dim = DECAYED_TYPE_OF( nb_dims )::value;
    using TF = DECAYED_TYPE_OF( positions )::TF;

    /// Un seul morceau, la cellule elle-même, et pas une coupe : rien à découper quand la densité
    /// est définie partout par la même formule. Le scratch de découpe n'est donc pas touché (et
    /// `extra_cuts_per_piece` rend 0, donc il n'est même pas alloué).
    ///
    /// COMMENT on s'intègre dépend de la dimension, et c'est un choix qui se fait ici, par
    /// composition -- l'intégrateur, lui, ne voit que le contrat :
    ///   * en 2D on SAIT le faire (voir `wedge_measure`), donc on se passe soi-même ;
    ///   * au-delà on ne sait pas, donc on se déclare BOÎTE NOIRE en s'emballant dans
    ///     `PointwiseDensity`, qui n'a besoin que de `value_at` / `gradient_at`.
    void for_each_piece( const auto &cell, auto &&/*ws*/, auto &&func ) const {
        if constexpr ( ct_dim == 2 )
            func( cell, *this );
        else
            func( cell, PointwiseDensity{ *this } );
    }

    // ---- l'intégration EXACTE en 2D ---------------------------------------------------------------
    // Une gaussienne isotrope sur un triangle n'a pas de forme close élémentaire -- c'est la fonction
    // T d'Owen -- mais elle se RÉDUIT à une intégrale 1D lisse et bornée, dont la précision ne dépend
    // plus de la forme de la cellule. C'est toute la différence avec une règle de quadrature sur le
    // triangle, dont l'erreur est en `( taille de cellule / sigma ) ^ 4`.
    //
    // La réduction : après translation/mise à l'échelle, le triangle est décomposé en trois coins
    // signés `( 0, P, Q )`, et un coin s'intègre en polaires --
    //
    //     W( P, Q ) = signe( p ) / 2pi * Int_{t_P}^{t_Q} [ 1 - exp( -( p^2 + t^2 ) / 2 ) ] p dt / ( p^2 + t^2 )
    //
    // où `p` est la distance signée de l'origine à la droite `PQ` et `t` l'abscisse le long d'elle.
    // Écrite AINSI -- la différence faite sous l'intégrale, pas entre deux arctangentes -- elle n'a
    // aucune compensation catastrophique, y compris quand l'origine frôle la droite.
    //
    // Au-delà de `|t| = tail_cut`, l'exponentielle ne vaut plus rien et il ne reste que la
    // lorentzienne, dont la primitive est `arctan( t / p )` : les queues sont donc EXACTES et
    // gratuites, et la quadrature ne travaille que sur un intervalle borné où l'intégrande a une
    // échelle `>= 1`. Mesuré : `8 x 4 = 32` évaluations par arête donnent 4e-14 d'erreur absolue
    // sur des configurations choisies pour être méchantes (germe sur le centre, triangle immense,
    // sliver rasant).
    static constexpr bool is_constant = false;

    static constexpr TF  tail_cut   = 8;    ///< `exp( -t^2/2 ) < 1e-14` au-delà : la queue est exacte
    static constexpr int nb_panels  = 4;    ///< panneaux de Gauss-Legendre sur le coeur

    /// La mesure normale standard SIGNÉE du triangle `( 0, P, Q )` -- le coin.
    TF wedge_measure( const auto &P, const auto &Q ) const;

    /// La mesure normale standard du triangle `ys` (positive, orientation quelconque).
    TF std_triangle_measure( const auto &ys ) const;

    /// `Int_T rho`, exact. `pts` : les 3 sommets.
    TF integrate_over_simplex( const auto &pts ) const;

    /// L'adjoint, ÉLÉMENTAIRE -- c'est le point remarquable : la valeur demande une fonction
    /// spéciale, ses dérivées non. Tout se ramène à des intégrales de BORD, qui pour une gaussienne
    /// le long d'un segment sont des `erf` :
    ///   * un sommet : le bouger balaie ses deux arêtes, d'où `Int rho * lambda` sur chacune ;
    ///   * le centre : `d/dc = -Int grad rho = -Contour rho n ds` (divergence) -- égal, et c'est une
    ///     bonne vérification, à MOINS la somme des dérivées par sommet ;
    ///   * sigma : `d rho / d sigma = sigma * laplacien( rho )` (identité de la chaleur, `t = sigma^2/2`),
    ///     donc encore un flux au bord, et `y . n` y est CONSTANT le long d'une arête ;
    ///   * le poids : la mesure elle-même, déjà calculée.
    void integrate_over_simplex_bwd( const auto &pts, TF g, auto &&grad_pts, auto &&grad_dist ) const;

    /// Pour une arête `A -> B` du triangle `A, B, C`, en repère standard : la normale SORTANTE, la
    /// distance signée `y . n` (constante le long de l'arête), `Int phi ds`, et `Int phi lambda_A ds`.
    struct EdgeInfo { Vector<TF,2> n; TF p; TF j0; TF j1a; };
    EdgeInfo edge_info( const auto &A, const auto &B, const auto &C ) const;

    /// Le noyau NORMALISÉ de la gaussienne `i` en `x` (masse 1), et le carré de la distance --
    /// les deux quantités dont tout le reste se déduit, calculées une fois.
    auto kernel_at( SI i, const auto &x ) const;

    TF   value_at         ( const auto &x ) const;   ///< rho( x )
    auto gradient_at      ( const auto &x ) const;   ///< grad rho( x ), un `Vector<TF,ct_dim>`

    /// Accumule `g * d rho( x ) / d paramètre` dans la cotangente de chaque paramètre.
    ///
    /// C'est la moitié « distribution » de l'adjoint, et elle ne connaît que `x` : l'intégrateur
    /// lui passe le poids de quadrature du noeud (`g`), sans savoir ce qu'il y a derrière. Les
    /// ajouts sont ATOMIQUES -- une gaussienne large est vue par les work-items de beaucoup de
    /// cellules à la fois -- et chacun est gardé par la validité de sa cible, un paramètre non
    /// dérivé arrivant en `NoneTensor`.
    void add_value_grad_at( auto &&grad_dist, const auto &x, TF g ) const;
};

}

#include "SumOfGaussians.cxx"

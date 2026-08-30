#pragma once

#include <loom/support/common_macros.h>
#include <loom/support/containers/Matrix.h>
#include <loom/support/containers/Vector.h>
#include <SYCL/sycl.hpp>

// `sycl::` et non `std::` pour les mathématiques -- voir `SumOfGaussians.cxx` pour ce que `std::`
// coûte sur la cible CUDA AOT.

namespace sdot {

// « Je ne sais pas m'intégrer, mais je sais me VALOIR en un point » -- la densité boîte noire,
// emballée dans une quadrature adaptative.
//
// C'est une IMPLÉMENTATION du contrat de morceau (voir `distributions/Distribution.py`), pas un
// régime de l'intégrateur : `PowerDiagram::integrate_into` ne connaît aucune quadrature, il demande
// à la densité l'intégrale sur un simplexe et lui rend la cotangente des sommets. Une densité qui
// SAIT s'intégrer -- une formule fermée, une réduction à une fonction spéciale, comme
// `SumOfGaussians` en 2D -- répond directement et ne passe jamais par ici. Une qui ne sait pas
// s'emballe là-dedans, en une ligne :
//
//     void for_each_piece( const auto &cell, auto &&, auto &&func ) const {
//         func( cell, PointwiseDensity{ *this } );
//     }
//
// Ce qu'elle demande à `D` : `value_at( x )`, `gradient_at( x )`, `add_value_grad_at( gd, x, g )`.
//
// = La règle, et pourquoi elle ne suffit pas seule
//
// `d + 1` noeuds de poids égal `1 / ( d + 1 )`, chacun portant `alpha` sur un sommet et `beta` sur
// les `d` autres : la règle symétrique exacte jusqu'au degré 2. `alpha` sort de l'exactitude sur les
// `lambda_i^2` : `alpha^2 + d beta^2 = 2 / ( d + 2 )` avec `alpha + d beta = 1`, donc
// `( d + 1 ) alpha^2 - 2 alpha + ( 2 - d ) / ( d + 2 ) = 0`. En 2D on retrouve `( 2/3, 1/6 )`, en 3D
// `( 0.585410, 0.138197 )` : les règles classiques du triangle et du tétraèdre.
//
// Son erreur est en `( taille du simplexe / échelle de la densité ) ^ 4` -- inacceptable dès qu'une
// cellule est grande devant une gaussienne étroite. D'où la SUBDIVISION : on bissecte le simplexe
// sur sa plus longue arête tant que la règle et ses deux moitiés ne s'accordent pas. C'est le seul
// moyen générique de rattraper une échelle qu'on ne connaît pas -- on ne demande justement pas à la
// densité de la déclarer, elle est une boîte noire.
//
// Une bissection DÉPILE UN élément et en EMPILE DEUX, donc la pile est bornée par la PROFONDEUR et
// non par le nombre de feuilles : `max_depth + 2` entrées, chacune une matrice barycentrique
// `( d + 1 ) ^ 2`. C'est ce qui la rend utilisable dans un kernel -- pas d'allocation, une borne
// écrite en dur. Ce n'est pas gratuit pour autant (en 3D, ~1 Ko par work-item) et ce budget-là
// n'est PAS compté par `PowerDiagram.measures`, qui ne connaît pas la densité.
//
// = L'adjoint
//
// Celui de CE QU'ON CALCULE, pas de l'intégrale idéale -- la seule façon d'être cohérent avec le
// forward, et c'est pourquoi le backward REFAIT la même subdivision (le critère est déterministe,
// donc il retombe sur les mêmes feuilles). Sur une feuille, deux termes : le volume bouge avec ses
// sommets (cofacteur du déterminant), et un noeud est une combinaison barycentrique FIXE de ces
// sommets, donc la cotangente de `gradient_at` s'y répartit avec ces poids-là. Les sommets d'une
// feuille étant eux-mêmes des combinaisons barycentriques des sommets d'ORIGINE, une dernière
// multiplication ramène tout où l'intégrateur l'attend.
template<class D>
struct PointwiseDensity {
    using TF = typename D::TF;
    static constexpr int ct_dim = D::ct_dim;

    /// pas constante sur un morceau : l'intégrateur passera par le découpage en simplices.
    static constexpr bool is_constant = false;

    /// Bissecter au moins jusque là avant d'avoir le droit de s'arrêter : la règle ne voit que
    /// `d + 1` points, et une densité étroite peut tomber pile entre eux -- auquel cas le simplexe
    /// et ses moitiés s'accorderaient sur zéro. C'est le garde-fou classique d'une quadrature
    /// adaptative, et il coûte quatre évaluations.
    static constexpr int min_depth = 2;
    static constexpr int max_depth = 8;     ///< borne la pile ET le pire cas de coût
    /// Écart relatif toléré entre un simplexe et ses deux moitiés -- ce qui borne l'erreur
    /// d'intégration, et c'est son seul rôle.
    ///
    /// Il borne AUSSI, involontairement, le saut que la valeur fait quand un sous-simplexe bascule
    /// de « raffiner » à « accepter » : un schéma adaptatif par la valeur n'est lisse qu'à `rtol`
    /// près, par construction. Ce n'est pas un réglage à durcir -- une différence finie divise ce
    /// saut par son pas, donc AUCUN `rtol` raisonnable ne rend le schéma vérifiable à pas serré,
    /// et le durcir ne fait que payer des niveaux de bissection pour rien. C'est au VÉRIFICATEUR
    /// de prendre un pas à la mesure du schéma (voir les tests), et à un optimiseur de savoir qu'il
    /// travaille sur une fonction lisse par morceaux.
    static constexpr TF  rtol      = 1e-5;

    D dens;

    /// Un sous-simplexe : ses `d + 1` sommets en coordonnées BARYCENTRIQUES du simplexe d'origine.
    /// C'est cette représentation-là, et pas les points, qui permet à l'adjoint de revenir aux
    /// sommets d'origine sans rien résoudre.
    using Bary = Vector<Vector<TF,ct_dim+1>,ct_dim+1>;

    static Bary whole() {
        return Bary( Function(), []( PI k ) {
            return Vector<TF,ct_dim+1>( Function(), [&]( PI j ) { return TF( j == k ); } ); } );
    }

    static auto points_of( const Bary &b, const auto &pts ) {
        return Vector<Vector<TF,ct_dim>,ct_dim+1>( Function(), [&]( PI k ) {
            return Vector<TF,ct_dim>( Function(), [&]( PI c ) {
                TF s = 0;
                for ( SI j = 0; j <= ct_dim; ++j )
                    s += b[ k ][ j ] * pts[ j ][ c ];
                return s;
            } );
        } );
    }

    /// Bissection de MAUBACH : on coupe TOUJOURS l'arête `( sommet 0, sommet d )`, et les enfants
    /// remettent le nouveau sommet en deuxième position, les autres décalés d'un cran.
    ///
    /// La règle est COMBINATOIRE -- elle ne regarde aucune longueur -- et c'est ce qui compte le
    /// plus ici. Couper « la plus longue arête » paraît mieux, mais c'est un choix DISCONTINU :
    /// deux arêtes presque égales, et un déplacement infinitésimal d'un germe fait basculer tout le
    /// motif, donc la valeur calculée saute de l'erreur de quadrature -- et l'adjoint ne peut plus
    /// correspondre à une différence finie. Avec une règle d'indices, le motif étant fixé, chaque
    /// sous-simplexe est une fonction AFFINE des sommets d'origine : la valeur est lisse, et la
    /// seule discontinuité restante est la décision de raffiner, bornée par `rtol`.
    ///
    /// Le roulement des sommets est ce qui borne la dégradation des formes (Maubach 1995). Mesuré
    /// sur 8 niveaux, la qualité minimale tient : 0.32 en 2D, 0.14 en 3D, 0.13 en 4D -- contre
    /// 0.50 / 0.22 / 0.18 pour la plus longue arête, et 0.13 / 0.09 / 0.08 pour un cyclique naïf.
    static void bisect( const Bary &b, const auto &/*pts*/, Bary &lo, Bary &hi ) {
        Vector<TF,ct_dim+1> mid;
        for ( SI j = 0; j <= ct_dim; ++j )
            mid[ j ] = ( b[ 0 ][ j ] + b[ ct_dim ][ j ] ) / 2;

        lo[ 0 ] = b[ 0 ];
        hi[ 0 ] = b[ ct_dim ];
        lo[ 1 ] = mid;
        hi[ 1 ] = mid;
        for ( SI k = 1; k < ct_dim; ++k ) {
            lo[ k + 1 ] = b[ k ];
            hi[ k + 1 ] = b[ k ];
        }
    }

    /// Les feuilles de la subdivision, avec la valeur de la règle sur chacune. Le forward les somme,
    /// le backward les redérive -- même parcours, donc mêmes feuilles.
    void for_each_leaf( const auto &pts, auto &&func ) const {
        Vector<Bary,max_depth+2> stack;
        Vector<int,max_depth+2>  depth;
        Vector<TF,max_depth+2>   value;

        SI top = 0;
        stack[ 0 ] = whole();
        depth[ 0 ] = 0;
        value[ 0 ] = rule( points_of( stack[ 0 ], pts ) );

        while ( top >= 0 ) {
            const Bary b = stack[ top ];
            const int dp = depth[ top ];
            const TF  cv = value[ top ];
            --top;

            if ( dp >= max_depth ) {
                func( b, cv );
                continue;
            }

            Bary lo, hi;
            bisect( b, pts, lo, hi );
            const TF v0 = rule( points_of( lo, pts ) );
            const TF v1 = rule( points_of( hi, pts ) );
            const TF fine = v0 + v1;

            const TF diff = fine > cv ? fine - cv : cv - fine;
            const TF mag  = fine < 0 ? -fine : fine;
            if ( dp >= min_depth && diff <= rtol * mag ) {
                // on garde les DEUX MOITIÉS comme feuilles, pas le père : c'est `fine` que le
                // forward additionne, donc c'est `fine` que le backward doit dériver.
                func( lo, v0 );
                func( hi, v1 );
                continue;
            }

            ++top; stack[ top ] = lo; depth[ top ] = dp + 1; value[ top ] = v0;
            ++top; stack[ top ] = hi; depth[ top ] = dp + 1; value[ top ] = v1;
        }
    }

    TF integrate_over_simplex( const auto &pts ) const {
        TF res = 0;
        for_each_leaf( pts, [&]( const Bary &, TF v ) { res += v; } );
        return res;
    }

    void integrate_over_simplex_bwd( const auto &pts, TF g, auto &&grad_pts, auto &&grad_dist ) const {
        for_each_leaf( pts, [&]( const Bary &b, TF /*v*/ ) {
            auto gl = Vector<Vector<TF,ct_dim>,ct_dim+1>( Function(), []( PI ) {
                return Vector<TF,ct_dim>::zeros(); } );

            rule_bwd( points_of( b, pts ), g, gl, grad_dist );

            // le sommet `k` de la feuille est `somme_j b[k][j] * pts[j]` : sa cotangente se
            // redistribue avec exactement ces poids-là.
            for ( SI k = 0; k <= ct_dim; ++k )
                for ( SI j = 0; j <= ct_dim; ++j )
                    for ( PI c = 0; c < ct_dim; ++c )
                        grad_pts[ j ][ c ] += b[ k ][ j ] * gl[ k ][ c ];
        } );
    }

    // ---- la règle elle-même, sur UN simplexe donné par ses points ---------------------------------

    static auto barycentric() {
        const TF d = ct_dim;
        const TF alpha = ( 1 + sycl::sqrt( 1 - ( d + 1 ) * ( 2 - d ) / ( d + 2 ) ) ) / ( d + 1 );
        return Vector<TF,2>( Values(), alpha, ( 1 - alpha ) / d );
    }

    static TF factorial() {
        TF res = 1;
        for ( int i = 2; i <= ct_dim; ++i )
            res *= i;
        return res;
    }

    /// `M` = les `d` arêtes issues de `P[ 0 ]`, en colonnes : son déterminant donne le volume, et
    /// ses cofacteurs la dérivée de ce volume.
    static auto edge_matrix( const auto &P ) {
        return Matrix<TF,ct_dim>::with_func( [&]( auto r, auto c ) { return P[ c + 1 ][ r ] - P[ 0 ][ r ]; } );
    }

    /// le noeud `q` : `beta` partout, `alpha` sur le sommet `q`.
    static auto node( const auto &P, SI q ) {
        const auto ab = barycentric();
        auto tot = Vector<TF,ct_dim>::zeros();
        for ( SI k = 0; k <= ct_dim; ++k )
            tot = tot + P[ k ];
        return ab[ 1 ] * tot + ( ab[ 0 ] - ab[ 1 ] ) * P[ q ];
    }

    TF rule( const auto &P ) const {
        const TF det = edge_matrix( P ).determinant();
        const TF vol = ( det < 0 ? -det : det ) / factorial();

        TF s = 0;
        for ( SI q = 0; q <= ct_dim; ++q )
            s += dens.value_at( node( P, q ) );
        return vol * s / ( ct_dim + 1 );
    }

    void rule_bwd( const auto &P, TF g, auto &&gP, auto &&grad_dist ) const {
        const auto ab = barycentric();
        const auto M = edge_matrix( P );
        const TF det = M.determinant();
        const TF vol = ( det < 0 ? -det : det ) / factorial();

        TF s = 0;
        for ( SI q = 0; q <= ct_dim; ++q )
            s += dens.value_at( node( P, q ) );
        s /= ( ct_dim + 1 );

        // ---- la part du VOLUME : `d|det|/dM = signe( det ) * cofacteur( M )`, et chaque colonne de
        // `M` est un sommet moins le premier, donc le premier ramasse MOINS la somme des colonnes.
        const TF gv = ( det < 0 ? -g : g ) * s / factorial();
        for ( PI r = 0; r < ct_dim; ++r ) {
            TF row_sum = 0;
            for ( PI c = 0; c < ct_dim; ++c ) {
                const TF minor = M.without_row_and_col( r, c ).determinant();
                const TF cof = ( ( r + c ) % 2 ? -minor : minor ) * gv;
                gP[ c + 1 ][ r ] += cof;
                row_sum += cof;
            }
            gP[ 0 ][ r ] -= row_sum;
        }

        // ---- la part des NOEUDS, et au même point celle des PARAMÈTRES de la densité
        const TF gn = g * vol / ( ct_dim + 1 );
        for ( SI q = 0; q <= ct_dim; ++q ) {
            const auto x = node( P, q );
            const auto gr = dens.gradient_at( x );
            for ( SI k = 0; k <= ct_dim; ++k ) {
                const TF w = ab[ 1 ] + ( q == k ? ab[ 0 ] - ab[ 1 ] : TF( 0 ) );
                for ( PI c = 0; c < ct_dim; ++c )
                    gP[ k ][ c ] += gn * gr[ c ] * w;
            }
            dens.add_value_grad_at( grad_dist, x, gn );
        }
    }
};

} // namespace sdot

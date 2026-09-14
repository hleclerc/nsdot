#pragma once

// =====================================================================================
// CE QU'UN DIAGRAMME DE PUISSANCE CALCULE, quelle que soit la facon dont il range ses germes.
//
// La cellule du germe `i` est la ou sa DISTANCE DE PUISSANCE gagne : `|x - d_i|^2 - w_i <=
// |x - d_j|^2 - w_j` pour tout autre `j`. Developpee, l'inegalite perd son `|x|^2` et devient un
// demi-espace : un diagramme de puissance coute exactement ce que coute un Voronoi, un plan par
// rival, la meme coupe ( `cell/Plane.h::bisector` ).
//
// Il n'y a PAS de diagramme ici : pas un sommet, pas une facette. Tout est refait, cellule par
// cellule, dans le scratch du work-item ( `Cell.py::CellScratch` ) : une cellule locale, une seconde
// pour les morceaux d'une distribution, et les cotangentes de l'adjoint. C'est LA CELLULE QUI
// DIRIGE ( `cell/Moteur.h` ) : elle demande ses plans a un fournisseur, et c'est le STOCKAGE du
// diagramme qui dit lequel -- tous les germes dans l'ordre pour `PowerDiagram_Plain`, l'arbre BSP
// retourne pour `PowerDiagram_Bsp`.
//
// = CE QU'UN STOCKAGE FOURNIT
//
//     ct_dim, TF, has_weights
//     SI   nb_seeds() const                   les germes, DANS SON ORDRE
//     auto point( SI k ) const                le germe `k` ( Vector<TF,ct_dim> ), poids `weight( k )`
//     SI   user_id( SI k ) const              l'indice de ce germe pour l'UTILISATEUR
//     auto fournisseur<TK>( SI k0 ) const     ce qui rend les plans de la cellule de `k0`
//
// Les cellules se construisent dans l'ORDRE DU STOCKAGE ( `k`, ce qu'un fournisseur met dans les
// identifiants de coupe, et l'indice des gradients sur les germes ) ; ce qui sort vers l'utilisateur
// -- une mesure, une cellule gardee et ses `cut_ids` -- est ecrit a l'indice `user_id( k )`.
// =====================================================================================

#include <loom/support/common_macros.h>
#include <loom/support/containers/Matrix.h>
#include <loom/support/containers/Vector.h>
#include <loom/support/atomic_add.h>
#include "../cell/Fournisseurs.h"
#include "../cell/Moteur.h"
#include "../cell/Ops.h"
#include "../PieceWorkspace.h"
#include "../UnitDensity.h"

#include <limits>

namespace sdot {
namespace diagram {

/// une cotangente par sommet, dans le scratch
template<class TF>
struct GradVp {
    TF *g;
    SI  cap;
    TF &operator()( int i, int d ) { return g[ d * cap + i ]; }
    TF  operator()( int i, int d ) const { return g[ d * cap + i ]; }
};

/// une distribution qui DECOUPE demande une seconde cellule ; `UnitDensity` et les densites
/// lisses n'en demandent pas
template<class Dist>
constexpr int nb_work_cells() {
    if constexpr ( requires { Dist::cuts_pieces; } ) return Dist::cuts_pieces ? 2 : 1;
    else return 1;
}

/// LE DECOUPAGE DU SCRATCH d'un work-item : `nb_cells` cellules locales de `cap` sommets, et ( pour
/// l'adjoint ) une cotangente par sommet -- LA MEME FORMULE que `PowerDiagram._scratch_words`.
template<class Local,class TF>
SI words_for( SI cap, int nb_cells, bool with_grad ) {
    return nb_cells * Local::words_for( cap ) + ( with_grad ? words_of<TF>( Local::ct_dim * cap ) : 0 );
}

template<class Local,class TF>
SI cap_in( SI nb_words, int nb_cells, bool with_grad ) {
    return cap_for_words( nb_words, [&]( SI c ) { return words_for<Local,TF>( c, nb_cells, with_grad ); } );
}

/// La cellule de `k0`, construite dans `c` a partir du domaine `dom`. Rend `false` si le scratch
/// n'a pas suffi -- la cellule est alors restee au dernier etat valide.
template<class PD,class Local>
bool make_cell( const PD &pd, Local &c, SI k0, const auto &dom ) {
    using TK = typename Local::TKernel;
    if ( ! c.load( dom ) )
        return false;
    auto f = pd.template fournisseur<TK>( k0 );
    return run<PD::on_cpu>( c, f ) != CutStatus::OVERFLOW;
}

/// le plan de la coupe `k` de `cell`, dans le flottant des positions : la bissectrice REFAITE
/// depuis les germes quand la coupe fait face a un germe, le plan relu sur la geometrie sinon
template<class PD>
void plane_of( const PD &pd, const auto &cell, SI k0, int k, auto &dir, typename PD::TF &off ) {
    using TF = typename PD::TF;
    constexpr int D = PD::ct_dim;
    const int id = cell.cid[ k ];
    if ( id >= 0 ) {
        const auto p0 = pd.point( k0 ), p1 = pd.point( id );
        off = ( pd.weight( k0 ) - pd.weight( id ) ) / 2;
        for ( int d = 0; d < D; ++d ) {
            dir[ d ] = p1[ d ] - p0[ d ];
            off += dir[ d ] * ( p0[ d ] + p1[ d ] ) / 2;
        }
    } else {
        TF dk[ D ];
        cell.plane( k, dk, off );
        for ( int d = 0; d < D; ++d )
            dir[ d ] = dk[ d ];
    }
}

// ---- CONTRE QUOI on integre ----------------------------------------------------------------------

/// les `D + 1` sommets du simplexe `chain`, comme points -- ce qu'une densite recoit
template<class TF,int D>
auto simplex_points( const auto &cell, const auto &chain ) {
    return Vector<Vector<TF,D>,D+1>( Function(), [&]( PI k ) {
        return Vector<TF,D>::with_func( [&]( PI c ) { return TF( cell.coord( int( chain[ k ] ), int( c ) ) ); } );
    } );
}

/// `res` = l'integrale de `dist` sur `cell`. La distribution DECOUPE, on INTEGRE : sur un morceau a
/// densite constante, `valeur * mesure` ; sinon le morceau part en simplexes et c'est la densite
/// qui s'integre sur chacun. Rend `false` si un morceau n'a pas tenu.
template<class TF,class Local>
bool integrate_into( auto &&res, const Local &cell, Local &piece, const auto &dist ) {
    constexpr int D = Local::ct_dim;
    PieceWorkspace<Local> ws{ piece };
    TF sum = 0;
    dist.for_each_piece( cell, ws, [&]( const auto &pc, const auto &dens ) {
        if constexpr ( DECAYED_TYPE_OF( dens )::is_constant ) {
            sum += dens.value * pc.template measure<TF>();
        } else {
            // une cellule non bornee n'a pas de simplices qui veuillent dire quoi que ce soit
            if ( ! pc.bounded() ) {
                sum = std::numeric_limits<TF>::max();
                return;
            }
            pc.for_each_simplex( [&]( const auto &chain ) {
                sum += dens.integrate_over_simplex( simplex_points<TF,D>( pc, chain ) );
            } );
        }
    } );
    res = sum;
    return ! ws.overflow;
}

/// le volume du simplexe `pts` : `| det( p_i - p_0 ) | / D!`
template<class TF,int D>
TF simplex_volume( const auto &pts ) {
    const auto M = Matrix<TF,D>::with_func( [&]( auto r, auto c ) { return pts[ int( c ) + 1 ][ int( r ) ] - pts[ 0 ][ int( r ) ]; } );
    TF det = M.determinant();
    if ( det < 0 ) det = - det;
    for ( int i = 2; i <= D; ++i )
        det /= i;
    return det;
}

/// les MOMENTS de `dist` sur `cell` : `mass = int rho`, `first = int x rho`, `second = int |x|^2 rho`
/// -- ce qu'il faut a un cout de transport ( `sum_i int_{cell_i} |x - p_i|^2 rho` ) et a ses
/// barycentres. Chaque morceau part en simplexes : a densite CONSTANTE ( `Image`, Lebesgue ) leurs
/// moments sont des formes fermees -- `int_T x = |T| g`, `g` le centre, et
/// `int_T |x|^2 = |T| ( sum_i |v_i|^2 + |sum_i v_i|^2 ) / ( ( D + 1 )( D + 2 ) )` -- sinon c'est la
/// quadrature de la densite qui les accumule ( `PointwiseDensity::integrate_moments_over_simplex` ).
template<class TF,class Local>
bool integrate_moments_into( auto &&mass, auto &&first, auto &&second, const Local &cell, Local &piece, const auto &dist ) {
    constexpr int D = Local::ct_dim;
    PieceWorkspace<Local> ws{ piece };
    TF m = 0, m2 = 0;
    auto mx = Vector<TF,D>::zeros();
    dist.for_each_piece( cell, ws, [&]( const auto &pc, const auto &dens ) {
        if ( ! pc.bounded() ) {
            m = std::numeric_limits<TF>::max();
            return;
        }
        if constexpr ( ! DECAYED_TYPE_OF( dens )::is_constant ) {
            pc.for_each_simplex( [&]( const auto &chain ) {
                dens.integrate_moments_over_simplex( simplex_points<TF,D>( pc, chain ), m, mx, m2 );
            } );
        } else {
            const TF rho = TF( dens.value );
            pc.for_each_simplex( [&]( const auto &chain ) {
                const auto pts = simplex_points<TF,D>( pc, chain );
                const TF w = rho * simplex_volume<TF,D>( pts );
                auto sum = Vector<TF,D>::zeros();
                TF sq = 0;
                for ( int k = 0; k <= D; ++k ) {
                    for ( int d = 0; d < D; ++d )
                        sum[ d ] += pts[ k ][ d ];
                    sq += norm_2_p2( pts[ k ] );
                }
                m += w;
                for ( int d = 0; d < D; ++d )
                    mx[ d ] += w * sum[ d ] / ( D + 1 );
                m2 += w * ( sq + norm_2_p2( sum ) ) / ( ( D + 1 ) * ( D + 2 ) );
            } );
        }
    } );
    mass = m;
    for ( int d = 0; d < D; ++d )
        first( d ) = mx[ d ];
    second = m2;
    return ! ws.overflow;
}

/// `res( k )` = la mesure de la cellule `k`, pour les germes de ce work-item -- une boucle striee :
/// `nb_threads` work-items se partagent les cellules, dans l'ordre du stockage ( deux germes
/// consecutifs y sont voisins dans l'espace ). `scratch` est le tenseur de travail du work-item --
/// et ou l'on dit qu'il a manque.
template<class PD>
void measures( const PD &pd, auto &&res, const auto &dom, auto &&scratch, const auto &dist,
               SI thread_index, SI nb_threads ) {
    using TF    = typename PD::TF;
    using TK    = KernelType<DECAYED_TYPE_OF( scratch )>;
    using Local = typename DECAYED_TYPE_OF( dom )::template Local<TK>;
    constexpr int nbc = nb_work_cells<DECAYED_TYPE_OF( dist )>();
    Carver cv = carver_of( scratch );
    const SI cap = cap_in<Local,TF>( cv.nb_words, nbc, false );
    Local c, piece;
    c.attach( cv, cap );
    if constexpr ( nbc > 1 ) piece.attach( cv, cap );
    else                     piece = c;                  // jamais touchee : la densite ne decoupe pas

    const SI n = pd.nb_seeds();
    for ( SI k = thread_index; k < n; k += nb_threads ) {
        if ( ! make_cell( pd, c, k, dom ) || ! integrate_into<TF>( res( pd.user_id( k ) ), c, piece, dist ) ) {
            ask_more( scratch, cv );
            return;
        }
    }
}

/// les moments de chaque cellule ( voir `integrate_moments_into` ), meme balayage que `measures`
template<class PD>
void moments( const PD &pd, auto &&mass, auto &&first, auto &&second, const auto &dom, auto &&scratch, const auto &dist,
              SI thread_index, SI nb_threads ) {
    using TF    = typename PD::TF;
    using TK    = KernelType<DECAYED_TYPE_OF( scratch )>;
    using Local = typename DECAYED_TYPE_OF( dom )::template Local<TK>;
    constexpr int nbc = nb_work_cells<DECAYED_TYPE_OF( dist )>();
    Carver cv = carver_of( scratch );
    const SI cap = cap_in<Local,TF>( cv.nb_words, nbc, false );
    Local c, piece;
    c.attach( cv, cap );
    if constexpr ( nbc > 1 ) piece.attach( cv, cap );
    else                     piece = c;

    const SI n = pd.nb_seeds();
    for ( SI k = thread_index; k < n; k += nb_threads ) {
        const SI u = pd.user_id( k );
        if ( ! make_cell( pd, c, k, dom ) || ! integrate_moments_into<TF>( mass( u ), first( u ), second( u ), c, piece, dist ) ) {
            ask_more( scratch, cv );
            return;
        }
    }
}

// ---- l'adjoint -----------------------------------------------------------------------------------
// La chaine est `m_k <- sommets <- plans <- germes`, et chaque fleche est une forme fermee :
// `measure_bwd` repond a la premiere ; un sommet est le COIN de ses `D` coupes, donc il resout
// `A x = b` avec les directions des coupes en lignes, et une petite resolution par sommet renvoie
// sa cotangente sur ses plans ( `scatter_cell_grad` ) ; un plan est la bissectrice ponderee de
// deux germes, qui se derive en deux lignes. Rien du forward n'est garde : la cellule est REFAITE.

/// `grad_vp` ( une cotangente par sommet de `cell` ) -> les germes. Tout autre germe que la cellule
/// touche recoit un ajout atomique ; la part de `k0`, a laquelle CHAQUE sommet contribue, est
/// sommee en registre et ajoutee une fois.
template<class PD>
void scatter_cell_grad( const PD &pd, SI k0, const auto &cell, const auto &grad_vp, auto &&grad_positions, auto &&grad_weights ) {
    using TF = typename PD::TF;
    constexpr int D = PD::ct_dim;
    if constexpr ( CT_VALUE( grad_positions.surely_null() ) && CT_VALUE( grad_weights.surely_null() ) ) {
        return;
    } else {
        if ( ! cell.bounded() || cell.nb_vertices() == 0 )
            return;

        auto atomic_add_to = []( auto &&dst, TF v ) {
            if constexpr ( ! CT_VALUE( dst.surely_null() ) )
                atomic_add( dst.ref(), v );
        };

        const auto p0 = pd.point( k0 );
        auto acc_p0 = Vector<TF,D>::zeros();
        TF acc_w0 = 0;

        const int nv = cell.nb_vertices();
        for ( int v = 0; v < nv; ++v ) {
            const auto q = Vector<TF,D>::with_func( [&]( PI d ) { return grad_vp( v, int( d ) ); } );
            const auto x = Vector<TF,D>::with_func( [&]( PI d ) { return TF( cell.coord( v, int( d ) ) ); } );

            // `A x = b`, lignes = les directions des coupes du sommet. Une cotangente `q` sur `x`
            // atteint les plans par `u` avec `A^T u = q` : `d off_r -> u[ r ]`, `d dir_r -> - u[ r ] * x`.
            // Les lignes qui ne font pas face a un germe sont relues sur la geometrie : leur echelle
            // ne change rien aux `u` des lignes qui comptent.
            Matrix<TF,D> At;
            int cuts[ D ];
            for ( int r = 0; r < D; ++r ) {
                cuts[ r ] = cell.vertex_cut( v, r );
                Vector<TF,D> dir;
                TF off;
                plane_of( pd, cell, k0, cuts[ r ], dir, off );
                for ( int c = 0; c < D; ++c )
                    At( c, r ) = dir[ c ];
            }
            const auto u = Matrix<TF,D>::solve_ge( At, q );

            for ( int r = 0; r < D; ++r ) {
                const int k1 = cell.cid[ cuts[ r ] ];
                if ( k1 < 0 )                            // le domaine, un morceau, ou une paroi factice
                    continue;
                // le plan de la paire `( k0, k1 )` : `dir = p1 - p0`,
                // `off = ( |p1|^2 - |p0|^2 ) / 2 + ( w0 - w1 ) / 2`. Les deux se derivent sur place.
                const TF g_off = u[ r ];
                const auto p1 = pd.point( k1 );
                for ( int d = 0; d < D; ++d ) {
                    const TF g_dir = - g_off * x[ d ];
                    atomic_add_to( grad_positions( k1, d ), g_dir + g_off * p1[ d ] );
                    acc_p0[ d ] -= g_dir + g_off * p0[ d ];
                }
                atomic_add_to( grad_weights( k1 ), - g_off / 2 );
                acc_w0 += g_off / 2;
            }
        }

        for ( int d = 0; d < D; ++d )
            atomic_add_to( grad_positions( k0, d ), acc_p0[ d ] );
        atomic_add_to( grad_weights( k0 ), acc_w0 );
    }
}

template<class PD,class Local>
bool integrate_bwd_into( const PD &pd, SI k0, auto &&grad_res, const Local &cell, Local &piece,
                         GradVp<typename PD::TF> &grad_vp, auto &&grad_positions, auto &&grad_weights,
                         auto &&grad_dist, const auto &dist ) {
    using TF = typename PD::TF;
    constexpr int D = PD::ct_dim;
    PieceWorkspace<Local> ws{ piece };
    const TF g = grad_res;
    dist.for_each_piece( cell, ws, [&]( const auto &pc, const auto &dens ) {
        if constexpr ( DECAYED_TYPE_OF( dens )::is_constant ) {
            // la part de la DENSITE : la masse est lineaire en elle, donc la derivee par rapport a
            // la valeur portee par ce morceau EST son volume ( un morceau infini n'en a pas )
            if ( pc.bounded() )
                dens.add_value_grad( grad_dist, g * pc.template measure<TF>() );
            // ... et la part de la GEOMETRIE, par la chaine habituelle
            pc.template measure_bwd<TF>( g * dens.value, grad_vp );
            scatter_cell_grad( pd, k0, pc, grad_vp, grad_positions, grad_weights );
        } else {
            if ( ! pc.bounded() )
                return;
            const int nv = pc.nb_vertices();
            for ( int v = 0; v < nv; ++v )
                for ( int c = 0; c < D; ++c )
                    grad_vp( v, c ) = 0;
            pc.for_each_simplex( [&]( const auto &chain ) {
                auto grad_pts = Vector<Vector<TF,D>,D+1>( Function(), [&]( PI ) { return Vector<TF,D>::zeros(); } );
                dens.integrate_over_simplex_bwd( simplex_points<TF,D>( pc, chain ), g, grad_pts, grad_dist );
                for ( int k = 0; k <= D; ++k )
                    for ( int c = 0; c < D; ++c )
                        grad_vp( int( chain[ k ] ), c ) += grad_pts[ k ][ c ];
            } );
            scatter_cell_grad( pd, k0, pc, grad_vp, grad_positions, grad_weights );
        }
    } );
    return ! ws.overflow;
}

template<class PD>
void measures_bwd( const PD &pd, auto &&res, const auto &dom, auto &&grad_res, auto &&grad_positions, auto &&grad_weights,
                   auto &&scratch, const auto &dist, auto &&grad_dist, SI thread_index, SI nb_threads ) {
    using TF    = typename PD::TF;
    using TK    = KernelType<DECAYED_TYPE_OF( scratch )>;
    using Local = typename DECAYED_TYPE_OF( dom )::template Local<TK>;
    constexpr int nbc = nb_work_cells<DECAYED_TYPE_OF( dist )>();
    Carver cv = carver_of( scratch );
    const SI cap = cap_in<Local,TF>( cv.nb_words, nbc, true );
    Local c, piece;
    c.attach( cv, cap );
    if constexpr ( nbc > 1 ) piece.attach( cv, cap );
    else                     piece = c;
    GradVp<TF> grad_vp{ cv.take<TF>( PD::ct_dim * cap ), cap };

    const SI n = pd.nb_seeds();
    for ( SI k = thread_index; k < n; k += nb_threads ) {
        if ( ! make_cell( pd, c, k, dom )
          || ! integrate_bwd_into( pd, k, grad_res( pd.user_id( k ) ), c, piece, grad_vp, grad_positions, grad_weights, grad_dist, dist ) ) {
            ask_more( scratch, cv );
            return;
        }
    }
}

/// La cellule du germe `k`, GARDEE : construite comme une autre, puis posee dans `res` -- avec ses
/// identifiants de coupe traduits pour l'utilisateur. La seule requete dont la memoire est fonction
/// du nombre de germes : ce qu'est un AFFICHAGE.
template<class PD>
void build_cell( const PD &pd, SI k, const auto &dom, auto &&res, auto &&scratch, SI thread_index ) {
    using TK    = KernelType<DECAYED_TYPE_OF( scratch )>;
    using Local = typename DECAYED_TYPE_OF( dom )::template Local<TK>;
    Carver cv = carver_of( scratch, thread_index );
    Local c = local_on<Local>( scratch, cv );
    if ( ! make_cell( pd, c, k, dom ) ) {
        ask_more( scratch, cv );
        return;
    }
    c.tidy();
    for ( int q = 0; q < c.nb_cuts(); ++q )
        if ( c.cid[ q ] >= 0 )
            c.cid[ q ] = int( pd.user_id( c.cid[ q ] ) );
    c.store( res );
}

/// UNE LIGNE DE LA HESSIENNE du transport, `d m_k / d w_j` pour les voisins `j` de la cellule `k` :
/// la bissectrice `( k, j )` glisse de `dw / ( 2 | p_k - p_j | )` quand `w_j` monte de `dw`, et ce
/// qu'elle emporte est la densite integree sur la FACETTE commune. Donc
/// `d m_k / d w_j = - int_{facette} rho / ( 2 | p_k - p_j | )`, et `d m_k / d w_k` en est l'oppose
/// somme ( une ligne somme a zero -- ce que l'appelant recompose ). Le domaine ( ids negatifs ) ne
/// bouge pas. `res` : `ids( r )` ( identifiants utilisateur ) et `vals( r )` ( les valeurs, POSITIVES ),
/// `nb_nbrs` le compte -- une capacite que loom double si elle manque.
///
/// A densite constante par morceau seulement ( `Image`, Lebesgue ) : la facette d'un morceau est
/// plate et la densite y est un nombre.
template<class PD>
void hessian_row( const PD &pd, SI k, const auto &dom, auto &&res, auto &&scratch, SI thread_index, const auto &dist ) {
    using TF    = typename PD::TF;
    using TK    = KernelType<DECAYED_TYPE_OF( scratch )>;
    using Local = typename DECAYED_TYPE_OF( dom )::template Local<TK>;
    constexpr int D = PD::ct_dim;
    constexpr int nbc = nb_work_cells<DECAYED_TYPE_OF( dist )>();
    Carver cv = carver_of( scratch, thread_index );
    const SI cap = cap_in<Local,TF>( cv.nb_words, nbc, false );
    Local c, piece;
    c.attach( cv, cap );
    if constexpr ( nbc > 1 ) piece.attach( cv, cap );
    else                     piece = c;
    if ( ! make_cell( pd, c, k, dom ) ) {
        ask_more( scratch, cv );
        return;
    }
    c.tidy();

    // les voisins de la cellule sont ses coupes vivantes : une case par coupe, accumulee morceau
    // par morceau ( un morceau porte les memes identifiants, plus ceux de son pave )
    const int nc = c.nb_cuts();
    if ( ! res.nb_nbrs.set( nc ) )
        return;                                          // trop de voisins : loom double et rappelle
    for ( int q = 0; q < nc; ++q ) {
        res.ids( q ) = c.cid[ q ] >= 0 ? int( pd.user_id( c.cid[ q ] ) ) : int( c.cid[ q ] );
        res.vals( q ) = 0;
    }
    const auto p0 = pd.point( k );

    PieceWorkspace<Local> ws{ piece };
    dist.for_each_piece( c, ws, [&]( const auto &pc, const auto &dens ) {
        static_assert( DECAYED_TYPE_OF( dens )::is_constant, "hessian : densite constante par morceau seulement" );
        const TF rho = TF( dens.value );
        pc.template for_each_facet<TF>( [&]( int cut, TF mes ) {
            const int id = pc.cid[ cut ];
            if ( id < 0 )
                return;                                  // le domaine, ou un bord de pave : immobile
            const auto pj = pd.point( id );
            TF d2 = 0;
            for ( int d = 0; d < D; ++d )
                d2 += ( pj[ d ] - p0[ d ] ) * ( pj[ d ] - p0[ d ] );
            const TF val = rho * mes / ( 2 * sycl::sqrt( d2 ) );
            for ( int q = 0; q < nc; ++q )
                if ( c.cid[ q ] == id ) {
                    res.vals( q ) += val;
                    break;
                }
        } );
    } );
    if ( ws.overflow )
        ask_more( scratch, cv );
}

} // namespace diagram
} // namespace sdot

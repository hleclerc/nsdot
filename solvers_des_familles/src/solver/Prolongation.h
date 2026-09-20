#pragma once

// =====================================================================================
// LE MULTI-ECHELLE, COTE PROLONGATION : on a resolu sur les REPRESENTANTS ( un germe par paquet,
// la masse du paquet ), et il faut des poids pour tous les autres germes du niveau fin.
//
// = Les paquets
//
// Un BSP a feuilles de `R` germes : une feuille est un paquet, spatialement compact, et son
// representant est le germe le plus proche de son centre. Le niveau grossier est le nuage des
// representants, la cible `nu_k = |paquet| / n`.
//
// = Les prolongations
//
//   COPIE        `w_i = w_k` : la litterature. Un SAUT entre deux paquets voisins, et une cellule
//                est vide des que le saut depasse le carre de la distance des deux germes.
//   HARMONIQUE   la proposition : les representants sont IMPOSES, les autres poids resolvent
//                l'equation de la chaleur du graphe de Voronoi fin, `( L w )_i = 0` -- chaque
//                poids libre est la moyenne ( ponderee par `c_ij` ) de ses voisins. Continue, et
//                sans saut. Le parametre `mu` RELACHE la condition imposee : on minimise
//                    1/2 sum_ij c_ij ( w_i - w_j )^2 + mu/2 sum_k d_k ( w_k - w_k^c )^2
//                soit `w_k = ( moy_j w_j + mu w_k^c ) / ( 1 + mu )` sur un representant ; `mu =
//                INFINI` est la condition exacte, `mu -> 0` la constante ( Voronoi ). C'est la
//                lecture INVARIANTE PAR JAUGE de « ( (1-t) I + t M ) w* = t w » : on ne peut pas
//                faire decroitre `w*` vers ZERO ( ajouter une constante aux poids ne change
//                aucune cellule, il ne faut pas que la prolongation en depende ), on le fait
//                decroitre vers la moyenne de ses voisins. `t = mu / ( 1 + mu )`.
//   MLS          l'interpolation QUADRATIQUE par moindres carres mobiles : pour un germe fin, un
//                polynome de degre 2 ajuste sur les representants a deux anneaux ( graphe de
//                Voronoi grossier ), ponderes par une gaussienne centree sur le germe. Ce qu'une
//                prolongation harmonique ne peut pas rendre, c'est la COURBURE : la solution `w`
//                du transport a une hessienne proche de `2 I` la ou les cellules sont etirees
//                ( `|p|^2 - w` est presque plat ), et l'harmonique, qui annule le laplacien
//                entre les representants, la concentre en PLIS sur eux -- du mauvais cote : la
//                cellule du representant se vide. Le quadratique porte la courbure.
//   CTRANSF      `w_i = max_l ( w_l - |p_i - p_l|^2 )`, `l` parmi le representant et ses voisins
//                de Laguerre grossiers : la parabole du germe fin touche le potentiel grossier
//                ( `2d_des_familles`, JOURNAL ). Dans une cellule grossiere, les cellules fines
//                sont alors le Voronoi du paquet DILATE d'un facteur 2 depuis le representant,
//                coupe par la cellule grossiere : les germes du fond du paquet n'y sont plus.
//
// = Quand ca ne passe pas
//
// Le diagramme fin dit s'il reste des cellules sous `seuil * nu_i`. Trois corrections, chacune
// avec son `t`, divise par deux tant que ca ne passe pas :
//   PENAL        `mu = t / ( 1 - t )` dans HARMONIQUE ( la proposition ) ;
//   RETRAIT      `t * w` : l'homothetie vers Voronoi, `t = 0` est toujours admissible ;
//   JACOBI       `k = 1, 2, 4, ...` balayages de Jacobi amorti ( `w <- w - 2/3 D^-1 L w` ) sur
//                tout le niveau, representants compris : le lissage litteral.
// =====================================================================================

#include "accel/AaBsp.h"
#include "diagram/PowerDiagram.h"
#include "solver/Laplacien.h"
#include "solver/Lineaire.h"
#include <Eigen/Dense>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>
#include <vector>

namespace sf {

/// LES PAQUETS d'un niveau, et le representant de chacun.
struct Paquets {
    SI              nb = 0;
    std::vector<SI> paquet;        ///< par germe fin : son paquet
    std::vector<SI> rep;           ///< par paquet : le germe fin qui le represente
    std::vector<SI> taille;        ///< par paquet
};

template<int D>
Paquets fait_paquets( const TF *const *P, SI n, SI R ) {
    AaBspT<D> arbre;
    arbre.build( P, nullptr, n, R );
    Paquets pq;
    pq.paquet.assign( n, -1 );
    for ( const auto &nd : arbre.nodes ) {
        if ( nd.right >= 0 ) continue;
        const SI k = pq.nb++;
        TF c[ D ] = {};
        for ( SI q = nd.beg; q < nd.end; ++q )
            for ( int d = 0; d < D; ++d ) c[ d ] += arbre.p[ d ][ q ];
        for ( int d = 0; d < D; ++d ) c[ d ] /= TF( nd.end - nd.beg );
        SI best = nd.beg; TF bd = std::numeric_limits<TF>::infinity();
        for ( SI q = nd.beg; q < nd.end; ++q ) {
            TF dd = 0;
            for ( int d = 0; d < D; ++d ) { const TF e = arbre.p[ d ][ q ] - c[ d ]; dd += e * e; }
            if ( dd < bd ) { bd = dd; best = q; }
            pq.paquet[ arbre.order[ q ] ] = k;
        }
        pq.rep.push_back( arbre.order[ best ] );
        pq.taille.push_back( nd.end - nd.beg );
    }
    return pq;
}

/// LE LAPLACIEN DU DIAGRAMME que `pd` porte ( Voronoi si `W == nullptr`, sinon `set_weights` ).
template<class PD>
SI laplacien_de( PD &pd, const TF *const *P, const TF *W, const Parallel &par, Laplacien &L, std::vector<TF> *aires = nullptr ) {
    constexpr int D = PD::dim;
    if ( W ) pd.set_weights( W, par );
    std::vector<std::vector<Facette>> par_th( std::max( par.threads, 1 ) );
    std::vector<TF> res;
    const SI deb = pd.measures_and_facets( res, par, [ & ]( int t, SI i, SI j, TF mes ) {
        TF d2 = 0;
        for ( int d = 0; d < D; ++d ) { const TF e = P[ d ][ j ] - P[ d ][ i ]; d2 += e * e; }
        if ( d2 > 0 ) par_th[ t ].push_back( Facette{ i, j, mes / ( 2 * std::sqrt( d2 ) ) } );
    } );
    std::vector<Facette> fa;
    for ( auto &v : par_th ) fa.insert( fa.end(), v.begin(), v.end() );
    L.assemble( pd.n, fa );
    if ( aires ) aires->swap( res );
    return deb;
}

// ------------------------------------------------------------------------------------ COPIE
inline void prolonge_copie( const Paquets &pq, const std::vector<TF> &wc, std::vector<TF> &w ) {
    const SI n = SI( pq.paquet.size() );
    w.resize( n );
    for ( SI i = 0; i < n; ++i ) w[ i ] = wc[ pq.paquet[ i ] ];
}

// ------------------------------------------------------------------------------------ HARMONIQUE
/// `mu = INFINI` : les representants exactement imposes ( systeme sur les libres seuls ).
/// Rend `false` si la factorisation echoue.
inline bool prolonge_harmonique( const Laplacien &L, const Paquets &pq, const std::vector<TF> &wc, TF mu, std::vector<TF> &w ) {
    using SpM = Eigen::SparseMatrix<double>;
    const SI n = L.n;
    std::vector<SI> imp( n, -1 );                       // germe -> paquet s'il est representant
    for ( SI k = 0; k < pq.nb; ++k ) imp[ pq.rep[ k ] ] = k;
    std::vector<Eigen::Triplet<double>> tri;
    tri.reserve( size_t( L.row[ n ] ) / 2 + n );
    Eigen::VectorXd b = Eigen::VectorXd::Zero( n );
    const bool exact = ! ( mu < std::numeric_limits<TF>::infinity() );
    for ( SI i = 0; i < n; ++i ) {
        if ( exact && imp[ i ] >= 0 ) {                 // ligne identite, valeur imposee
            tri.emplace_back( i, i, 1.0 );
            b[ i ] = double( wc[ imp[ i ] ] );
            continue;
        }
        double dia = double( L.dia[ i ] );
        if ( imp[ i ] >= 0 ) { dia += double( mu * L.dia[ i ] ); b[ i ] = double( mu * L.dia[ i ] * wc[ imp[ i ] ] ); }
        tri.emplace_back( i, i, dia );
        for ( SI e = L.row[ i ]; e < L.row[ i + 1 ]; ++e ) {
            const SI j = L.col[ e ];
            if ( exact && imp[ j ] >= 0 ) { b[ i ] += double( L.c[ e ] * wc[ imp[ j ] ] ); continue; }
            if ( j < i ) tri.emplace_back( i, j, -double( L.c[ e ] ) );
        }
    }
    SpM A( n, n );
    A.setFromTriplets( tri.begin(), tri.end() );
    Eigen::SimplicialLDLT<SpM, Eigen::Lower, Eigen::AMDOrdering<int>> so;
    so.compute( A );
    if ( so.info() != Eigen::Success ) return false;
    const Eigen::VectorXd x = so.solve( b );
    if ( so.info() != Eigen::Success ) return false;
    w.resize( n );
    for ( SI i = 0; i < n; ++i ) w[ i ] = TF( x[ i ] );
    return true;
}

// ------------------------------------------------------------------------------------ MLS
/// `Pc` les positions des representants ( ordre des paquets ), `Lc` leur graphe ( Voronoi
/// grossier ), `wc` leurs poids. Rend le nombre de germes retombes sur un ajustement lineaire ou
/// sur la copie ( pochoir trop pauvre ou singulier ).
template<int D>
SI prolonge_mls( const TF *const *P, const Paquets &pq, const TF *const *Pc, const Laplacien &Lc,
                 const std::vector<TF> &wc, const Parallel &par, std::vector<TF> &w, int anneaux = 2, TF largeur = 1 ) {
    constexpr int NQ = D == 2 ? 6 : 10;                  // 1, p, p (x) p
    const SI n = SI( pq.paquet.size() );
    w.resize( n );
    // les germes de chaque paquet, contigus
    std::vector<SI> deb( pq.nb + 1, 0 ), membres( n );
    for ( SI i = 0; i < n; ++i ) ++deb[ pq.paquet[ i ] + 1 ];
    for ( SI k = 0; k < pq.nb; ++k ) deb[ k + 1 ] += deb[ k ];
    { std::vector<SI> at( deb.begin(), deb.end() - 1 ); for ( SI i = 0; i < n; ++i ) membres[ at[ pq.paquet[ i ] ]++ ] = i; }

    std::atomic<SI> retombes{ 0 };
    parallel_for( pq.nb, par, [ & ]( SI k, int ) {
        // le pochoir : deux anneaux autour de `k`, et l'echelle = distance moyenne au premier
        std::vector<SI> st{ k };
        TF H = 0; int nh = 0;
        for ( SI e = Lc.row[ k ]; e < Lc.row[ k + 1 ]; ++e ) {
            const SI l = Lc.col[ e ];
            st.push_back( l );
            TF dd = 0; for ( int d = 0; d < D; ++d ) { const TF x = Pc[ d ][ l ] - Pc[ d ][ k ]; dd += x * x; }
            H += std::sqrt( dd ); ++nh;
        }
        for ( int an = 1, deb = 1; an < anneaux; ++an ) {
            const SI fin = SI( st.size() );
            for ( SI q = deb; q < fin; ++q )
                for ( SI e = Lc.row[ st[ q ] ]; e < Lc.row[ st[ q ] + 1 ]; ++e )
                    if ( std::find( st.begin(), st.end(), Lc.col[ e ] ) == st.end() ) st.push_back( Lc.col[ e ] );
            deb = fin;
        }
        H = ( nh ? H / nh : TF( 1 ) ) * largeur;
        for ( SI m = deb[ k ]; m < deb[ k + 1 ]; ++m ) {
            const SI i = membres[ m ];
            // l'ajustement, centre sur `p_i`, coordonnees en unites de `H`
            Eigen::Matrix<double,NQ,NQ> A = Eigen::Matrix<double,NQ,NQ>::Zero();
            Eigen::Matrix<double,NQ,1>  r = Eigen::Matrix<double,NQ,1>::Zero();
            for ( SI l : st ) {
                double q[ NQ ], x[ D ], dd = 0;
                for ( int d = 0; d < D; ++d ) { x[ d ] = double( Pc[ d ][ l ] - P[ d ][ i ] ) / double( H ); dd += x[ d ] * x[ d ]; }
                const double om = std::exp( -dd );
                int c = 0;
                q[ c++ ] = 1;
                for ( int d = 0; d < D; ++d ) q[ c++ ] = x[ d ];
                for ( int d = 0; d < D; ++d ) for ( int e = d; e < D; ++e ) q[ c++ ] = x[ d ] * x[ e ];
                for ( int a = 0; a < NQ; ++a ) { r[ a ] += om * q[ a ] * double( wc[ l ] ); for ( int b = 0; b < NQ; ++b ) A( a, b ) += om * q[ a ] * q[ b ]; }
            }
            bool ok = false;
            if ( SI( st.size() ) >= NQ + 2 ) {
                Eigen::LDLT<Eigen::Matrix<double,NQ,NQ>> so( A );
                if ( so.info() == Eigen::Success && so.isPositive() ) {
                    const Eigen::Matrix<double,NQ,1> s = so.solve( r );
                    if ( ( A * s - r ).norm() <= 1e-8 * ( r.norm() + 1e-300 ) ) { w[ i ] = TF( s[ 0 ] ); ok = true; }
                }
            }
            if ( ! ok ) {                                // lineaire, puis copie
                ++retombes;
                const Eigen::Matrix<double,D+1,D+1> Al = A.template topLeftCorner<D+1,D+1>();
                const Eigen::Matrix<double,D+1,1>  rl = r.template head<D+1>();
                Eigen::LDLT<Eigen::Matrix<double,D+1,D+1>> so( Al );
                if ( SI( st.size() ) >= D + 2 && so.info() == Eigen::Success && so.isPositive() ) w[ i ] = TF( so.solve( rl )[ 0 ] );
                else w[ i ] = wc[ k ];
            }
        }
    } );
    return retombes.load();
}

// ------------------------------------------------------------------------------------ CTRANSF
/// `Lc` le graphe de LAGUERRE grossier ( la solution grossiere ) : le max se cherche sur le
/// representant et ses voisins de Laguerre.
template<int D>
void prolonge_ctransf( const TF *const *P, const Paquets &pq, const TF *const *Pc, const Laplacien &Lc,
                       const std::vector<TF> &wc, std::vector<TF> &w ) {
    const SI n = SI( pq.paquet.size() );
    w.resize( n );
    for ( SI i = 0; i < n; ++i ) {
        const SI k = pq.paquet[ i ];
        auto val = [ & ]( SI l ) {
            TF dd = 0; for ( int d = 0; d < D; ++d ) { const TF e = P[ d ][ i ] - Pc[ d ][ l ]; dd += e * e; }
            return wc[ l ] - dd;
        };
        TF best = val( k );
        for ( SI e = Lc.row[ k ]; e < Lc.row[ k + 1 ]; ++e ) best = std::max( best, val( Lc.col[ e ] ) );
        w[ i ] = best;
    }
}

// ------------------------------------------------------------------------------------ corrections
/// `nb` balayages de Jacobi amorti sur le graphe `L`.
inline void lisse_jacobi( const Laplacien &L, int nb, std::vector<TF> &w ) {
    const SI n = L.n;
    std::vector<TF> v( n );
    for ( int s = 0; s < nb; ++s ) {
        for ( SI i = 0; i < n; ++i ) {
            TF lw = L.dia[ i ] * w[ i ];
            for ( SI e = L.row[ i ]; e < L.row[ i + 1 ]; ++e ) lw -= L.c[ e ] * w[ L.col[ e ] ];
            v[ i ] = w[ i ] - TF( 2. / 3 ) * lw / L.dia[ i ];
        }
        w.swap( v );
    }
}

/// `-psi( x ) = max_{j != hors} ( w_j - |x - p_j|^2 )`, cherche dans l'arbre de `pd` ( qui porte `w` ) :
/// sur une boite, `w` est majore par son majorant affine, donc `w_j - |x - p_j|^2 <= max_boite( w )
/// - dist^2( x, boite )` -- l'argument de l'elagage, sans cellule.
template<class PD>
TF moins_psi( const PD &pd, const TF *x, SI hors ) {
    constexpr int D = PD::dim;
    const auto &tr = pd.arbre;
    TF best = -std::numeric_limits<TF>::infinity();
    std::vector<SI> pile{ 0 };
    while ( ! pile.empty() ) {
        const auto &nd = tr.nodes[ pile.back() ];
        pile.pop_back();
        TF dd = 0, wmax = nd.wm.b;
        for ( int d = 0; d < D; ++d ) {
            const TF e = x[ d ] < nd.lo[ d ] ? nd.lo[ d ] - x[ d ] : ( x[ d ] > nd.hi[ d ] ? x[ d ] - nd.hi[ d ] : TF( 0 ) );
            dd += e * e;
            const TF s = TF( nd.wm.a[ d ] ) * nd.lo[ d ], t = TF( nd.wm.a[ d ] ) * nd.hi[ d ];
            wmax += s > t ? s : t;
        }
        if ( ! ( wmax - dd > best ) ) continue;
        if ( nd.right < 0 ) {
            for ( SI k = nd.beg; k < nd.end; ++k ) {
                if ( tr.order[ k ] == hors ) continue;
                TF d2 = 0;
                for ( int d = 0; d < D; ++d ) { const TF e = x[ d ] - tr.p[ d ][ k ]; d2 += e * e; }
                best = std::max( best, tr.pw[ k ] - d2 );
            }
        } else {
            pile.push_back( SI( &nd - tr.nodes.data() ) + 1 );
            pile.push_back( nd.right );
        }
    }
    return best;
}

/// LE RATTRAPAGE : les cellules sous `plancher` sont relevees a `-psi( p_i ) + marge * h_i^2` ( `h_i^2`
/// la distance moyenne au carre aux voisins de Voronoi ) -- le germe rentre dans sa cellule, avec un
/// peu d'air -- et on recommence tant qu'il en reste, `passes` fois au plus. Rend le nombre de cellules
/// encore sous le plancher ; `releves` compte les relevements.
template<class PD>
SI rattrape( PD &pd, const TF *const *P, const Laplacien &Lvor, std::vector<TF> &w, TF plancher, TF marge, int passes,
             const Parallel &par, SI &releves, int &faites ) {
    constexpr int D = PD::dim;
    const SI n = pd.n;
    std::vector<TF> a;
    releves = 0;
    for ( faites = 0; faites < passes; ++faites ) {
        pd.set_weights( w.data(), par );
        pd.measures( a, par );
        std::vector<SI> vides;
        for ( SI i = 0; i < n; ++i ) if ( a[ i ] < plancher ) vides.push_back( i );
        if ( vides.empty() ) return 0;
        std::vector<TF> neuf( vides.size() );
        parallel_for( SI( vides.size() ), par, [ & ]( SI q, int ) {
            const SI i = vides[ q ];
            TF x[ D ], h2 = 0; int nh = 0;
            for ( int d = 0; d < D; ++d ) x[ d ] = P[ d ][ i ];
            for ( SI e = Lvor.row[ i ]; e < Lvor.row[ i + 1 ]; ++e ) {
                const SI j = Lvor.col[ e ];
                TF d2 = 0; for ( int d = 0; d < D; ++d ) { const TF t = P[ d ][ j ] - P[ d ][ i ]; d2 += t * t; }
                h2 += d2; ++nh;
            }
            neuf[ q ] = moins_psi( pd, x, i ) + marge * ( nh ? h2 / nh : TF( 0 ) );
        } );
        for ( SI q = 0; q < SI( vides.size() ); ++q ) w[ vides[ q ] ] = std::max( w[ vides[ q ] ], neuf[ q ] );
        releves += SI( vides.size() );
    }
    pd.set_weights( w.data(), par );
    pd.measures( a, par );
    SI reste = 0;
    for ( SI i = 0; i < n; ++i ) reste += a[ i ] < plancher;
    return reste;
}

/// LE RELEVEMENT MINIMAL ( `scripts/adoucissement_1d.py` ) : une cellule vide NAIT en un sommet du
/// diagramme des autres, et le poids qui l'y fait naitre est le plus petit qui la rende non vide. On
/// le cherche par BISSECTION sur le poids de `i` seul ( `cellule_avec_poids` ), entre `w_i` ( vide )
/// et `-psi( p_i )` ( `p_i` dans sa cellule : non vide a coup sur ), jusqu'a une aire dans
/// `[ cible, 2 cible ]`, `cible = eps * min( nu_i, |Vor_i| )`. Toutes les vides d'une passe sur le
/// meme diagramme, puis on recommence tant qu'il en reste ( deux vides nees au meme sommet se
/// disputent la place -- un ping-pong a l'echelle de `cible`, que `passes` borne ; doubler la cible
/// a chaque reprise, essaye, fait tout exploser ). Rend le nombre de cellules encore sous `plancher`.
template<class PD>
SI releve_minimal( PD &pd, const TF *const *P, const std::vector<TF> &nu, const std::vector<TF> &avor, std::vector<TF> &w,
                   TF plancher, TF eps, int passes, const Parallel &par, SI &releves, int &faites, SI &cellules ) {
    constexpr int D = PD::dim;
    const SI n = pd.n;
    std::vector<TF> a;
    std::vector<SI> rang( n );                              // identifiant -> rang dans l'arbre
    for ( SI k = 0; k < n; ++k ) rang[ pd.ids[ k ] ] = k;
    releves = 0; cellules = 0;
    std::atomic<SI> nb_cel{ 0 };
    for ( faites = 0; faites < passes; ++faites ) {
        pd.set_weights( w.data(), par );
        pd.measures( a, par );
        std::vector<SI> vides;
        for ( SI i = 0; i < n; ++i ) if ( a[ i ] < plancher ) vides.push_back( i );
        if ( vides.empty() ) { cellules = nb_cel.load(); return 0; }
        std::vector<TF> neuf( vides.size() );
        parallel_for( SI( vides.size() ), par, [ & ]( SI q, int ) {
            const SI i = vides[ q ], k = rang[ i ];
            TF x[ D ];
            for ( int d = 0; d < D; ++d ) x[ d ] = P[ d ][ i ];
            const TF cible = eps * std::min( nu[ i ], avor[ i ] );
            TF lo = w[ i ], hi = std::max( moins_psi( pd, x, i ) + cible, w[ i ] + cible );
            typename PD::Cell cel;
            // une cellule qui DEBORDE `MaxNv` est une cellule enorme : on la compte infinie, la bissection redescend
            auto aire = [ & ]( TF wk ) { ++nb_cel; return pd.cellule_avec_poids( k, wk, cel ) ? PD::mesure( cel ) : std::numeric_limits<TF>::infinity(); };
            TF ah = aire( hi );
            for ( int it = 0; it < 20 && ah < cible; ++it ) { hi += std::max( hi - lo, cible ); ah = aire( hi ); }   // par securite
            for ( int it = 0; it < 60; ++it ) {
                if ( ah <= 2 * cible ) break;
                const TF mid = ( lo + hi ) / 2, am = aire( mid );
                if ( am >= cible ) { hi = mid; ah = am; } else lo = mid;
            }
            neuf[ q ] = hi;
        } );
        for ( SI q = 0; q < SI( vides.size() ); ++q ) w[ vides[ q ] ] = neuf[ q ];
        releves += SI( vides.size() );
    }
    cellules = nb_cel.load();
    pd.set_weights( w.data(), par );
    pd.measures( a, par );
    SI reste = 0;
    for ( SI i = 0; i < n; ++i ) reste += a[ i ] < plancher;
    return reste;
}

/// LE TEST : le diagramme fin en `w`, et le nombre de cellules sous `seuil * nu_i`.
template<class PD>
SI teste_admissible( PD &pd, const std::vector<TF> &w, const std::vector<TF> &nu, TF seuil, const Parallel &par, TF &amin_rel, TF &pire ) {
    pd.set_weights( w.data(), par );
    std::vector<TF> a;
    pd.measures( a, par );
    SI mauvaises = 0;
    amin_rel = std::numeric_limits<TF>::infinity(); pire = 0;
    for ( SI i = 0; i < pd.n; ++i ) {
        amin_rel = std::min( amin_rel, a[ i ] / nu[ i ] );
        pire = std::max( pire, std::fabs( a[ i ] - nu[ i ] ) / nu[ i ] );
        mauvaises += a[ i ] < seuil * nu[ i ];
    }
    return mauvaises;
}

} // namespace sf

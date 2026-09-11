#pragma once

#include "spatial_accel/AaBsp.h"
#include "geometry/Cell.h"
#include "geometry/PowerDiagram.h"
#include "util/parallel.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <chrono>
#include <thread>
#include <vector>

namespace pd {

inline SI hull_rate = 12;       ///< germes par paquet
inline bool hull_init = true;   ///< partir de la sur-cellule plutot que du domaine
inline int hull_threads = 0;    ///< 0 = autant que de coeurs. Doit suivre `--threads`, sinon on
                                ///< compare une preparation PARALLELE a une passe sequentielle --
                                ///< ce qui faisait croire a un amortissement en 0.5 passe.

/// LA SUR-CELLULE ANISOTROPE : l'enveloppe convexe des cellules du paquet, et non plus une
/// paraboloide abaissee.
///
/// = Ce que ca change, et pourquoi c'est plus simple
///
/// L'idee precedente resumait un paquet par UN scalaire `delta`, ce qui donnait un DISQUE de rayon
/// `sqrt( delta )` -- une forme qui ne sait rien du paquet. Ici on prend directement
///
///     H( u ) = enveloppe convexe de l'union des `E_m`, pour `m` du paquet `u`
///
/// ou `E_m` est la cellule de `m` contre le seul sous-ensemble `S`. Comme `psi_S >= psi`, on a
/// `Lag( m ) contenu dans E_m`, donc `H( u )` contient toutes les cellules du paquet -- c'est un
/// certificat, par construction, et il ne reste NI delta, NI paraboloide.
///
/// = Le k-DOP, et pourquoi il suffit
///
/// L'enveloppe exacte est chere a intersecter. On garde sa fonction d'appui dans `K` directions
/// fixes : `H( u )` est alors contenue dans le polytope `{ x : d_i . x <= sup_i }`, donc travailler
/// sur ce dernier ne fait qu'AGRANDIR -- le certificat tient. Le test de chevauchement devient
/// `K` comparaisons :
///
///     disjoints s'il existe `i` tel que  `sup_i( u ) + sup_{ i + K/2 }( v ) < 0`
///
/// = Et la cellule part de la
///
/// `init_cell` initialise la cellule avec `H( u )` au lieu du carre unite. Le resultat est le meme
/// ensemble (`H ∩ tous les demi-plans = Lag`, puisque `Lag ⊂ H`), mais le test d'eviction mord des
/// la premiere boite au lieu d'attendre que la cellule ait retreci.
struct AaBspHull {
    static constexpr int dim = 2;

    /// Directions du k-DOP. MESURE : `K = 16` raccourcit bien les listes (25.5 -> 19.7 par
    /// cellule sur le cas dur a rho = 12) mais double les sommets de la sur-cellule, donc le cout
    /// de chaque `cut` au depart. Bilan a rho = 12 : 1.10x contre 1.23x sur l'uniforme, 1.09x
    /// contre 1.19x sur le Voronoi groupe, 1.73x contre 1.70x sur le cas dur -- et 50 % de memoire
    /// en plus. On reste a 8.
    static constexpr SI K = 8;
    struct Pk { SI node, beg, end; };

    AaBsp full;
    std::vector<Pk> pk;
    std::vector<SI> leaf_of;
    std::vector<SI> loff, lval;
    std::vector<TF> ldq;
    std::vector<TF> sup;                    ///< `K` par paquet
    std::vector<SI> hoff;                   ///< CSR des sommets de la sur-cellule
    std::vector<TF> hx, hy;
    TF wmax = 0;
    bool secu = true;

    static constexpr const char *name = "hull";

    static TF dirx( SI i ) { return std::cos( TF( 6.283185307179586 ) * i / K ); }
    static TF diry( SI i ) { return std::sin( TF( 6.283185307179586 ) * i / K ); }

    Vec<2> seed( SI k ) const { return full.seed( k ); }
    TF seed_x( SI k ) const { return full.seed_x( k ); }
    TF seed_y( SI k ) const { return full.seed_y( k ); }
    TF seed_w( SI k ) const { return full.seed_w( k ); }
    SI seed_id( SI k ) const { return full.order[ k ]; }
    SI nb_seeds() const { return full.nb_seeds(); }

    /// La cellule part de la SUR-CELLULE. Les plans des aretes se retrouvent depuis les sommets
    /// consecutifs -- inutile de les ranger : le polygone est en ordre direct, donc la normale
    /// SORTANTE de l'arete `i -> i+1` est `( vy_j - vy_i, vx_i - vx_j )`.
    template<class Cell>
    void init_cell( Cell &c, SI k0 ) const {
        if ( ! hull_init ) {
            c.init_as_unit_square();
            return;
        }
        const SI u = leaf_of[ k0 ], b = hoff[ u ], nb = hoff[ u + 1 ] - b;
        c.nb = nb;
        for ( SI i = 0; i < nb; ++i ) {
            const SI j = i + 1 < nb ? i + 1 : 0;
            c.vx[ i ] = hx[ b + i ]; c.vy[ i ] = hy[ b + i ];
            const TF dx = hy[ b + j ] - hy[ b + i ], dy = hx[ b + i ] - hx[ b + j ];
            c.cdx[ i ] = dx; c.cdy[ i ] = dy;
            c.co[ i ] = dx * hx[ b + i ] + dy * hy[ b + i ];
            c.cid[ i ] = -1;                    // une arete du DOMAINE elargi, pas un voisin
        }
    }

    size_t bytes() const {
        return sizeof( Pk ) * pk.size() + sizeof( SI ) * ( leaf_of.size() + loff.size()
                                                           + lval.size() + hoff.size() )
             + sizeof( TF ) * ( ldq.size() + sup.size() + hx.size() + hy.size() );
    }

    template<class MayCut, class CutWith, class Reach2>
    void for_each_candidate( SI k0, MayCut &&may_cut, CutWith &&cut_with, Reach2 &&reach2 ) const {
        const SI u = leaf_of[ k0 ], i0 = full.order[ k0 ];
        const Vec<2> p0 = full.seed( k0 );
        const TF w0 = full.seed_w( k0 );
        TF seuil = TF( 1e30 );
        SI suiv = 1;
        for ( SI q = loff[ u ]; q < loff[ u + 1 ]; ++q ) {
            if ( secu && q - loff[ u ] == suiv ) {
                const TF R2 = reach2(), R = std::sqrt( R2 );
                const TF s2 = R2 + wmax - w0;
                const TF t = R + ( s2 > 0 ? std::sqrt( s2 ) : TF( 0 ) );
                seuil = t * t;
                suiv *= 2;
            }
            if ( ldq[ q ] > seuil )
                return;
            if ( ! full.for_each_candidate_from( pk[ lval[ q ] ].node, p0, i0,
                                                 may_cut, cut_with, reach2 ) )
                return;
        }
    }

    static TF boite2( const AaBsp::Node &a, const AaBsp::Node &b ) {
        TF s = 0;
        for ( int d = 0; d < 2; ++d ) {
            const TF e = a.lo[ d ] > b.hi[ d ] ? a.lo[ d ] - b.hi[ d ]
                       : ( b.lo[ d ] > a.hi[ d ] ? b.lo[ d ] - a.hi[ d ] : TF( 0 ) );
            s += e * e;
        }
        return s;
    }

    void build( const TF *const *P, const TF *W, SI n, SI leaf ) { build( P[ 0 ], P[ 1 ], W, n, leaf ); }
    void build( const TF *X, const TF *Y, const TF *W, SI n, SI leaf ) {
        using Cell = CellSoAT<64>;
        const int nth = hull_threads > 0 ? hull_threads
                                         : int( std::max( 1u, std::thread::hardware_concurrency() ) );
        secu = ! std::getenv( "PD2D_HULL_NOSECU" );

        // LE DECOUPAGE DE LA PREPARATION. Ce qui decide s'il y a quelque chose a mutualiser d'une
        // iteration de Newton a l'autre : les positions ne bougent pas, les poids si.
        auto top = [] { return std::chrono::duration<double>(
            std::chrono::steady_clock::now().time_since_epoch() ).count(); };
        double t0 = top(), tarbre = 0, tpaq = 0, tenc = 0, thull = 0, tlst = 0;

        full.build( X, Y, W, n, leaf );
        tarbre = top() - t0; t0 = top();

        // ---- les paquets : les noeuds les plus HAUTS dont le sous-arbre tient dans `rho`.
        const SI rho = std::max<SI>( 1, hull_rate );
        pk.clear();
        leaf_of.assign( n, -1 );
        {
            std::vector<SI> st{ 0 };
            while ( ! st.empty() ) {
                const SI h = st.back(); st.pop_back();
                const AaBsp::Node &nd = full.nodes[ h ];
                if ( nd.right < 0 || nd.end - nd.beg <= rho ) {
                    pk.push_back( Pk{ h, nd.beg, nd.end } );
                    for ( SI k = nd.beg; k < nd.end; ++k )
                        leaf_of[ k ] = SI( pk.size() ) - 1;
                } else {
                    st.push_back( h + 1 );
                    st.push_back( nd.right );
                }
            }
        }
        const SI m = SI( pk.size() );

        // ---- un representant REEL par paquet : `S` doit etre un sous-ensemble des germes, c'est
        //      ce qui donne `psi_S >= psi` et donc `E_m` contient `Lag( m )`.
        std::vector<SI> rep( m );
        for ( SI u = 0; u < m; ++u ) {
            const AaBsp::Node &nd = full.nodes[ pk[ u ].node ];
            const TF cx = ( nd.lo[ 0 ] + nd.hi[ 0 ] ) / 2, cy = ( nd.lo[ 1 ] + nd.hi[ 1 ] ) / 2;
            TF best = 1e30; SI bk = pk[ u ].beg;
            for ( SI k = pk[ u ].beg; k < pk[ u ].end; ++k ) {
                const TF ex = full.p[ 0 ][ k ] - cx, ey = full.p[ 1 ][ k ] - cy;
                if ( ex * ex + ey * ey < best ) { best = ex * ex + ey * ey; bk = k; }
            }
            rep[ u ] = full.order[ bk ];
        }
        AaBsp sub;
        sub.build_sel( X, Y, W, rep.data(), m, leaf );
        tpaq = top() - t0; t0 = top();

        // ---- LA FONCTION D'APPUI de chaque `E_m`, dans les `K` directions. Un seul balayage des
        //      sommets : il n'y a rien de plus a calculer que dans la version paraboloide.
        std::vector<TF> dv( size_t( n ) * K, TF( -1e30 ) );
        parallel_for( n, nth, Split::blocks, false, [ & ]( SI k, int ) {
            Cell ce;
            PowerDiagram<Cell, OneSeed, true, false, false>{
                OneSeed{ sub, Vec<2>{ full.p[ 0 ][ k ], full.p[ 1 ][ k ] }, full.seed_w( k ), full.order[ k ] } }
                .make_cell( ce, 0 );
            for ( SI v = 0; v < ce.nb; ++v )
                for ( SI i = 0; i < K; ++i ) {
                    const TF s = dirx( i ) * ce.vx[ v ] + diry( i ) * ce.vy[ v ];
                    TF &d = dv[ size_t( k ) * K + i ];
                    d = s > d ? s : d;
                }
        } );
        sup.assign( size_t( m ) * K, TF( -1e30 ) );
        for ( SI k = 0; k < n; ++k ) {
            const SI u = leaf_of[ k ];
            for ( SI i = 0; i < K; ++i ) {
                TF &d = sup[ size_t( u ) * K + i ];
                const TF s = dv[ size_t( k ) * K + i ];
                d = s > d ? s : d;
            }
        }
        tenc = top() - t0; t0 = top();

        // la MARGE de tangence : deux paquets adjacents se touchent en mesure nulle et l'arrondi
        // les separe. Grossir la sur-cellule ne peut qu'allonger les listes -- ca reste un
        // certificat. Meme lecon que `Cell::cut` et que le critere de la grille.
        std::vector<char> vide( m, 0 );
        for ( SI u = 0; u < m; ++u ) {
            vide[ u ] = sup[ size_t( u ) * K ] < TF( -1e29 );
            if ( ! vide[ u ] )
                for ( SI i = 0; i < K; ++i )
                    sup[ size_t( u ) * K + i ] += TF( 1e-9 );
        }

        // ---- la SUR-CELLULE comme polygone, pour en faire partir les cellules.
        hoff.assign( m + 1, 0 );
        hx.clear(); hy.clear();
        std::vector<TF> bb( 4 * size_t( m ), 0 );
        for ( SI u = 0; u < m; ++u ) {
            if ( ! vide[ u ] ) {
                Cell c;
                c.init_as_unit_square();
                for ( SI i = 0; i < K; ++i )
                    c.cut( dirx( i ), diry( i ), sup[ size_t( u ) * K + i ], -1 );
                for ( SI v = 0; v < c.nb; ++v ) { hx.push_back( c.vx[ v ] ); hy.push_back( c.vy[ v ] ); }
                // les indices des quatre directions AXIALES, en fonction de `K` -- ecrits `4` et
                // `6` (justes pour `K = 8` seulement), ils donnent des diagonales des que `K`
                // change : boites fausses, grille degeneree, et 12 s de preparation au lieu de 90 ms.
                static_assert( K % 4 == 0, "il faut les quatre directions axiales" );
                bb[ 4 * u + 0 ] = -sup[ size_t( u ) * K + K / 2 ];       // -x
                bb[ 4 * u + 1 ] = -sup[ size_t( u ) * K + 3 * K / 4 ];   // -y
                bb[ 4 * u + 2 ] =  sup[ size_t( u ) * K + 0 ];           // +x
                bb[ 4 * u + 3 ] =  sup[ size_t( u ) * K + K / 4 ];       // +y
                for ( int d = 0; d < 2; ++d ) {
                    bb[ 4 * u + d ]     = std::max<TF>( 0, std::min<TF>( 1, bb[ 4 * u + d ] ) );
                    bb[ 4 * u + d + 2 ] = std::max<TF>( 0, std::min<TF>( 1, bb[ 4 * u + d + 2 ] ) );
                }
            }
            hoff[ u + 1 ] = SI( hx.size() );
        }

        thull = top() - t0; t0 = top();

        // deux k-DOP sont disjoints s'il existe une direction ou leurs appuis s'excluent.
        auto disjoint = [ & ]( SI a, SI b ) {
            if ( vide[ a ] || vide[ b ] )
                return true;
            for ( SI i = 0; i < K; ++i )
                if ( sup[ size_t( a ) * K + i ] + sup[ size_t( b ) * K + ( i + K / 2 ) % K ] < 0 )
                    return true;
            return false;
        };

        // ---- les listes, par une grille sur les boites des sur-cellules.
        SI g = std::max<SI>( 4, std::min<SI>( 512, SI( std::sqrt( double( m ) ) ) ) );
        auto plage = [ & ]( SI u, SI gg, SI &i0, SI &j0, SI &i1, SI &j1 ) {
            auto cl = [ & ]( TF v ) { return std::max<SI>( 0, std::min<SI>( gg - 1, SI( v * gg ) ) ); };
            i0 = cl( bb[ 4 * u + 0 ] ); j0 = cl( bb[ 4 * u + 1 ] );
            i1 = cl( bb[ 4 * u + 2 ] ); j1 = cl( bb[ 4 * u + 3 ] );
        };
        for ( ;; ) {
            long long tot = 0;
            for ( SI u = 0; u < m; ++u )
                if ( ! vide[ u ] ) {
                    SI i0, j0, i1, j1; plage( u, g, i0, j0, i1, j1 );
                    tot += (long long)( i1 - i0 + 1 ) * ( j1 - j0 + 1 );
                }
            if ( tot <= 64ll * m || g <= 4 )
                break;
            g /= 2;
        }
        std::vector<SI> goff( size_t( g ) * g + 2, 0 ), gval;
        for ( int pass = 0; pass < 2; ++pass ) {
            for ( SI u = 0; u < m; ++u ) {
                if ( vide[ u ] )
                    continue;
                SI i0, j0, i1, j1; plage( u, g, i0, j0, i1, j1 );
                for ( SI j = j0; j <= j1; ++j )
                    for ( SI i = i0; i <= i1; ++i )
                        if ( pass == 0 ) ++goff[ j * g + i + 2 ];
                        else             gval[ goff[ j * g + i + 1 ]++ ] = u;
            }
            if ( pass == 0 ) {
                for ( size_t q = 1; q < goff.size(); ++q )
                    goff[ q ] += goff[ q - 1 ];
                gval.resize( goff.back() );
            }
        }

        std::vector<std::vector<SI>> lst( m );
        parallel_for( m, nth, Split::blocks, false, [ & ]( SI u, int ) {
            auto &L = lst[ u ];
            L.push_back( u );
            if ( vide[ u ] )
                return;
            std::vector<SI> vus;
            SI i0, j0, i1, j1; plage( u, g, i0, j0, i1, j1 );
            for ( SI j = j0; j <= j1; ++j )
                for ( SI i = i0; i <= i1; ++i )
                    for ( SI q = goff[ j * g + i ]; q < goff[ j * g + i + 1 ]; ++q )
                        vus.push_back( gval[ q ] );
            std::sort( vus.begin(), vus.end() );
            vus.erase( std::unique( vus.begin(), vus.end() ), vus.end() );
            for ( SI v : vus )
                if ( v != u && ! disjoint( u, v ) )
                    L.push_back( v );
            const AaBsp::Node &nu = full.nodes[ pk[ u ].node ];
            std::sort( L.begin() + 1, L.end(), [ & ]( SI a, SI b ) {
                return boite2( nu, full.nodes[ pk[ a ].node ] )
                     < boite2( nu, full.nodes[ pk[ b ].node ] );
            } );
        } );

        wmax = 0;
        if ( W )
            for ( SI i = 0; i < n; ++i )
                wmax = std::max( wmax, W[ i ] );

        loff.assign( m + 1, 0 );
        for ( SI u = 0; u < m; ++u )
            loff[ u + 1 ] = loff[ u ] + SI( lst[ u ].size() );
        lval.resize( loff[ m ] );
        ldq.resize( loff[ m ] );
        long long tt = 0;
        for ( SI u = 0; u < m; ++u ) {
            std::copy( lst[ u ].begin(), lst[ u ].end(), lval.begin() + loff[ u ] );
            for ( size_t q = 0; q < lst[ u ].size(); ++q )
                ldq[ loff[ u ] + SI( q ) ] = boite2( full.nodes[ pk[ u ].node ],
                                                     full.nodes[ pk[ lst[ u ][ q ] ].node ] );
            tt += SI( lst[ u ].size() );
        }
        tlst = top() - t0;
        std::printf( "  hull: rho=%d, %d paquets, liste moyenne %.1f, index %.1f Mo, "
                     "sur-cellule %.1f sommets\n", int( rho ), int( m ), double( tt ) / m,
                     bytes() / 1048576.0, double( hx.size() ) / m );
        std::printf( "  hull: preparation %.0f ms = arbre %.0f + paquets/sous-arbre %.0f "
                     "| enclos E_m %.0f + sur-cellules %.0f + listes %.0f  (POIDS-DEPENDANT %.0f%%)\n",
                     1e3 * ( tarbre + tpaq + tenc + thull + tlst ), 1e3 * tarbre, 1e3 * tpaq,
                     1e3 * tenc, 1e3 * thull, 1e3 * tlst,
                     100 * ( tenc + thull + tlst ) / ( tarbre + tpaq + tenc + thull + tlst ) );
    }
};

} // namespace pd

#pragma once

#include "spatial_accel/AaBsp.h"
#include "geometry/Cell.h"
#include "geometry/PowerDiagram.h"
#include "util/parallel.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <thread>
#include <vector>

namespace pd {

/// La taille du PAQUET, c'est-a-dire l'unite de l'index. Distincte de `--leaf`, qui reste l'unite
/// d'elagage : le paquet est un NOEUD du Bsp, et on descend dans son sous-arbre.
inline SI pack_rate = 16;

/// L'INDEX PAR PAQUETS : plus de descente d'arbre, une LISTE precalculee par paquet.
///
/// C'est la « fin de l'algorithme » de la parabole abaissee, isolee pour etre CHRONOMETREE. Le
/// paquet est ici une FEUILLE du Bsp -- la demonstration n'exige aucune partition en particulier,
/// et une feuille a l'avantage d'etre contigue en memoire, avec sa boite et son majorant de poids
/// deja calcules.
///
/// La preparation (les representants, les `delta`, les enclos, les chevauchements) est faite dans
/// `build`, donc comptee dans « arbre XX ms » et HORS du chrono. Elle n'est pas optimisee : ce
/// qu'on mesure ici est le regime permanent, c'est-a-dire
///
///     pour chaque paquet candidat, dans l'ordre du plus proche :
///         le test d'eviction sur SA boite ; s'il passe, ses germes, d'un seul tenant
///
/// contre ce que fait le Bsp aujourd'hui : une descente qui teste une quinzaine de boites
/// emboitees avant d'arriver a la meme feuille. La question est de savoir si une liste plate,
/// parcourue sans indirection, bat la descente -- ou si la descente elague tellement mieux que la
/// liste, forcement statique, ne rattrape pas.
///
/// `--leaf` est la taille du paquet (`rho`) : le paquet EST la feuille.
struct AaBspPack {
    static constexpr int dim = 2;

    struct Pk {
        SI node;                    ///< le noeud du Bsp -- on descend DEDANS, on ne le balaie pas
        SI beg, end;
    };

    AaBsp full;                     ///< le nuage complet ; ses FEUILLES sont les paquets
    std::vector<Pk> pk;
    std::vector<SI> leaf_of;        ///< par place dans `full.order` : son paquet
    std::vector<SI> loff, lval;     ///< CSR : les paquets candidats, LE PLUS PROCHE D'ABORD
    std::vector<TF> ldq;            ///< ... et le carre de la distance BOITE A BOITE, croissante
    TF wmax = 0;                    ///< le poids maximum du nuage, pour le rayon de securite
    bool secu = true;               ///< le rayon de securite (`PD2D_PACK_NOSECU` pour comparer
                                    ///< DANS LE MEME BINAIRE -- la disposition du code deplace la
                                    ///< reference de 10 % d'une compilation a l'autre)

    static constexpr const char *name = "pack";

    Vec<2> seed( SI k ) const { return full.seed( k ); }
    TF seed_x( SI k ) const { return full.seed_x( k ); }
    TF seed_y( SI k ) const { return full.seed_y( k ); }
    TF seed_w( SI k ) const { return full.seed_w( k ); }
    SI seed_id( SI k ) const { return full.order[ k ]; }
    SI nb_seeds() const { return full.nb_seeds(); }

    /// Le regime permanent : la liste, puis une descente BORNEE au sous-arbre du paquet. Le
    /// premier test de cette descente est celui de la boite du paquet lui-meme, donc l'eviction
    /// grossiere est faite sans code en plus.
    template<class MayCut, class CutWith, class Reach2>
    void for_each_candidate( SI k0, MayCut &&may_cut, CutWith &&cut_with, Reach2 &&reach2 ) const {
        const SI u = leaf_of[ k0 ], i0 = full.order[ k0 ];
        const Vec<2> p0 = full.seed( k0 );
        const TF w0 = full.seed_w( k0 );

        // LE RAYON DE SECURITE, sur une liste triee par distance BOITE A BOITE. Comme `p0` est
        // dans la boite de son paquet, `dist( p0, boite_v ) >= d( boite_u, boite_v )` : la cle du
        // tri MINORE la vraie distance, donc un seul `break` termine la liste.
        //
        //     |v - p_j| >= D - R  =>  pas de coupe des que  ( D - R )^2 >= R^2 + wmax - w0
        //
        // `reach2` est recalcule en des indices GEOMETRIQUES (1, 2, 4, 8...) : la cellule retrecit
        // surtout au debut -- le premier paquet est le sien -- et le balayage des sommets ne doit
        // pas coûter plus que ce qu'il economise.
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

    /// Octets de l'index PROPREMENT DIT (hors Bsp, qui existerait de toute facon).
    size_t bytes() const {
        return sizeof( Pk ) * pk.size() + sizeof( SI ) * ( leaf_of.size() + loff.size()
                                                           + lval.size() )
             + sizeof( TF ) * ldq.size();
    }

    /// Carre de la distance entre DEUX BOITES -- separable par axe, 0 si elles se touchent.
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
        secu = ! std::getenv( "PD2D_PACK_NOSECU" );
        const int nth = std::max( 1u, std::thread::hardware_concurrency() );

        full.build( X, Y, W, n, leaf );

        // ---- les paquets : les noeuds les plus HAUTS dont le sous-arbre tient dans `rho`. Le
        //      Bsp garde sa taille de feuille (`--leaf`), qui reste l'unite d'elagage.
        const SI rho = std::max<SI>( 1, pack_rate );
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

        // ---- un REPRESENTANT REEL par paquet, le plus proche du centre de sa boite. `S` doit etre
        //      un sous-ensemble des germes : c'est ce qui donne `psi_S >= psi`, donc `E_m` contient
        //      `Lag( m )`. Un germe virtuel ne majorerait rien.
        std::vector<SI> rep( m );
        for ( SI u = 0; u < m; ++u ) {
            const Pk &b = pk[ u ];
            const AaBsp::Node &nd = full.nodes[ b.node ];
            const TF cx = ( nd.lo[ 0 ] + nd.hi[ 0 ] ) / 2, cy = ( nd.lo[ 1 ] + nd.hi[ 1 ] ) / 2;
            TF best = 1e30; SI bk = b.beg;
            for ( SI k = b.beg; k < b.end; ++k ) {
                const TF ex = full.p[ 0 ][ k ] - cx, ey = full.p[ 1 ][ k ] - cy;
                if ( ex * ex + ey * ey < best ) { best = ex * ex + ey * ey; bk = k; }
            }
            rep[ u ] = full.order[ bk ];
        }
        AaBsp sub;
        sub.build_sel( X, Y, W, rep.data(), m, leaf );

        // ---- le germe VIRTUEL de chaque paquet : son barycentre. Rien ne l'oblige a etre un
        //      germe du nuage -- la demonstration ne demande `h_q - delta <= h_m` que sur `E_m`.
        std::vector<TF> qx( m, 0 ), qy( m, 0 );
        for ( SI u = 0; u < m; ++u ) {
            for ( SI k = pk[ u ].beg; k < pk[ u ].end; ++k ) { qx[ u ] += full.p[ 0 ][ k ];
                                                              qy[ u ] += full.p[ 1 ][ k ]; }
            const TF c = TF( 1 ) / TF( pk[ u ].end - pk[ u ].beg );
            qx[ u ] *= c; qy[ u ] *= c;
        }

        // ---- l'ABAISSEMENT. `h_q - h_m` est affine, donc son max sur `E_m` est a un SOMMET.
        std::vector<TF> dv( n, TF( -1e30 ) );
        parallel_for( n, nth, Split::blocks, false, [ & ]( SI k, int ) {
            const TF mx = full.p[ 0 ][ k ], my = full.p[ 1 ][ k ], mw = full.seed_w( k );
            Cell ce;
            PowerDiagram<Cell, OneSeed, true, false, false>{
                OneSeed{ sub, Vec<2>{ mx, my }, mw, full.order[ k ] } }.make_cell( ce, 0 );
            const SI u = leaf_of[ k ];
            const TF dx = qx[ u ] - mx, dy = qy[ u ] - my;
            const TF e = dx * ( qx[ u ] + mx ) + dy * ( qy[ u ] + my ) + mw;
            TF best = -1e30;
            for ( SI v = 0; v < ce.nb; ++v ) {
                const TF s = e - 2 * ( dx * ce.vx[ v ] + dy * ce.vy[ v ] );
                best = s > best ? s : best;
            }
            dv[ k ] = best;
        } );
        std::vector<TF> delta( m, TF( -1e30 ) );
        for ( SI k = 0; k < n; ++k )
            delta[ leaf_of[ k ] ] = std::max( delta[ leaf_of[ k ] ], dv[ k ] );

        // LA MARGE, et elle n'est pas cosmetique. Deux paquets adjacents se touchent EXACTEMENT
        // le long d'une arete : leur intersection est de mesure nulle, et l'arrondi les separe.
        // MESURE : sans marge, 310 voisins reels manquaient des listes sur le cas dur a rho = 2,
        // et la somme des aires valait 3.86. Les ecarts mesures allaient de 3.5e-18 a 1.5e-11 --
        // de la tangence, pas un trou : `delta` lui-meme etait exact (zero violation de
        // `h_q - delta <= h_m` sur les sommets des VRAIES cellules).
        //
        // On corrige en UN endroit plutot qu'en deux tolerances de comparaison : grossir `delta`
        // ne peut qu'AGRANDIR `I_u`, donc reste un certificat. `1e-9` ecarte les deux convexes de
        // `eps / ( 2 d )`, soit ~5e-7 pour des cellules de 1e-3 -- quatre ordres au-dessus du
        // bruit mesure, et cinq ordres sous la taille d'une cellule.
        for ( SI u = 0; u < m; ++u )
            delta[ u ] += TF( 1e-9 );
        if ( const char *e = std::getenv( "PD2D_PACK_EPS" ) )
            for ( SI u = 0; u < m; ++u )
                delta[ u ] += TF( std::atof( e ) );

        // DIAGNOSTIC : la demonstration exige `h_q - delta_u <= h_m` sur `Lag( m )`, et l'obtient
        // en prenant le max sur `E_m` qui le contient. On le VERIFIE sur les sommets des vraies
        // cellules -- si ca viole, le trou est dans `delta` ; sinon il est dans les listes.
        if ( std::getenv( "PD2D_PACK_VERIF" ) ) {
            std::vector<double> bad( n, 0 );
            PowerDiagram<Cell, AaBsp, true, false, false> pf{ full };
            parallel_for( n, nth, Split::blocks, false, [ & ]( SI k, int ) {
                Cell c;
                pf.make_cell( c, k );
                const TF mx = full.p[ 0 ][ k ], my = full.p[ 1 ][ k ], mw = full.seed_w( k );
                const SI u = leaf_of[ k ];
                for ( SI v = 0; v < c.nb; ++v ) {
                    const TF ex = c.vx[ v ] - qx[ u ], ey = c.vy[ v ] - qy[ u ];
                    const TF fx = c.vx[ v ] - mx,      fy = c.vy[ v ] - my;
                    const double d = double( ex * ex + ey * ey - delta[ u ] )
                                   - double( fx * fx + fy * fy - mw );
                    bad[ k ] = std::max( bad[ k ], d );
                }
            } );
            SI nb = 0; double wo = 0;
            for ( SI k = 0; k < n; ++k ) { nb += bad[ k ] > 1e-14; wo = std::max( wo, bad[ k ] ); }
            std::printf( "  pack: VERIF delta -- %d germes violent h_q - delta <= h_m sur leur "
                         "vraie cellule, pire %.3e\n", int( nb ), wo );
        }

        // ---- les convexes `I_u`, et leurs boites.
        std::vector<SI> voff( m + 1, 0 );
        std::vector<TF> vxs, vys, bb( 4 * size_t( m ), 0 );
        vxs.reserve( size_t( 10 ) * m ); vys.reserve( size_t( 10 ) * m );
        for ( SI u = 0; u < m; ++u ) {
            Cell c;
            PowerDiagram<Cell, OneSeed, true, false, false>{
                OneSeed{ sub, Vec<2>{ qx[ u ], qy[ u ] }, delta[ u ], SI( -1 ) } }.make_cell( c, 0 );
            TF x0 = 1e30, y0 = 1e30, x1 = -1e30, y1 = -1e30;
            for ( SI v = 0; v < c.nb; ++v ) {
                vxs.push_back( c.vx[ v ] ); vys.push_back( c.vy[ v ] );
                x0 = std::min( x0, c.vx[ v ] ); x1 = std::max( x1, c.vx[ v ] );
                y0 = std::min( y0, c.vy[ v ] ); y1 = std::max( y1, c.vy[ v ] );
            }
            voff[ u + 1 ] = SI( vxs.size() );
            bb[ 4 * u + 0 ] = x0; bb[ 4 * u + 1 ] = y0;
            bb[ 4 * u + 2 ] = x1; bb[ 4 * u + 3 ] = y1;
        }

        // deux convexes sont disjoints ssi une normale d'arete de l'un les separe. La TOLERANCE
        // n'est pas cosmetique : deux paquets adjacents se touchent le long d'une arete, donc
        // s'intersectent en mesure nulle, et un test strict les declarerait separes.
        auto demi = [ & ]( SI p, SI q ) {
            const SI b0 = voff[ p ], e0 = voff[ p + 1 ], b1 = voff[ q ], e1 = voff[ q + 1 ];
            for ( SI i = b0; i < e0; ++i ) {
                const SI j = i + 1 < e0 ? i + 1 : b0;
                const TF nx = vys[ j ] - vys[ i ], ny = vxs[ i ] - vxs[ j ];
                TF a0 = 1e30, a1 = -1e30, c0 = 1e30, c1 = -1e30;
                for ( SI k = b0; k < e0; ++k ) {
                    const TF s = nx * vxs[ k ] + ny * vys[ k ];
                    a0 = std::min( a0, s ); a1 = std::max( a1, s );
                }
                for ( SI k = b1; k < e1; ++k ) {
                    const TF s = nx * vxs[ k ] + ny * vys[ k ];
                    c0 = std::min( c0, s ); c1 = std::max( c1, s );
                }
                const TF eps = TF( 1e-11 ) * std::sqrt( nx * nx + ny * ny );
                if ( c0 > a1 + eps || a0 > c1 + eps )
                    return true;
            }
            return false;
        };

        // ---- les chevauchements, par une grille sur les boites. La maille se resserre tant que
        //      les insertions debordent -- sur un nuage groupe une boite peut couvrir le domaine.
        SI g = std::max<SI>( 4, std::min<SI>( 512, SI( std::sqrt( double( m ) ) ) ) );
        auto plage = [ & ]( SI u, SI gg, SI &i0, SI &j0, SI &i1, SI &j1 ) {
            auto cl = [ & ]( TF v ) { return std::max<SI>( 0, std::min<SI>( gg - 1, SI( v * gg ) ) ); };
            i0 = cl( bb[ 4 * u + 0 ] ); j0 = cl( bb[ 4 * u + 1 ] );
            i1 = cl( bb[ 4 * u + 2 ] ); j1 = cl( bb[ 4 * u + 3 ] );
        };
        for ( ;; ) {
            long long tot = 0;
            for ( SI u = 0; u < m; ++u ) {
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
            std::vector<SI> vus;
            SI i0, j0, i1, j1; plage( u, g, i0, j0, i1, j1 );
            for ( SI j = j0; j <= j1; ++j )
                for ( SI i = i0; i <= i1; ++i )
                    for ( SI q = goff[ j * g + i ]; q < goff[ j * g + i + 1 ]; ++q )
                        vus.push_back( gval[ q ] );
            std::sort( vus.begin(), vus.end() );
            vus.erase( std::unique( vus.begin(), vus.end() ), vus.end() );
            auto &L = lst[ u ];
            for ( SI v : vus ) {
                if ( v == u )
                    continue;
                if ( bb[ 4 * v + 0 ] > bb[ 4 * u + 2 ] || bb[ 4 * u + 0 ] > bb[ 4 * v + 2 ] ||
                     bb[ 4 * v + 1 ] > bb[ 4 * u + 3 ] || bb[ 4 * u + 1 ] > bb[ 4 * v + 3 ] )
                    continue;
                // DIAGNOSTIC : sans l'axe separateur, la liste est le SUR-ENSEMBLE par boites.
                if ( ! std::getenv( "PD2D_PACK_NOSAT" ) && ( demi( u, v ) || demi( v, u ) ) )
                    continue;
                L.push_back( v );
            }
            // LE PLUS PROCHE D'ABORD, comme la descente du Bsp : la cellule retrecit tout de suite,
            // et tout le reste est ensuite elague contre une cellule deja petite.
            const AaBsp::Node &nu = full.nodes[ pk[ u ].node ];
            std::sort( L.begin(), L.end(), [ & ]( SI a, SI b ) {
                return boite2( nu, full.nodes[ pk[ a ].node ] )
                     < boite2( nu, full.nodes[ pk[ b ].node ] );
            } );
            L.insert( L.begin(), u );               // son propre paquet EN PREMIER
            if ( std::getenv( "PD2D_PACK_VERIF" ) )
                std::sort( L.begin(), L.end() );    // pour la recherche du diagnostic
        } );

        // DIAGNOSTIC : la liste contient-elle TOUS les voisins reels ? C'est la seule chose que
        // le certificat promet, et la seule dont l'algorithme ait besoin.
        if ( std::getenv( "PD2D_PACK_VERIF" ) ) {
            std::vector<SI> pos( n, -1 );
            for ( SI k = 0; k < n; ++k ) pos[ full.order[ k ] ] = k;
            std::vector<long long> mv( n, 0 );
            std::vector<double> gg2( n, -1e30 );
            PowerDiagram<Cell, AaBsp, true, false, false> pf{ full };
            parallel_for( n, nth, Split::blocks, false, [ & ]( SI k, int ) {
                Cell c;
                pf.make_cell( c, k );
                const SI u = leaf_of[ k ];
                for ( SI v = 0; v < c.nb; ++v ) {
                    const SI id = c.cid[ v ];
                    if ( id < 0 )
                        continue;
                    const SI w2 = leaf_of[ pos[ id ] ];
                    if ( w2 != u && ! std::binary_search( lst[ u ].begin(), lst[ u ].end(), w2 ) ) {
                        ++mv[ k ];
                        // de COMBIEN les deux convexes sont-ils separes ? A 1e-16 c'est de la
                        // tangence, a 1e-6 c'est autre chose.
                        TF g2 = -1e30;
                        for ( int t = 0; t < 2; ++t ) {
                            const SI P0 = t ? w2 : u, Q0 = t ? u : w2;
                            for ( SI i = voff[ P0 ]; i < voff[ P0 + 1 ]; ++i ) {
                                const SI j = i + 1 < voff[ P0 + 1 ] ? i + 1 : voff[ P0 ];
                                const TF nx = vys[ j ] - vys[ i ], ny = vxs[ i ] - vxs[ j ];
                                const TF ln = std::sqrt( nx * nx + ny * ny );
                                if ( ln == 0 ) continue;
                                TF a0 = 1e30, a1 = -1e30, c0 = 1e30, c1 = -1e30;
                                for ( SI q = voff[ P0 ]; q < voff[ P0 + 1 ]; ++q ) {
                                    const TF s3 = ( nx * vxs[ q ] + ny * vys[ q ] ) / ln;
                                    a0 = std::min( a0, s3 ); a1 = std::max( a1, s3 ); }
                                for ( SI q = voff[ Q0 ]; q < voff[ Q0 + 1 ]; ++q ) {
                                    const TF s3 = ( nx * vxs[ q ] + ny * vys[ q ] ) / ln;
                                    c0 = std::min( c0, s3 ); c1 = std::max( c1, s3 ); }
                                g2 = std::max( g2, std::max( c0 - a1, a0 - c1 ) );
                            }
                        }
                        gg2[ k ] = std::max( gg2[ k ], double( g2 ) );
                    }
                }
            } );
            long long ms = 0; double gw = -1e30, gs = 1e30;
            for ( SI k = 0; k < n; ++k ) {
                ms += mv[ k ];
                if ( mv[ k ] ) { gw = std::max( gw, gg2[ k ] ); gs = std::min( gs, gg2[ k ] ); }
            }
            std::printf( "  pack: VERIF listes -- %lld voisins reels ABSENTS ; ecart des convexes "
                         "de %.3e a %.3e\n", ms, gs, gw );
        }

        // DIAGNOSTIC. Un `I_u` vide n'est insere nulle part : sa liste se reduit a lui-meme, et
        // si ses cellules ne sont PAS vides le resultat est faux. `delta = -inf` en est la cause
        // possible -- tous les membres du paquet auraient un `E_m` vide.
        {
            SI nv = 0, ni = 0;
            for ( SI u = 0; u < m; ++u ) {
                ni += delta[ u ] < TF( -1e29 );
                nv += voff[ u + 1 ] == voff[ u ];
            }
            long long tt = 0;
            for ( SI u = 0; u < m; ++u ) tt += SI( lst[ u ].size() );
            std::printf( "  pack: rho=%d, %d paquets, %d I_u vides (%d sans delta), liste "
                         "moyenne %.1f, index %.1f Mo\n", int( rho ), int( m ), int( nv ),
                         int( ni ), double( tt ) / m,
                         ( sizeof( Pk ) * m + sizeof( SI ) * ( n + m + 1 + tt ) ) / 1048576.0 );
        }

        wmax = 0;
        if ( W )
            for ( SI i = 0; i < n; ++i )
                wmax = std::max( wmax, W[ i ] );

        loff.assign( m + 1, 0 );
        for ( SI u = 0; u < m; ++u )
            loff[ u + 1 ] = loff[ u ] + SI( lst[ u ].size() );
        lval.resize( loff[ m ] );
        ldq.resize( loff[ m ] );
        for ( SI u = 0; u < m; ++u ) {
            std::copy( lst[ u ].begin(), lst[ u ].end(), lval.begin() + loff[ u ] );
            for ( size_t q = 0; q < lst[ u ].size(); ++q )
                ldq[ loff[ u ] + SI( q ) ] = boite2( full.nodes[ pk[ u ].node ],
                                                     full.nodes[ pk[ lst[ u ][ q ] ].node ] );
        }
    }
};

} // namespace pd

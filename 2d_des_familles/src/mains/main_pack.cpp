// L'INDEX PAR PAQUETS : chaque paquet porte la liste des paquets qui peuvent l'atteindre, triee par
// distance, et la cellule ne descend que dans les sous-arbres qu'elle nomme.
//
// Ce banc porte aussi les deux sondes qui ont mene a cet index et qui n'ont de sens que pour lui :
// `--enclos` (de combien la cellule calculee contre un sous-echantillon est trop grande) et
// `--baisse` (la parabole abaissee, un enclos par paquet).
//
//   xmake run pd_pack --help

#include "spatial_accel/AaBspPack.h"
#include "spatial_accel/AaBspPre.h"
#include "bench/Bench.h"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <string>
#include <vector>

using namespace pd;
using namespace pd::bench;

namespace {

/// L'ENCLOS PAR SOUS-ECHANTILLON, mesure : de combien la cellule calculee contre `S` seul est trop
/// grande, ce qu'elle coute, et CE QU'ELLE PERMETTRAIT si on s'en servait comme index.
///
/// = Premiere table : l'enclos
///
/// Le pendant 2D exact du tableau de `cases/sub1d.py`. La 1D predisait que le rapport des MESURES
/// vaut `n / |S|` (un enclos contient environ `n / |S|` cellules), donc que celui des RAYONS vaut
/// `sqrt( n / |S| )` en 2D.
///
/// = Seconde table : les PAQUETS, c'est-a-dire l'index qu'on n'a pas encore
///
/// L'idee testee ici : se passer du BSP et n'utiliser que le pavage GROSSIER. On range chaque germe
/// fin `j` dans les cellules grossieres que son enclos rencontre ; les candidats de `k` sont
/// l'union des paquets des cellules grossieres que SON enclos rencontre. Plus une boite a tester,
/// plus de descente -- deux listes contigues.
///
/// Pourquoi c'est un sur-ensemble : si `C( j )` et `C( k )` se touchent en `x`, alors `x` est dans
/// les deux enclos (chacun contient sa cellule), et la cellule grossiere `D_r` qui gagne en `x`
/// rencontre donc les deux enclos. `r` est dans les deux listes, et `j` sort du paquet de `r`.
///
/// Les cellules grossieres qu'un enclos rencontre se lisent GRATUITEMENT : ce sont les `cid` de ses
/// aretes. En effet `D_r inter C_S( j ) = D_r inter { h_j <= h_r }` -- sur `D_r`, `h_r` EST le
/// minimum sur `S` -- donc l'intersection est d'aire non nulle exactement quand la bissectrice
/// `( j, r )` coupe `D_r`, c'est-a-dire quand `r` borde l'enclos. Le seul cas qui echappe est celui
/// d'une cellule grossiere AVALEE tout entiere : elle ne borde rien. C'est pourquoi la colonne
/// « manques » est la : elle compte, sur les vrais voisins de Laguerre de `k`, ceux que la regle ne
/// proposerait pas. Elle doit valoir zero, sans quoi la regle demande un correctif.
int enclos_stats( const Args &a, const std::vector<TF> &X, const std::vector<TF> &Y, const TF *W ) {
    using Cell = CellSoAT<64>;

    auto quant = []( std::vector<double> &v, double q ) {
        return v.empty() ? 0.0 : v[ std::min( v.size() - 1, size_t( q * v.size() ) ) ];
    };
    auto reach = []( const Cell &c, TF p0x, TF p0y ) {
        TF m = 0;
        for ( SI v = 0; v < c.nb; ++v ) {
            const TF ex = c.vx[ v ] - p0x, ey = c.vy[ v ] - p0y;
            m = std::max( m, ex * ex + ey * ey );
        }
        return std::sqrt( double( m ) );
    };

    const SI rates[] = { 2, 4, 8, 16, 32, 64 };
    struct Row { int r; double qr[3], qm[3], nn, qc[3]; long long miss; double bpre, bfull, tried; };
    std::vector<Row> rows;

    for ( SI r : rates ) {
        pre_rate = r;
        AaBspSub t;
        t.build( X.data(), Y.data(), W, a.n, a.leaf );

        const SI n = t.nb_seeds(), m = t.sub.nb_seeds();
        std::vector<SI> loc( a.n, -1 ), pos( a.n, -1 );
        for ( SI u = 0; u < m; ++u ) loc[ t.sub.order[ u ] ] = u;
        for ( SI k = 0; k < n; ++k ) pos[ t.full.order[ k ] ] = k;

        PowerDiagram<Cell, AaBspSub, true, false, true> pe{ t };        // contre `S` seul
        PowerDiagram<Cell, AaBsp,    true, false, true> pf{ t.full };   // la vraie cellule

        // ---- passe 1 : les enclos, et pour chacun la liste des cellules grossieres qu'il borde.
        std::vector<SI> noff( n + 1, 0 ), nval;
        std::vector<double> re( n ), me( n );
        nval.reserve( size_t( 7 ) * n );
        for ( SI k = 0; k < n; ++k ) {
            Cell ce;
            pe.make_cell( ce, k );
            re[ k ] = reach( ce, t.seed_x( k ), t.seed_y( k ) );
            me[ k ] = double( ce.measure() );
            const size_t b0 = nval.size();
            for ( SI v = 0; v < ce.nb; ++v ) {
                const SI id = ce.cid[ v ];
                if ( id < 0 )
                    continue;                       // une arete du DOMAINE, pas un voisin
                bool seen = false;
                for ( size_t u = b0; u < nval.size(); ++u )
                    seen |= nval[ u ] == id;
                if ( ! seen )
                    nval.push_back( id );
            }
            noff[ k + 1 ] = SI( nval.size() );
        }

        // ---- les PAQUETS : la relation inverse, en CSR.
        std::vector<SI> boff( m + 2, 0 ), bval( nval.size() );
        for ( SI v : nval )
            ++boff[ loc[ v ] + 2 ];
        for ( SI u = 1; u < m + 2; ++u )
            boff[ u ] += boff[ u - 1 ];
        for ( SI k = 0; k < n; ++k )
            for ( SI u = noff[ k ]; u < noff[ k + 1 ]; ++u )
                bval[ boff[ loc[ nval[ u ] ] + 1 ]++ ] = k;

        // ---- passe 2 : la vraie cellule, la liste de candidats, et les MANQUES.
        std::vector<SI> stamp( n, -1 );
        std::vector<double> qr, qm, qc;
        qr.reserve( n ); qm.reserve( n ); qc.reserve( n );
        long long miss = 0, snn = 0;
        for ( SI k = 0; k < n; ++k ) {
            Cell cf;
            pf.make_cell( cf, k );

            SI cnt = 0;
            stamp[ k ] = k;                         // le germe lui-meme n'est pas son candidat
            for ( SI u = noff[ k ]; u < noff[ k + 1 ]; ++u ) {
                const SI rr = loc[ nval[ u ] ];
                for ( SI w = boff[ rr ]; w < boff[ rr + 1 ]; ++w ) {
                    const SI j = bval[ w ];
                    if ( stamp[ j ] != k ) { stamp[ j ] = k; ++cnt; }
                }
            }
            snn += noff[ k + 1 ] - noff[ k ];
            qc.push_back( double( cnt ) );

            for ( SI v = 0; v < cf.nb; ++v ) {      // les VRAIS voisins sont-ils tous proposes ?
                const SI id = cf.cid[ v ];
                if ( id >= 0 && stamp[ pos[ id ] ] != k )
                    ++miss;
            }

            const double rf = reach( cf, t.seed_x( k ), t.seed_y( k ) );
            if ( rf > 0 ) qr.push_back( re[ k ] / rf );
            const double mf = double( cf.measure() );
            if ( mf > 0 ) qm.push_back( me[ k ] / mf );
        }
        std::sort( qr.begin(), qr.end() );
        std::sort( qm.begin(), qm.end() );
        std::sort( qc.begin(), qc.end() );

        const double d = double( n );
        rows.push_back( Row{ int( r ),
            { quant( qr, .5 ), quant( qr, .9 ), qr.empty() ? 0 : qr.back() },
            { quant( qm, .5 ), quant( qm, .9 ), qm.empty() ? 0 : qm.back() },
            double( snn ) / d,
            { quant( qc, .5 ), quant( qc, .9 ), qc.empty() ? 0 : qc.back() },
            miss, pe.st_boxes / d, pf.st_boxes / d, pf.st_tried / d } );
    }

    std::printf( "  %-6s %28s %28s %17s\n", "", "RAYON enclos / vrai",
                 "MESURE enclos / vraie", "boites/cellule" );
    std::printf( "  %-6s %8s %8s %8s   %8s %8s %8s   %8s %8s\n", "n/|S|",
                 "median", "dec. 9", "max", "median", "dec. 9", "max", "pre", "pleine" );
    for ( const Row &w : rows )
        std::printf( "  %-6d %8.2f %8.2f %8.1f   %8.1f %8.1f %8.0f   %8.1f %8.1f\n", w.r,
                     w.qr[ 0 ], w.qr[ 1 ], w.qr[ 2 ], w.qm[ 0 ], w.qm[ 1 ], w.qm[ 2 ],
                     w.bpre, w.bfull );

    std::printf( "\n  LES PAQUETS : ce que couterait un index fait du seul pavage grossier\n" );
    std::printf( "  %-6s %9s %27s %8s %19s\n", "n/|S|", "cellules", "CANDIDATS par cellule",
                 "", "le BSP d'aujourd'hui" );
    std::printf( "  %-6s %9s %8s %8s %8s %8s   %9s %9s\n", "", "grossieres",
                 "median", "dec. 9", "max", "manques", "tentees", "boites" );
    for ( const Row &w : rows )
        std::printf( "  %-6d %9.2f %8.0f %8.0f %8.0f %8lld   %9.1f %9.1f\n", w.r, w.nn,
                     w.qc[ 0 ], w.qc[ 1 ], w.qc[ 2 ], w.miss, w.tried, w.bfull );
    return 0;
}


/// LA PARABOLE ABAISSEE en 2D : combien de germes reste-t-il a tester ?
///
/// Le portage direct de `cases/sub1d_baisse.py`. On prend `|S| = n / rho` germes (stratifies : un
/// par sous-arbre du Bsp), on rattache chaque germe `m` au `k` de `S` le PLUS PROCHE -- son paquet
/// -- et on ABAISSE la parabole de `k` juste assez pour qu'elle minore tout son paquet :
///
///     h_k( x ) - h_m( x ) = -2 ( p_k - p_m ) . x + |p_k|^2 - |p_m|^2 - w_k + w_m
///
/// est AFFINE (le `|x|^2` est commun), donc son maximum sur `E_m` -- l'enclos de `m` contre `S`
/// seul, deja un convexe -- est atteint a un SOMMET, et se lit sans approximation.
///
/// `I_k = { x : h_k - delta_k <= h_j , j de S }` contient alors toutes les cellules du paquet :
///
///     x dans Lag( m ) ==> h_m <= h_j sur S ==> h_k - delta_k <= h_j sur S ==> x dans I_k
///
/// C'est un diagramme de puissance ou CHAQUE cellule est calculee avec son propre poids augmente
/// -- donc `|S|` convexes qui se CHEVAUCHENT au lieu d'un pavage. Deux germes ne peuvent se couper
/// que si leurs paquets se chevauchent : la colonne « manq. » verifie que le certificat tient, en
/// reprenant tous les couples reellement adjacents du vrai diagramme.
///
/// Ce qui est simule et ce qui ne l'est pas : on suppose, comme demande, qu'on sait trouver les
/// chevauchements vite. La colonne « boite » dit ce que couterait la version bon marche -- ne
/// tester que les boites englobantes des `I_k` au lieu des convexes eux-memes.
int baisse_stats( const Args &a, const std::vector<TF> &X, const std::vector<TF> &Y, const TF *W ) {
    using Cell = CellSoAT<64>;
    const SI rates[] = { 2, 4, 8, 16, 32, 64 };

    auto quant = []( std::vector<double> v, double q ) {
        if ( v.empty() ) return 0.0;
        std::sort( v.begin(), v.end() );
        return v[ std::min( v.size() - 1, size_t( q * v.size() ) ) ];
    };

    // OU EST LE DEBORDEMENT ? `delta_k` est le max de `h_k - h_m` sur `E_m`, qui CONTIENT
    // `Lag( m )` mais lui est bien plus gros. La demonstration, elle, n'a besoin de
    // `h_k - delta_k <= h_m` que sur `Lag( m )` : un `delta` calcule sur la vraie cellule reste
    // donc VALIDE (la colonne « manq. » le confirme), simplement pas calculable sans la reponse.
    // C'est un ORACLE, et il partage la question en deux -- si les chiffres s'effondrent, c'est
    // `E_m` qu'il faut resserrer ; s'ils ne bougent pas, c'est la GEOMETRIE du paquet qui coute, et
    // aucun raffinement de `delta` n'y changera rien.
    //
    // ESSAYE ET MESURE par ailleurs : rattacher `m` au proprietaire de `p_m` dans le diagramme de
    // puissance grossier au lieu du germe le plus proche. Sans poids les deux regles COINCIDENT
    // (`argmin_j h_j( p_m )` est alors le plus proche) ; avec poids la seconde s'effondre -- 22436
    // candidats contre 42 sur le cas dur. La pente de `h_k - h_m` vaut `2 | p_k - p_m |`, une
    // quantite de l'espace des GERMES : le proprietaire de `p_m` peut y etre tres loin.
    // TROIS VARIANTES de la parabole abaissee. `h_q( x ) - omega = |x|^2 + a . x + b` avec
    // `q = -a/2` : abaisser seulement `w` fixe `a = -2 p_k` et n'optimise que `b`, donc ne corrige
    // pas la PENTE de `h_k - h_m`, qui vaut `2 | p_k - p_m |`. Deplacer aussi le germe, c'est
    // choisir `a` -- et le barycentre du paquet est le choix bon marche.
    static const char *noms[] = { "delta seul, germe = le plus proche de S",
                                  "delta + germe VIRTUEL au barycentre du paquet",
                                  "delta seul, pris sur Lag( m ) -- ORACLE, non calculable" };
    for ( int mode = 0; mode < 3; ++mode ) {
    std::printf( "  -- %s\n", noms[ mode ] );
    std::printf( "  %5s %8s %8s %8s %8s %8s %8s %9s %8s %7s\n", "n/|S|", "|S|", "paq/paq",
                 "vrais", "aire I/p", "cand.", "cd. d9", "cd. max", "cd.boite", "manq." );

    for ( SI r : rates ) {
        pre_rate = r;
        AaBspSub t;
        t.build( X.data(), Y.data(), W, a.n, a.leaf );
        const SI n = t.nb_seeds(), m = t.sub.nb_seeds();

        std::vector<SI> loc( a.n, -1 ), pos( a.n, -1 );
        for ( SI u = 0; u < m; ++u ) loc[ t.sub.order[ u ] ] = u;
        for ( SI k = 0; k < n; ++k ) pos[ t.full.order[ k ] ] = k;

        // ---- 1. LES PAQUETS : le germe de `S` le plus proche. Le parcours du Bsp sert TEL QUEL --
        //         `may_cut` devient « cette boite peut-elle contenir plus proche que le meilleur ».
        std::vector<SI> pack( n, -1 );
        parallel_for( n, a.threads, Split::blocks, a.pin, [ & ]( SI k, int ) {
            const TF px = t.full.p[ 0 ][ k ], py = t.full.p[ 1 ][ k ];
            TF best = 1e30;
            SI bu = -1;
            t.sub.for_each_candidate_at( Vec<2>{ px, py }, -1,
                [ & ]( Vec<2> lo, Vec<2> hi, const WMaj & ) {
                    const TF ex = px < lo[ 0 ] ? lo[ 0 ] - px : ( px > hi[ 0 ] ? px - hi[ 0 ] : TF( 0 ) );
                    const TF ey = py < lo[ 1 ] ? lo[ 1 ] - py : ( py > hi[ 1 ] ? py - hi[ 1 ] : TF( 0 ) );
                    return ex * ex + ey * ey <= best;
                },
                [ & ]( Vec<2> q, TF, SI id ) {
                    const TF ex = q[ 0 ] - px, ey = q[ 1 ] - py;
                    const TF d = ex * ex + ey * ey;
                    if ( d < best ) { best = d; bu = loc[ id ]; }
                    return true;
                },
                [ & ]() { return best; } );
            pack[ k ] = bu;
        } );
        std::vector<SI> psz( m, 0 );
        for ( SI k = 0; k < n; ++k )
            ++psz[ pack[ k ] ];

        // le germe VIRTUEL du paquet : son barycentre. Il n'appartient plus au nuage, ce que la
        // demonstration autorise -- elle ne demande `h_q - delta <= h_m` que sur `B_m`.
        std::vector<TF> qx( m, 0 ), qy( m, 0 ), qw( m, 0 );
        for ( SI u = 0; u < m; ++u ) { qx[ u ] = t.sub.p[ 0 ][ u ]; qy[ u ] = t.sub.p[ 1 ][ u ];
                                       qw[ u ] = t.sub.seed_w( u ); }
        if ( mode == 1 ) {
            std::fill( qx.begin(), qx.end(), TF( 0 ) );
            std::fill( qy.begin(), qy.end(), TF( 0 ) );
            std::fill( qw.begin(), qw.end(), TF( 0 ) );
            for ( SI k = 0; k < n; ++k ) { qx[ pack[ k ] ] += t.full.p[ 0 ][ k ];
                                           qy[ pack[ k ] ] += t.full.p[ 1 ][ k ]; }
            for ( SI u = 0; u < m; ++u ) { qx[ u ] /= psz[ u ]; qy[ u ] /= psz[ u ]; }
        }

        // ---- 2. L'ABAISSEMENT. `E_m` est la cellule de `m` contre `S` seul : `AaBspSub` la donne
        //         directement. Le maximum de la difference affine y est a un SOMMET.
        PowerDiagram<Cell, AaBspSub, true, false, false> pe{ t };
        PowerDiagram<Cell, AaBsp,    true, false, false> pf{ t.full };
        std::vector<TF> dv( n, TF( 0 ) );
        parallel_for( n, a.threads, Split::blocks, a.pin, [ & ]( SI k, int ) {
            Cell ce;
            if ( mode == 2 ) pf.make_cell( ce, k ); else pe.make_cell( ce, k );
            const SI u = pack[ k ];
            const TF mx = t.full.p[ 0 ][ k ], my = t.full.p[ 1 ][ k ], mw = t.full.seed_w( k );
            const TF rx = qx[ u ], ry = qy[ u ], rw = qw[ u ];
            const TF dx = rx - mx, dy = ry - my;
            // `|p_r|^2 - |p_m|^2` FACTORISE, et pas litteralement : ecrit `rx*rx - mx*mx` il ne rend
            // pas exactement zero quand `r == m`, et le paquet s'abaisserait d'un cran pour rien --
            // c'est le defaut qui avait fait rater 8700 voisins au critere de la grille.
            const TF e = dx * ( rx + mx ) + dy * ( ry + my ) - rw + mw;
            TF best = mode == 1 ? TF( -1e30 ) : TF( 0 );
            for ( SI v = 0; v < ce.nb; ++v ) {
                const TF s = e - 2 * ( dx * ce.vx[ v ] + dy * ce.vy[ v ] );
                best = s > best ? s : best;
            }
            dv[ k ] = best;
        } );
        std::vector<TF> delta( m, mode == 1 ? TF( -1e30 ) : TF( 0 ) );
        for ( SI k = 0; k < n; ++k )
            delta[ pack[ k ] ] = std::max( delta[ pack[ k ] ], dv[ k ] );
        // la MARGE de tangence, cf. `AaBspPack` : deux paquets adjacents se touchent en mesure
        // nulle, et l'arrondi les separe. Grossir `delta` reste un certificat.
        for ( SI u = 0; u < m; ++u )
            delta[ u ] += TF( 1e-9 );

        // ---- 3. LES CONVEXES `I_k`, ranges en CSR. C'est le diagramme de `S` ou chaque cellule est
        //         calculee avec SON poids augmente -- donc `|S|` convexes qui se recouvrent.
        std::vector<SI> voff( m + 1, 0 );
        std::vector<TF> vxs, vys, bb( 4 * size_t( m ), 0 );
        std::vector<double> ai( m, 0 );
        vxs.reserve( size_t( 10 ) * m ); vys.reserve( size_t( 10 ) * m );
        for ( SI u = 0; u < m; ++u ) {
            const TF ux = qx[ u ], uy = qy[ u ];
            const TF uw = qw[ u ] + delta[ u ];
            // le germe virtuel n'est PAS dans `S` : aucun plan a sauter. Le representant, lui,
            // peut se sauter parce que `delta >= 0` rend `h_k - delta <= h_k` automatique.
            const SI uid = mode == 1 ? SI( -1 ) : t.sub.order[ u ];
            Cell c;
            const OneSeed os{ t.sub, ux, uy, uw, uid };
            PowerDiagram<Cell, OneSeed, true, false, false>{ os }.make_cell( c, 0 );
            TF x0 = 1e30, y0 = 1e30, x1 = -1e30, y1 = -1e30;
            for ( SI v = 0; v < c.nb; ++v ) {
                vxs.push_back( c.vx[ v ] ); vys.push_back( c.vy[ v ] );
                x0 = std::min( x0, c.vx[ v ] ); x1 = std::max( x1, c.vx[ v ] );
                y0 = std::min( y0, c.vy[ v ] ); y1 = std::max( y1, c.vy[ v ] );
            }
            voff[ u + 1 ] = SI( vxs.size() );
            bb[ 4 * u + 0 ] = x0; bb[ 4 * u + 1 ] = y0;
            bb[ 4 * u + 2 ] = x1; bb[ 4 * u + 3 ] = y1;
            ai[ u ] = double( c.measure() );
        }

        // deux convexes sont disjoints SSI une normale d'arete de l'un les separe (axe separateur).
        // `<=` partout : deux convexes qui se touchent par une arete comptent comme se chevauchant,
        // et c'est bien ce qu'on veut -- deux cellules adjacentes se touchent exactement ainsi.
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
                // TOLERANCE, et elle n'est pas cosmetique : deux paquets ADJACENTS se touchent
                // exactement le long d'une arete, donc s'intersectent en mesure nulle. Un test
                // strict les declare separes des que l'arrondi ecarte les deux convexes de 1e-17,
                // et le certificat se met a « manquer » 3754 couples a rho = 2 -- la ou delta est
                // petit et ou les `I_k` sont donc presque le pavage grossier lui-meme. La normale
                // n'est pas unitaire : sa longueur est celle de l'arete, d'ou le facteur.
                const TF eps = TF( 1e-12 ) * std::sqrt( nx * nx + ny * ny );
                if ( c0 > a1 + eps || a0 > c1 + eps )
                    return true;
            }
            return false;
        };
        // l'ECART des deux convexes, normalise : > 0 = separes. Sert au diagnostic des manques --
        // de la tangence a 1e-16 et un vrai trou ne se corrigent pas de la meme facon.
        auto ecart = [ & ]( SI p, SI q ) {
            TF g = -1e30;
            for ( int s2 = 0; s2 < 2; ++s2 ) {
                const SI P0 = s2 ? q : p, Q0 = s2 ? p : q;
                const SI b0 = voff[ P0 ], e0 = voff[ P0 + 1 ], b1 = voff[ Q0 ], e1 = voff[ Q0 + 1 ];
                for ( SI i = b0; i < e0; ++i ) {
                    const SI j = i + 1 < e0 ? i + 1 : b0;
                    const TF nx = vys[ j ] - vys[ i ], ny = vxs[ i ] - vxs[ j ];
                    const TF ln = std::sqrt( nx * nx + ny * ny );
                    if ( ln == 0 )
                        continue;
                    TF a0 = 1e30, a1 = -1e30, c0 = 1e30, c1 = -1e30;
                    for ( SI k = b0; k < e0; ++k ) {
                        const TF s3 = ( nx * vxs[ k ] + ny * vys[ k ] ) / ln;
                        a0 = std::min( a0, s3 ); a1 = std::max( a1, s3 );
                    }
                    for ( SI k = b1; k < e1; ++k ) {
                        const TF s3 = ( nx * vxs[ k ] + ny * vys[ k ] ) / ln;
                        c0 = std::min( c0, s3 ); c1 = std::max( c1, s3 );
                    }
                    g = std::max( g, std::max( c0 - a1, a0 - c1 ) );
                }
            }
            return double( g );
        };
        auto chevauche = [ & ]( SI p, SI q ) {
            if ( voff[ p + 1 ] == voff[ p ] || voff[ q + 1 ] == voff[ q ] )
                return false;
            const TF eb = TF( 1e-12 );
            if ( bb[ 4 * q + 0 ] > bb[ 4 * p + 2 ] + eb || bb[ 4 * p + 0 ] > bb[ 4 * q + 2 ] + eb ||
                 bb[ 4 * q + 1 ] > bb[ 4 * p + 3 ] + eb || bb[ 4 * p + 1 ] > bb[ 4 * q + 3 ] + eb )
                return false;
            return ! demi( p, q ) && ! demi( q, p );
        };

        // ---- 4. LES CHEVAUCHEMENTS. On suppose qu'on sait les trouver vite ; ici une grille sur
        //         les boites suffit, sa maille etant reduite tant que les insertions debordent.
        SI g = std::max<SI>( 4, std::min<SI>( 512, SI( std::sqrt( double( m ) ) ) ) );
        auto plage = [ & ]( SI u, SI gg, SI &i0, SI &j0, SI &i1, SI &j1 ) {
            auto cl = [ & ]( TF v ) { return std::max<SI>( 0, std::min<SI>( gg - 1, SI( v * gg ) ) ); };
            i0 = cl( bb[ 4 * u + 0 ] ); j0 = cl( bb[ 4 * u + 1 ] );
            i1 = cl( bb[ 4 * u + 2 ] ); j1 = cl( bb[ 4 * u + 3 ] );
        };
        for ( ;; ) {
            long long tot = 0;
            for ( SI u = 0; u < m; ++u )
                if ( voff[ u + 1 ] > voff[ u ] ) {
                    SI i0, j0, i1, j1; plage( u, g, i0, j0, i1, j1 );
                    tot += (long long)( i1 - i0 + 1 ) * ( j1 - j0 + 1 );
                }
            if ( tot <= 64ll * m || g <= 4 )
                break;
            g /= 2;
        }
        std::vector<SI> goff( size_t( g ) * g + 2, 0 ), gval;
        for ( SI u = 0; u < m; ++u )
            if ( voff[ u + 1 ] > voff[ u ] ) {
                SI i0, j0, i1, j1; plage( u, g, i0, j0, i1, j1 );
                for ( SI j = j0; j <= j1; ++j )
                    for ( SI i = i0; i <= i1; ++i )
                        ++goff[ j * g + i + 2 ];
            }
        for ( size_t q = 1; q < goff.size(); ++q )
            goff[ q ] += goff[ q - 1 ];
        gval.resize( goff.back() );
        for ( SI u = 0; u < m; ++u )
            if ( voff[ u + 1 ] > voff[ u ] ) {
                SI i0, j0, i1, j1; plage( u, g, i0, j0, i1, j1 );
                for ( SI j = j0; j <= j1; ++j )
                    for ( SI i = i0; i <= i1; ++i )
                        gval[ goff[ j * g + i + 1  ]++ ] = u;
            }

        std::vector<double> cand( m, 0 ), cbox( m, 0 ), nov( m, 0 );
        std::vector<std::vector<SI>> mark( a.threads, std::vector<SI>( m, -1 ) );
        parallel_for( m, a.threads, Split::blocks, a.pin, [ & ]( SI u, int th ) {
            if ( voff[ u + 1 ] == voff[ u ] )
                return;
            auto &mk = mark[ th ];
            SI i0, j0, i1, j1; plage( u, g, i0, j0, i1, j1 );
            double cb = 0, ce = 0, no = 0;
            for ( SI j = j0; j <= j1; ++j )
                for ( SI i = i0; i <= i1; ++i )
                    for ( SI q = goff[ j * g + i ]; q < goff[ j * g + i + 1 ]; ++q ) {
                        const SI v = gval[ q ];
                        if ( mk[ v ] == u )
                            continue;
                        mk[ v ] = u;
                        if ( bb[ 4 * v + 0 ] > bb[ 4 * u + 2 ] + TF( 1e-12 ) ||
                             bb[ 4 * u + 0 ] > bb[ 4 * v + 2 ] + TF( 1e-12 ) ||
                             bb[ 4 * v + 1 ] > bb[ 4 * u + 3 ] + TF( 1e-12 ) ||
                             bb[ 4 * u + 1 ] > bb[ 4 * v + 3 ] + TF( 1e-12 ) )
                            continue;
                        cb += psz[ v ];
                        if ( demi( u, v ) || demi( v, u ) )
                            continue;
                        ce += psz[ v ]; no += 1;
                    }
            cand[ u ] = ce; cbox[ u ] = cb; nov[ u ] = no;
        } );

        // ---- 5. LE CERTIFICAT, et l'aire vraie de chaque paquet. On reprend TOUS les couples
        //         reellement adjacents du vrai diagramme : aucun ne doit manquer.
        std::vector<double> ar( n, 0 );
        std::vector<long long> mv( n, 0 ), rv( n, 0 );
        std::vector<double> gv( n, 0 );
        std::vector<std::vector<std::pair<SI,SI>>> adj( a.threads );
        parallel_for( n, a.threads, Split::blocks, a.pin, [ & ]( SI k, int th ) {
            Cell c;
            pf.make_cell( c, k );
            ar[ k ] = double( c.measure() );
            for ( SI v = 0; v < c.nb; ++v ) {
                const SI id = c.cid[ v ];
                if ( id < 0 )
                    continue;
                ++rv[ k ];
                if ( ! chevauche( pack[ k ], pack[ pos[ id ] ] ) ) {
                    ++mv[ k ];
                    gv[ k ] = std::max( gv[ k ], ecart( pack[ k ], pack[ pos[ id ] ] ) );
                }
                adj[ th ].emplace_back( pack[ k ], pack[ pos[ id ] ] );
            }
        } );
        // LA REFERENCE : combien de paquets sont REELLEMENT adjacents. Sans elle « 13.3 paquets
        // retenus » ne dit pas si c'est le critere qui est large ou le regroupement qui est cher.
        std::vector<std::pair<SI,SI>> ap;
        for ( auto &v : adj ) ap.insert( ap.end(), v.begin(), v.end() );
        std::sort( ap.begin(), ap.end() );
        const double vrai = double( std::unique( ap.begin(), ap.end() ) - ap.begin() ) / m;

        long long miss = 0, rel = 0;
        double gmax = 0;
        std::vector<double> par( m, 0 );
        for ( SI k = 0; k < n; ++k ) {
            miss += mv[ k ]; rel += rv[ k ]; gmax = std::max( gmax, gv[ k ] );
            par[ pack[ k ] ] += ar[ k ];
        }

        // par GERME et non par paquet : un paquet gros pese plus lourd, et c'est le germe qui paie.
        std::vector<double> cg( n ), deb;
        for ( SI k = 0; k < n; ++k )
            cg[ k ] = cand[ pack[ k ] ];
        double sov = 0, sbo = 0;
        for ( SI u = 0; u < m; ++u ) {
            sov += nov[ u ];
            sbo += cbox[ u ] * psz[ u ];
            if ( par[ u ] > 0 )
                deb.push_back( ai[ u ] / par[ u ] );
        }
        std::printf( "  %5d %8d %8.1f %8.1f %8.2f %8.0f %8.0f %9.0f %8.0f %7lld\n",
                     int( r ), int( m ), sov / m, vrai, quant( deb, 0.5 ), quant( cg, 0.5 ),
                     quant( cg, 0.9 ), quant( cg, 1.0 ), sbo / n, miss );
        if ( miss )
            std::printf( "        ^ ecart max des couples manques : %.2e\n", gmax );
        (void)rel;
    }
    }

    AaBsp bs;
    bs.build( X.data(), Y.data(), W, a.n, a.leaf );
    PowerDiagram<Cell, AaBsp, true, false, true> pr{ bs };
    for ( SI k = 0; k < a.n; ++k ) { Cell c; pr.make_cell( c, k ); }
    std::printf( "  (le BSP d'aujourd'hui : %.1f coupes tentees et %.1f boites par cellule)\n",
                 double( pr.st_tried ) / a.n, double( pr.st_boxes ) / a.n );
    return 0;
}


} // namespace

int main( int argc, char **argv ) {
    Args a;
    bool stats = false, enclos = false, baisse = false;
    for ( int i = 1; i < argc; ++i ) {
        const std::string s = argv[ i ];
        auto val = [ & ]() { return i + 1 < argc ? argv[ ++i ] : ""; };
        if ( parse_commun( a, s, i, argc, argv ) ) continue;
        if ( s == "--stats" ) stats = true;
        else if ( s == "--pack-rate" ) pack_rate = std::atoi( val() );
        else if ( s == "--enclos" )    enclos = true;
        else if ( s == "--baisse" )    baisse = true;
        else if ( s == "--pre-rate" )  pre_rate = std::atoi( val() );
        else {
            std::printf( "usage: pd_pack [options]\n" );
            usage_commun();
            std::printf( "  --pack-rate R   germes par paquet (rho) ; --leaf reste l unite d elagage\n"
                         "  --stats         compte boites/coupes par cellule au lieu de chronometrer\n"
                         "  --enclos        la SONDE : de combien la cellule calculee contre un\n"
                         "                  sous-echantillon S est trop grande, et ce qu elle permettrait\n"
                         "  --baisse        la SONDE : la parabole ABAISSEE, un enclos par paquet\n"
                         "  --pre-rate R    le sous-echantillon des sondes : un germe sur R      (16)\n" );
            return s == "--help" || s == "-h" ? 0 : 1;
        }
    }
    finalise( a );

    auto pour_chaque = [ & ]( auto &&fn ) {
        int bad = 0;
        for ( const Cloud<2> &cl : suite_2d( a ) ) {
            if ( cl.absent ) { std::printf( "  %-28s : ABSENT\n", cl.nom.c_str() ); continue; }
            std::printf( "=== %s\n", cl.nom.c_str() );
            Args b = a;
            b.n = cl.n;
            bad += fn( b, cl.c[ 0 ], cl.c[ 1 ], cl.W );
        }
        return bad;
    };
    if ( enclos )
        return pour_chaque( []( const Args &b, const std::vector<TF> &X, const std::vector<TF> &Y,
                                const TF *W ) { return enclos_stats( b, X, Y, W ); } );
    if ( baisse )
        return pour_chaque( []( const Args &b, const std::vector<TF> &X, const std::vector<TF> &Y,
                                const TF *W ) { return baisse_stats( b, X, Y, W ); } );

    if ( stats ) {
        std::printf( "=== 2D  ce que le parcours fait par cellule\n" );
        for ( const Cloud<2> &cl : suite_2d( a ) ) {
            if ( cl.absent ) continue;
            std::printf( "  %s\n", cl.nom.c_str() );
            AaBspPack tr; tr.build( cl.P, cl.W, cl.n, a.leaf );
            stats_une<AaBspPack, CellSoAT<64>, true>( tr );
        }
        return 0;
    }

    return banc<AaBspPack, Absent>( a );
}

// LE FRONT : remplacer la marche dans le BSP par un ETALEMENT de proche en proche, tant que la
// parabole du germe peut passer sous un MAJORANT AFFINE de psi.
//
// Ce banc porte l'accelerateur `FrontPd` et les trois sondes qui y ont mene, dans l'ordre ou elles
// ont ete faites :
//
//   --psigrid   le critere `min_B h_i > M( B )` sur une grille reguliere -- exact, et statique
//   --front     l ETALEMENT sur cette grille : amorce par descente, taille du front, manques
//   --front-rate R  le meme, sur les CELLULES D UN DIAGRAMME GROSSIER : le majorant y est GRATUIT
//
// Resultat : 2.1 a 4.3x sur le diagramme, mais l'index coute cinq diagrammes a construire, donc
// une boucle de Newton y perd. Voir README, « LE PAVAGE PAR UN DIAGRAMME GROSSIER ».
//
//   xmake run pd_front --help

#include "spatial_accel/AaBsp.h"
#include "spatial_accel/FrontPd.h"
#include "spatial_accel/AaBspPre.h"
#include "spatial_accel/Grid.h"
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

/// LE CRITERE DE L'ENVELOPPE SUR UNE GRILLE : le troisieme index candidat, mesure avant d'etre ecrit.
///
/// = Le critere, et il est exact
///
/// Sur une boite `B` de l'espace, `min_B psi = min_i min_B h_i` se calcule EXACTEMENT, et
///
///     M( B ) := min_i max_B h_i     majore     max_B psi
///
/// puisque `psi <= h_i` partout, pour tout `i`. D'ou, pour un dirac `i` :
///
///     min_B h_i > M( B )   =>   C( i ) inter B = vide
///
/// car alors `h_i > M( B ) >= max_B psi >= psi` en tout point de `B`. C'est un critere que la
/// grille actuelle N'A PAS : elle s'arrete sur un rayon, d'ou ses 4546 boites par cellule sur le
/// cas dur. Et il est STATIQUE -- `M` se calcule une fois pour tous les diracs, alors que le test
/// du BSP depend de la cellule en cours.
///
/// = Ce qui est mesure ici
///
/// `M( B )` est pris comme `max_B h_r` ou `r` est le dirac qui GAGNE au centre de `B` (obtenu en
/// rasterisant les vraies cellules). C'est un `min` sur un singleton, donc un majorant valide, et
/// le plus serre qu'un seul dirac puisse donner.
///
/// Puis, boite par boite, on compte les diracs que le critere retient ; on inverse ; et pour chaque
/// dirac on prend l'union des listes des boites ou il a ete retenu -- ses candidats. La colonne
/// « manques » verifie que ses VRAIS voisins de Laguerre y sont tous : ici elle doit valoir zero
/// par construction, le critere etant une implication et non une heuristique.
///
/// = L'ecueil qu'on ne mesure pas ici
///
/// L'enonce parle d'INONDATION : partir de la boite du dirac et s'etendre aux voisines. Mais rien
/// ne garantit que la boite de `p_i` soit retenue -- sur le cas dur, un autre dirac gagne en `p_i`,
/// c'est toute la difficulte du nuage. On construit donc par BOITE et non par dirac, ce qui evite
/// la question du point de depart et celle de la connexite du domaine retenu.
/// QUI GAGNE EN `x` : `argmin_j ( |x - p_j|^2 - w_j )`, cherche dans l'arbre. Rend l'indice
/// D'ORIGINE. C'est la seule chose qu'un pavage ait besoin de stocker pour porter un majorant
/// AFFINE de `psi` : toutes les paraboles partagent `|x|^2`, donc `psi - |x|^2` est un min de
/// fonctions affines, donc CONCAVE, et un majorant affine minimal en est un hyperplan d'appui --
/// c'est-a-dire la partie affine de la parabole d'un dirac qui gagne quelque part dans la tuile.
/// Stocker le proprietaire EST stocker un majorant affine.
SI gagnant( const AaBsp &tr, TF x, TF y ) {
    TF best = 1e300;
    SI bi = -1;
    tr.for_each_candidate_at( Vec<2>{ x, y }, -1,
        [ & ]( Vec<2> lo, Vec<2> hi, const WMaj &wm ) {
            const TF ex = x < lo[ 0 ] ? lo[ 0 ] - x : ( x > hi[ 0 ] ? x - hi[ 0 ] : TF( 0 ) );
            const TF ey = y < lo[ 1 ] ? lo[ 1 ] - y : ( y > hi[ 1 ] ? y - hi[ 1 ] : TF( 0 ) );
            const TF sx = TF( wm.a[ 0 ] ) * lo[ 0 ], tx = TF( wm.a[ 0 ] ) * hi[ 0 ];
            const TF sy = TF( wm.a[ 1 ] ) * lo[ 1 ], ty = TF( wm.a[ 1 ] ) * hi[ 1 ];
            const TF wmax = ( sx > tx ? sx : tx ) + ( sy > ty ? sy : ty ) + wm.b;
            return ex * ex + ey * ey - wmax < best;
        },
        [ & ]( Vec<2> q, TF w, SI id ) {
            const TF dx = q[ 0 ] - x, dy = q[ 1 ] - y;
            const TF h = dx * dx + dy * dy - w;
            if ( h < best ) { best = h; bi = id; }
            return true;
        },
        [] { return TF( 0 ); } );
    return bi;
}

/// LE FRONT SUR UNE GRILLE REGULIERE.
///
/// L'objectif : remplacer la marche dans le BSP par un ETALEMENT de proche en proche, tant que la
/// parabole du dirac peut passer sous le majorant. Trois choses a etablir, et c'est ce que cette
/// mesure fait :
///
///   1. L'AMORCE. Le front est valide depuis n'importe quelle tuile rencontrant `Lag_i` -- celle-ci
///      etant convexe, les tuiles qu'elle rencontre forment un ensemble CONNEXE, et elles passent
///      toutes le critere. Mais la tuile de `p_i` n'est PAS garantie retenue : sur le cas dur c'est
///      un autre dirac qui gagne en `p_i`. On amorce donc par une DESCENTE sur
///      `phi_i( x ) = h_i( x ) - psi( x )`, qui est un MAX de fonctions affines donc CONVEXE, et
///      qui vaut zero exactement sur `Lag_i`. Au centre d'une tuile, `psi` vaut exactement
///      `h_proprietaire`, donc `phi_i` s'y evalue sans rien chercher.
///
///   2. LA TAILLE DU FRONT. C'est elle, et non le nombre de candidats, qui remplacera les 42 a 135
///      tests de boite du BSP.
///
///   3. QU'IL NE MANQUE RIEN. On rasterise la VRAIE cellule et on verifie que chacune de ses tuiles
///      est dans le front.
int front_stats( const Args &a, const std::vector<TF> &X, const std::vector<TF> &Y, const TF *W ) {
    using Cell = CellSoAT<64>;
    AaBsp bs;
    bs.build( X.data(), Y.data(), W, a.n, a.leaf );
    PowerDiagram<Cell, AaBsp, true, false, true> pf{ bs };
    const SI n = a.n;
    auto poids = [ & ]( SI i ) { return W ? W[ i ] : TF( 0 ); };

    std::printf( "  %-5s %9s %8s   %8s %8s %8s   %8s %8s %9s\n", "g", "tuiles", "orphel.",
                 "marche", "echecs", "hors", "front", "front max", "manques" );

    for ( SI g : { SI( 64 ), SI( 128 ), SI( 256 ), SI( 512 ) } ) {
        const SI nb = g * g;
        const TF hh = TF( 1 ) / g;

        // ---- A. LE PROPRIETAIRE de chaque tuile, par requete. Parallele sans partage, une requete
        //         par tuile -- et c'est la seule chose pour laquelle le BSP reste necessaire.
        std::vector<SI> owner( nb, -1 );
        const double t0 = now();
        parallel_for( nb, a.threads, Split::blocks, false, [ & ]( SI b, int ) {
            owner[ b ] = gagnant( bs, ( b % g + TF( 0.5 ) ) * hh, ( b / g + TF( 0.5 ) ) * hh );
        } );
        const double t_own = now() - t0;

        SI orphelines = 0;
        for ( SI b = 0; b < nb; ++b )
            orphelines += owner[ b ] < 0;

        // `min_B ( h_i - h_r )`, exact : les deux paraboles ont le meme `|x|^2`, donc la difference
        // est AFFINE et son minimum sur la boite est a un coin. `|p_i|^2 - |p_r|^2` est ecrit
        // `dx ( px + rx )` et non litteralement : developpe, il ne rend pas zero quand `i == r`
        // (cf. le meme piege dans `--psigrid`), et le proprietaire s'excluait de sa propre liste.
        auto ecart = [ & ]( SI i, SI b, bool au_centre ) {
            const SI r = owner[ b ];
            const TF dx = X[ i ] - X[ r ], dy = Y[ i ] - Y[ r ];
            const TF e = dx * ( X[ i ] + X[ r ] ) + dy * ( Y[ i ] + Y[ r ] ) - poids( i ) + poids( r );
            const TF x0 = ( b % g ) * hh, y0 = ( b / g ) * hh;
            if ( au_centre )
                return e - 2 * ( dx * ( x0 + hh / 2 ) + dy * ( y0 + hh / 2 ) );
            return e - 2 * ( dx > 0 ? dx * ( x0 + hh ) : dx * x0 )
                     - 2 * ( dy > 0 ? dy * ( y0 + hh ) : dy * y0 );
        };

        // ---- B. LE FRONT, par dirac
        std::vector<double> lg_desc( n, 0 ), lg_front( n, 0 );
        std::atomic<SI> echecs{ 0 }, hors{ 0 }, manques{ 0 }, front_max{ 0 };
        const int nth = std::max( a.threads, 1 );
        std::vector<std::vector<SI>> vu( nth, std::vector<SI>( nb, -1 ) );
        std::vector<std::vector<SI>> pile( nth );

        const double t1 = now();
        parallel_for( n, a.threads, Split::blocks, false, [ & ]( SI k, int th ) {
            const SI i = bs.seed_id( k );

            // 1. L'AMORCE. Le front n'est valide que depuis une tuile qui RENCONTRE `Lag_i` -- pas
            //    seulement une tuile retenue : l'ensemble retenu peut avoir plusieurs composantes,
            //    et partir de la mauvaise fait manquer la cellule.
            //
            //    On descend `phi_i( x ) = h_i( x ) - psi( x )`, qui est un MAX de fonctions affines
            //    donc CONVEXE, nul exactement sur `Lag_i`. Au centre d'une tuile, `psi` vaut
            //    exactement `h_proprietaire`, donc `phi_i` s'y evalue sans rien chercher. Le
            //    CERTIFICAT est `proprietaire == i` : le centre est alors dans `Lag_i`.
            //
            //    ESSAYE ET REJETE : suivre la direction de descente `p_i - p_r` (« s'eloigner du
            //    concurrent »), qui est le vrai gradient de la piece affine courante. C'est une
            //    marche de visibilite ordinaire, mais elle n'a pas de critere d'arret utilisable :
            //    quand le dirac ne possede AUCUNE tuile -- 96 % d'entre eux a g=64 -- elle court
            //    jusqu'a sa borne. Mesure : 480 a 2810 pas contre 7 a 58 pour le glouton.
            SI b = std::min<SI>( g - 1, SI( Y[ i ] / hh ) ) * g + std::min<SI>( g - 1, SI( X[ i ] / hh ) );
            SI pas = 0;
            for ( ; owner[ b ] != i && pas < 4 * g; ++pas ) {
                const TF cur = ecart( i, b, true );
                SI best = b;
                TF bv = cur;
                const SI bx = b % g, by = b / g;
                for ( int dy = -1; dy <= 1; ++dy )
                    for ( int dx = -1; dx <= 1; ++dx ) {
                        const SI nx = bx + dx, ny = by + dy;
                        if ( ( dx == 0 && dy == 0 ) || nx < 0 || ny < 0 || nx >= g || ny >= g )
                            continue;
                        const TF v = ecart( i, ny * g + nx, true );
                        if ( v < bv ) { bv = v; best = ny * g + nx; }
                    }
                if ( best == b )
                    break;
                b = best;
            }
            lg_desc[ i ] = pas;

            // `hors` : on n'a PAS le certificat -- soit la cellule est plus petite qu'une tuile et
            // aucun centre ne lui appartient, soit le glouton a cale. On part quand meme de la
            // tuile atteinte si elle est retenue, et on compte separement.
            if ( owner[ b ] != i ) {
                ++hors;
                if ( ! ( ecart( i, b, false ) <= 0 ) ) {
                    ++echecs;
                    ++manques;                          // sans amorce du tout, la cellule est manquee
                    return;
                }
            }

            // 2. L'ETALEMENT : tant que la parabole peut passer sous le majorant, au sens large.
            std::vector<SI> &pi = pile[ th ];
            std::vector<SI> &vi = vu[ th ];
            pi.clear();
            pi.push_back( b );
            vi[ b ] = i;
            SI nf = 0;
            for ( SI t = 0; t < SI( pi.size() ); ++t ) {
                const SI c = pi[ t ];
                ++nf;
                const SI cx = c % g, cy = c / g;
                for ( int dy = -1; dy <= 1; ++dy )
                    for ( int dx = -1; dx <= 1; ++dx ) {
                        const SI nx = cx + dx, ny = cy + dy;
                        if ( nx < 0 || ny < 0 || nx >= g || ny >= g )
                            continue;
                        const SI d = ny * g + nx;
                        if ( vi[ d ] == i )
                            continue;
                        if ( ecart( i, d, false ) <= 0 ) { vi[ d ] = i; pi.push_back( d ); }
                    }
            }
            lg_front[ i ] = nf;
            SI fm = front_max.load( std::memory_order_relaxed );
            while ( nf > fm && ! front_max.compare_exchange_weak( fm, nf ) )
                ;

            // 3. LE CONTROLE : la vraie cellule, rasterisee, doit etre entierement dans le front.
            Cell c;
            pf.make_cell( c, k );
            if ( ! c.nb )
                return;
            TF lox, loy, hix, hiy;
            c.bounds( lox, loy, hix, hiy );
            const SI i0 = std::max<SI>( 0, SI( lox / hh ) ), i1 = std::min<SI>( g - 1, SI( hix / hh ) );
            const SI j0 = std::max<SI>( 0, SI( loy / hh ) ), j1 = std::min<SI>( g - 1, SI( hiy / hh ) );
            for ( SI jj = j0; jj <= j1; ++jj )
                for ( SI ii = i0; ii <= i1; ++ii ) {
                    const TF qx = ( ii + TF( 0.5 ) ) * hh, qy = ( jj + TF( 0.5 ) ) * hh;
                    bool in = true;
                    for ( SI v = 0; v < c.nb && in; ++v )
                        in = c.cdx[ v ] * qx + c.cdy[ v ] * qy <= c.co[ v ];
                    if ( in && vu[ th ][ jj * g + ii ] != i )
                        ++manques;
                }
        } );
        const double t_front = now() - t1;

        double sd = 0, sf = 0;
        for ( SI i = 0; i < n; ++i ) { sd += lg_desc[ i ]; sf += lg_front[ i ]; }
        std::printf( "  %-5d %9d %8d   %8.2f %8d %8d   %8.1f %8d %9d   [%.0f ms + %.0f ms]\n",
                     int( g ), int( nb ), int( orphelines ), sd / n, int( echecs.load() ),
                     int( hors.load() ), sf / n, int( front_max.load() ), int( manques.load() ),
                     1e3 * t_own, 1e3 * t_front );
    }
    return 0;
}

/// LE FRONT SUR LES CELLULES D'UN DIAGRAMME GROSSIER.
///
/// Le pavage est le diagramme de puissance d'un germe sur `--front-rate`, avec leurs vrais poids.
/// Deux proprietes le rendent bien meilleur qu'une grille reguliere, et aucune n'est un reglage :
///
/// = LE MAJORANT EST GRATUIT
///
/// Sur la cellule grossiere `T_k`, le germe `k` est lui-meme un VRAI dirac, donc `psi <= h_k`
/// partout. Le majorant affine de `T_k` est donc la parabole de son propre germe : pas de requete,
/// pas de rasterisation, rien a stocker que le pavage.
///
/// = L'ENSEMBLE RETENU EST CONNEXE, donc l'amorce suffit
///
/// Sur `T_k`, `psi_S` vaut exactement `h_k`. Le critere `min_{T_k} ( h_i - h_k ) <= 0` dit donc
/// exactement « `T_k` rencontre `E_i` », ou `E_i = { x : h_i <= psi_S }` est l'ENCLOS de `i` contre
/// `S` seul -- une cellule de puissance, donc CONVEXE. Les cellules grossieres qu'elle rencontre
/// forment un ensemble CONNEXE, et il contient toutes celles que rencontre `Lag_i` puisque
/// `Lag_i` est inclus dans `E_i`. Le piege de la grille reguliere -- plusieurs composantes, on part
/// dans la mauvaise, 12 a 2101 manques -- ne peut plus se produire.
///
/// = Et l'adjacence est donnee
///
/// `Cell::cid` porte deja les voisins de chaque cellule grossiere. Le front n'a rien a construire.
int front_pd_stats( const Args &a, const std::vector<TF> &X, const std::vector<TF> &Y, const TF *W ) {
    using Cell = CellSoAT<64>;
    const SI n = a.n;
    auto poids = [ & ]( SI i ) { return W ? W[ i ] : TF( 0 ); };

    std::printf( "  %-6s %8s %8s   %8s %8s %8s   %8s %8s   %9s %8s\n", "rho", "|S|", "sommets",
                 "amorce", "descente", "echecs", "front", "front max", "candidats", "manques" );

    for ( SI rho : { SI( 4 ), SI( 8 ), SI( 16 ), SI( 32 ), SI( 64 ) } ) {
        pre_rate = rho;
        AaBspSub t;
        t.build( X.data(), Y.data(), W, n, a.leaf );
        const SI ns = t.sub.nb_seeds();

        std::vector<SI> cidx( n, -1 );                  // id d'origine -> indice grossier
        for ( SI k = 0; k < ns; ++k )
            cidx[ t.sub.seed_id( k ) ] = k;

        // ---- A. LES CELLULES GROSSIERES, contre `S` seul : sommets et voisins.
        PowerDiagram<Cell, AaBsp, true, false, false> pc{ t.sub };
        std::vector<SI> coff( ns + 1, 0 );
        std::vector<TF> cvx, cvy;
        std::vector<SI> cadj;
        double som = 0;
        for ( SI k = 0; k < ns; ++k ) {
            Cell c;
            pc.make_cell( c, k );
            som += c.nb;
            for ( SI v = 0; v < c.nb; ++v ) {
                cvx.push_back( c.vx[ v ] );
                cvy.push_back( c.vy[ v ] );
                cadj.push_back( c.cid[ v ] < 0 ? SI( -1 ) : cidx[ c.cid[ v ] ] );
            }
            coff[ k + 1 ] = SI( cvx.size() );
        }

        // `min_T ( h_i - h_k )` : la difference de deux paraboles de meme courbure est AFFINE, donc
        // son minimum sur un convexe est a un SOMMET. `|p_i|^2 - |p_k|^2` en `dx ( px + kx )` et
        // non litteralement -- developpe il ne rend pas zero quand `i == k`.
        // LA MARGE DE TANGENCE, et c'est la troisieme fois dans ce banc. Quand `i` et `j` sont
        // tous deux des germes grossiers, leur arete FINE est portee par leur arete GROSSIERE :
        // le minimum de `h_i - h_k` sur la tuile vaut alors exactement zero, et l'arrondi le rend
        // POSITIF -- mesure, 1.7e-18. Le critere rejetait donc une tuile qu'il devait garder, et
        // 30 aretes sur 600 000 disparaissaient des listes. Elargir le retenu est toujours sur :
        // c'est un SUR-ENSEMBLE, donc l'implication reste vraie.
        const TF marge = 1e-12;
        auto ecart = [ & ]( SI i, SI c ) {
            const SI k = t.sub.seed_id( c );
            const TF dx = X[ i ] - X[ k ], dy = Y[ i ] - Y[ k ];
            const TF e = dx * ( X[ i ] + X[ k ] ) + dy * ( Y[ i ] + Y[ k ] ) - poids( i ) + poids( k );
            TF m = 1e300;
            for ( SI v = coff[ c ]; v < coff[ c + 1 ]; ++v )
                m = std::min( m, e - 2 * ( dx * cvx[ v ] + dy * cvy[ v ] ) );
            return m;
        };

        // ---- B. LE FRONT, par dirac fin
        std::vector<SI> fpos( n + 1, 0 );
        std::vector<std::vector<SI>> fro( n );
        std::atomic<SI> amorce_ok{ 0 }, echecs{ 0 }, front_max{ 0 };
        std::vector<double> lg_desc( n, 0 );

        parallel_for( n, a.threads, Split::blocks, false, [ & ]( SI i, int ) {
            // L'AMORCE : la cellule grossiere qui contient `p_i`. Elle n'est PAS garantie retenue
            // -- un germe grossier lourd peut battre `i` en `p_i` -- d'ou la descente sur
            // `omega_i = h_i - psi_S`, qui est un MAX de fonctions affines donc CONVEXE et nul
            // exactement sur `E_i`. Sur une cellule grossiere `omega_i` vaut `h_i - h_k` : le
            // critere lui-meme sert de guide.
            SI c = cidx[ gagnant( t.sub, X[ i ], Y[ i ] ) ];
            TF v = ecart( i, c );
            if ( v <= marge )
                ++amorce_ok;
            SI pas = 0;
            for ( ; v > marge && pas < 4 * ns; ++pas ) {
                SI best = c;
                TF bv = v;
                for ( SI u = coff[ c ]; u < coff[ c + 1 ]; ++u ) {
                    const SI d = cadj[ u ];
                    if ( d < 0 )
                        continue;
                    const TF w = ecart( i, d );
                    if ( w < bv ) { bv = w; best = d; }
                }
                if ( best == c )
                    break;
                c = best;
                v = bv;
            }
            lg_desc[ i ] = pas;
            if ( v > marge ) { ++echecs; return; }

            // L'ETALEMENT, sur l'adjacence des cellules grossieres. `E_i` etant convexe, l'ensemble
            // retenu est connexe : rien a craindre d'une seconde composante.
            std::vector<SI> &f = fro[ i ];
            f.push_back( c );
            for ( SI q = 0; q < SI( f.size() ); ++q )
                for ( SI u = coff[ f[ q ] ]; u < coff[ f[ q ] + 1 ]; ++u ) {
                    const SI d = cadj[ u ];
                    if ( d < 0 )
                        continue;
                    bool vu = false;
                    for ( SI z : f )
                        vu |= z == d;
                    if ( ! vu && ecart( i, d ) <= marge )
                        f.push_back( d );
                }
            SI fm = front_max.load( std::memory_order_relaxed );
            while ( SI( f.size() ) > fm && ! front_max.compare_exchange_weak( fm, SI( f.size() ) ) )
                ;
        } );

        // ---- C. la relation inverse : par cellule grossiere, les diracs retenus
        std::vector<SI> loff( ns + 2, 0 );
        double sf = 0;
        for ( SI i = 0; i < n; ++i ) {
            sf += fro[ i ].size();
            for ( SI c : fro[ i ] )
                ++loff[ c + 2 ];
        }
        for ( SI u = 1; u < ns + 2; ++u )
            loff[ u ] += loff[ u - 1 ];
        std::vector<SI> lval( loff[ ns + 1 ] );
        for ( SI i = 0; i < n; ++i )
            for ( SI c : fro[ i ] )
                lval[ loff[ c + 1 ]++ ] = i;

        // ---- D. les candidats, et le controle : les VRAIS voisins doivent tous y etre
        AaBsp bs;
        bs.build( X.data(), Y.data(), W, n, a.leaf );
        PowerDiagram<Cell, AaBsp, true, false, false> pf{ bs };
        std::atomic<SI> manques{ 0 };
        std::atomic<long long> scand{ 0 };
        std::atomic<TF> pire{ -1e300 };
        parallel_for( n, a.threads, Split::blocks, false, [ & ]( SI k, int ) {
            const SI i = bs.seed_id( k );
            std::vector<SI> cand;
            for ( SI c : fro[ i ] )
                for ( SI u = loff[ c ]; u < loff[ c + 1 ]; ++u )
                    cand.push_back( lval[ u ] );
            std::sort( cand.begin(), cand.end() );
            cand.erase( std::unique( cand.begin(), cand.end() ), cand.end() );
            scand += SI( cand.size() );

            Cell c;
            pf.make_cell( c, k );
            for ( SI v = 0; v < c.nb; ++v ) {
                const SI j = c.cid[ v ];
                if ( j < 0 || std::binary_search( cand.begin(), cand.end(), j ) )
                    continue;
                ++manques;
                // LE DIAGNOSTIC : l'arete partagee avec `j` a un milieu, ce milieu est dans une
                // cellule grossiere `T`, et `T` devrait etre retenue par `i` ET par `j`. On regarde
                // laquelle des deux retenues a echoue, et DE COMBIEN -- un ecart de l'ordre de
                // 1e-16 dit une tangence, un ecart franc dit un defaut de raisonnement.
                const SI u = v + 1 < c.nb ? v + 1 : 0;
                const TF mx = ( c.vx[ v ] + c.vx[ u ] ) / 2, my = ( c.vy[ v ] + c.vy[ u ] ) / 2;
                const SI T = cidx[ gagnant( t.sub, mx, my ) ];
                const TF ei = ecart( i, T ), ej = ecart( j, T );
                bool dans = false;
                for ( SI z : fro[ i ] ) dans |= z == T;
                TF prec = pire.load( std::memory_order_relaxed );
                const TF q = std::max( ei, ej );
                while ( q > prec && ! pire.compare_exchange_weak( prec, q ) )
                    ;
                if ( manques.load() <= 3 )
                    std::printf( "      manque %d->%d : arete %.2e, T=%d dans le front %d,"
                                 " ecart_i %.3e ecart_j %.3e\n", int( i ), int( j ),
                                 double( std::hypot( c.vx[ u ] - c.vx[ v ], c.vy[ u ] - c.vy[ v ] ) ),
                                 int( T ), int( dans ), double( ei ), double( ej ) );
            }
        } );

        std::printf( "  %-6d %8d %8.2f   %7.1f%% %8.2f %8d   %8.2f %8d   %9.1f %8d\n",
                     int( rho ), int( ns ), som / ns, 100.0 * amorce_ok.load() / n,
                     std::accumulate( lg_desc.begin(), lg_desc.end(), 0.0 ) / n, int( echecs.load() ),
                     sf / n, int( front_max.load() ), double( scand.load() ) / n,
                     int( manques.load() ) );
        if ( manques.load() )
            std::printf( "      (pire ecart d'un manque : %.3e)\n", double( pire.load() ) );
    }
    return 0;
}
int psigrid_stats( const Args &a, const std::vector<TF> &X, const std::vector<TF> &Y, const TF *W ) {
    using Cell = CellSoAT<64>;
    auto quant = []( std::vector<double> &v, double q ) {
        return v.empty() ? 0.0 : v[ std::min( v.size() - 1, size_t( q * v.size() ) ) ];
    };

    AaBsp bs;
    bs.build( X.data(), Y.data(), W, a.n, a.leaf );
    PowerDiagram<Cell, AaBsp, true, false, true> pf{ bs };
    const SI n = bs.nb_seeds();

    std::vector<SI> pos( a.n, -1 );
    for ( SI k = 0; k < n; ++k )
        pos[ bs.order[ k ] ] = k;

    struct Row { int g; double ql[ 3 ], qc[ 3 ]; long long miss, vrais; double maxlen; };

    std::printf( "  %-7s %7s %31s   %31s\n", "", "", "M SCALAIRE", "BISSECTRICE (proprietaire)" );
    std::printf( "  %-7s %7s %7s %7s %6s   %7s %7s %6s %6s %10s\n", "g x g", "diracs",
                 "retenus", "candid.", "manq.", "retenus", "candid.", "manq.", "vrais", "arete max" );

    for ( SI g : { SI( 32 ), SI( 64 ), SI( 128 ), SI( 256 ) } ) {
        const SI nb = g * g;
        const TF hh = TF( 1 ) / g;

        // ---- A. le proprietaire du centre de chaque boite, par RASTERISATION des vraies cellules.
        //         Les cellules pavent le domaine, donc chaque centre en a exactement un.
        std::vector<SI> owner( nb, -1 );
        for ( SI k = 0; k < n; ++k ) {
            Cell c;
            pf.make_cell( c, k );
            if ( ! c.nb )
                continue;
            TF lox, loy, hix, hiy;
            c.bounds( lox, loy, hix, hiy );
            const SI i0 = std::max<SI>( 0, SI( lox / hh ) ), i1 = std::min<SI>( g - 1, SI( hix / hh ) );
            const SI j0 = std::max<SI>( 0, SI( loy / hh ) ), j1 = std::min<SI>( g - 1, SI( hiy / hh ) );
            for ( SI j = j0; j <= j1; ++j )
                for ( SI i = i0; i <= i1; ++i ) {
                    const TF cx = ( i + TF( 0.5 ) ) * hh, cy = ( j + TF( 0.5 ) ) * hh;
                    bool in = true;
                    for ( SI v = 0; v < c.nb && in; ++v )
                        in = c.cdx[ v ] * cx + c.cdy[ v ] * cy <= c.co[ v ];
                    if ( in )
                        owner[ j * g + i ] = k;
                }
        }

        // ---- B. `M[ b ] = max_B h_owner`, donc le maximum sur les QUATRE COINS (`|x-p|^2` est
        //         convexe, son maximum sur une boite est a un coin).
        std::vector<TF> M( nb, 0 );
        SI orphelines = 0;
        for ( SI b = 0; b < nb; ++b ) {
            const SI k = owner[ b ];
            if ( k < 0 ) { M[ b ] = 1e300; ++orphelines; continue; }
            const TF px = bs.seed_x( k ), py = bs.seed_y( k ), pw = bs.seed_w( k );
            const TF x0 = ( b % g ) * hh, y0 = ( b / g ) * hh;
            TF m = -1e300;
            for ( int u = 0; u < 4; ++u ) {
                const TF ex = x0 + ( u & 1 ) * hh - px, ey = y0 + ( u >> 1 ) * hh - py;
                m = std::max( m, ex * ex + ey * ey - pw );
            }
            M[ b ] = m;
        }

        // ---- C/D. par boite, les diracs que le critere retient ; puis la relation inverse et,
        //           pour chaque dirac, l'union des listes des boites ou il a ete retenu.
        //
        // DEUX criteres, et la difference est tout l'interet de la mesure :
        //
        //   `M` SCALAIRE  -- `min_B h_i > M( B )`, ce que la grille stocke litteralement. Son mou
        //      vaut au moins l'OSCILLATION de `psi` sur la boite, puisque `M( B )` doit majorer
        //      `max_B psi` alors qu'on le compare a un minimum.
        //
        //   BISSECTRICE   -- `min_B ( h_i - h_r ) > 0` avec `r` le proprietaire de la boite. C'est
        //      strictement plus fort, au meme cout : `psi <= h_r`, donc `h_i > h_r` sur `B` suffit.
        //      Et `h_i - h_r` est AFFINE -- le `|x|^2` s'en va -- donc son minimum sur la boite est
        //      exact, a un coin. Le mou en `osc( psi )` disparait : ce qui reste est « le dirac `i`
        //      bat-il le proprietaire quelque part dans la boite ».
        auto mesure = [ & ]( bool bissectrice, Row &w ) {
            std::vector<std::vector<SI>> lst( nb );
            parallel_for( nb, a.threads, Split::blocks, false, [ & ]( SI b, int ) {
                const TF x0 = ( b % g ) * hh, y0 = ( b / g ) * hh;
                const TF x1 = x0 + hh, y1 = y0 + hh;
                const TF mb = M[ b ];
                const SI r = owner[ b ];
                const TF rx = bs.seed_x( r ), ry = bs.seed_y( r ), rw = bs.seed_w( r );
                std::vector<SI> &o = lst[ b ];
                for ( SI k = 0; k < n; ++k ) {
                    const TF px = bs.seed_x( k ), py = bs.seed_y( k ), pw = bs.seed_w( k );
                    bool keep;
                    if ( bissectrice ) {
                        // `|p_i|^2 - |p_r|^2` NON pas litteralement mais en `dx ( px + rx )` :
                        // ecrite telle quelle, la difference des carres ne rend pas zero quand
                        // `i == r` (`fl( px^2 + py^2 ) - px^2 - py^2` vaut environ 1e-19, parfois
                        // POSITIF), et le proprietaire de la boite s'excluait alors de sa propre
                        // liste -- 8700 voisins manques sur l'uniforme a g=256, dont des aretes de
                        // 7.5e-3, plus longues qu'une cellule entiere. Meme lecon que pour
                        // l'interpolation de `Cell::cut` : la forme factorisee est exacte la ou
                        // l'autre ne l'est pas.
                        const TF dx = px - rx, dy = py - ry;
                        const TF e = dx * ( px + rx ) + dy * ( py + ry ) - pw + rw;
                        keep = e - 2 * ( dx > 0 ? dx * x1 : dx * x0 )
                                 - 2 * ( dy > 0 ? dy * y1 : dy * y0 ) <= 0;
                    } else {
                        const TF ex = px < x0 ? x0 - px : ( px > x1 ? px - x1 : TF( 0 ) );
                        const TF ey = py < y0 ? y0 - py : ( py > y1 ? py - y1 : TF( 0 ) );
                        keep = ex * ex + ey * ey - pw <= mb;
                    }
                    if ( keep )
                        o.push_back( k );
                }
            } );

            std::vector<SI> boff( n + 2, 0 );
            for ( const std::vector<SI> &o : lst )
                for ( SI k : o )
                    ++boff[ k + 2 ];
            for ( SI u = 1; u < n + 2; ++u )
                boff[ u ] += boff[ u - 1 ];
            std::vector<SI> bval( boff[ n + 1 ] );
            for ( SI b = 0; b < nb; ++b )
                for ( SI k : lst[ b ] )
                    bval[ boff[ k + 1 ]++ ] = b;

            std::vector<double> ql, qc;
            for ( const std::vector<SI> &o : lst )
                ql.push_back( double( o.size() ) );
            std::vector<SI> stamp( n, -1 );
            long long miss = 0, vrais = 0;
            double maxlen = 0;
            for ( SI k = 0; k < n; ++k ) {
                SI cnt = 0;
                stamp[ k ] = k;
                for ( SI u = boff[ k ]; u < boff[ k + 1 ]; ++u )
                    for ( SI j : lst[ bval[ u ] ] )
                        if ( stamp[ j ] != k ) { stamp[ j ] = k; ++cnt; }
                qc.push_back( double( cnt ) );

                // Un voisin « manque » -- mais `cid` marque une coupe meme quand l'arete qu'elle
                // porte a ete reduite a RIEN par les coupes suivantes. Deux cellules qui ne se
                // touchent qu'en un point ne sont pas voisines, et aucun index n'a a les proposer.
                // On mesure donc AUSSI la longueur de l'arete manquee : si elle est nulle, c'est la
                // reference qui est sale, pas le critere.
                Cell cf;
                pf.make_cell( cf, k );
                for ( SI v = 0; v < cf.nb; ++v ) {
                    const SI id = cf.cid[ v ];
                    if ( id < 0 || stamp[ pos[ id ] ] == k )
                        continue;
                    ++miss;
                    const SI vn = ( v + 1 ) % cf.nb;
                    const TF ex = cf.vx[ vn ] - cf.vx[ v ], ey = cf.vy[ vn ] - cf.vy[ v ];
                    const double len = std::sqrt( double( ex * ex + ey * ey ) );
                    maxlen = std::max( maxlen, len );
                    if ( len > 1e-9 )
                        ++vrais;
                    if ( bissectrice && len > 1e-6 && vrais <= 3 ) {
                        // le MILIEU de l'arete manquee est, par construction, dans les deux
                        // cellules : la boite qui le contient doit retenir les deux germes.
                        const TF mx = ( cf.vx[ v ] + cf.vx[ vn ] ) / 2;
                        const TF my = ( cf.vy[ v ] + cf.vy[ vn ] ) / 2;
                        const SI bi = std::min<SI>( g - 1, SI( mx / hh ) );
                        const SI bj = std::min<SI>( g - 1, SI( my / hh ) );
                        const SI b  = bj * g + bi;
                        const SI kj = pos[ id ], r = owner[ b ];
                        bool ini = false, inj = false;
                        for ( SI q : lst[ b ] ) { ini |= q == k; inj |= q == kj; }
                        std::printf( "    MANQUE g=%d : k=%d (%.6f,%.6f) j=%d (%.6f,%.6f) "
                                     "arete %.3e milieu (%.6f,%.6f) boite %d "
                                     "[%d germes, proprio %d] k dedans=%d j dedans=%d\n",
                                     int( g ), int( k ), double( bs.seed_x( k ) ), double( bs.seed_y( k ) ),
                                     int( kj ), double( bs.seed_x( kj ) ), double( bs.seed_y( kj ) ),
                                     len, double( mx ), double( my ), int( b ),
                                     int( lst[ b ].size() ), int( r ), int( ini ), int( inj ) );
                    }
                }
            }
            std::sort( ql.begin(), ql.end() );
            std::sort( qc.begin(), qc.end() );
            w = Row{ int( g ), { quant( ql, .5 ), quant( ql, .9 ), ql.back() },
                                { quant( qc, .5 ), quant( qc, .9 ), qc.back() }, miss, vrais, maxlen };
        };

        if ( orphelines )
            std::printf( "  (g=%d : %d boites sur %d SANS proprietaire)\n",
                         int( g ), int( orphelines ), int( nb ) );
        Row wa, wb;
        mesure( false, wa );
        mesure( true,  wb );
        std::printf( "  %-7d %7.1f %7.0f %7.0f %6lld   %7.0f %7.0f %6lld %6lld %10.2e\n",
                     int( g ), double( n ) / nb,
                     wa.ql[ 0 ], wa.qc[ 0 ], wa.miss,
                     wb.ql[ 0 ], wb.qc[ 0 ], wb.miss, wb.vrais, wb.maxlen );
    }
    // la reference, sur des compteurs NEUFS : `pf` en a accumule autant de passes qu'il y a de
    // resolutions, et un diagnostic qui se compare a un chiffre faux ne vaut rien.
    PowerDiagram<Cell, AaBsp, true, false, true> pr{ bs };
    for ( SI k = 0; k < n; ++k ) { Cell c; pr.make_cell( c, k ); }
    std::printf( "  (le BSP d'aujourd'hui : %.1f coupes tentees et %.1f boites par cellule)\n",
                 pr.st_tried / double( n ), pr.st_boxes / double( n ) );
    return 0;
}

} // namespace

int main( int argc, char **argv ) {
    Args a;
    bool psigrid = false, front = false;
    SI rho = 0;
    for ( int i = 1; i < argc; ++i ) {
        const std::string s = argv[ i ];
        auto val = [ & ]() { return i + 1 < argc ? argv[ ++i ] : ""; };
        if ( parse_commun( a, s, i, argc, argv ) ) continue;
        if ( s == "--psigrid" )              psigrid = true;
        else if ( s == "--front" )           front = true;
        else if ( s == "--front-rate" )    { rho = std::atoi( val() ); front_rate = rho; }
        else if ( s == "--front-bouclier" )  front_bouclier = true;
        else {
            std::printf( "usage: pd_front [options]\n" );
            usage_commun();
            std::printf(
                "  --front-rate R  un germe grossier sur R                     (4)\n"
                "  --front-bouclier  garder l index entre iterations avec une marge 2 eps\n"
                "                  (EXACT et INUTILISABLE : la marge vaut 3 a 45 fois l echelle\n"
                "                   d une cellule -- voir README, « LE BOUCLIER »)\n"
                "  --psigrid       la SONDE : le critere min_B h_i > M(B) sur une grille reguliere\n"
                "  --front         la SONDE : l etalement, amorce / front / manques\n"
                "                  ... avec --front-rate R : sur les cellules d un diagramme grossier\n" );
            return s == "--help" || s == "-h" ? 0 : 1;
        }
    }
    finalise( a );
    front_threads = a.threads;

    auto pour_chaque = [ & ]( auto &&fn ) {
        int bad = 0;
        for ( const Cloud<2> &cl : suite_2d( a ) ) {
            if ( cl.absent ) { std::printf( "  %-28s : ABSENT\n", cl.nom.c_str() ); continue; }
            std::printf( "=== %s\n", cl.nom.c_str() );
            Args b = a;
            b.n = cl.n;                     // les sondes travaillent sur le nuage, pas sur `-n`
            bad += fn( b, cl.c[ 0 ], cl.c[ 1 ], cl.W );
        }
        return bad;
    };

    if ( psigrid )
        return pour_chaque( []( const Args &b, const std::vector<TF> &X, const std::vector<TF> &Y,
                                const TF *W ) { return psigrid_stats( b, X, Y, W ); } );
    if ( front && rho > 0 )
        return pour_chaque( []( const Args &b, const std::vector<TF> &X, const std::vector<TF> &Y,
                                const TF *W ) { return front_pd_stats( b, X, Y, W ); } );
    if ( front )
        return pour_chaque( []( const Args &b, const std::vector<TF> &X, const std::vector<TF> &Y,
                                const TF *W ) { return front_stats( b, X, Y, W ); } );

    return banc<FrontPd, Absent>( a );
}

#pragma once

#include "AaBsp.h"
#include "Cell.h"
#include "PowerDiagram.h"
#include "parallel.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace pd2d {

/// Le taux du pavage : un germe sur `front_rate` porte une cellule grossiere.
inline SI front_rate = 4;
inline int front_threads = 0;   ///< doit suivre `--threads`, sinon la preparation ment

/// LE FRONT SUR UN DIAGRAMME GROSSIER : plus de marche dans l'arbre du tout.
///
/// = LE PAVAGE
///
/// Les cellules du diagramme de puissance d'un germe sur `front_rate`, avec leurs vrais poids.
/// Elles pavent le domaine, elles sont placees la ou il y a de l'information, et leur adjacence est
/// deja portee par `Cell::cid`.
///
/// = LE MAJORANT EST GRATUIT
///
/// Sur la cellule grossiere `T_k`, le germe `k` est lui-meme un VRAI dirac, donc `psi <= h_k`
/// partout : le majorant affine de `T_k` est la parabole de son propre germe. Rien a calculer,
/// rien a stocker que le pavage. Et `h_i - h_k` est AFFINE -- les deux paraboles ont la meme
/// courbure -- donc son minimum sur un convexe est a un SOMMET : le critere
///
///     min_{T_k} ( h_i - h_k ) > 0   ==>   Lag_i ne rencontre pas T_k
///
/// est exact, en `O( nb sommets )`, et il NE DEPEND PAS de la cellule en cours de construction.
///
/// = L'ENSEMBLE RETENU EST CONNEXE, donc un front suffit
///
/// Sur `T_k`, `psi_S` vaut exactement `h_k`, donc le critere dit exactement « `T_k` rencontre
/// `E_i` », ou `E_i = { x : h_i <= psi_S }` est l'enclos de `i` contre `S` seul -- une cellule de
/// puissance, donc CONVEXE. Les cellules grossieres qu'elle rencontre forment un ensemble connexe,
/// qui contient toutes celles que rencontre `Lag_i` puisque `Lag_i` est inclus dans `E_i`.
///
/// = CE QUI EST PAYE OU
///
/// La PREPARATION porte tout : le diagramme grossier, l'amorce de chaque front (une localisation
/// dans le diagramme grossier, puis une descente), l'etalement, et l'inversion. La PASSE ne fait
/// plus que lire une liste et couper -- il n'y a plus ni pile, ni boite, ni test d'eviction.
struct FrontPd {
    AaBsp full;                     ///< tous les germes : l'enumeration et l'ordre spatial
    AaBsp sub;                      ///< le sous-echantillon, avec les identites d'origine

    std::vector<SI> coff, cadj;     ///< les cellules grossieres : voisins, par tranche
    std::vector<TF> cvx, cvy;       ///< ... et leurs sommets
    std::vector<SI> cidx;           ///< id d'origine -> indice grossier, ou `-1`

    std::vector<SI> loff, lval;     ///< par cellule grossiere, les germes qui la retiennent
                                    ///< (en places dans `full`, pour la localite)

    static constexpr const char *name = "front";
    static constexpr TF marge = 1e-12;  ///< la tangence, cf. le README : 1.7e-18 mesure

    TF seed_x( SI k ) const { return full.px[ k ]; }
    TF seed_y( SI k ) const { return full.py[ k ]; }
    TF seed_w( SI k ) const { return full.seed_w( k ); }
    SI seed_id( SI k ) const { return full.order[ k ]; }
    SI nb_seeds() const { return full.nb_seeds(); }

    /// `min_T ( h_i - h_k )`, exact : la difference de deux paraboles de meme courbure est affine,
    /// donc son minimum sur un convexe est a un sommet. `|p_i|^2 - |p_k|^2` est ecrit
    /// `dx ( px + kx )` et non litteralement -- developpe, il ne rend pas zero quand `i == k`.
    TF ecart( TF px, TF py, TF pw, SI c ) const {
        const SI k = cidx_seed[ c ];
        const TF kx = full.px[ k ], ky = full.py[ k ], kw = full.seed_w( k );
        const TF dx = px - kx, dy = py - ky;
        const TF e = dx * ( px + kx ) + dy * ( py + ky ) - pw + kw;
        TF m = 1e300;
        for ( SI v = coff[ c ]; v < coff[ c + 1 ]; ++v )
            m = std::min( m, e - 2 * ( dx * cvx[ v ] + dy * cvy[ v ] ) );
        return m;
    }

    /// LA PASSE : lire la liste, couper. Plus de pile, plus de boite, plus de test d'eviction.
    ///
    /// La liste est MATERIALISEE a la preparation, dedoublonnee et TRIEE DU PLUS PROCHE AU PLUS
    /// LOIN. Les deux comptent :
    ///
    ///  * dedoublonner, parce qu'un germe retient plusieurs cellules grossieres et que couper deux
    ///    fois avec le meme plan n'est pas neutre -- `Cell::cut` n'est pas idempotente ;
    ///  * trier par distance, parce que sans ordre le polygone INTERMEDIAIRE enfle : mesure sans
    ///    tri, 247 cellules debordent 64 sommets sur le nuage a aires egales, et la somme des aires
    ///    part a 1.000000086. Le BSP obtenait cet ordre gratuitement en descendant
    ///    fils-le-plus-proche d'abord ; ici il faut le payer, mais UNE FOIS.
    ///
    /// ESSAYE ET REJETE : faire l'union et le tri A CHAQUE CELLULE, depuis le front et la relation
    /// inverse. C'est six fois moins de memoire, mais le tri de ~144 entrees par cellule coutait
    /// plus que tout le reste : 4837 ns par germe contre 1600 pour le BSP.
    /// LE REPLI, et il est necessaire. Un germe dont la CELLULE EST VIDE a un enclos vide -- il
    /// perd contre `S` partout -- donc aucune cellule grossiere ne le retient, donc aucun candidat,
    /// donc la cellule construite reste le DOMAINE ENTIER. Mesure avant correction : `--check
    /// --weights 1.0` sortait un ecart de 1.000e+00, soit une cellule d'aire 1 la ou la vraie est
    /// vide. Vider une cellule demande jusqu'a trois demi-plans (en 2D, une intersection vide de
    /// demi-plans en a une sous-famille vide d'au plus trois), et rien ne dit lesquels : pour ces
    /// germes-la, rares, on repasse par l'arbre. C'est exact, et ca ne coute que sur eux.
    template<class MayCut, class CutWith, class Reach2>
    void for_each_candidate( SI k0, MayCut &&may_cut, CutWith &&cut_with, Reach2 &&reach2 ) const {
        const SI i0 = full.order[ k0 ];
        if ( qoff[ k0 ] == qoff[ k0 + 1 ] ) {
            full.for_each_candidate_at( full.px[ k0 ], full.py[ k0 ], i0, may_cut, cut_with, reach2 );
            return;
        }
        for ( SI u = qoff[ k0 ]; u < qoff[ k0 + 1 ]; ++u ) {
            const SI p = qval[ u ];
            const SI id = full.order[ p ];
            if ( id != i0 && ! cut_with( full.px[ p ], full.py[ p ], full.seed_w( p ), id ) )
                return;
        }
    }

    void build( const TF *X, const TF *Y, const TF *W, SI n, SI leaf ) {
        const int th = front_threads > 0 ? front_threads : 1;
        full.build( X, Y, W, n, leaf );

        const SI r = front_rate < 1 ? SI( 1 ) : front_rate;
        std::vector<SI> in;
        in.reserve( n / r + 1 );
        for ( SI k = 0; k < n; k += r )
            in.push_back( full.order[ k ] );
        sub.build_sel( X, Y, W, in.data(), SI( in.size() ), leaf );
        const SI ns = sub.nb_seeds();

        cidx.assign( n, -1 );
        cidx_seed.assign( ns, -1 );
        std::vector<SI> pos( n, -1 );
        for ( SI k = 0; k < n; ++k )
            pos[ full.order[ k ] ] = k;
        for ( SI c = 0; c < ns; ++c ) {
            cidx[ sub.seed_id( c ) ] = c;
            cidx_seed[ c ] = pos[ sub.seed_id( c ) ];   // la place du germe grossier dans `full`
        }

        // ---- les cellules grossieres : sommets et voisins
        {
            PowerDiagram<CellSoAT<64>, AaBsp, true, false, false> pc{ sub };
            coff.assign( ns + 1, 0 );
            cvx.clear(); cvy.clear(); cadj.clear();
            for ( SI c = 0; c < ns; ++c ) {
                CellSoAT<64> cl;
                pc.make_cell( cl, c );
                for ( SI v = 0; v < cl.nb; ++v ) {
                    cvx.push_back( cl.vx[ v ] );
                    cvy.push_back( cl.vy[ v ] );
                    cadj.push_back( cl.cid[ v ] < 0 ? SI( -1 ) : cidx[ cl.cid[ v ] ] );
                }
                coff[ c + 1 ] = SI( cvx.size() );
            }
        }

        // ---- le front de chaque germe : amorce, descente, etalement
        std::vector<std::vector<SI>> fro( n );
        parallel_for( n, th, Split::blocks, false, [ & ]( SI k, int ) {
            const TF px = full.px[ k ], py = full.py[ k ], pw = full.seed_w( k );

            SI c = cidx[ localise( px, py ) ];
            TF v = ecart( px, py, pw, c );
            for ( SI pas = 0; v > marge && pas < 4 * ns; ++pas ) {
                SI best = c;
                TF bv = v;
                for ( SI u = coff[ c ]; u < coff[ c + 1 ]; ++u ) {
                    const SI d = cadj[ u ];
                    if ( d < 0 )
                        continue;
                    const TF w = ecart( px, py, pw, d );
                    if ( w < bv ) { bv = w; best = d; }
                }
                if ( best == c )
                    break;
                c = best;
                v = bv;
            }
            if ( v > marge )                            // pas d'amorce : on ne peut rien affirmer
                return;

            std::vector<SI> &f = fro[ k ];
            f.push_back( c );
            for ( SI q = 0; q < SI( f.size() ); ++q )
                for ( SI u = coff[ f[ q ] ]; u < coff[ f[ q ] + 1 ]; ++u ) {
                    const SI d = cadj[ u ];
                    if ( d < 0 )
                        continue;
                    bool vu = false;
                    for ( SI z : f )
                        vu |= z == d;
                    if ( ! vu && ecart( px, py, pw, d ) <= marge )
                        f.push_back( d );
                }
        } );

        foff.assign( n + 1, 0 );
        for ( SI k = 0; k < n; ++k )
            foff[ k + 1 ] = foff[ k ] + SI( fro[ k ].size() );
        fval.resize( foff[ n ] );
        for ( SI k = 0; k < n; ++k )
            std::copy( fro[ k ].begin(), fro[ k ].end(), fval.begin() + foff[ k ] );

        // ---- la relation inverse : par cellule grossiere, les germes qui la retiennent
        loff.assign( ns + 2, 0 );
        for ( SI c : fval )
            ++loff[ c + 2 ];
        for ( SI u = 1; u < ns + 2; ++u )
            loff[ u ] += loff[ u - 1 ];
        lval.resize( loff[ ns + 1 ] );
        for ( SI k = 0; k < n; ++k )
            for ( SI u = foff[ k ]; u < foff[ k + 1 ]; ++u )
                lval[ loff[ fval[ u ] + 1 ]++ ] = k;

        // ---- les listes de candidats, dedoublonnees et triees par distance
        std::vector<std::vector<SI>> qq( n );
        parallel_for( n, th, Split::blocks, false, [ & ]( SI k, int ) {
            std::vector<SI> &q = qq[ k ];
            for ( SI u = foff[ k ]; u < foff[ k + 1 ]; ++u ) {
                const SI c = fval[ u ];
                q.insert( q.end(), lval.begin() + loff[ c ], lval.begin() + loff[ c + 1 ] );
            }
            std::sort( q.begin(), q.end() );
            q.erase( std::unique( q.begin(), q.end() ), q.end() );
            const TF px = full.px[ k ], py = full.py[ k ];
            std::sort( q.begin(), q.end(), [ & ]( SI a, SI b ) {
                const TF ax = full.px[ a ] - px, ay = full.py[ a ] - py;
                const TF bx = full.px[ b ] - px, by = full.py[ b ] - py;
                return ax * ax + ay * ay < bx * bx + by * by;
            } );
        } );
        qoff.assign( n + 1, 0 );
        for ( SI k = 0; k < n; ++k )
            qoff[ k + 1 ] = qoff[ k ] + SI( qq[ k ].size() );
        qval.resize( qoff[ n ] );
        for ( SI k = 0; k < n; ++k )
            std::copy( qq[ k ].begin(), qq[ k ].end(), qval.begin() + qoff[ k ] );

        if ( std::getenv( "PD2D_FRONT_INFO" ) )
            std::printf( "  front: rho=%d, %d cellules grossieres, front moyen %.2f,"
                         " %.1f candidats, index %.1f Mo\n",
                         int( r ), int( ns ), double( foff[ n ] ) / n, double( qoff[ n ] ) / n,
                         ( sizeof( SI ) * ( qoff[ n ] + foff[ n ] + lval.size() ) ) / 1048576.0 );
    }

    /// L'amorce : qui gagne en `( x, y )` parmi `S`. C'est le seul endroit ou un arbre est encore
    /// parcouru -- sur un arbre `front_rate` fois plus petit, et une fois par germe, a la
    /// PREPARATION.
    SI localise( TF x, TF y ) const {
        TF best = 1e300;
        SI bi = -1;
        sub.for_each_candidate_at( x, y, -1,
            [ & ]( TF lox, TF loy, TF hix, TF hiy, const WMaj &wm ) {
                const TF ex = x < lox ? lox - x : ( x > hix ? x - hix : TF( 0 ) );
                const TF ey = y < loy ? loy - y : ( y > hiy ? y - hiy : TF( 0 ) );
                const TF sx = TF( wm.ax ) * lox, tx = TF( wm.ax ) * hix;
                const TF sy = TF( wm.ay ) * loy, ty = TF( wm.ay ) * hiy;
                const TF wm2 = ( sx > tx ? sx : tx ) + ( sy > ty ? sy : ty ) + wm.b;
                return ex * ex + ey * ey - wm2 < best;
            },
            [ & ]( TF px, TF py, TF w, SI id ) {
                const TF dx = px - x, dy = py - y;
                const TF h = dx * dx + dy * dy - w;
                if ( h < best ) { best = h; bi = id; }
                return true;
            },
            [] { return TF( 0 ); } );
        return bi;
    }

    std::vector<SI> foff, fval;     ///< le front de chaque germe (places dans `full`)
    std::vector<SI> qoff, qval;     ///< les CANDIDATS de chaque germe, tries par distance
    std::vector<SI> cidx_seed;      ///< indice grossier -> place du germe grossier dans `full`
};

} // namespace pd2d

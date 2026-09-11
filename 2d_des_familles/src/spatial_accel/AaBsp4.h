#pragma once

#include "geometry/WeightMajorant.h"
#include "util/common.h"
#include <algorithm>
#include <vector>

namespace pd {

/// LE MEME BSP, MAIS A QUATRE FILS : on coupe comme avant -- mediane sur l'axe le plus long -- et
/// on recoupe tout de suite chaque moitie, mais les deux niveaux sont ranges dans UN SEUL noeud.
/// Les feuilles sont donc EXACTEMENT les memes que celles de `AaBsp` a `--leaf` egal, ce qui rend
/// la comparaison honnete : seule la facon de descendre change.
///
/// = Ce que ca cherche, et ce n'est pas « moins d'octets »
///
/// Une visite de `AaBsp` touche TROIS noeuds : le sien, pour la boite et le majorant, puis ceux
/// des deux fils, pour la `nearness` qui decide de l'ordre d'empilement. Les deux derniers sont
/// des chargements DEPENDANTS -- leur adresse sort du noeud courant -- donc leur latence ne se
/// recouvre pas. Ici les quatre boites, les quatre majorants et les quatre `nearness` sont dans le
/// noeud deja charge : une visite touche UN noeud, et les quatre tests sont independants.
///
/// = Ce que ca coute, et c'est le vrai risque
///
/// `AaBsp` teste la boite d'un fils A SA SORTIE DE PILE, donc contre la cellule telle qu'elle est
/// alors -- chaque coupe faite depuis l'empilement rend le « non » plus probable. Ici les quatre
/// fils sont testes A L'ENTREE, tous ensemble, donc les trois lointains sont juges TROP TOT. On
/// echange de la latence contre de la precision d'elagage, et c'est le banc qui tranche.
///
/// = Le noeud ne tient plus dans une ligne de cache, et c'est assume
///
/// 244 octets, quatre lignes. Mais il y a quatre fois moins de noeuds internes (0.85 Mo contre
/// 1.28 pour `AaBsp` a n=1e5), et surtout une visite lit ces quatre lignes SANS DEPENDANCE entre
/// elles, la ou `AaBsp` en lit trois en chaine.
/// = Les deux variantes, et pourquoi il en faut deux
///
/// `Lazy = false` teste les quatre boites A LA SORTIE DU PERE et n'empile que celles qui passent.
/// `Lazy = true` empile les quatre fils sans les tester et teste chacun A SA PROPRE SORTIE, comme
/// `AaBsp` : la pile porte alors un COUPLE ( pere, rang ), la boite d'un fils etant chez son pere.
/// C'est ce qui separe les deux effets -- l'arite, et le moment du test.
template<bool Lazy>
struct AaBsp4T {
    static constexpr int dim = 2;

    struct alignas( 64 ) Node {
        TF   lo[ 2 ][ 4 ], hi[ 2 ][ 4 ];    ///< les boites des QUATRE fils, en SoA
        WMaj wm[ 4 ];                       ///< leurs majorants de poids
        SI   ch[ 4 ];                       ///< le noeud fils, ou `< 0` pour une FEUILLE
        SI   beg[ 4 ], end[ 4 ];            ///< sa tranche (ce qu'une feuille fait balayer)
        SI   nb;                            ///< fils utilises : 1 (racine degeneree) a 4
    };

    std::vector<Node> nodes;
    std::vector<SI>   order;
    std::vector<TF>   px, py;
    std::vector<TF>   pw;
    SI                leaf_size = 10;

    static constexpr const char *name = Lazy ? "bsp4l" : "bsp4";

    Vec<2> seed( SI k ) const { return { px[ k ], py[ k ] }; }
    TF seed_x( SI k ) const { return px[ k ]; }
    TF seed_y( SI k ) const { return py[ k ]; }
    TF seed_w( SI k ) const { return pw.empty() ? TF( 0 ) : pw[ k ]; }
    SI seed_id( SI k ) const { return order[ k ]; }
    SI nb_seeds() const { return SI( order.size() ); }

    template<class MayCut, class CutWith, class Reach2>
    void for_each_candidate( SI k0, MayCut &&may_cut, CutWith &&cut_with, Reach2 &&reach2 ) const {
        for_each_candidate_at( Vec<2>{ px[ k0 ], py[ k0 ] }, order[ k0 ], may_cut, cut_with, reach2 );
    }

    /// La pile porte soit un NOEUD (`b < 0`), soit une FEUILLE deja retenue (`[ a, b )`). Une
    /// feuille n'est pas un noeud : elle n'a ni boite ni majorant a elle, sa boite est rangee chez
    /// son pere -- c'est ce qui evite de payer 244 octets pour dix germes.
    template<class MayCut, class CutWith, class Reach2>
    void for_each_candidate_at( Vec<2> p0, SI i0, MayCut &&may_cut, CutWith &&cut_with,
                                Reach2 && ) const {
        const TF p0x = p0[ 0 ], p0y = p0[ 1 ];
        // le carre de la distance du point a la boite du rang `i` de `nd` -- la cle d'ordre
        auto near = [ & ]( const Node &nd, SI i ) {
            const TF ex = p0x < nd.lo[ 0 ][ i ] ? nd.lo[ 0 ][ i ] - p0x
                        : ( p0x > nd.hi[ 0 ][ i ] ? p0x - nd.hi[ 0 ][ i ] : TF( 0 ) );
            const TF ey = p0y < nd.lo[ 1 ][ i ] ? nd.lo[ 1 ][ i ] - p0y
                        : ( p0y > nd.hi[ 1 ][ i ] ? p0y - nd.hi[ 1 ][ i ] : TF( 0 ) );
            return ex * ex + ey * ey;
        };
        // tri par cle DECROISSANTE -- quatre elements, donc par insertion : le plus proche est
        // empile en dernier, donc depile en premier, comme dans `AaBsp`.
        auto tri = [ & ]( TF *key, SI *cs, SI nc ) {
            for ( SI i = 1; i < nc; ++i ) {
                const TF k = key[ i ];
                const SI c = cs[ i ];
                SI j = i;
                for ( ; j > 0 && key[ j - 1 ] < k; --j ) { key[ j ] = key[ j - 1 ]; cs[ j ] = cs[ j - 1 ]; }
                key[ j ] = k;
                cs[ j ] = c;
            }
        };

        if constexpr ( Lazy ) {
            // la pile porte `4 * noeud + rang` : un fils n'a pas de boite a lui, on garde donc de
            // quoi retrouver celle que son pere lui reserve.
            SI stack[ 96 ];
            SI top = 0;
            auto empile = [ & ]( SI h ) {
                const Node &nd = nodes[ h ];
                SI cs[ 4 ];
                TF key[ 4 ];
                for ( SI i = 0; i < nd.nb; ++i ) { cs[ i ] = i; key[ i ] = near( nd, i ); }
                tri( key, cs, nd.nb );
                for ( SI i = 0; i < nd.nb; ++i )
                    stack[ top++ ] = 4 * h + cs[ i ];
            };
            empile( 0 );

            while ( top > 0 ) {
                const SI e = stack[ --top ];
                const Node &nd = nodes[ e >> 2 ];
                const SI s = e & 3;

                if ( ! may_cut( Vec<2>{ nd.lo[ 0 ][ s ], nd.lo[ 1 ][ s ] },
                                Vec<2>{ nd.hi[ 0 ][ s ], nd.hi[ 1 ][ s ] }, nd.wm[ s ] ) )
                    continue;

                if ( nd.ch[ s ] < 0 ) {
                    for ( SI k = nd.beg[ s ]; k < nd.end[ s ]; ++k ) {
                        const SI id = order[ k ];
                        if ( id != i0 && ! cut_with( Vec<2>{ px[ k ], py[ k ] }, seed_w( k ), id ) )
                            return;
                    }
                    continue;
                }
                empile( nd.ch[ s ] );
            }
            return;
        } else {

        struct Ent { SI a, b; };
        Ent stack[ 96 ];                    // depile 1, empile 4 : `3 * profondeur + 4` suffit
        SI top = 0;
        stack[ top++ ] = Ent{ 0, -1 };

        while ( top > 0 ) {
            const Ent e = stack[ --top ];

            if ( e.b >= 0 ) {
                for ( SI k = e.a; k < e.b; ++k ) {
                    const SI id = order[ k ];
                    if ( id != i0 && ! cut_with( Vec<2>{ px[ k ], py[ k ] }, seed_w( k ), id ) )
                        return;
                }
                continue;
            }

            const Node &nd = nodes[ e.a ];

            // les quatre tests, sur des boites deja chargees : aucun n'attend l'adresse d'un autre
            SI cs[ 4 ];
            TF key[ 4 ];
            SI nc = 0;
            for ( SI i = 0; i < nd.nb; ++i ) {
                if ( ! may_cut( Vec<2>{ nd.lo[ 0 ][ i ], nd.lo[ 1 ][ i ] },
                                Vec<2>{ nd.hi[ 0 ][ i ], nd.hi[ 1 ][ i ] }, nd.wm[ i ] ) )
                    continue;
                key[ nc ] = near( nd, i );
                cs[ nc ] = i;
                ++nc;
            }
            tri( key, cs, nc );
            for ( SI i = 0; i < nc; ++i ) {
                const SI s = cs[ i ];
                stack[ top++ ] = nd.ch[ s ] >= 0 ? Ent{ nd.ch[ s ], -1 }
                                                 : Ent{ nd.beg[ s ], nd.end[ s ] };
            }
        }
        }
    }

    void build( const TF *const *P, const TF *W, SI n, SI leaf ) { build( P[ 0 ], P[ 1 ], W, n, leaf ); }
    void build( const TF *X, const TF *Y, const TF *W, SI n, SI leaf ) {
        leaf_size = leaf;
        order.resize( n );
        for ( SI i = 0; i < n; ++i )
            order[ i ] = i;
        px.resize( n ); py.resize( n );
        if ( W )
            pw.resize( n );

        nodes.clear();
        nodes.reserve( n / std::max<SI>( 3 * leaf, 1 ) + 2 );

        const Box b0 = box_of( X, Y, 0, n );
        if ( fissile( n, b0 ) ) {
            _build( X, Y, W, 0, n, b0 );
        } else {
            // le nuage tient dans une feuille : un noeud a UN fils, pour que le parcours n'ait pas
            // de cas particulier
            nodes.push_back( Node{} );
            nodes[ 0 ].nb = 1;
            set_child( X, Y, W, 0, 0, 0, n, b0 );
            nodes[ 0 ].ch[ 0 ] = -1;
        }

        for ( SI k = 0; k < n; ++k ) {
            px[ k ] = X[ order[ k ] ];
            py[ k ] = Y[ order[ k ] ];
            if ( W )
                pw[ k ] = W[ order[ k ] ];
        }
    }

private:
    struct Box { TF lox, loy, hix, hiy; };

    Box box_of( const TF *X, const TF *Y, SI beg, SI end ) const {
        Box b{ X[ order[ beg ] ], Y[ order[ beg ] ], X[ order[ beg ] ], Y[ order[ beg ] ] };
        for ( SI k = beg + 1; k < end; ++k ) {
            const TF x = X[ order[ k ] ], y = Y[ order[ k ] ];
            b.lox = x < b.lox ? x : b.lox;  b.hix = x > b.hix ? x : b.hix;
            b.loy = y < b.loy ? y : b.loy;  b.hiy = y > b.hiy ? y : b.hiy;
        }
        return b;
    }

    bool fissile( SI m, const Box &b ) const {
        return m > leaf_size && ( b.hix - b.lox > 0 || b.hiy - b.loy > 0 );
    }

    /// La coupe MEDIANE sur l'axe le plus long, exactement celle de `AaBsp`.
    SI median( const TF *X, const TF *Y, SI beg, SI end, const Box &b ) {
        const SI mid = beg + ( end - beg ) / 2;
        const TF *C = ( b.hix - b.lox ) >= ( b.hiy - b.loy ) ? X : Y;
        std::nth_element( order.begin() + beg, order.begin() + mid, order.begin() + end,
                          [ & ]( SI p, SI q ) { return C[ p ] < C[ q ]; } );
        return mid;
    }

    void set_child( const TF *X, const TF *Y, const TF *W, SI me, SI i, SI beg, SI end,
                    const Box &b ) {
        nodes[ me ].lo[ 0 ][ i ] = b.lox; nodes[ me ].lo[ 1 ][ i ] = b.loy;
        nodes[ me ].hi[ 0 ][ i ] = b.hix; nodes[ me ].hi[ 1 ][ i ] = b.hiy;
        nodes[ me ].beg[ i ] = beg;
        nodes[ me ].end[ i ] = end;
        if ( W )
            nodes[ me ].wm[ i ] = weight_majorant<2>( beg, end, [ & ]( SI k, Vec<2> &q, TF &w ) {
                const SI j = order[ k ];
                q[ 0 ] = X[ j ]; q[ 1 ] = Y[ j ]; w = W[ j ];
            } );
    }

    /// Deux coupes emboitees, puis les fils. L'appelant garantit que `[ beg, end )` est fissile.
    SI _build( const TF *X, const TF *Y, const TF *W, SI beg, SI end, const Box &b ) {
        const SI me = SI( nodes.size() );
        nodes.push_back( Node{} );

        const SI mid = median( X, Y, beg, end, b );
        SI  cut[ 5 ];
        Box bx[ 4 ];
        SI  nb = 0;
        cut[ 0 ] = beg;
        for ( int h = 0; h < 2; ++h ) {                 // la moitie gauche, puis la droite
            const SI hb = h ? mid : beg, he = h ? end : mid;
            const Box hbx = box_of( X, Y, hb, he );
            if ( fissile( he - hb, hbx ) ) {
                const SI q = median( X, Y, hb, he, hbx );
                bx[ nb ] = box_of( X, Y, hb, q );  cut[ ++nb ] = q;
                bx[ nb ] = box_of( X, Y, q, he );  cut[ ++nb ] = he;
            } else {
                bx[ nb ] = hbx;                    cut[ ++nb ] = he;
            }
        }
        nodes[ me ].nb = nb;

        for ( SI i = 0; i < nb; ++i )
            set_child( X, Y, W, me, i, cut[ i ], cut[ i + 1 ], bx[ i ] );

        // les recursions APRES les boites : `nodes` se realloue, donc `nodes[ me ]` ne doit pas
        // etre tenu a travers un appel
        for ( SI i = 0; i < nb; ++i ) {
            const SI cb = cut[ i ], ce = cut[ i + 1 ];
            const SI c = fissile( ce - cb, bx[ i ] ) ? _build( X, Y, W, cb, ce, bx[ i ] ) : -1;
            nodes[ me ].ch[ i ] = c;
        }
        return me;
    }
};

using AaBsp4  = AaBsp4T<false>;
using AaBsp4L = AaBsp4T<true>;

} // namespace pd

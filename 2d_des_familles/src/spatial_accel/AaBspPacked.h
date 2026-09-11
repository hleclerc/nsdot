#pragma once

#include "geometry/WeightMajorant.h"
#include "util/common.h"
#include <algorithm>
#include <vector>

namespace pd {

/// LE MEME BSP, mais tout dans UNE ARENE, et les points COLLES A LEUR FEUILLE.
///
/// = Ce que ca change, et pourquoi ca devrait compter
///
/// Dans `AaBsp`, visiter une feuille touche QUATRE regions memoire eloignees : le noeud (dans
/// `nodes`), les abscisses (`px`), les ordonnees (`py`), les indices (`order`). Quatre lignes de
/// cache a quatre endroits, pour une seule feuille -- et la marche est du pointer-chasing, donc ce
/// qui coute n'est pas le nombre d'octets lus mais le nombre de LIGNES touchees.
///
/// Ici, une feuille est `[ en-tete ][ germe 0 ][ germe 1 ] ...` d'un seul tenant : les memes
/// donnees tiennent dans quatre lignes CONSECUTIVES, que le prefetcheur suit tout seul. Et un
/// germe est `( x, y, w, id )` ensemble, parce que la boucle des candidats se sert des quatre.
///
/// = Des offsets plutot que des index
///
/// Un noeud interne n'a pas de charge utile, donc son fils gauche le suit IMMEDIATEMENT (preordre).
/// Le fils droit est un offset stocke. Tout est en « slots » de `TF` : c'est ce qui garde
/// l'alignement naturel sans arithmetique d'octets.
struct AaBspPacked {
    static constexpr int dim = 2;

    struct Node {
        TF lo[ 2 ], hi[ 2 ];    ///< la boite du sous-arbre
        WMaj wm;                ///< MAJORANT AFFINE des poids du sous-arbre
        int32_t right;          ///< offset (en slots) du fils droit ; `< 0` dit FEUILLE
        int32_t nb;             ///< feuille : combien de germes SUIVENT
    };
    struct Seed {
        TF x, y;                ///< la position
        TF w;                   ///< le poids
        SI id, _pad;            ///< l'indice d'origine (le `cut_id`)
    };

    static constexpr SI node_slots = SI( sizeof( Node ) / sizeof( TF ) );
    static constexpr SI seed_slots = SI( sizeof( Seed ) / sizeof( TF ) );
    static_assert( sizeof( Node ) % sizeof( TF ) == 0 && sizeof( Seed ) % sizeof( TF ) == 0 );

    std::vector<TF> arena;      ///< l'arbre ENTIER : en-tetes et points melanges, en preordre

    // la boucle EXTERNE (un germe apres l'autre) veut un acces sequentiel ordinaire : elle garde
    // des tableaux plats a cote. C'est une copie des points, assumee -- elle est lue en flux, pas
    // en acces disperse, donc elle ne coute que sa bande passante.
    std::vector<SI> order;
    std::vector<TF> px, py, pw;
    SI              leaf_size = 10;

    static constexpr const char *name = "packed";

    Vec<2> seed( SI k ) const { return { px[ k ], py[ k ] }; }
    TF seed_x( SI k ) const { return px[ k ]; }
    TF seed_y( SI k ) const { return py[ k ]; }
    TF seed_w( SI k ) const { return pw.empty() ? TF( 0 ) : pw[ k ]; }
    SI seed_id( SI k ) const { return order[ k ]; }
    SI nb_seeds() const { return SI( order.size() ); }

    /// Le meme parcours que `AaBsp` -- profondeur d'abord, fils le plus proche en premier -- sur la
    /// meme forme d'arbre. Ce qui change est la LECTURE : un seul flux, les germes derriere leur
    /// en-tete.
    template<class MayCut, class CutWith, class Reach2>
    void for_each_candidate( SI k0, MayCut &&may_cut, CutWith &&cut_with, Reach2 && ) const {
        const TF p0x = px[ k0 ], p0y = py[ k0 ];
        const SI i0 = order[ k0 ];

        SI stack[ 64 ];
        SI top = 0;
        stack[ top++ ] = 0;

        while ( top > 0 ) {
            const SI h = stack[ --top ];
            const Node &nd = node( h );

            if ( ! may_cut( vec_of<2>( nd.lo ), vec_of<2>( nd.hi ), nd.wm ) )
                continue;

            if ( nd.right < 0 ) {
                const Seed *sd = reinterpret_cast<const Seed *>(
                                     reinterpret_cast<const TF *>( &nd ) + node_slots );
                for ( SI j = 0; j < nd.nb; ++j )
                    if ( sd[ j ].id != i0 && ! cut_with( Vec<2>{ sd[ j ].x, sd[ j ].y }, sd[ j ].w, sd[ j ].id ) )
                        return;
                continue;
            }

            const SI l = h + node_slots, r = nd.right;
            if ( nearness( l, p0x, p0y ) <= nearness( r, p0x, p0y ) ) {
                stack[ top++ ] = r;
                stack[ top++ ] = l;
            } else {
                stack[ top++ ] = l;
                stack[ top++ ] = r;
            }
        }
    }

    void build( const TF *const *P, const TF *W, SI n, SI leaf ) { build( P[ 0 ], P[ 1 ], W, n, leaf ); }
    void build( const TF *X, const TF *Y, const TF *W, SI n, SI leaf ) {
        leaf_size = leaf;
        order.resize( n );
        for ( SI i = 0; i < n; ++i )
            order[ i ] = i;

        arena.clear();
        arena.reserve( size_t( n ) * seed_slots + size_t( 2 * n / std::max<SI>( leaf, 1 ) + 2 ) * node_slots );
        _build( X, Y, W, 0, n );

        px.resize( n ); py.resize( n );
        if ( W )
            pw.resize( n );
        for ( SI k = 0; k < n; ++k ) {
            px[ k ] = X[ order[ k ] ];
            py[ k ] = Y[ order[ k ] ];
            if ( W )
                pw[ k ] = W[ order[ k ] ];
        }
    }

private:
    const Node &node( SI h ) const { return *reinterpret_cast<const Node *>( arena.data() + h ); }

    TF nearness( SI h, TF x, TF y ) const {
        const Node &nd = node( h );
        const TF ex = x < nd.lo[ 0 ] ? nd.lo[ 0 ] - x : ( x > nd.hi[ 0 ] ? x - nd.hi[ 0 ] : TF( 0 ) );
        const TF ey = y < nd.lo[ 1 ] ? nd.lo[ 1 ] - y : ( y > nd.hi[ 1 ] ? y - nd.hi[ 1 ] : TF( 0 ) );
        return ex * ex + ey * ey;
    }

    SI _build( const TF *X, const TF *Y, const TF *W, SI beg, SI end ) {
        const SI me = SI( arena.size() );
        arena.resize( me + node_slots );

        TF lox = X[ order[ beg ] ], hix = lox, loy = Y[ order[ beg ] ], hiy = loy;
        for ( SI k = beg + 1; k < end; ++k ) {
            const TF x = X[ order[ k ] ], y = Y[ order[ k ] ];
            lox = x < lox ? x : lox;  hix = x > hix ? x : hix;
            loy = y < loy ? y : loy;  hiy = y > hiy ? y : hiy;
        }

        // `arena` peut REALLOUER a chaque `resize`, donc on ne garde jamais de pointeur : tout
        // passe par l'offset `me`.
        auto self = [ & ]() -> Node & { return *reinterpret_cast<Node *>( arena.data() + me ); };
        self().lo[ 0 ] = lox; self().lo[ 1 ] = loy;
        self().hi[ 0 ] = hix; self().hi[ 1 ] = hiy;
        self().wm = WMaj{};
        if ( W )
            self().wm = weight_majorant<2>( beg, end, [ & ]( SI k, Vec<2> &q, TF &w ) {
                const SI i = order[ k ];
                q[ 0 ] = X[ i ]; q[ 1 ] = Y[ i ]; w = W[ i ];
            } );

        const int ax = ( hix - lox ) >= ( hiy - loy ) ? 0 : 1;
        const TF span = ax ? hiy - loy : hix - lox;

        if ( end - beg <= leaf_size || ! ( span > 0 ) ) {
            const SI nb = end - beg;
            arena.resize( me + node_slots + nb * seed_slots );
            self().right = -1;
            self().nb = int32_t( nb );
            Seed *sd = reinterpret_cast<Seed *>( arena.data() + me + node_slots );
            for ( SI j = 0; j < nb; ++j ) {
                const SI i = order[ beg + j ];
                sd[ j ] = Seed{ X[ i ], Y[ i ], W ? W[ i ] : TF( 0 ), i, 0 };
            }
            return me;
        }

        const SI mid = beg + ( end - beg ) / 2;
        const TF *C = ax ? Y : X;
        std::nth_element( order.begin() + beg, order.begin() + mid, order.begin() + end,
                          [ & ]( SI a, SI b ) { return C[ a ] < C[ b ]; } );

        _build( X, Y, W, beg, mid );                    // == me + node_slots
        const SI r = _build( X, Y, W, mid, end );
        self().right = int32_t( r );
        self().nb = 0;
        return me;
    }
};

} // namespace pd

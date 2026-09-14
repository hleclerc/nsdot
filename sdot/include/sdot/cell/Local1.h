#pragma once

// =====================================================================================
// LA CELLULE 1D : un SEGMENT, ou une demi-droite, ou la droite entiere.
//
// Deux sommets au plus, `vx[ 0 ] <= vx[ 1 ]`, et une coupe par bout : `cid[ 0 ]` porte le bout
// gauche, `cid[ 1 ]` le droit. Le non borne suit la meme convention que les autres dimensions --
// un bout marque `INFINITE` est pose a une distance inventee ( `+-1` de l'autre bout ) et repousse
// avant chaque coupe qui pourrait le classer ; ici « repousser » est immediat : un bout infini est
// toujours du bon cote, il suffit de le regarder comme tel.
//
// Il n'y a pas de noyau a registres ni de diagramme en 1D ( `OtPlan1d` a le sien ) : c'est la
// forme de `Cell_1.py`, et rien de plus.
// =====================================================================================

#include <loom/support/common_types.h>
#include <loom/support/containers/Vector.h>
#include "Scratch.h"
#include "Plane.h"
#include "Etat.h"
#include "Ids.h"

#include <limits>

namespace sdot {

template<class TK>
struct Local1 {
    static constexpr int ct_dim = 1;
    using TKernel = TK;
    using PlaneT  = Plane<TK,1>;

    int  nb  = 0;                                        ///< 0 ( vide ) ou 2
    int  cap = 0;
    TK  *vx  = nullptr;
    int *cid = nullptr;

    static constexpr SI words_for( SI cap ) { return words_of<TK>( cap ) + words_of<int>( cap ); }

    bool attach( Carver &c, SI capacity ) {
        cap = int( capacity );
        vx = c.take<TK>( cap );
        cid = c.take<int>( cap );
        nb = 0;
        return ! c.overflow && cap >= 2;
    }

    bool copy_from( const Local1 &o ) {
        nb = o.nb;
        for ( int i = 0; i < nb; ++i ) { vx[ i ] = o.vx[ i ]; cid[ i ] = o.cid[ i ]; }
        return true;
    }

    int  nb_vertices() const { return nb; }
    int  nb_cuts    () const { return nb; }
    bool unbounded_at( int i ) const { return cid[ i ] == cell_ids::INFINITE; }
    bool bounded    () const { return nb == 0 || ( ! unbounded_at( 0 ) && ! unbounded_at( 1 ) ); }
    TK   coord      ( int i, int ) const { return vx[ i ]; }
    int  vertex_cut ( int i, int ) const { return i; }

    bool init_hypercube( const auto &origin, const auto &axes, int cut_id ) {
        const TK a = TK( origin[ 0 ] ), b = a + TK( axes( 0, 0 ) );
        vx[ 0 ] = a < b ? a : b;
        vx[ 1 ] = a < b ? b : a;
        cid[ 0 ] = cid[ 1 ] = cut_id;
        nb = 2;
        return true;
    }

    bool init_unbounded() {
        vx[ 0 ] = 0; vx[ 1 ] = 1;
        cid[ 0 ] = cid[ 1 ] = cell_ids::INFINITE;
        nb = 2;
        return true;
    }

    void make_empty() { nb = 0; }
    void tidy() {}

    /// `dir * x <= off`
    int cut( const PlaneT &p ) {
        if ( nb == 0 || p.dir[ 0 ] == 0 )
            return p.dir[ 0 ] == 0 && p.off < 0 ? ( nb = 0, CutStatus::EMPTY ) : CutStatus::UNCHANGED;
        const TK x = p.off / p.dir[ 0 ];
        // le bout que la coupe retient est `x` ; l'autre reste s'il est du bon cote -- un bout
        // infini l'est toujours
        const int keep = p.dir[ 0 ] > 0 ? 0 : 1, kill = 1 - keep;
        const bool kill_inf = unbounded_at( kill ), keep_inf = unbounded_at( keep );
        if ( ! kill_inf && ( p.dir[ 0 ] > 0 ? vx[ kill ] <= x : vx[ kill ] >= x ) )
            return CutStatus::UNCHANGED;
        if ( ! keep_inf && ( p.dir[ 0 ] > 0 ? vx[ keep ] > x : vx[ keep ] < x ) ) {
            nb = 0;
            return CutStatus::EMPTY;
        }
        vx[ kill ] = x;
        cid[ kill ] = p.id;
        if ( keep_inf )                                  // l'autre bout, factice, suit a distance 1
            vx[ keep ] = keep ? x + 1 : x - 1;
        return CutStatus::CUT;
    }

    template<class TF>
    TF measure() const {
        if ( nb == 0 )
            return 0;
        if ( ! bounded() )
            return std::numeric_limits<TF>::max();
        return TF( vx[ 1 ] ) - TF( vx[ 0 ] );
    }

    template<class TF>
    void measure_bwd( TF grad_res, auto &&grad_vp ) const {
        if ( nb == 0 || ! bounded() )
            return;
        grad_vp( 0, 0 ) = - grad_res;
        grad_vp( 1, 0 ) = grad_res;
    }

    void for_each_simplex( auto &&func ) const {
        if ( nb == 2 ) {
            Vector<SI,2> chain;
            chain[ 0 ] = 0; chain[ 1 ] = 1;
            func( chain );
        }
    }

    template<class T>
    void plane( int i, T *dir, T &off ) const {
        dir[ 0 ] = i ? T( 1 ) : T( -1 );
        off = dir[ 0 ] * T( vx[ i ] );
    }

    void bbox( TK *lo, TK *hi ) const { lo[ 0 ] = nb ? vx[ 0 ] : TK( 0 ); hi[ 0 ] = nb ? vx[ 1 ] : TK( 0 ); }

    bool load( const auto &c ) {
        nb = int( SI( c.nb_vertices ) );
        for ( int i = 0; i < nb; ++i ) {
            vx[ i ] = TK( c.vertex_positions( i, 0 ) );
            cid[ i ] = int( SI( c.cut_ids( i ) ) );
        }
        return true;
    }

    bool store( auto &&c ) const {
        if ( ! c.nb_vertices.set( nb ) )
            return false;
        c.nb_cuts.set( nb );
        for ( int i = 0; i < nb; ++i ) {
            c.vertex_positions( i, 0 ) = vx[ i ];
            c.cut_ids( i ) = cid[ i ];
        }
        return true;
    }
};

} // namespace sdot

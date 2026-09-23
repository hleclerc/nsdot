#pragma once

// =====================================================================================
// CE QUE LES TROIS `Cell_*.h` ONT EN COMMUN : poser une cellule locale sur le scratch d'un item, et
// les operations de `Cell_*.py` ( init, coupe, mesure ) ecrites une fois sur `Local1 / 2 / N`.
//
// LE FLOTTANT DU NOYAU est celui que le scratch declare ( `kernel_fp_size`, une constante de
// compilation ) : `float` par defaut, `double` au choix. La cellule STOCKEE est dans le flottant
// de l'appelant ( `TF`, celui de `vertex_positions` ) ; `load` / `store` convertissent.
//
// LA CAPACITE se deduit du scratch : `cap_for` est le plus grand nombre de sommets dont les
// tableaux tiennent dans les mots recus ( `Local::words_for`, la formule que `Cell_*.py` a
// utilisee pour dimensionner ). Un scratch trop petit se signale sur `nb_words` -- ce que loom
// sait faire grossir -- et la cellule n'ecrit rien.
// =====================================================================================

#include <loom/support/common_macros.h>
#include "Scratch.h"

#include <type_traits>

namespace sdot {

/// le flottant du noyau, lu sur le scratch
template<class Scr>
using KernelType = std::conditional_t<DECAYED_TYPE_OF( std::declval<Scr>().kernel_fp_size )::value == 64, double, float>;

/// le plus grand `cap` tel que `words( cap ) <= nb_words` ( `words` croissante )
HD SI cap_for_words( SI nb_words, auto &&words ) {
    SI lo = 0, hi = 1;
    while ( words( hi ) <= nb_words )
        hi *= 2;
    while ( hi - lo > 1 ) {
        const SI mid = ( lo + hi ) / 2;
        if ( words( mid ) <= nb_words ) lo = mid;
        else hi = mid;
    }
    return lo;
}

/// le plus grand `cap` tel que `Local::words_for( cap ) <= nb_words`
template<class Local>
HD SI cap_for( SI nb_words ) {
    return cap_for_words( nb_words, []( SI c ) { return Local::words_for( c ); } );
}

/// une cellule locale posee sur le scratch d'un item ( `Local` attache, `cap` deduit )
template<class Local,class Scr>
HD Local local_on( Scr &sc, Carver &cv ) {
    Local c;
    c.attach( cv, cap_for<Local>( cv.nb_words ) );
    return c;
}

/// la ligne `row` du scratch ( `words` est `[ nb_threads, nb_words ]` : une ligne par work-item,
/// ou une seule quand l'appel est batche sur les cellules )
template<class Scr>
HD Carver carver_of( Scr &sc, SI row = 0 ) {
    auto w = sc.words( row );
    return Carver{ w.data().raw, SI( w.shape( 0 ) ) };
}

/// le scratch n'a pas suffi : on le dit ( loom double et relance ), sans rien ecrire
template<class Scr>
HD void ask_more( Scr &sc, const Carver &cv ) {
    sc.nb_words.set( 2 * cv.nb_words + 64 );
}

// ---- les operations de `Cell_*.py`, une fois pour toutes -------------------------------------

namespace cell_ops {

template<class Local>
HD void init_as_hypercube( auto &&cell, auto &&scratch, auto &&origin, auto &&axes, SI cut_id ) {
    Carver cv = carver_of( scratch );
    Local c = local_on<Local>( scratch, cv );
    if ( ! c.init_hypercube( origin, axes, int( cut_id ) ) ) { ask_more( scratch, cv ); return; }
    c.tidy();
    c.store( cell );
}

template<class Local>
HD void init_as_unbounded( auto &&cell, auto &&scratch ) {
    Carver cv = carver_of( scratch );
    Local c = local_on<Local>( scratch, cv );
    if ( ! c.init_unbounded() ) { ask_more( scratch, cv ); return; }
    c.store( cell );
}

/// intersecte avec `direction . x <= offset`, le resultat allant dans `res` ( les entrees et
/// les sorties d'un appel sont disjointes ). Un debordement est signale sur `res.nb_vertices`.
template<class Local>
HD void cut( const auto &cell, auto &&res, auto &&scratch, auto &&direction, auto &&offset, SI cut_id ) {
    using TK = typename Local::TKernel;
    constexpr int D = Local::ct_dim;
    Carver cv = carver_of( scratch );
    Local c = local_on<Local>( scratch, cv );
    if ( ! c.load( cell ) ) { ask_more( scratch, cv ); return; }
    typename Local::PlaneT p;
    for ( int d = 0; d < D; ++d )
        p.dir[ d ] = TK( direction( d ) );
    p.off = TK( offset );
    p.id  = int( cut_id );
    if ( c.cut( p ) == CutStatus::OVERFLOW ) { ask_more( scratch, cv ); return; }
    c.tidy();
    c.store( res );
}

template<class Local>
HD void measure( const auto &cell, auto &&res, auto &&scratch ) {
    using TF = DECAYED_TYPE_OF( res )::TF;
    Carver cv = carver_of( scratch );
    Local c = local_on<Local>( scratch, cv );
    if ( ! c.load( cell ) ) { ask_more( scratch, cv ); return; }
    res = c.template measure<TF>();
}

template<class Local>
HD void measure_bwd( const auto &cell, auto &&res, auto &&grad_res, auto &&grad_vertex_positions, auto &&scratch ) {
    if constexpr ( ! CT_VALUE( grad_vertex_positions.surely_null() ) ) {
        using TF = DECAYED_TYPE_OF( grad_res )::TF;
        constexpr int D = Local::ct_dim;
        Carver cv = carver_of( scratch );
        Local c = local_on<Local>( scratch, cv );
        if ( ! c.load( cell ) ) { ask_more( scratch, cv ); return; }
        // la cotangente a une valeur partout ou sa primale en a une : tout le tampon, padding compris
        const SI capv = SI( grad_vertex_positions.shape( 0 ) );
        for ( SI i = 0; i < capv; ++i )
            for ( int d = 0; d < D; ++d )
                grad_vertex_positions( i, d ) = 0;
        c.template measure_bwd<TF>( TF( grad_res ), [&]( int i, int d ) -> auto & { return grad_vertex_positions( i, d ).ref(); } );
    }
}

} // namespace cell_ops
} // namespace sdot

#pragma once

#include <loom/support/common_macros.h>
#include <loom/support/containers/Vector.h>

namespace sdot {

// UN NIVEAU de la construction du BSP, pour UN noeud : `AaBsp.py::_build`, mis dans un kernel.
//
// = Pourquoi un niveau et pas l'arbre entier
//
// Un niveau ne peut pas commencer avant que le precedent soit fini (il lit les tranches qu'il
// produit), et il n'y a pas de barriere GLOBALE dans un kernel SYCL -- seulement au sein d'un
// work-group. La barriere est donc la FIN DU LANCEMENT : l'hote enchaine `depth` appels, ce qui
// est le motif habituel sur GPU et ne coute que `depth` lancements (une quinzaine a 1e6 germes).
//
// Rien de tout ca n'a de capacite a deviner : `depth` est `max_depth_for( n, leaf_size )`, une
// fonction de `n` seul, donc la boucle se deroule sous un `jit` comme ailleurs.
//
// = Ce qu'un work-item fait, et ce qui rend l'ecriture disjointe
//
// Un work-item par noeud du niveau. Les tranches `[ begin, end )` d'un niveau PARTITIONNENT
// `[ 0, n )` -- c'est la raison d'etre de la propagation decrite dans `AaBsp.py::_build` -- donc
// deux work-items n'ecrivent jamais la meme case, ni dans le nuage de sortie ni dans le scratch de
// permutation. Aucun atomique, aucune barriere.
//
// Le nuage est DOUBLE-TAMPONNE (`pos_in` -> `pos_out`) parce que les entrees et les sorties d'un
// appel sont disjointes (voir `driver.call`), et il porte les positions et les poids PERMUTES a
// cote des indices : un noeud lit alors ses points d'un seul tenant, la ou une indirection par
// `seed_indices` en ferait une collecte eparse -- ce qui compte d'autant plus que les niveaux du
// haut sont traites par tres peu de work-items.


// Quickselect (Hoare, pivot median de trois) : rearrange `perm[ b .. e )` de sorte que le rang
// `t - b` soit a l'indice `t`, tout ce qui precede <= et tout ce qui suit >=.
//
// Ecrit a la main plutot que `std::nth_element` : rien de la libstdc++ n'est utilise dans les
// kernels d'ici, et l'introselect en ferait dependre la compilation device. La cle est lue par
// indirection (`pos( perm( k ), ax )`), mais `perm` part de l'identite et les deux balayages de
// Hoare sont lineaires, donc les acces restent quasi sequentiels dans la tranche du noeud.
void bsp_select( auto &&perm, const auto &pos, SI b, SI e, SI t, int ax ) {
    using TF = typename DECAYED_TYPE_OF( pos )::TF;

    auto key = [&]( SI k ) { return TF( pos( SI( perm( k ) ), ax ) ); };
    auto swp = [&]( SI i, SI j ) { const SI x = SI( perm( i ) ); perm( i ) = SI( perm( j ) ); perm( j ) = x; };

    SI lo = b, hi = e;
    while ( hi - lo > 2 ) {
        const SI c = lo + ( hi - lo ) / 2;
        const TF k0 = key( lo ), k1 = key( c ), k2 = key( hi - 1 );
        // le pivot est TOUJOURS une valeur presente dans la tranche : c'est ce qui garantit que
        // les deux balayages ci-dessous s'arretent sans test de borne.
        const TF pivot = k0 < k1 ? ( k1 < k2 ? k1 : ( k0 < k2 ? k2 : k0 ) )
                                 : ( k0 < k2 ? k0 : ( k1 < k2 ? k2 : k1 ) );

        SI i = lo - 1, j = hi;
        while ( true ) {
            do { ++i; } while ( key( i ) < pivot );
            do { --j; } while ( key( j ) > pivot );
            if ( i >= j )
                break;
            swp( i, j );
        }

        // `[ lo, j ]` et `[ j + 1, hi )`, tous deux non vides (Hoare avec un pivot present coupe
        // au milieu meme quand toutes les valeurs sont egales, donc la recursion converge).
        if ( t <= j )
            hi = j + 1;
        else
            lo = j + 1;
    }

    if ( hi - lo == 2 && key( lo ) > key( lo + 1 ) )
        swp( lo, lo + 1 );
}


// `( a, b )` tels que `w_k <= a . y_k + b` pour tout germe de la tranche -- voir
// `AaBsp.py::_weight_majorant` pour POURQUOI le majorant est affine et comment le candidat est
// retenu. Meme regle, meme seuil ; seul l'ajustement differe (voir plus bas).
template<int ct_dim>
void bsp_weight_majorant( const auto &pos, const auto &w, SI b, SI e, auto &&wa_out, auto &&wb_out ) {
    using TF = typename DECAYED_TYPE_OF( pos )::TF;

    const SI m = e - b;

    TF wmin = TF( w( b ) ), wmax = wmin, wsum = 0;
    auto psum = Vector<TF,ct_dim>::zeros();
    for ( SI k = b; k < e; ++k ) {
        const TF v = TF( w( k ) );
        wmin = v < wmin ? v : wmin;
        wmax = v > wmax ? v : wmax;
        wsum += v;
        for ( int d = 0; d < ct_dim; ++d )
            psum[ d ] += TF( pos( k, d ) );
    }
    const TF spread = wmax - wmin;

    auto a = Vector<TF,ct_dim>::zeros();
    if ( m >= 2 * ( ct_dim + 1 ) && spread > 0 ) {
        const TF inv = TF( 1 ) / TF( m );
        const auto pm = Vector<TF,ct_dim>::with_func( [&]( PI d ) { return psum[ d ] * inv; } );
        const TF wm = wsum * inv;

        // les EQUATIONS NORMALES du moindre carre centre, `[ q^T q | q^T dw ]`, resolues par Gauss
        // avec pivot partiel. L'hote, lui, passe par une SVD (`lstsq`), mieux conditionnee -- et ca
        // n'a pas a l'etre ici : quel que soit le `a` qui sort, le `b` calcule plus bas est RELEVE
        // jusqu'a majorer, donc un ajustement mediocre ne peut qu'elaguer moins, jamais mentir.
        TF A[ ct_dim ][ ct_dim + 1 ];
        for ( int i = 0; i < ct_dim; ++i )
            for ( int j = 0; j <= ct_dim; ++j )
                A[ i ][ j ] = 0;
        for ( SI k = b; k < e; ++k ) {
            const TF dw = TF( w( k ) ) - wm;
            for ( int i = 0; i < ct_dim; ++i ) {
                const TF qi = TF( pos( k, i ) ) - pm[ i ];
                for ( int j = 0; j < ct_dim; ++j )
                    A[ i ][ j ] += qi * ( TF( pos( k, j ) ) - pm[ j ] );
                A[ i ][ ct_dim ] += qi * dw;
            }
        }

        bool ok = true;
        for ( int c = 0; c < ct_dim && ok; ++c ) {
            int p = c;
            for ( int i = c + 1; i < ct_dim; ++i )
                if ( sycl::fabs( A[ i ][ c ] ) > sycl::fabs( A[ p ][ c ] ) )
                    p = i;
            if ( ! ( sycl::fabs( A[ p ][ c ] ) > 0 ) ) {     // colonne nulle -> pas d'ajustement
                ok = false;
                break;
            }
            if ( p != c )
                for ( int j = c; j <= ct_dim; ++j ) {
                    const TF t = A[ c ][ j ]; A[ c ][ j ] = A[ p ][ j ]; A[ p ][ j ] = t;
                }
            for ( int i = c + 1; i < ct_dim; ++i ) {
                const TF f = A[ i ][ c ] / A[ c ][ c ];
                for ( int j = c; j <= ct_dim; ++j )
                    A[ i ][ j ] -= f * A[ c ][ j ];
            }
        }

        if ( ok ) {
            auto fit = Vector<TF,ct_dim>::zeros();
            for ( int i = ct_dim - 1; i >= 0; --i ) {
                TF s = A[ i ][ ct_dim ];
                for ( int j = i + 1; j < ct_dim; ++j )
                    s -= A[ i ][ j ] * fit[ j ];
                fit[ i ] = s / A[ i ][ i ];
            }

            TF rmin = 0, rmax = 0;
            for ( SI k = b; k < e; ++k ) {
                TF r = TF( w( k ) );
                for ( int d = 0; d < ct_dim; ++d )
                    r -= fit[ d ] * TF( pos( k, d ) );
                if ( k == b ) { rmin = r; rmax = r; }
                else { rmin = r < rmin ? r : rmin; rmax = r > rmax ? r : rmax; }
            }

            // le resserrement que le HASARD donne deja a `d + 1` parametres sur `m` points : sans
            // cette correction un noeud de poids purement aleatoires retiendrait l'affine une fois
            // sur trois. Voir `AaBsp.py::_weight_majorant`.
            const TF u = TF( 1 ) - TF( ct_dim ) / TF( m - 1 );
            const TF by_chance = sycl::sqrt( u > 0 ? u : TF( 0 ) );
            if ( rmax - rmin < TF( 0.85 ) * by_chance * spread )
                a = fit;
        }
    }

    TF bb = 0, amax = 0;
    for ( SI k = b; k < e; ++k ) {
        TF ay = 0;
        for ( int d = 0; d < ct_dim; ++d )
            ay += a[ d ] * TF( pos( k, d ) );
        const TF v = TF( w( k ) ) - ay;
        if ( k == b ) bb = v; else bb = v > bb ? v : bb;
        amax = sycl::fabs( ay ) > amax ? sycl::fabs( ay ) : amax;
    }

    for ( int d = 0; d < ct_dim; ++d )
        wa_out( d ) = a[ d ];

    // une MARGE d'arrondi sur la constante, et sur elle seule -- voir `_weight_majorant` : `b` est
    // le seul terme que l'hote et le kernel calculeraient differemment, et un `b` arrondi vers le
    // bas cesserait de majorer.
    wb_out = bb + TF( 1e-6 ) * ( sycl::fabs( bb ) + spread + amax );
}


// Le corps du niveau, pour le noeud dont la tranche est `[ beg_in, end_in )`.
//
// `mid_out` dit ou couper : le fils gauche recoit `[ beg, mid )` et le droit `[ mid, end )`. Un
// noeud qui n'a plus rien a couper rend `mid = end`, donc passe tout a gauche -- c'est la
// PROPAGATION de `AaBsp.py::_build`, ce qui garde la partition de `[ 0, n )` d'un niveau au
// suivant, donc l'ecriture disjointe.
void bsp_build_level( const auto &src, auto &&dst, auto &&perm,
                      const auto &beg_in, const auto &end_in,
                      auto &&lo_out, auto &&hi_out, auto &&wa_out, auto &&wb_out, auto &&mid_out,
                      SI leaf_size ) {
    // la dimension est un compte COMPILE-TIME (`nb_dims : CtShapeVar`), et c'est LUI qui la porte :
    // la forme d'un tenseur, elle, traverse en entiers d'execution. C'est la raison pour laquelle
    // cette fonction prend les nuages entiers et non leurs membres.
    constexpr int ct_dim = CT_VALUE( src.nb_dims );
    using TF = typename DECAYED_TYPE_OF( src.positions )::TF;

    const auto &pos_in = src.positions;
    const auto &w_in   = src.weights;
    const auto &ord_in = src.order;
    auto &&pos_out = dst.positions;
    auto &&w_out   = dst.weights;
    auto &&ord_out = dst.order;

    const SI b = SI( beg_in );
    const SI e = SI( end_in );

    // un emplacement VIDE : le fils droit d'un noeud qui a tout passe a gauche. Il n'a rien a lire
    // ni a ecrire dans le nuage, mais ses sorties par noeud sont a lui et personne d'autre ne les
    // ecrira -- un tampon de sortie n'est pas remis a zero.
    mid_out = e;
    if ( e <= b ) {
        for ( int d = 0; d < ct_dim; ++d ) {
            lo_out( d ) = 0;
            hi_out( d ) = 0;
        }
        if constexpr ( CT_VALUE( wa_out.is_valid() ) ) {
            for ( int d = 0; d < ct_dim; ++d )
                wa_out( d ) = 0;
            wb_out = 0;
        }
        return;
    }

    // ---- la boite du sous-arbre
    auto lo = Vector<TF,ct_dim>::with_func( [&]( PI d ) { return TF( pos_in( b, d ) ); } );
    auto hi = lo;
    for ( SI k = b + 1; k < e; ++k )
        for ( int d = 0; d < ct_dim; ++d ) {
            const TF v = TF( pos_in( k, d ) );
            lo[ d ] = v < lo[ d ] ? v : lo[ d ];
            hi[ d ] = v > hi[ d ] ? v : hi[ d ];
        }
    for ( int d = 0; d < ct_dim; ++d ) {
        lo_out( d ) = lo[ d ];
        hi_out( d ) = hi[ d ];
    }

    // ---- le majorant des poids. Pas de poids -> les deux tenseurs sont des `NoneTensor` et tout
    // ce bloc disparait a la COMPILATION, comme dans `AaBsp.cxx`.
    if constexpr ( CT_VALUE( wa_out.is_valid() ) )
        bsp_weight_majorant<ct_dim>( pos_in, w_in, b, e, wa_out, wb_out );

    // ---- couper, ou propager
    int ax = 0;
    for ( int d = 1; d < ct_dim; ++d )
        if ( hi[ d ] - lo[ d ] > hi[ ax ] - lo[ ax ] )
            ax = d;

    for ( SI k = b; k < e; ++k )
        perm( k ) = k;

    SI mid = e;
    // `hi[ ax ] <= lo[ ax ]` : tous les germes au meme endroit, aucune coupe ne les separerait.
    if ( e - b > leaf_size && hi[ ax ] > lo[ ax ] ) {
        // la MEDIANE, pas le milieu de la boite : c'est ce qui borne la profondeur par
        // `log2( n / leaf_size )` quelle que soit la distribution.
        mid = b + ( e - b ) / 2;
        bsp_select( perm, pos_in, b, e, mid, ax );
    }
    mid_out = mid;

    // ---- le nuage permute. Une feuille (ou un noeud qui propage) recopie sa tranche telle
    // quelle : `perm` y est reste l'identite, donc c'est le meme code.
    for ( SI j = b; j < e; ++j ) {
        const SI s = SI( perm( j ) );
        ord_out( j ) = ord_in( s );
        for ( int d = 0; d < ct_dim; ++d )
            pos_out( j, d ) = pos_in( s, d );
        if constexpr ( CT_VALUE( w_in.is_valid() ) )
            w_out( j ) = w_in( s );
    }
}

} // namespace sdot

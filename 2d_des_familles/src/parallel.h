#pragma once

#include "common.h"
#include <thread>
#include <vector>
#include <functional>

#ifdef __linux__
#include <pthread.h>
#include <sched.h>
#endif

namespace pd2d {

/// Comment le travail est DECOUPE entre les threads. Deux facons, et le banc les compare parce que
/// ce n'est pas un detail : elles ne donnent pas la meme localite.
enum class Split {
    blocks,   ///< thread `t` prend `[ t*n/T, (t+1)*n/T )` -- des germes CONTIGUS dans l'ordre de
              ///< l'accelerateur, donc voisins dans l'espace. Chaque thread reste dans SA region.
    strided,  ///< thread `t` prend `t, t+T, t+2T, ...` -- tous les threads balaient la meme region
              ///< au meme instant, donc partagent les memes noeuds d'arbre... et se disputent les
              ///< memes lignes de cache. C'est ce que fait `sdot` aujourd'hui.
};

/// Un `parallel_for` sans dependance : `T` threads, chacun sa tranche, un `join`. Pas de pool
/// persistant -- le banc mesure des boucles d'une seconde, la creation de seize threads s'y perd,
/// et un pool ajouterait des barrieres dont on n'a pas besoin ici.
///
/// `pin` epingle le thread `t` sur le CPU `t`. Sur cette machine les CPU 0..7 sont les huit coeurs
/// PHYSIQUES et 8..15 leurs jumeaux SMT, donc epingler sur 0..7 donne un thread par coeur -- ce
/// qu'on veut pour mesurer une montee en charge sans que l'ordonnanceur en empile deux au meme
/// endroit.
template<class F>
void parallel_for( SI n, int nb_threads, Split split, bool pin, F &&body ) {
    if ( nb_threads <= 1 ) {
        for ( SI i = 0; i < n; ++i )
            body( i, 0 );
        return;
    }

    std::vector<std::thread> ths;
    ths.reserve( nb_threads );
    for ( int t = 0; t < nb_threads; ++t ) {
        ths.emplace_back( [ &, t ]() {
            if ( pin ) {
#ifdef __linux__
                cpu_set_t set;
                CPU_ZERO( &set );
                CPU_SET( t, &set );
                pthread_setaffinity_np( pthread_self(), sizeof( set ), &set );
#endif
            }
            if ( split == Split::blocks ) {
                const SI b = SI( ( long long ) t * n / nb_threads );
                const SI e = SI( ( long long ) ( t + 1 ) * n / nb_threads );
                for ( SI i = b; i < e; ++i )
                    body( i, t );
            } else {
                for ( SI i = t; i < n; i += nb_threads )
                    body( i, t );
            }
        } );
    }
    for ( auto &th : ths )
        th.join();
}

} // namespace pd2d

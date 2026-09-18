#pragma once

// =====================================================================================
// LES FILS. Un `parallel_for` sans dependance : `T` threads, chacun une tranche CONTIGUE, un
// `join`. Pas de pool persistant -- on mesure des boucles d'une seconde ou plus, la creation de
// seize threads s'y perd.
//
// La tranche est contigue dans l'ordre de l'arbre, donc chaque thread reste dans SA region de
// l'espace : c'est le decoupage qui a gagne dans le banc (`blocks` contre `strided`).
// =====================================================================================

#include "util/common.h"
#include <thread>
#include <vector>

#ifdef __linux__
#include <pthread.h>
#include <sched.h>
#endif

namespace sf {

/// Combien de fils, et s'ils sont epingles. `pin` met le thread `t` sur le CPU `t` : sur une
/// machine a SMT les CPU `0..coeurs-1` sont les coeurs physiques, donc `--threads 8` epingle donne
/// un thread par coeur, ce qu'on veut pour mesurer une montee en charge.
struct Parallel {
    int  threads = 1;
    bool pin     = true;
};

/// `body( i, t )` pour `i` dans `[ 0, n )`, `t` etant le numero du thread -- pour accumuler par
/// thread sans partage.
template<class F>
void parallel_for( SI n, const Parallel &par, F &&body ) {
    const int T = par.threads;
    if ( T <= 1 ) {
        for ( SI i = 0; i < n; ++i )
            body( i, 0 );
        return;
    }

    std::vector<std::thread> ths;
    ths.reserve( T );
    for ( int t = 0; t < T; ++t ) {
        ths.emplace_back( [ &, t ]() {
            if ( par.pin ) {
#ifdef __linux__
                cpu_set_t set;
                CPU_ZERO( &set );
                CPU_SET( t, &set );
                pthread_setaffinity_np( pthread_self(), sizeof( set ), &set );
#endif
            }
            const SI b = SI( ( long long ) t * n / T );
            const SI e = SI( ( long long ) ( t + 1 ) * n / T );
            for ( SI i = b; i < e; ++i )
                body( i, t );
        } );
    }
    for ( auto &th : ths )
        th.join();
}

/// une case par thread, sur sa propre ligne de cache : huit threads qui incrementent huit
/// `double` voisins passeraient leur temps a s'invalider mutuellement.
template<class T>
struct alignas( 64 ) ParThread {
    T v{};
};

} // namespace sf

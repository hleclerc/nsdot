#pragma once

#include <loom/support/kernels/CpuQueue.h> // IWYU pragma: keep -- la queue par défaut

namespace sdot {

/// La queue sur laquelle tourne un kernel : le contexte d'exécution d'un `driver.call`.
///
/// Le choix du device est un TYPEDEF, pas un test runtime : la zone mémoire dans laquelle vit un
/// pointeur fait partie de son type (voir `Ptr.h`), donc c'est le device qui décide du type des
/// vues que le kernel manipule. Le source généré inclut l'en-tête de SA queue et fixe `SDOT_QUEUE`
/// avant d'inclure celui-ci (voir `Device.cpp_queue_include` / `cpp_queue_type` côté python) ;
/// par défaut, le CPU -- ce qu'un source compilé à la main sans rien définir obtient.
///
/// `run_parallel` accepte cette queue seule, ou une liste de queues quand il y a un contexte à
/// choisir (il prend alors le moins coûteux, transferts compris).
#ifndef SDOT_QUEUE
#   define SDOT_QUEUE CpuQueue
#endif

using Queue = SDOT_QUEUE;

} // namespace sdot

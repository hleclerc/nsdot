#pragma once

#include "support/kernels/CudaQueue.h" // IWYU pragma: keep -- SDOT_QUEUE peut la désigner
#include "support/kernels/CpuQueue.h" // IWYU pragma: keep -- idem

namespace sdot {

/// La queue sur laquelle tourne un kernel : le contexte d'exécution d'un `driver.call`.
///
/// Le choix du device est un TYPEDEF, pas un test runtime : la zone mémoire dans laquelle vit un
/// pointeur fait partie de son type (voir `Ptr.h`), donc c'est le device qui décide du type des
/// vues que le kernel manipule. Le source généré fixe `SDOT_QUEUE` (voir `Device.cpp_queue_type`
/// côté python) ; par défaut, le CPU.
///
/// `run_parallel` accepte cette queue seule, ou une liste de queues quand il y a un contexte à
/// choisir (il prend alors le moins coûteux, transferts compris).
#ifndef SDOT_QUEUE
#   define SDOT_QUEUE CpuQueue
#endif

using Queue = SDOT_QUEUE;

} // namespace sdot

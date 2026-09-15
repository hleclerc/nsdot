 1. Pourquoi jax.devices()[0].memory_stats() affiche 2.76 GiB alors que nvidia-smi affiche 4096 MiB (4 GiB) ?
Explications :
Différence entre mémoire totale et mémoire disponible pour JAX :
nvidia-smi affiche la mémoire totale du GPU (4 GiB).
jax.devices()[0].memory_stats() affiche la mémoire disponible pour JAX, qui peut être inférieure à la mémoire totale pour plusieurs raisons :Mémoire réservée par d'autres processus (ex: CUDA, autres programmes).
Mémoire réservée par JAX pour ses propres besoins (ex: cache, buffers internes).
Fragmentation de la mémoire : JAX peut ne pas pouvoir utiliser toute la mémoire libre en continu.


Pourquoi 2.76 GiB ?
JAX ne peut pas utiliser toute la mémoire libre en raison de :Allocation par blocs : JAX alloue la mémoire par blocs de taille fixe (ex: 256 MiB, 512 MiB).
Mémoire réservée pour le système : CUDA et le driver NVIDIA réservent une partie de la mémoire.
Limites internes de JAX : JAX peut limiter volontairement la mémoire utilisable pour éviter les Out of Memory.


Comment obtenir la mémoire réellement disponible ?
Utilisez nvidia-smi pour voir la mémoire totale et utilisée :bash
Copier

nvidia-smi





Utilisez jax.devices()[0].memory_stats() pour voir la mémoire disponible pour JAX.
Différence = Mémoire utilisée par d'autres processus + fragmentation.


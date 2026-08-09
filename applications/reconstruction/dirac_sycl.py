"""Référence SYCL (CPU pour l'instant) du coût + gradient du modèle DIRACS (`models.DiracModel`),
en UN SEUL `driver.call` fwd-only -- sans `bwd_code`, donc sans passer par l'autodiff Jax : le
gradient est écrit directement par le kernel, formule fermée `(point - barycentre) * direction`,
au lieu d'un `jax.grad` à travers `_pure_jax_cost1d.cost_1d_ot` (le chemin par défaut,
`Image.try_update_otplan1d`) ou du couple fwd/bwd `OtPlan1d.cxx` (le chemin C++ général).

Motivation : comparer la vitesse d'un noyau qui fusionne les DEUX passes (coût ET gradient, la
même formule que `OtPlan1d.cxx::sweep_outputs_bwd`'s barycentre-recompute branch, mais calculé
UNE FOIS au lieu de deux) contre le pipeline pur Jax actuellement utilisé par
`Reconstruction.diracs`. Volontairement plus simple que `OtPlan1d` : un seul work-item par angle
(`std::sort` + comparateur, pas le radix sort O(n) coopératif de `OtPlan1d.cxx::sort_diracs`, et
pas la marche `udp_at`/`cell_cum_mass` -- un simple `udp_start` séquentiel suffit puisqu'un seul
thread traite tout l'angle). Repousser ces optimisations à une étape ultérieure si cette
référence s'avère prometteuse.

Réutilise `ProjectedSumOfDiracs`/`Image` TELS QUELS (mêmes structs C++, mêmes méthodes
`position(i)`/`udp_start`/`udp_cont` que `OtPlan1d.cxx`) -- seule l'orchestration (tri + balayage
+ dispersion du gradient) est nouvelle.

S'est avéré assez prometteur (10-30x plus rapide que le chemin Jax mesuré sur le poumon, voir
`benchmarks/execution_speed/benchmark_fused.py`) pour être BRANCHÉ : `models.DiracModel.value_and_grad`
expose `diracs_cost_grad` avec le même contrat que `optimizers.FusedLBFGS` attend, et
`Reconstruction.diracs( backend = "sycl" )` route dessus (voir `experiments/lung_alveoli.py`).
"""
import numpy as np

from sdot import Tensor, driver
from sdot.compilation.FfiCode import FfiCodeParallel
from sdot.distributions.ProjectedSumOfDiracs import ProjectedSumOfDiracs

from .Sinogram import Sinogram


def diracs_cost_grad( points, sinogram: Sinogram ):
    """`(cost, grad)` du modèle DIRACS pour `points` (`[n,2]`, Tensor ou tableau) face à
    `sinogram`, calculés par le kernel SYCL fusionné ci-dessus -- `cost` un flottant Python,
    `grad` un tableau numpy `[n,2]` (même convention de signe que `jax.grad(model.cost)`, un pas
    de descente de gradient fait donc `points - lr * grad`).
    """
    pts = points if isinstance( points, Tensor ) else Tensor( points )

    # `src`/`dst` normalisés à masse 1 -- exactement ce que fait `OtPlan1d.__init__` avant de
    # balayer (voir sa docstring) : sans ça la marche `udp_cont` (dont les prises `w` par dirac
    # doivent épuiser exactement la masse totale de l'image) ne balaierait pas l'image en entier.
    # `src` n'a pas de poids explicites -> `normalized_version()` donnerait des poids UNIFORMES
    # `1 / n`, exactement ce que le kernel calcule lui-même (`w = TF(1)/TF(n)`) -- inutile de
    # matérialiser un tenseur de poids rien que pour ça.
    src = ProjectedSumOfDiracs( points = pts, normal = sinogram.normals_t,
                                batch_axes = [ sinogram.num_angle ] )
    dst = sinogram.batched_image().normalized_version()

    sorted_idx = Tensor[ sinogram.num_angle, src.num_dirac, dict( dtype = int ) ]()
    cost = Tensor[ sinogram.num_angle ]()
    grad = Tensor[ src.num_dirac, src.proj_dim ]()

    driver.call(
        FfiCodeParallel(
            name = "diracs_fused_cost_grad",
            includes = [ "sdot/support/atomic_add.h" ],
            # zéro AVANT la boucle parallèle sur les angles : `grad` est PARTAGÉ (les points sont
            # les mêmes à tous les angles) et accumulé par `atomic_add` -- un buffer FFI fraîchement
            # alloué n'est pas garanti démarrer à zéro (voir `ProjectedSumOfDiracs::zero_position_grad`).
            fwd_setup_code = "grad.fill_with( queue, 0 );",
            fwd_code = """
            {
                const SI n = SI( src.points.shape( 0 ) );
                auto order = sorted_idx( batch_index );
                for ( SI i = 0; i < n; ++i )
                    order( i ) = i;

                // vue tranchée à CET angle, construite UNE FOIS -- `src( batch_index )` referait
                // sinon cette résolution (normale par angle, etc.) à CHAQUE comparaison du tri
                // (O(n log n) fois) et à chaque pas de la marche `udp_cont` (O(n) fois).
                auto s = src( batch_index );

                // projection à la volée (pas de tableau [nb_angles, n] matérialisé) : la même
                // méthode que le chemin C++ général, `ProjectedSumOfDiracs::position`.
                auto proj = [&]( SI i ) { return s.position( i ); };

                // Tri par tas ITÉRATIF (pas de récursion) : `std::sort` (introsort récursif côté
                // libstdc++) fait planter la passe "loop splitter" de la compilation SSCP générique
                // d'AdaptiveCpp -- récursion infinie observée dans `fillTransitiveSplitterCallers`
                // en isolant le crash (SIGSEGV, stack overflow pendant la compilation JIT, pas à
                // l'exécution). Simple/lisible avant d'être rapide : c'est le tri radix O(n) de
                // `OtPlan1d.cxx::sort_diracs` qui vise la vitesse, pas cette référence.
                auto sift_down = [&]( SI start, SI end ) {
                    SI root = start;
                    while ( 2 * root + 1 < end ) {
                        SI child = 2 * root + 1;
                        if ( child + 1 < end && proj( order( child ) ) < proj( order( child + 1 ) ) )
                            ++child;
                        if ( proj( order( root ) ) < proj( order( child ) ) ) {
                            const SI tmp = order( root );
                            order( root ) = order( child );
                            order( child ) = tmp;
                            root = child;
                        } else
                            break;
                    }
                };
                for ( SI start = n / 2 - 1; start >= 0; --start )
                    sift_down( start, n );
                for ( SI end = n - 1; end > 0; --end ) {
                    const SI tmp = order( 0 );
                    order( 0 ) = order( end );
                    order( end ) = tmp;
                    sift_down( 0, end );
                }

                dst( batch_index ).with_defaults( [&]( auto &&img ) {
                    using TF = DECAYED_TYPE_OF( img.values )::TF;
                    const TF w = TF( 1 ) / TF( n );

                    // un seul work-item pour tout l'angle -> pas besoin de `udp_at`/`cell_cum_mass`
                    // (la marche part du tout début, comme `udp_at( cell_cum_mass, 0 )` le ferait).
                    auto udp = img.udp_start();
                    TF local_cost = 0;
                    for ( SI k = 0; k < n; ++k ) {
                        const SI di = order( k );
                        const TF p = proj( di );

                        // coût ET barycentre (premier moment) EN UNE SEULE marche -- pas de
                        // buffer `barycenters` [nb_angles, n], pas de second passage bwd.
                        TF moment = 0;
                        img.udp_cont( udp, w, [&]( auto &&item ) {
                            local_cost += item.w2_dist( p );
                            moment += item.first_moment();
                        } );
                        const TF b = moment / w;

                        // d cost / d position(i) = 2 w (p - b) ; d position / d point = normal
                        // (voir `ProjectedSumOfDiracs::add_position_grad`) -- même formule, mais
                        // écrite directement dans `grad` (sortie brute) plutôt que via l'indirection
                        // `grad_src` que le protocole de différentiation Jax construirait pour un bwd.
                        const TF grad_s = TF( 2 ) * w * ( p - b );
                        atomic_add( grad( num_dirac = di, proj_dim = 0 ).ref(),
                                    TF( grad_s * TF( s.normal( proj_dim = 0 ) ) ) );
                        atomic_add( grad( num_dirac = di, proj_dim = 1 ).ref(),
                                    TF( grad_s * TF( s.normal( proj_dim = 1 ) ) ) );
                    }
                    cost( batch_index ) = local_cost;
                } );
            }
            """,
        ),
        output_attributes = [ "cost", "grad", "sorted_idx" ],
        scratch_attributes = [ "sorted_idx" ],
        has_dynamic_capacity = False,
        src = src,
        dst = dst,
        sorted_idx = sorted_idx,
        cost = cost,
        grad = grad,
    )

    return float( np.asarray( cost.value ).sum() ), np.asarray( grad.value )

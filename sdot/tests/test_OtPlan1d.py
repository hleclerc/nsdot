import numpy

from sdot import Image, OtPlan1d, SumOfDiracs1d, driver
from sdot.devices.Cpu import Cpu
from sdot.testing import test, check_grad

# driver.ftype = "FP64"


if test( "basic" ):
    src = SumOfDiracs1d( positions = [ 0, 1 ] )
    dst = Image( values = [ 1, 0, 1 ] )

    otp = OtPlan1d( src, dst, with_barycenters = True )

    info( otp.cost )
    info( otp.barycenters )

if test( "cost_uniform" ):
    # Un seul dirac (masse 1) face à une densité uniforme sur [0,1] (masse 1) : le coût est
    # exactement Integral_0^1 (x - 0.5)^2 dx = 1/12, et le barycentre de la tranche cible est 0.5.
    # Ce cas franchit aussi la garde de bornes `udp_cont` (une seule cellule, boucle non entrée).
    otp = OtPlan1d( SumOfDiracs1d( positions = [ 0.5 ] ), Image( values = [ 1 ] ), with_barycenters = True )

    assert abs( float( otp.cost ) - 1 / 12 ) < 1e-6
    assert abs( float( otp.barycenters.sum() ) - 0.5 ) < 1e-6

if test( "cost_two_cells" ):
    # Densité uniforme 0.5 sur [0,2] (masse 1), un dirac en 1.0 : le prélèvement traverse DEUX
    # cellules (la boucle `while` de `udp_cont` s'exécute une fois), coût attendu 1/3.
    otp = OtPlan1d( SumOfDiracs1d( positions = [ 1.0 ] ), Image( values = [ 1, 1 ] ) )

    assert abs( float( otp.cost ) - 1 / 3 ) < 1e-6

if test( "grad_cost" ):
    # Dérivée de `cost` par rapport aux positions des diracs (via `update_outputs_bwd`). Positions
    # bien séparées pour que la perturbation de `check_grad` ne change ni l'ordre trié ni
    # l'assignation des tranches ; le coût est alors lisse et l'adjoint 2 w_i ( p_i - b_i ) exact.
    positions = driver.array( [ 0.2, 0.5, 0.9 ] )

    check_grad( lambda p: OtPlan1d( SumOfDiracs1d( positions = p ), Image( values = [ 1, 0, 1 ] ) ).cost, positions )

if test( "grad_values" ):
    # Dérivée de `cost` par rapport aux valeurs de l'image : terme direct Integral (x-p)^2 + terme
    # de bord -Phi_k. Valeurs strictement positives et bords de tranches (W = 1/3, 2/3) intérieurs à
    # la cellule centrale -> CDF C1 en ces points, donc adjoint exact.
    values = driver.array( [ 1.0, 3.0, 1.0 ] )
    info( values )

    check_grad( lambda v: OtPlan1d( SumOfDiracs1d( positions = [ 0.2, 0.5, 0.9 ] ), Image( values = v ) ).cost, values )

if test( "group_size_cooperative" ):
    # Force `local_size > 1` on CPU (never the perf path there -- see `Cpu.group_size`'s docstring --
    # but the only place to cheaply exercise the cooperative code against a `local_size == 1` case it
    # would trivially degenerate around). The SORT stays bit-identical regardless of `local_size` (each
    # pass is stable w.r.t. its own input order, chunk-by-chunk, ascending index -- ties still resolve
    # by original index, exactly like the sequential algorithm). The SWEEP no longer does, by design:
    # `Image::udp_at`'s jump-start reads a `cell_cum_mass`/chunk-weight-prefix built via chunked
    # (order-dependent) floating-point reduction, so results match only within numeric tolerance, not
    # bit-for-bit -- see [[group-cooperative-sort]].
    positions = [ 0.9, 0.1, 0.5, 0.3, 0.7, 0.05, 0.95, 0.42, 0.63, 0.18 ]
    values = [ 1, 2, 0, 3, 1 ]

    otp_ref = OtPlan1d( SumOfDiracs1d( positions = positions ), Image( values = values ), with_barycenters = True )
    cost_ref = float( otp_ref.cost )
    bary_ref = numpy.asarray( otp_ref.barycenters )

    orig_group_size = Cpu.group_size
    Cpu.group_size = lambda self, **_: 4
    try:
        otp = OtPlan1d( SumOfDiracs1d( positions = positions ), Image( values = values ), with_barycenters = True )
        assert abs( float( otp.cost ) - cost_ref ) < 1e-9 * max( 1.0, abs( cost_ref ) )
        assert numpy.allclose( numpy.asarray( otp.barycenters ), bary_ref, rtol = 1e-9, atol = 1e-9 )

        weights = driver.array( [ 1.0 ] * len( positions ) )
        check_grad( lambda w: OtPlan1d( SumOfDiracs1d( positions = positions, weights = w ), Image( values = values ) ).cost, weights )
    finally:
        Cpu.group_size = orig_group_size

if test( "zero_density_cells" ):
    # Cellules de densité nulle en tête ET en queue -- vise directement le cas non trivial de
    # `Image::udp_at` : une limite tombant EXACTEMENT sur une masse cumulée nulle doit atterrir sur la
    # PREMIÈRE cellule qui partage cette valeur (règle du plus petit `c`), pas la dernière -- sinon
    # `udp_start()`'s comportement (jamais d'avance anticipée sur une cellule de tête nulle) ne serait
    # pas reproduit, et une cellule nulle de fin ne recevrait jamais sa pièce (perdant sa contribution
    # directe à `grad_values`, non nulle même à densité nulle -- voir `second_moment_about`).
    positions = [ 0.1, 0.3, 0.5, 0.7, 0.9 ]

    for values in ( [ 0, 0, 1, 2, 1 ], [ 1, 2, 1, 0, 0 ] ):
        otp_ref = OtPlan1d( SumOfDiracs1d( positions = positions ), Image( values = values ) )
        cost_ref = float( otp_ref.cost )

        orig_group_size = Cpu.group_size
        Cpu.group_size = lambda self, **_: 4
        try:
            otp = OtPlan1d( SumOfDiracs1d( positions = positions ), Image( values = values ) )
            assert abs( float( otp.cost ) - cost_ref ) < 1e-9 * max( 1.0, abs( cost_ref ) )

            weights = driver.array( [ 1.0 ] * len( positions ) )
            check_grad( lambda w: OtPlan1d( SumOfDiracs1d( positions = positions, weights = w ), Image( values = values ) ).cost, weights )
        finally:
            Cpu.group_size = orig_group_size

if test( "single_cell_target" ):
    # `nb_cells (1) < group_size (4)` -- exercise `build_cell_cum_mass`'s cell-chunking when there are
    # FEWER cells than cooperating work-items (most work-items own an empty cell sub-range).
    positions = [ 0.2, 0.5, 0.7, 0.9 ]
    values = [ 1 ]

    otp_ref = OtPlan1d( SumOfDiracs1d( positions = positions ), Image( values = values ) )
    cost_ref = float( otp_ref.cost )

    orig_group_size = Cpu.group_size
    Cpu.group_size = lambda self, **_: 4
    try:
        otp = OtPlan1d( SumOfDiracs1d( positions = positions ), Image( values = values ) )
        assert abs( float( otp.cost ) - cost_ref ) < 1e-9 * max( 1.0, abs( cost_ref ) )
    finally:
        Cpu.group_size = orig_group_size

if test( "more_threads_than_diracs" ):
    # `local_size (8) > nb (2)` -- most work-items own an EMPTY dirac chunk (`lo == hi`): exercises
    # `chunked_weight_prefix`/the sweep's per-work-item loop bounds when several chunks never execute.
    positions = [ 0.3, 0.7 ]
    values = [ 1, 2, 1 ]

    otp_ref = OtPlan1d( SumOfDiracs1d( positions = positions ), Image( values = values ) )
    cost_ref = float( otp_ref.cost )

    orig_group_size = Cpu.group_size
    Cpu.group_size = lambda self, **_: 8
    try:
        otp = OtPlan1d( SumOfDiracs1d( positions = positions ), Image( values = values ) )
        assert abs( float( otp.cost ) - cost_ref ) < 1e-9 * max( 1.0, abs( cost_ref ) )

        weights = driver.array( [ 1.0, 1.0 ] )
        check_grad( lambda w: OtPlan1d( SumOfDiracs1d( positions = positions, weights = w ), Image( values = values ) ).cost, weights )
    finally:
        Cpu.group_size = orig_group_size

if test( "boundary_straddling_cell_grad" ):
    # 6 diracs uniformément répartis, poids uniformes -> à `group_size = 3` (chunks de 2), les limites
    # de chunk tombent en masse-cible 1/3 et 2/3 -- STRICTEMENT À L'INTÉRIEUR des deux cellules de masse
    # égale (bords cumulés 0, 1/2, 1), pas sur un bord. Vise directement l'ajout ATOMIQUE de
    # `update_outputs_bwd` sur `grad_values` : sans lui, la cellule scindée entre deux work-items
    # concurrents perdrait une des deux contributions (écriture non protégée). Note : AdaptiveCpp exécute
    # les work-items coopératifs du CPU via Boost.Fiber (coopératif, pas préemptif) -- ce test valide
    # l'ARITHMÉTIQUE du chemin phase-1/phase-2/atomic_add (via différence finie), pas la détection de la
    # course elle-même (qui demanderait un thread sanitizer ou une exécution GPU sous stress).
    positions = [ 0.1, 0.2, 0.3, 0.7, 0.8, 0.9 ]
    values = driver.array( [ 1.0, 1.0 ] )

    check_grad( lambda v: OtPlan1d( SumOfDiracs1d( positions = positions ), Image( values = v ) ).cost, values )

    orig_group_size = Cpu.group_size
    Cpu.group_size = lambda self, **_: 3
    try:
        check_grad( lambda v: OtPlan1d( SumOfDiracs1d( positions = positions ), Image( values = v ) ).cost, values )
    finally:
        Cpu.group_size = orig_group_size

if test( "grad_weights" ):
    # Dérivée de `cost` par rapport aux poids des diracs : somme suffixe des sauts de potentiel Phi.
    # Poids positifs ; les bords restent intérieurs à une cellule (coût lisse). La normalisation
    # (Python, dérivée par le framework) est traversée de bout en bout par `check_grad`.
    weights = driver.array( [ 1.0, 1.0, 2.0 ] )

    check_grad( lambda w: OtPlan1d( SumOfDiracs1d( positions = [ 0.2, 0.5, 0.9 ], weights = w ), Image( values = [ 1, 3, 1 ] ) ).cost, weights )

if test( "joint_position_and_weight_value_grad" ):
    # Différencie positions ET poids ET valeurs de l'image EN MÊME TEMPS (barycentres NON stockés,
    # le défaut) -- le seul cas où `update_outputs_bwd`'s position-grad block (recompute-b_i path)
    # ET son weights/values-grad block s'exécutent tous les deux dans le MÊME appel, exerçant donc
    # le tri hissé (hoisted `sort_diracs`) partagé entre les deux.
    positions = driver.array( [ 0.2, 0.5, 0.9 ] )
    weights   = driver.array( [ 1.0, 1.0, 2.0 ] )
    values    = driver.array( [ 1.0, 3.0, 1.0 ] )

    check_grad( lambda p, w, v: OtPlan1d( SumOfDiracs1d( positions = p, weights = w ), Image( values = v ) ).cost,
                positions, weights, values )

    orig_group_size = Cpu.group_size
    Cpu.group_size = lambda self, **_: 4
    try:
        check_grad( lambda p, w, v: OtPlan1d( SumOfDiracs1d( positions = p, weights = w ), Image( values = v ) ).cost,
                    positions, weights, values )
    finally:
        Cpu.group_size = orig_group_size

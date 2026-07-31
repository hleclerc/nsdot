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
    # but the only place to cheaply exercise the cooperative chunked-histogram-scatter `sort_diracs`
    # against a case `local_size == 1` would trivially degenerate around). The final 4-pass sort is
    # bit-identical regardless of `local_size` (each pass is stable w.r.t. its own input order,
    # chunk-by-chunk, ascending index -- ties still resolve by original index, exactly like the
    # sequential algorithm), so the WHOLE computation (leader-only sweep, unchanged) must come out
    # bit-identical too -- not just close.
    positions = [ 0.9, 0.1, 0.5, 0.3, 0.7, 0.05, 0.95, 0.42, 0.63, 0.18 ]
    values = [ 1, 2, 0, 3, 1 ]

    otp_ref = OtPlan1d( SumOfDiracs1d( positions = positions ), Image( values = values ), with_barycenters = True )
    cost_ref = float( otp_ref.cost )
    bary_ref = numpy.asarray( otp_ref.barycenters ).tolist()

    orig_group_size = Cpu.group_size
    Cpu.group_size = lambda self, **_: 4
    try:
        otp = OtPlan1d( SumOfDiracs1d( positions = positions ), Image( values = values ), with_barycenters = True )
        assert float( otp.cost ) == cost_ref
        assert numpy.asarray( otp.barycenters ).tolist() == bary_ref

        weights = driver.array( [ 1.0 ] * len( positions ) )
        check_grad( lambda w: OtPlan1d( SumOfDiracs1d( positions = positions, weights = w ), Image( values = values ) ).cost, weights )
    finally:
        Cpu.group_size = orig_group_size

if test( "grad_weights" ):
    # Dérivée de `cost` par rapport aux poids des diracs : somme suffixe des sauts de potentiel Phi.
    # Poids positifs ; les bords restent intérieurs à une cellule (coût lisse). La normalisation
    # (Python, dérivée par le framework) est traversée de bout en bout par `check_grad`.
    weights = driver.array( [ 1.0, 1.0, 2.0 ] )

    check_grad( lambda w: OtPlan1d( SumOfDiracs1d( positions = [ 0.2, 0.5, 0.9 ], weights = w ), Image( values = [ 1, 3, 1 ] ) ).cost, weights )

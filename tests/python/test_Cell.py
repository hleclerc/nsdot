from sdot import Cell, driver, new_batch_axis
from . import test, check_grad

if test( "basic" ):
    c = Cell.make_hypercube( 2, [ 0, 0 ], [ [ 2, 0 ], [ 0, 1 ] ] )
    info( c.vertex_positions )
    info( c.measure )

if test( "batch" ):
    c = Cell.make_hypercube( 3, [ 0, 0, 0 ], [ [ 2, 0, 0 ], [ 0, 1, 0 ], [ 0, 0, 1 ] ], batch_axes = [ new_batch_axis( 2 ) ] )
    info( c.vertex_positions )
    info( c.measure )


if test( "grad_hypercube" ):
    # Dérivées des sorties d'un hypercube 2D par rapport à ses entrées `origin` et `axes`.
    # `check_grad` est agnostique : il confronte l'adjoint du driver à une différence finie, sans
    # jamais importer le framework. `f` renvoie un `Tensor` : `check_grad` en prend la vue dense
    # (le padding de capacité est retiré tout seul, sans `.raw[ :n ]`).
    origin = driver.array( [ 0.3, -0.2 ] )
    axes   = driver.array( [ [ 2.0, 0.1 ], [ -0.3, 1.0 ] ] )

    check_grad( lambda o, a: Cell.make_hypercube( 2, o, a ).vertex_positions, origin, axes )
    check_grad( lambda o, a: Cell.make_hypercube( 2, o, a ).cut_directions  , origin, axes )
    check_grad( lambda o, a: Cell.make_hypercube( 2, o, a ).cut_offsets     , origin, axes )

    # une seule entrée dérivée à la fois : l'autre n'est pas perturbée, donc son gradient arrive
    # au backward en NoneTensor ( is_valid() == faux ) et le bloc qui l'écrit est supprimé à la
    # compilation. On couvre ainsi les deux branches `! is_valid()`.
    check_grad( lambda o: Cell.make_hypercube( 2, o, axes   ).vertex_positions, origin )
    check_grad( lambda o: Cell.make_hypercube( 2, o, axes   ).cut_offsets     , origin )
    check_grad( lambda a: Cell.make_hypercube( 2, origin, a ).cut_directions  , axes   )
    check_grad( lambda a: Cell.make_hypercube( 2, origin, a ).vertex_positions, axes   )


if test( "grad_measure" ):
    # `measure` (l'aire, en 2D) par rapport à `origin`/`axes` -- passe par `measure_bwd`'s
    # shoelace adjoint, et par `vertex_positions` seul (`cut_directions`/`cut_offsets` sont hors
    # du call, voir `input_exceptions` dans `Cell.py::measure`).
    origin = driver.array( [ 0.3, -0.2 ] )
    axes   = driver.array( [ [ 2.0, 0.1 ], [ -0.3, 1.0 ] ] )

    check_grad( lambda o, a: Cell.make_hypercube( 2, o, a ).measure, origin, axes )

    # une seule entrée dérivée : couvre la branche `! grad_for_cell.vertex_positions.is_valid()`
    # côté `init_as_hypercube_bwd` (l'autre entrée n'est pas perturbée).
    check_grad( lambda o: Cell.make_hypercube( 2, o, axes   ).measure, origin )
    check_grad( lambda a: Cell.make_hypercube( 2, origin, a ).measure, axes   )

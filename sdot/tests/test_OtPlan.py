import numpy

from loom.testing import test, experiment, Param

from sdot import Image, OtPlan, PowerDiagram, SumOfDiracs, SumOfGaussians, Visualizer, box_half_spaces, write_convergence_html


# le domaine déborde `[ 0, 1 ]^d`, où vivent diracs et gaussiennes : la masse gaussienne hors du
# domaine est PERDUE, et le solveur remet les masses cibles à l'échelle de ce que le domaine contient
# ( `otplan/Solve.h` ) -- le plan est celui du transport vers la densité restreinte au domaine.
# `atol` des tests : la masse cible d'un dirac est `1 / n`, et Newton converge au bruit du noyau.
_BOX = ( [ -0.5, -0.5 ], [ 1.5, 1.5 ] )


def _overlapping_target( d, nb_gaussians, seed, spread = 0.12 ):
    """Quelques gaussiennes PROCHES les unes des autres plutôt que des bosses séparées : leur
    densité combinée ne s'annule nulle part sur la zone utile -- le cas DOUX."""
    rng = numpy.random.default_rng( seed )
    center = numpy.full( d, 0.5 )
    pos = center + rng.uniform( -spread, spread, size = ( nb_gaussians, d ) )
    sigmas = rng.uniform( 0.18, 0.24, nb_gaussians )
    weights = rng.uniform( 0.6, 1.4, nb_gaussians )
    return SumOfGaussians( pos, sigmas, weights = weights )


def _scattered_target( d, nb_gaussians, seed, sigma = 0.13 ):
    """Des bosses ÉTROITES et SÉPARÉES sur tout `[ 0.3, 0.7 ]^d` -- le cas DUR : un dirac tiré
    uniformément a de bonnes chances de tomber dans un DÉSERT de densité, entre deux bosses. C'est
    là que l'amortissement de KMT ( le plancher de masse ) travaille."""
    rng = numpy.random.default_rng( seed )
    pos = rng.uniform( 0.3, 0.7, size = ( nb_gaussians, d ) )
    sigmas = numpy.full( nb_gaussians, sigma )
    weights = rng.uniform( 0.7, 1.3, nb_gaussians )
    return SumOfGaussians( pos, sigmas, weights = weights )


def _target_masses( plan ):
    return numpy.asarray( plan.target_masses ).reshape( -1 )


if test( "newton_matches_the_target_masses" ):
    # le test de base : les masses des CELLULES, une fois l'ajustement fini, doivent retomber sur
    # les masses des DIRACS -- c'est la seule chose que `OtPlan` promet. UNE gaussienne, large et
    # bien centrée sur le nuage de diracs : le cas le plus simple, sans aucun désert de densité.
    rng = numpy.random.default_rng( 3 )
    pos = rng.uniform( 0.15, 0.85, size = ( 20, 2 ) )
    src = SumOfDiracs( pos )
    dst = SumOfGaussians( numpy.array( [ [ 0.5, 0.5 ] ] ), numpy.array( [ 0.15 ] ),
                          weights = numpy.array( [ 1.0 ] ) )

    plan = OtPlan( src, dst, boundaries = box_half_spaces( *_BOX ), max_iter = 50, mass_tol = 1e-12 )

    assert plan.converged, plan.stats
    got = numpy.asarray( plan.cell_masses ).reshape( -1 )
    assert numpy.allclose( got, _target_masses( plan ), atol = 1e-10 ), numpy.abs( got - _target_masses( plan ) ).max()
    # la masse cible est celle des diracs, à la masse perdue hors du domaine près ( négligeable ici )
    assert abs( plan.stats[ "masse_domaine" ] - 1 ) < 1e-6


if test( "starting_from_nonzero_weights_still_converges" ):
    # le point de départ ne devrait être qu'une question de vitesse, pas de résultat -- ici on
    # part déjà PRÈS de la solution ( `weights0` tiré au hasard mais petit ) plutôt que de zéro, et
    # le solveur garde ces poids-là ( `depart = "weights0"` ) : ils ne vident aucune cellule.
    rng = numpy.random.default_rng( 2 )
    pos = rng.uniform( 0.1, 0.9, size = ( 18, 2 ) )
    src = SumOfDiracs( pos )
    dst = _overlapping_target( 2, 2, seed = 3 )
    w0 = rng.uniform( -0.003, 0.003, 18 )

    plan = OtPlan( src, dst, boundaries = box_half_spaces( *_BOX ), weights0 = w0, max_iter = 50, mass_tol = 1e-12 )

    assert plan.converged and plan.stats[ "depart" ] == "weights0", plan.stats
    got = numpy.asarray( plan.cell_masses ).reshape( -1 )
    assert numpy.allclose( got, _target_masses( plan ), atol = 1e-10 ), numpy.abs( got - _target_masses( plan ) ).max()


if test( "no_cell_dies_even_with_scattered_targets" ):
    # le cas DUR ( `_scattered_target` ) : sans plancher, ce scénario vide plusieurs cellules et s'y
    # bloque. Ici on vérifie les DEUX choses que l'amortissement promet : aucune cellule ne meurt
    # EN COURS DE ROUTE ( `min_measure` reste `> 0` à CHAQUE pas de `plan.history` ), et
    # l'ajustement retombe quand même sur les masses cibles.
    rng = numpy.random.default_rng( 5 )
    pos = rng.uniform( 0.1, 0.9, size = ( 40, 2 ) )
    src = SumOfDiracs( pos )
    dst = _scattered_target( 2, 4, seed = 6 )

    plan = OtPlan( src, dst, boundaries = box_half_spaces( *_BOX ), max_iter = 200, mass_tol = 1e-12 )

    assert all( h[ "min_measure" ] > 0 for h in plan.history ), min( h[ "min_measure" ] for h in plan.history )
    assert plan.converged, plan.stats
    got = numpy.asarray( plan.cell_masses ).reshape( -1 )
    assert numpy.allclose( got, _target_masses( plan ), atol = 1e-10 ), numpy.abs( got - _target_masses( plan ) ).max()


if test( "the_hessian_rows_are_the_jacobian_of_the_measures" ):
    # `PowerDiagram.hessian_rows` contre la différence finie des mesures par rapport aux poids,
    # sur une image ( des facettes plates à densité constante ) : symétrique, lignes de somme nulle
    rng = numpy.random.default_rng( 31 )
    n = 25
    pos = rng.uniform( 0.1, 0.9, size = ( n, 2 ) )
    img = Image( values = rng.uniform( 0.2, 1, size = ( 10, 10 ) ), origin = [ 0.0, 0.0 ],
                 frame = [ [ 0.1, 0 ], [ 0, 0.1 ] ] )
    w0 = rng.uniform( -0.01, 0.01, n )
    pd = PowerDiagram( pos, w0, distribution = img, kernel_dtype = "FP64" )
    counts, ids, vals = pd.hessian_rows()
    H = numpy.zeros( ( n, n ) )
    for i in range( n ):
        for q in range( counts[ i ] ):
            if ids[ i, q ] >= 0:
                H[ i, ids[ i, q ] ] -= vals[ i, q ]
                H[ i, i ] += vals[ i, q ]
    assert numpy.abs( H - H.T ).max() < 1e-12
    assert numpy.abs( H.sum( axis = 1 ) ).max() < 1e-12
    h = 1e-6
    for j in ( 0, 8, 24 ):
        e = numpy.zeros( n ); e[ j ] = h
        pd.weights = w0 + e; mp = numpy.asarray( pd.measures ).reshape( -1 )
        pd.weights = w0 - e; mm = numpy.asarray( pd.measures ).reshape( -1 )
        assert numpy.abs( ( mp - mm ) / ( 2 * h ) - H[ :, j ] ).max() < 1e-7


if test( "the_hessian_rows_hold_in_3d_too" ):
    # en 3D la facette est une FACE, dont l'aire vient de l'accumulation de `LocalN::measure_3d`
    # ( `for_each_facet` ) : même vérification par différence finie, sans distribution ( Lebesgue )
    rng = numpy.random.default_rng( 32 )
    n = 14
    pos = rng.uniform( 0.1, 0.9, size = ( n, 3 ) )
    w0 = rng.uniform( -0.01, 0.01, n )
    pd = PowerDiagram( pos, w0, boundaries = box_half_spaces( [ 0 ] * 3, [ 1 ] * 3 ), kernel_dtype = "FP64" )
    counts, ids, vals = pd.hessian_rows()
    H = numpy.zeros( ( n, n ) )
    for i in range( n ):
        for q in range( counts[ i ] ):
            if ids[ i, q ] >= 0:
                H[ i, ids[ i, q ] ] -= vals[ i, q ]
                H[ i, i ] += vals[ i, q ]
    assert numpy.abs( H - H.T ).max() < 1e-12
    h = 1e-6
    for j in ( 0, 5, 13 ):
        e = numpy.zeros( n ); e[ j ] = h
        pd.weights = w0 + e; mp = numpy.asarray( pd.measures ).reshape( -1 )
        pd.weights = w0 - e; mm = numpy.asarray( pd.measures ).reshape( -1 )
        assert numpy.abs( ( mp - mm ) / ( 2 * h ) - H[ :, j ] ).max() < 1e-7


if test( "newton_converges_quadratically_on_an_image" ):
    # sur une image, quelques pas suffisent, le résidu chute quadratiquement à la fin, et un départ
    # chaud ( les poids d'un nuage voisin ) n'en demande que deux ou trois -- ce dont vit une
    # reconstruction ( `otrec.models.ProjectedDiracModel` )
    rng = numpy.random.default_rng( 41 )
    n = 300
    pos = rng.uniform( 0.05, 0.95, size = ( n, 2 ) )
    img = Image( values = 1 + 0.5 * rng.random( ( 24, 24 ) ), origin = [ 0.0, 0.0 ],
                 frame = [ [ 1 / 24, 0 ], [ 0, 1 / 24 ] ] )
    plan = OtPlan( SumOfDiracs( pos ), img, max_iter = 60, mass_tol = 1e-10 / n )
    res = [ h[ "max_abs_residual" ] * n for h in plan.history ]
    assert plan.converged and res[ -1 ] < 1e-9 and len( res ) < 30, ( plan.stats, res[ -1 ], len( res ) )
    # les deux derniers pas : au moins un ordre de grandeur chacun ( la phase quadratique )
    assert res[ -1 ] < 0.1 * res[ -2 ] < 0.01 * res[ -3 ]
    got  = numpy.asarray( plan.cell_masses ).reshape( -1 )
    assert numpy.allclose( got, _target_masses( plan ), atol = 1e-11 )
    # un diagramme par pas : aucun recul sur ce cas doux
    assert plan.stats[ "nb_recul" ] == 0 and plan.stats[ "nb_diag" ] == len( res ), plan.stats

    warm = OtPlan( SumOfDiracs( pos + 1e-4 * rng.normal( size = pos.shape ) ), img, max_iter = 60, mass_tol = 1e-10 / n,
                   weights0 = plan.weights )
    assert warm.stats[ "depart" ] == "weights0" and len( warm.history ) <= 6, ( warm.stats, len( warm.history ) )

    # un départ chaud qui VIDE une cellule ( des poids qui n'ont plus rien à voir avec le nuage )
    # est abandonné pour le Voronoï, et on converge quand même
    bad = OtPlan( SumOfDiracs( pos ), img, max_iter = 60, mass_tol = 1e-10 / n, weights0 = rng.uniform( -1, 1, n ) )
    assert bad.converged and bad.stats[ "depart" ] == "voronoi", bad.stats


if test( "newton_starts_from_a_similarity_when_the_voronoi_has_empty_cells" ):
    # des diracs HORS du domaine ( leur cellule de Voronoï restreinte au domaine est vide ) : le
    # départ est le Voronoï du nuage ramené dans le domaine par une similitude, écrit comme
    # diagramme de puissance du nuage d'origine -- toutes les cellules nourries, et Newton converge
    rng = numpy.random.default_rng( 51 )
    n = 40
    pos = rng.uniform( -2, 3, size = ( n, 2 ) )
    img = Image( values = 1 + 0.3 * rng.random( ( 16, 16 ) ), origin = [ 0.0, 0.0 ],
                 frame = [ [ 1 / 16, 0 ], [ 0, 1 / 16 ] ] )
    voronoi = PowerDiagram( pos, numpy.zeros( n ), distribution = img, kernel_dtype = "FP64" )
    assert ( numpy.asarray( voronoi.measures ) == 0 ).any()      # le problème existe bien
    plan = OtPlan( SumOfDiracs( pos ), img, max_iter = 80, mass_tol = 1e-10 / n, keep_weights = True )
    assert plan.stats[ "depart" ] == "similitude", plan.stats
    assert plan.history[ 0 ][ "min_measure" ] > 0                # ... et le départ l'a résolu
    assert plan.converged, plan.stats
    # la similitude elle-même : ses poids sont ceux du Voronoï du nuage contracté ( à la jauge près )
    w = numpy.asarray( plan.history[ 0 ][ "weights" ] ).reshape( -1 )
    q = 0.5 + 0.8 * ( pos - ( pos.min( axis = 0 ) + pos.max( axis = 0 ) ) / 2 ) / numpy.ptp( pos, axis = 0 ).max()
    a = 0.8 / numpy.ptp( pos, axis = 0 ).max()
    ws = ( pos ** 2 ).sum( 1 ) - ( q ** 2 ).sum( 1 ) / a
    assert numpy.allclose( w - w[ 0 ], ws - ws[ 0 ] )


if test( "newton_works_in_3d" ):
    # la même promesse en 3D, sans distribution ( Lebesgue sur le cube ) : la facette est une face
    rng = numpy.random.default_rng( 61 )
    n = 200
    pos = rng.uniform( 0.05, 0.95, size = ( n, 3 ) )
    # ( `OtPlan` demande une distribution : une image constante à un pavé est la mesure de Lebesgue )
    img = Image( values = numpy.ones( ( 1, 1, 1 ) ), origin = [ 0.0, 0.0, 0.0 ], frame = numpy.eye( 3 ) )
    plan = OtPlan( SumOfDiracs( pos ), img, max_iter = 60, mass_tol = 1e-10 / n )
    assert plan.converged and len( plan.history ) < 30, ( plan.stats, len( plan.history ) )
    got = numpy.asarray( plan.cell_masses ).reshape( -1 )
    assert numpy.allclose( got, _target_masses( plan ), atol = 1e-11 )


if test( "the_limits_step_reaches_the_same_plan_with_fewer_diagrams" ):
    # `step = "limits"` ( le défaut en 2D ) : les mêmes poids que les essais de KMT ( `"trials"` ), et
    # moins de diagrammes -- sur le cas DUR, où les essais reculent ( `solvers_des_familles` README § 7 )
    rng = numpy.random.default_rng( 81 )
    pos = rng.uniform( 0.1, 0.9, size = ( 60, 2 ) )
    src = SumOfDiracs( pos )
    dst = _scattered_target( 2, 4, seed = 6 )
    # ( sans la continuation : c'est le Newton direct, et ses reculs, qu'on compare ici )
    a = OtPlan( src, dst, boundaries = box_half_spaces( *_BOX ), max_iter = 200, mass_tol = 1e-12, step = "trials", continuation = "never" )
    b = OtPlan( src, dst, boundaries = box_half_spaces( *_BOX ), max_iter = 200, mass_tol = 1e-12, step = "limits", continuation = "never" )
    assert a.converged and b.converged, ( a.stats, b.stats )
    assert numpy.allclose( numpy.asarray( a.weights ), numpy.asarray( b.weights ), atol = 1e-9 )
    assert b.stats[ "nb_diag" ] <= a.stats[ "nb_diag" ], ( a.stats[ "nb_diag" ], b.stats[ "nb_diag" ] )
    assert b.stats[ "nb_cell_lim" ] > 0 and b.stats[ "nb_recul" ] == 0, b.stats


if test( "the_continuation_solves_what_direct_newton_cannot" ):
    # des bosses ÉTROITES ( le cas dur du banc, `solvers_des_familles` README § 9 ) : des cellules sans
    # masse au départ, Newton direct STAGNE ; la continuation en largeur ( `otplan/Continuation.h` )
    # converge, et `"auto"` la déclenche toute seule sur la plus petite masse du départ
    rng = numpy.random.default_rng( 91 )
    pos = rng.uniform( 0, 1, size = ( 400, 2 ) )
    centres = numpy.array( [ [ 0.3, 0.3 ], [ 0.7, 0.35 ], [ 0.4, 0.75 ], [ 0.75, 0.7 ] ] )
    dst = SumOfGaussians( centres, 0.04 * numpy.array( [ 1, 0.7, 1.3, 1 ] ), weights = numpy.array( [ 0.35, 0.25, 0.25, 0.15 ] ) )
    box = box_half_spaces( [ 0, 0 ], [ 1, 1 ] )
    direct = OtPlan( SumOfDiracs( pos ), dst, boundaries = box, max_iter = 100, mass_rtol = 1e-6, continuation = "never" )
    assert not direct.converged, direct.stats
    plan = OtPlan( SumOfDiracs( pos ), dst, boundaries = box, max_iter = 100, mass_rtol = 1e-6 )
    assert plan.converged and plan.stats[ "nb_etapes" ] > 1, plan.stats
    got = numpy.asarray( plan.cell_masses ).reshape( -1 )
    assert numpy.allclose( got, _target_masses( plan ), rtol = 1e-5 ), numpy.abs( got / _target_masses( plan ) - 1 ).max()
    # l'historique porte la largeur de chaque pas, décroissante jusqu'à 0
    ss = [ h[ "s" ] for h in plan.history ]
    assert ss[ 0 ] > 0 and ss[ -1 ] == 0 and all( b <= a for a, b in zip( ss, ss[ 1: ] ) )

    # une IMAGE aussi ( floutée sur sa grille ) : une image presque vide, sauf deux taches
    values = numpy.full( ( 32, 32 ), 1e-6 )
    values[ 6:10, 6:10 ] = 1.0
    values[ 20:26, 18:24 ] = 0.7
    img = Image( values = values, origin = [ 0.0, 0.0 ], frame = [ [ 1 / 32, 0 ], [ 0, 1 / 32 ] ] )
    plan = OtPlan( SumOfDiracs( pos ), img, max_iter = 100, mass_rtol = 1e-6 )
    assert plan.converged and plan.stats[ "nb_etapes" ] > 1, plan.stats
    got = numpy.asarray( plan.cell_masses ).reshape( -1 )
    assert numpy.allclose( got, _target_masses( plan ), rtol = 1e-5 ), numpy.abs( got / _target_masses( plan ) - 1 ).max()


if test( "an_unbounded_domain_is_closed_by_the_hull_of_the_diracs" ):
    # des gaussiennes SANS `boundaries` : le domaine est l'enveloppe des diracs ( `hull.py`, seize
    # demi-plans qui s'appuient sur le nuage, les axes compris -> un pavé de départ et douze coupes ),
    # chaque dirac y est, le transport est celui vers la gaussienne restreinte à ce domaine
    rng = numpy.random.default_rng( 3 )
    pos = rng.uniform( 0.15, 0.85, size = ( 60, 2 ) )
    dst = SumOfGaussians( numpy.array( [ [ 0.5, 0.5 ] ] ), numpy.array( [ 0.15 ] ), weights = numpy.array( [ 1.0 ] ) )
    plan = OtPlan( SumOfDiracs( pos ), dst, max_iter = 100, mass_tol = 1e-12 )
    assert plan.converged, plan.stats
    pd = plan._pd
    assert pd.box_min.is_defined and int( pd.bnd_offsets.shape[ 0 ] ) == 12
    assert numpy.allclose( numpy.asarray( pd.box_min ), pos.min( axis = 0 ) ) and numpy.allclose( numpy.asarray( pd.box_max ), pos.max( axis = 0 ) )
    assert 0.9 < plan.stats[ "masse_domaine" ] < 1.0
    got = numpy.asarray( plan.cell_masses ).reshape( -1 )
    assert numpy.allclose( got, _target_masses( plan ), atol = 1e-10 )
    # une marge écarte le domaine, qui contient alors plus de masse
    wide = OtPlan( SumOfDiracs( pos ), dst, max_iter = 100, mass_tol = 1e-12, domain_margin = 0.2 )
    assert wide.converged and wide.stats[ "masse_domaine" ] > plan.stats[ "masse_domaine" ]


if test( "the_plain_storage_gives_the_same_plan" ):
    # `accelerator = "plain"` : les mêmes poids que l'arbre BSP, aux arrondis près ( l'accélération
    # ne change que ce que le diagramme coûte )
    rng = numpy.random.default_rng( 71 )
    n = 60
    pos = rng.uniform( 0.05, 0.95, size = ( n, 2 ) )
    img = Image( values = 1 + 0.5 * rng.random( ( 8, 8 ) ), origin = [ 0.0, 0.0 ], frame = [ [ 1 / 8, 0 ], [ 0, 1 / 8 ] ] )
    a = OtPlan( SumOfDiracs( pos ), img, max_iter = 60, mass_tol = 1e-12 )
    b = OtPlan( SumOfDiracs( pos ), img, max_iter = 60, mass_tol = 1e-12, accelerator = "plain" )
    assert a.converged and b.converged
    assert numpy.allclose( numpy.asarray( a.weights ), numpy.asarray( b.weights ), atol = 1e-10 )


# -- les moments, et ce qu'un coût de transport en tire ---------------------------------------

if test( "moments_are_the_closed_forms" ):
    # UN dirac dans le carré unité : sa cellule est le carré, dont les moments sont connus --
    # masse 1, barycentre ( 1/2, 1/2 ), `int |x|^2 = 2/3`. Et sur une image à UN pixel allumé, le
    # barycentre est le centre de ce pixel : c'est aussi ce qui vérifie l'orientation de la grille
    # ( `values[ i, j ]` <-> `origin + i frame[ 0 ] + j frame[ 1 ]` ).
    pd = PowerDiagram( numpy.array( [ [ 0.3, 0.6 ] ] ), boundaries = box_half_spaces( [ 0, 0 ], [ 1, 1 ] ),
                       kernel_dtype = "FP64" )
    mass, first, second = pd.moments
    assert abs( float( numpy.asarray( mass ).reshape( -1 )[ 0 ] ) - 1 ) < 1e-12
    assert numpy.allclose( numpy.asarray( first ).reshape( -1 ), [ 0.5, 0.5 ], atol = 1e-12 )
    assert abs( float( numpy.asarray( second ).reshape( -1 )[ 0 ] ) - 2 / 3 ) < 1e-12

    values = numpy.zeros( ( 4, 3 ) )
    values[ 3, 1 ] = 1.0
    img = Image( values = values, origin = [ 0.0, 0.0 ], frame = [ [ 0.5, 0.0 ], [ 0.0, 0.25 ] ] )
    pd = PowerDiagram( numpy.array( [ [ 0.3, 0.6 ] ] ), distribution = img, kernel_dtype = "FP64" )
    mass, first, second = pd.moments
    m = float( numpy.asarray( mass ).reshape( -1 )[ 0 ] )
    bary = numpy.asarray( first ).reshape( -1 ) / m
    assert abs( m - 1 ) < 1e-12, m                                    # normalisée
    assert numpy.allclose( bary, [ 3.5 * 0.5, 1.5 * 0.25 ], atol = 1e-12 ), bary

    # plusieurs diracs : les moments d'ordre 0 sont les mesures, et les barycentres restent dans
    # le carré, leur moyenne pondérée étant le centre de masse du domaine
    rng = numpy.random.default_rng( 11 )
    pos = rng.uniform( 0.1, 0.9, size = ( 15, 2 ) )
    pd = PowerDiagram( pos, boundaries = box_half_spaces( [ 0, 0 ], [ 1, 1 ] ), kernel_dtype = "FP64" )
    mass, first, second = pd.moments
    m = numpy.asarray( mass ).reshape( -1 )
    mx = numpy.asarray( first ).reshape( -1, 2 )
    assert numpy.allclose( m, numpy.asarray( pd.measures ).reshape( -1 ), atol = 1e-12 )
    assert numpy.allclose( mx.sum( axis = 0 ), [ 0.5, 0.5 ], atol = 1e-12 )
    assert abs( float( numpy.asarray( second ).reshape( -1 ).sum() ) - 2 / 3 ) < 1e-12


if test( "the_transport_cost_derives_by_the_envelope_theorem" ):
    # `cost_and_position_grad` : la dérivée du coût par rapport aux positions des diracs, aux
    # poids ajustés, contre la différence finie du coût lui-même ( chaque évaluation réajustant
    # ses poids, en repartant des précédents ). Sur une IMAGE : ses moments sont exacts ( ceux d'une
    # gaussienne passent par la quadrature adaptative, à `rtol` près ).
    rng = numpy.random.default_rng( 21 )
    pos = rng.uniform( 0.2, 0.8, size = ( 8, 2 ) )
    img = Image( values = 1 + 0.5 * rng.random( ( 12, 12 ) ), origin = [ 0.0, 0.0 ], frame = [ [ 1 / 12, 0 ], [ 0, 1 / 12 ] ] )

    def plan_at( p, w0 = None ):
        return OtPlan( SumOfDiracs( p ), img, weights0 = w0, max_iter = 100, mass_tol = 1e-14 )

    plan = plan_at( pos )
    cost, grad = plan.cost_and_position_grad()
    cost, grad = float( cost ), numpy.asarray( grad )
    assert cost > 0 and numpy.isfinite( grad ).all()

    # le coût est bien `W_2^2` : la même chose que `sum_i m_i |p_i - b_i|^2 + sum_i var_i` -- on
    # vérifie au moins la borne `cost >= sum_i m_i |p_i - b_i|^2`
    _, bary, m = plan.transport()
    bary, m = numpy.asarray( bary ), numpy.asarray( m ).reshape( -1 )
    assert cost >= float( ( m * ( ( pos - bary ) ** 2 ).sum( axis = 1 ) ).sum() ) - 1e-12

    h = 1e-4
    for i, c in ( ( 0, 0 ), ( 3, 1 ), ( 7, 0 ) ):
        dp = numpy.zeros_like( pos ); dp[ i, c ] = h
        fd = ( float( plan_at( pos + dp, plan.weights ).cost ) - float( plan_at( pos - dp, plan.weights ).cost ) ) / ( 2 * h )
        assert abs( fd - grad[ i, c ] ) < 1e-4 * max( 1.0, abs( fd ) ), ( i, c, fd, grad[ i, c ] )


# -- ce qu'on REGARDE ------------------------------------------------------------------------
#
#   ./run experiment test_OtPlan

def _report( p, plan, pos, stem ):
    """Commun aux deux expériences ci-dessous : la même paire ( courbe, animation ), la même
    lecture d'historique."""
    last = plan.history[ -1 ]
    print( f"  { last[ 'step' ] } pas, { plan.stats[ 'nb_diag' ] } diagrammes, { plan.stats[ 'fin' ] }"
           f", résidu max { last[ 'max_abs_residual' ]:.3e}"
           f", mesure min finale { last[ 'min_measure' ]:.3e}" )

    write_convergence_html(
        { "résidu l2":                        [ h[ "residual_l2" ] for h in plan.history ],
          "résidu max":                       [ h[ "max_abs_residual" ] for h in plan.history ],
          "mesure minimale (jamais 0)":       [ h[ "min_measure" ] for h in plan.history ] },
        p.out_dir / f"{ stem }_convergence.html",
        title = f"OtPlan 2D -- { len( pos ) } diracs" )

    idx = numpy.unique( numpy.linspace(
        0, len( plan.history ) - 1, min( 40, len( plan.history ) ) ).astype( int ) )

    viz = Visualizer( title = f"OtPlan 2D, { len( pos ) } diracs -- convergence", frame_axis = "pas" )
    for j, i in enumerate( idx ):
        if j:
            viz.new_frame( int( plan.history[ i ][ "step" ] ) )
        w = plan.history[ i ][ "weights" ]
        plan.power_diagram( w ).add_to_viz( viz )
        viz.add_points( pos, color = "#ffffff" )
    viz.write_html( p.out_dir / f"{ stem }_anim.html" )


if p := experiment( "ot 2D newton",
                    nb_points    = Param( 30, help = "nombre de diracs" ),
                    nb_gaussians = Param( 2, help = "nombre de gaussiennes de la cible" ),
                    max_iter     = Param( 100, help = "nombre de pas" ),
                    seed         = Param( 5, help = "graine du tirage" ) ):
    # ce que fait `OtPlan` : PARTIR des poids nuls ( le Voronoï -- chaque cellule prend sa part
    # purement géométrique ) et les faire GLISSER jusqu'à ce que chaque cellule pèse, contre la
    # densité cible, exactement ce que pèse son dirac. La courbe de convergence dit SI ça converge
    # et à quelle vitesse ; l'animation montre COMMENT : les PLANS glissent d'un pas à l'autre,
    # pas les germes. Cible DOUCE ( `_overlapping_target` ).
    pos = numpy.random.default_rng( p.seed ).uniform( 0.1, 0.9, size = ( p.nb_points, 2 ) )
    src = SumOfDiracs( pos )
    dst = _overlapping_target( 2, p.nb_gaussians, seed = p.seed + 1 )

    plan = OtPlan( src, dst, boundaries = box_half_spaces( *_BOX ), max_iter = p.max_iter, keep_weights = True, verbose = True )
    _report( p, plan, pos, "ot_2d_newton" )


if p := experiment( "ot 2D newton scattered",
                    nb_points    = Param( 40, help = "nombre de diracs" ),
                    nb_gaussians = Param( 4, help = "nombre de bosses, séparées et étroites" ),
                    max_iter     = Param( 200, help = "nombre de pas" ),
                    seed         = Param( 5, help = "graine du tirage" ) ):
    # le cas DUR : des bosses étroites et séparées ( `_scattered_target` ). La courbe `mesure
    # minimale` est celle qui compte ici : elle part quasi nulle ( un dirac dans un désert de
    # densité, au Voronoï ) et doit REMONTER sans jamais retoucher 0.
    pos = numpy.random.default_rng( p.seed ).uniform( 0.1, 0.9, size = ( p.nb_points, 2 ) )
    src = SumOfDiracs( pos )
    dst = _scattered_target( 2, p.nb_gaussians, seed = p.seed + 1 )

    plan = OtPlan( src, dst, boundaries = box_half_spaces( *_BOX ), max_iter = p.max_iter, keep_weights = True, verbose = True )
    _report( p, plan, pos, "ot_2d_newton_scattered" )

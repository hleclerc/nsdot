import numpy

from loom.testing import test, experiment, Param

from sdot import Image, OtPlan, PowerDiagram, SumOfDiracs, SumOfGaussians, Visualizer, box_half_spaces, write_convergence_html


# le domaine déborde LARGEMENT `[ 0, 1 ]^d`, où vivent diracs et gaussiennes : la marge (plusieurs
# fois `sigma`, voir `_target`) rend négligeable la masse gaussienne perdue hors du domaine -- sans
# ça, la somme des mesures plafonne SOUS la masse totale des diracs, un résidu qui ne peut alors
# JAMAIS s'annuler, pas parce que l'ajustement a raté (voir `OtPlan.residual`, et
# `Distribution.bounding_half_spaces` sur ce que rendre le domaine plus grand que le support
# UTILE ne coûte rien -- BEAUCOUP plus grand, ici, coûte simplement de laisser le solveur
# retomber sur ses pieds).
_BOX = ( [ -0.5, -0.5 ], [ 1.5, 1.5 ] )


def _overlapping_target( d, nb_gaussians, seed, spread = 0.12 ):
    """Quelques gaussiennes PROCHES les unes des autres plutôt que des bosses séparées : leur
    densité combinée ne s'annule nulle part sur la zone utile. `spread` (le rayon du nuage de
    centres) reste petit devant `sigma` pour que les bosses se recouvrent largement -- le cas
    DOUX, opposé de `_scattered_target`."""
    rng = numpy.random.default_rng( seed )
    center = numpy.full( d, 0.5 )
    pos = center + rng.uniform( -spread, spread, size = ( nb_gaussians, d ) )
    sigmas = rng.uniform( 0.18, 0.24, nb_gaussians )
    weights = rng.uniform( 0.6, 1.4, nb_gaussians )
    return SumOfGaussians( pos, sigmas, weights = weights )


def _scattered_target( d, nb_gaussians, seed, sigma = 0.13 ):
    """Des bosses ÉTROITES et SÉPARÉES sur tout `[ 0.3, 0.7 ]^d` -- le cas DUR : un dirac tiré
    uniformément a de bonnes chances de tomber dans un DÉSERT de densité, entre deux bosses.
    C'est ce qui pousse `_fit` à vider une cellule (voir la docstring d'`OtPlan`) plutôt qu'une
    subtilité de LBFGS : le gradient d'un poids est proportionnel à la densité le long du bord de
    SA cellule, donc quasi nul loin de tout maximum -- et c'est exactement pour ce cas-là que la
    barrière et le plancher de `_fit` existent."""
    rng = numpy.random.default_rng( seed )
    pos = rng.uniform( 0.3, 0.7, size = ( nb_gaussians, d ) )
    sigmas = numpy.full( nb_gaussians, sigma )
    weights = rng.uniform( 0.7, 1.3, nb_gaussians )
    return SumOfGaussians( pos, sigmas, weights = weights )


if test( "lbfgs_matches_the_target_masses" ):
    # le test de base : les masses des CELLULES, une fois l'ajustement fini, doivent retomber sur
    # les masses des DIRACS -- c'est la seule chose que `OtPlan` promet, indépendamment de
    # comment elle y arrive (moindre-carrés aujourd'hui, Newton demain). UNE gaussienne, large et
    # bien centrée sur le nuage de diracs : le cas le plus simple, sans aucun désert de densité.
    rng = numpy.random.default_rng( 3 )
    pos = rng.uniform( 0.15, 0.85, size = ( 20, 2 ) )
    src = SumOfDiracs( pos )
    dst = SumOfGaussians( numpy.array( [ [ 0.5, 0.5 ] ] ), numpy.array( [ 0.15 ] ),
                          weights = numpy.array( [ 1.0 ] ) )

    plan = OtPlan( src, dst, boundaries = box_half_spaces( *_BOX ), max_iter = 150, ftol = 1e-15 )

    got    = numpy.asarray( plan.cell_masses ).reshape( -1 )
    target = numpy.asarray( src.normalized_version().weights ).reshape( -1 )
    assert numpy.allclose( got, target, atol = 5e-4 ), numpy.abs( got - target ).max()


if test( "starting_from_nonzero_weights_still_converges" ):
    # le point de départ ne devrait être qu'une question de vitesse, pas de résultat -- ici on
    # part déjà PRÈS de la solution ( `weights0` tiré au hasard mais petit ) plutôt que de zéro.
    # `pert` reste sous le seuil qui viderait déjà une cellule AVANT le premier pas -- `_fit`
    # refuse alors de démarrer (`ValueError`, voir sa docstring) plutôt que de faire semblant.
    rng = numpy.random.default_rng( 2 )
    pos = rng.uniform( 0.1, 0.9, size = ( 18, 2 ) )
    src = SumOfDiracs( pos )
    dst = _overlapping_target( 2, 2, seed = 3 )
    w0 = rng.uniform( -0.003, 0.003, 18 )

    plan = OtPlan( src, dst, boundaries = box_half_spaces( *_BOX ), weights0 = w0,
                  max_iter = 150, ftol = 1e-15 )

    got    = numpy.asarray( plan.cell_masses ).reshape( -1 )
    target = numpy.asarray( src.normalized_version().weights ).reshape( -1 )
    assert numpy.allclose( got, target, atol = 5e-3 ), numpy.abs( got - target ).max()


if test( "no_cell_dies_even_with_scattered_targets" ):
    # le cas DUR (`_scattered_target`) : sans barrière ni plancher, ce scénario précis vide
    # plusieurs cellules et s'y bloque (vérifié -- voir la docstring d'`OtPlan`). Ici on vérifie
    # les DEUX choses que `_fit` promet : aucune cellule ne meurt EN COURS DE ROUTE (`min_measure`
    # reste `> 0` à CHAQUE pas de `plan.history`), et l'ajustement retombe quand même sur les
    # masses cibles.
    rng = numpy.random.default_rng( 5 )
    pos = rng.uniform( 0.1, 0.9, size = ( 40, 2 ) )
    src = SumOfDiracs( pos )
    dst = _scattered_target( 2, 4, seed = 6 )

    plan = OtPlan( src, dst, boundaries = box_half_spaces( *_BOX ), max_iter = 400, ftol = 1e-15 )

    assert all( h[ "min_measure" ] > 0 for h in plan.history ), \
        min( h[ "min_measure" ] for h in plan.history )

    got    = numpy.asarray( plan.cell_masses ).reshape( -1 )
    target = numpy.asarray( src.normalized_version().weights ).reshape( -1 )
    assert numpy.allclose( got, target, atol = 1e-3 ), numpy.abs( got - target ).max()


if test( "the_dual_objective_matches_the_target_masses_too" ):
    # `objective = "dual"` : la fonctionnelle de Kantorovich, dont le gradient est le résidu lui-même
    # ( voir `OtPlan.__init__` ). Même promesse que la moindre-carrés -- les masses cibles -- sur les
    # deux cas, le doux et le DUR ( `_scattered_target`, où des diracs tombent dans des déserts de
    # densité : ni barrière ni plancher ici, une cellule vide a pour gradient sa masse cible ).
    # La précision atteinte est celle de la VALEUR de la fonctionnelle : la descente s'arrête là où
    # elle cesse de décroître. Sur une image ( des morceaux à densité constante, des moments EXACTS )
    # c'est la précision du noyau -- FP64 ici ; sur des gaussiennes, la quadrature ADAPTATIVE
    # ( `PointwiseDensity` ) rend la valeur discontinue à `rtol` ( 1e-5 ) près -- un saut que la
    # décroissance ne franchit plus dès que le gradient tombe sous ~1e-3 dans le cas dur. C'est la
    # limite CONNUE de cet objectif sur une densité lisse ( voir `OtPlan.__init__` ) : la
    # moindre-carrés, elle, a ses mesures exactes ( la réduction 2D de `SumOfGaussians` ).
    rng = numpy.random.default_rng( 7 )
    img = Image( values = rng.uniform( 0.2, 1, size = ( 12, 12 ) ), origin = [ -0.5, -0.5 ],
                 frame = [ [ 2 / 12, 0 ], [ 0, 2 / 12 ] ] )
    for seed, target, n, atol in ( ( 3, _overlapping_target( 2, 2, seed = 3 ), 20, 2e-4 ),
                                   ( 5, _scattered_target( 2, 4, seed = 6 ), 40, 5e-3 ),
                                   ( 7, img, 30, 1e-6 ) ):
        rng = numpy.random.default_rng( seed )
        pos = rng.uniform( 0.1, 0.9, size = ( n, 2 ) )
        src = SumOfDiracs( pos )
        plan = OtPlan( src, target, boundaries = box_half_spaces( *_BOX ), objective = "dual",
                       max_iter = 400, mass_tol = 1e-8, kernel_dtype = "FP64" )
        got    = numpy.asarray( plan.cell_masses ).reshape( -1 )
        want   = numpy.asarray( src.normalized_version().weights ).reshape( -1 )
        assert numpy.allclose( got, want, atol = atol ), ( seed, numpy.abs( got - want ).max(), len( plan.history ) )
        # la fonctionnelle décroît à chaque pas accepté ( Armijo )
        losses = [ h[ "loss" ] for h in plan.history ]
        assert all( b <= a + 1e-12 for a, b in zip( losses, losses[ 1: ] ) )


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
    # `objective = "newton"` : sur une image, quelques pas suffisent, le résidu chute
    # quadratiquement à la fin, et un départ chaud ( les poids d'un nuage voisin ) n'en demande
    # que deux ou trois -- ce dont vit une reconstruction ( `otrec.models.ProjectedDiracModel` )
    rng = numpy.random.default_rng( 41 )
    n = 300
    pos = rng.uniform( 0.05, 0.95, size = ( n, 2 ) )
    img = Image( values = 1 + 0.5 * rng.random( ( 24, 24 ) ), origin = [ 0.0, 0.0 ],
                 frame = [ [ 1 / 24, 0 ], [ 0, 1 / 24 ] ] )
    plan = OtPlan( SumOfDiracs( pos ), img, objective = "newton", max_iter = 60, mass_tol = 1e-10 / n,
                   kernel_dtype = "FP64" )
    res = [ h[ "max_abs_residual" ] * n for h in plan.history ]
    assert res[ -1 ] < 1e-9 and len( res ) < 30, ( res[ -1 ], len( res ) )
    # les deux derniers pas : au moins un ordre de grandeur chacun ( la phase quadratique )
    assert res[ -1 ] < 0.1 * res[ -2 ] < 0.01 * res[ -3 ]
    got  = numpy.asarray( plan.cell_masses ).reshape( -1 )
    want = numpy.asarray( SumOfDiracs( pos ).normalized_version().weights ).reshape( -1 )
    assert numpy.allclose( got, want, atol = 1e-11 )

    warm = OtPlan( SumOfDiracs( pos + 1e-4 * rng.normal( size = pos.shape ) ), img, objective = "newton",
                   max_iter = 60, mass_tol = 1e-10 / n, kernel_dtype = "FP64", weights0 = plan.weights )
    assert len( warm.history ) <= 6, len( warm.history )

    # un départ chaud qui VIDE une cellule ( des poids qui n'ont plus rien à voir avec le nuage )
    # est abandonné pour le Voronoï, et on converge quand même
    bad = OtPlan( SumOfDiracs( pos ), img, objective = "newton", max_iter = 60, mass_tol = 1e-10 / n,
                  kernel_dtype = "FP64", weights0 = rng.uniform( -1, 1, n ) )
    assert bad.history[ -1 ][ "max_abs_residual" ] * n < 1e-9
    assert numpy.all( bad.history[ 0 ][ "weights" ] == 0 )


if test( "newton_starts_from_a_similarity_when_the_voronoi_has_empty_cells" ):
    # des diracs HORS du domaine ( leur cellule de Voronoï restreinte au domaine est vide ) : le
    # départ est le Voronoï du nuage ramené dans le domaine par une similitude, écrit comme
    # diagramme de puissance du nuage d'origine -- toutes les cellules nourries, et Newton converge
    rng = numpy.random.default_rng( 51 )
    n = 40
    pos = rng.uniform( -2, 3, size = ( n, 2 ) )
    img = Image( values = 1 + 0.3 * rng.random( ( 16, 16 ) ), origin = [ 0.0, 0.0 ],
                 frame = [ [ 1 / 16, 0 ], [ 0, 1 / 16 ] ] )
    voronoi = OtPlan( SumOfDiracs( pos ), img, objective = "newton", max_iter = 0, kernel_dtype = "FP64" )
    f, g, m = voronoi._dual( numpy.zeros( n ) )
    assert ( m == 0 ).any()                           # le problème existe bien
    plan = OtPlan( SumOfDiracs( pos ), img, objective = "newton", max_iter = 80, mass_tol = 1e-10 / n,
                   kernel_dtype = "FP64" )
    assert plan.history[ 0 ][ "min_measure" ] > 0     # ... et le départ l'a résolu
    assert numpy.any( plan.history[ 0 ][ "weights" ] != 0 )
    assert plan.history[ -1 ][ "max_abs_residual" ] * n < 1e-9, plan.history[ -1 ][ "max_abs_residual" ] * n
    # la similitude elle-même : ses poids sont ceux du Voronoï du nuage contracté
    w = plan._similarity_start()
    q = 0.5 + 0.8 * ( pos - ( pos.min( axis = 0 ) + pos.max( axis = 0 ) ) / 2 ) / numpy.ptp( pos, axis = 0 ).max()
    a = 0.8 / numpy.ptp( pos, axis = 0 ).max()
    assert numpy.allclose( w - w[ 0 ], ( ( pos ** 2 ).sum( 1 ) - ( q ** 2 ).sum( 1 ) / a ) - ( ( pos ** 2 ).sum( 1 ) - ( q ** 2 ).sum( 1 ) / a )[ 0 ] )


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
    # ses poids, en repartant des précédents ). La tolérance est celle de l'ajustement.
    rng = numpy.random.default_rng( 21 )
    pos = rng.uniform( 0.2, 0.8, size = ( 8, 2 ) )
    dst = _overlapping_target( 2, 2, seed = 4 )

    def plan_at( p, w0 = None ):
        return OtPlan( SumOfDiracs( p ), dst, boundaries = box_half_spaces( *_BOX ), weights0 = w0,
                       max_iter = 300, ftol = 1e-16, kernel_dtype = "FP64" )

    plan = plan_at( pos )
    cost, grad = plan.cost_and_position_grad()
    assert cost > 0 and numpy.isfinite( grad ).all()

    # le coût est bien `W_2^2` : la même chose que `sum_i m_i |p_i - b_i|^2 + sum_i var_i` -- on
    # vérifie au moins la borne `cost >= sum_i m_i |p_i - b_i|^2`
    _, bary, m = plan.transport()
    assert cost >= float( ( m * ( ( pos - bary ) ** 2 ).sum( axis = 1 ) ).sum() ) - 1e-12

    h = 1e-4
    for i, c in ( ( 0, 0 ), ( 3, 1 ), ( 7, 0 ) ):
        dp = numpy.zeros_like( pos ); dp[ i, c ] = h
        fd = ( plan_at( pos + dp, plan.weights ).cost - plan_at( pos - dp, plan.weights ).cost ) / ( 2 * h )
        assert abs( fd - grad[ i, c ] ) < 2e-3 * max( 1.0, abs( fd ) ), ( i, c, fd, grad[ i, c ] )


# -- ce qu'on REGARDE ------------------------------------------------------------------------
#
#   ./run experiment test_OtPlan

def _report( p, plan, pos, stem ):
    """Commun aux deux expériences ci-dessous : la même paire ( courbe, animation ), la même
    lecture d'historique."""
    last = plan.history[ -1 ]
    print( f"  { last[ 'step' ] } pas, perte { last[ 'loss' ]:.3e}"
          f", résidu max { last[ 'max_abs_residual' ]:.3e}"
          f", mesure min finale { last[ 'min_measure' ]:.3e}" )

    write_convergence_html(
        { "perte (moindre-carrés + barrière)": [ h[ "loss" ] for h in plan.history ],
          "résidu max":                        [ h[ "max_abs_residual" ] for h in plan.history ],
          "mesure minimale (jamais 0)":        [ h[ "min_measure" ] for h in plan.history ] },
        p.out_dir / f"{ stem }_convergence.html",
        title = f"OtPlan 2D -- { len( pos ) } diracs" )

    # au plus 40 images, régulièrement choisies dans l'historique : la descente peut prendre plus
    # de pas que ça, et une image par pas ferait une page inutilement lourde pour ce qu'elle
    # montre de plus (la géométrie change peu d'un pas au suivant une fois la descente entamée).
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


if p := experiment( "ot 2D lbfgs",
                    nb_points    = Param( 30, help = "nombre de diracs" ),
                    nb_gaussians = Param( 2, help = "nombre de gaussiennes de la cible" ),
                    max_iter     = Param( 250, help = "nombre de pas" ),
                    seed         = Param( 5, help = "graine du tirage" ) ):
    # ce que fait `OtPlan` : PARTIR des poids nuls (le Voronoï -- chaque cellule prend sa part
    # purement géométrique) et les faire GLISSER jusqu'à ce que chaque cellule pèse, contre la
    # densité cible, exactement ce que pèse son dirac (la même masse pour tous, ici). La courbe
    # de convergence dit SI ça converge et à quelle vitesse ; l'animation montre COMMENT : les
    # PLANS glissent d'un pas à l'autre, pas les germes -- ils ne bougent jamais ici. Cible DOUCE
    # (`_overlapping_target`) : voir l'expérience `ot 2D lbfgs scattered` pour le cas dur.
    pos = numpy.random.default_rng( p.seed ).uniform( 0.1, 0.9, size = ( p.nb_points, 2 ) )
    src = SumOfDiracs( pos )
    dst = _overlapping_target( 2, p.nb_gaussians, seed = p.seed + 1 )

    plan = OtPlan( src, dst, boundaries = box_half_spaces( *_BOX ), max_iter = p.max_iter )
    _report( p, plan, pos, "ot_2d_lbfgs" )


if p := experiment( "ot 2D lbfgs scattered",
                    nb_points    = Param( 40, help = "nombre de diracs" ),
                    nb_gaussians = Param( 4, help = "nombre de bosses, séparées et étroites" ),
                    max_iter     = Param( 400, help = "nombre de pas" ),
                    seed         = Param( 5, help = "graine du tirage" ) ):
    # le cas DUR : des bosses étroites et séparées (`_scattered_target`), qui sans la barrière ni
    # le plancher de `_fit` videraient plusieurs cellules et s'y bloqueraient (vérifié -- voir
    # `OtPlan._fit`). La courbe `mesure minimale` est celle qui compte ici : elle part quasi nulle
    # (un dirac dans un désert de densité, au Voronoï) et doit REMONTER sans jamais retoucher 0.
    pos = numpy.random.default_rng( p.seed ).uniform( 0.1, 0.9, size = ( p.nb_points, 2 ) )
    src = SumOfDiracs( pos )
    dst = _scattered_target( 2, p.nb_gaussians, seed = p.seed + 1 )

    plan = OtPlan( src, dst, boundaries = box_half_spaces( *_BOX ), max_iter = p.max_iter )
    _report( p, plan, pos, "ot_2d_lbfgs_scattered" )

import numpy

from loom.testing import test, experiment, Param

from sdot import OtPlan, SumOfDiracs, SumOfGaussians, Visualizer, box_half_spaces, write_convergence_html


# le domaine déborde LARGEMENT `[ 0, 1 ]^d`, où vivent diracs et gaussiennes : la marge (plusieurs
# fois `sigma`, voir `_target`) rend négligeable la masse gaussienne perdue hors du domaine -- sans
# ça, la somme des mesures plafonne SOUS la masse totale des diracs, un résidu qui ne peut alors
# JAMAIS s'annuler, pas parce que l'ajustement a raté (voir `OtPlan.residual`, et
# `Distribution.bounding_half_spaces` sur ce que rendre le domaine plus grand que le support
# UTILE ne coûte rien -- BEAUCOUP plus grand, ici, coûte simplement de laisser le solveur
# retomber sur ses pieds).
_BOX = ( [ -0.5, -0.5 ], [ 1.5, 1.5 ] )


def _target( d, nb_gaussians, seed, spread = 0.12 ):
    """Quelques gaussiennes PROCHES les unes des autres plutôt que des bosses séparées : leur
    densité combinée ne s'annule nulle part sur la zone utile. Un DÉSERT de densité (deux bosses
    loin l'une de l'autre, un dirac coincé entre les deux) est ce qui bloque LBFGS bien avant
    convergence -- voir `OtPlan._fit_lbfgs` : le gradient d'un poids est proportionnel à la
    densité le long du bord de SA cellule, donc quasi nul loin de tout maximum, ce qu'aucune
    itération de plus ne répare. `spread` (le rayon du nuage de centres) reste petit devant
    `sigma` pour que les bosses se recouvrent largement."""
    rng = numpy.random.default_rng( seed )
    center = numpy.full( d, 0.5 )
    pos = center + rng.uniform( -spread, spread, size = ( nb_gaussians, d ) )
    sigmas = rng.uniform( 0.18, 0.24, nb_gaussians )
    weights = rng.uniform( 0.6, 1.4, nb_gaussians )
    return SumOfGaussians( pos, sigmas, weights = weights )


if test( "lbfgs_matches_the_target_masses" ):
    # le test de base : les masses des CELLULES, une fois l'ajustement fini, doivent retomber sur
    # les masses des DIRACS -- c'est la seule chose que `OtPlan` promet, indépendamment de
    # comment elle y arrive (moindre-carrés aujourd'hui, Newton demain). UNE gaussienne, large et
    # bien centrée sur le nuage de diracs (`_target`, avec `nb_gaussians = 1`, ne tire donc AUCUN
    # désert de densité) : le cas le plus simple où LBFGS n'a aucune raison de trébucher.
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
    rng = numpy.random.default_rng( 2 )
    pos = rng.uniform( 0.1, 0.9, size = ( 18, 2 ) )
    src = SumOfDiracs( pos )
    dst = _target( 2, 2, seed = 3 )
    w0 = rng.uniform( -0.01, 0.01, 18 )

    plan = OtPlan( src, dst, boundaries = box_half_spaces( *_BOX ), weights0 = w0,
                  max_iter = 150, ftol = 1e-15 )

    got    = numpy.asarray( plan.cell_masses ).reshape( -1 )
    target = numpy.asarray( src.normalized_version().weights ).reshape( -1 )
    assert numpy.allclose( got, target, atol = 5e-3 ), numpy.abs( got - target ).max()


# -- ce qu'on REGARDE ------------------------------------------------------------------------
#
#   ./run experiment test_OtPlan

if p := experiment( "ot 2D lbfgs",
                    nb_points    = Param( 30, help = "nombre de diracs" ),
                    nb_gaussians = Param( 2, help = "nombre de gaussiennes de la cible" ),
                    max_iter     = Param( 250, help = "nombre de pas LBFGS" ),
                    seed         = Param( 5, help = "graine du tirage" ) ):
    # ce que fait `OtPlan` : PARTIR des poids nuls (le Voronoï -- chaque cellule prend sa part
    # purement géométrique) et les faire GLISSER jusqu'à ce que chaque cellule pèse, contre la
    # densité cible, exactement ce que pèse son dirac (la même masse pour tous, ici). La courbe
    # de convergence dit SI ça converge et à quelle vitesse ; l'animation montre COMMENT : les
    # PLANS glissent d'un pas à l'autre, pas les germes -- ils ne bougent jamais ici.
    pos = numpy.random.default_rng( p.seed ).uniform( 0.1, 0.9, size = ( p.nb_points, 2 ) )
    src = SumOfDiracs( pos )
    dst = _target( 2, p.nb_gaussians, seed = p.seed + 1 )

    plan = OtPlan( src, dst, boundaries = box_half_spaces( *_BOX ), max_iter = p.max_iter )

    last = plan.history[ -1 ]
    print( f"  { last[ 'step' ] } pas, perte { last[ 'loss' ]:.3e}"
          f", résidu max { last[ 'max_abs_residual' ]:.3e}" )

    write_convergence_html(
        { "perte (0.5 |résidu|²)": [ h[ "loss" ] for h in plan.history ],
          "résidu max":            [ h[ "max_abs_residual" ] for h in plan.history ] },
        p.out_dir / "ot_2d_lbfgs_convergence.html",
        title = f"OtPlan 2D -- LBFGS, { p.nb_points } diracs" )

    # au plus 40 images, régulièrement choisies dans l'historique : LBFGS peut prendre plus de
    # pas que ça, et une image par pas ferait une page inutilement lourde pour ce qu'elle montre
    # de plus (la géométrie change peu d'un pas au suivant une fois la descente bien entamée).
    idx = numpy.unique( numpy.linspace(
        0, len( plan.history ) - 1, min( 40, len( plan.history ) ) ).astype( int ) )

    viz = Visualizer( title = f"OtPlan 2D, { p.nb_points } diracs -- convergence LBFGS",
                      frame_axis = "pas" )
    for j, i in enumerate( idx ):
        if j:
            viz.new_frame( int( plan.history[ i ][ "step" ] ) )
        w = plan.history[ i ][ "weights" ]
        plan.power_diagram( w ).add_to_viz( viz )
        viz.add_points( pos, color = "#ffffff" )
    viz.write_html( p.out_dir / "ot_2d_lbfgs_anim.html" )

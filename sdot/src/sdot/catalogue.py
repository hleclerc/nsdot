"""Ce qui PROVOQUE les compilations du catalogue : l'usage standard de sdot, exercé une fois.

    SDOT_CATALOGUE_RECORD=dir python -m sdot.catalogue [--device cpu|cuda]

Chaque appel ci-dessous passe par `compile_and_register`, qui dépose le source généré dans `dir`
(voir `loom.compilation.catalogue`) ; `scripts/build_catalogue.py` compile ensuite ce relevé par
variante. Un noyau qui n'est pas provoqué ici n'est pas dans le catalogue -- il se compilera sur
la machine de l'utilisateur (mode `auto`), ou manquera (`SDOT_KERNELS=catalogue`). La liste est
donc une DÉCISION : ce qu'un usage standard fait, en 2D et 3D, en Voronoi et en Laguerre, avec
les deux flottants du noyau, la densité uniforme et une image, ses dérivées.

Petit et rapide (quelques secondes hors compilation) : ce sont les SOURCES qui comptent, pas les
tailles.
"""
import argparse

import numpy as np


def provoke( device = None ):
    from loom import driver
    if device:
        driver.device = device
    from sdot import PowerDiagram, Cell, box_half_spaces, Image, SumOfGaussians

    rng = np.random.default_rng( 0 )
    for d in ( 2, 3 ):
        n = 200
        pos = rng.uniform( 0.05, 0.95, size = ( n, d ) )
        w = rng.uniform( -0.3, 0.3, n ) * n ** ( -2.0 / d )
        box = box_half_spaces( [ 0 ] * d, [ 1 ] * d )
        for kernel in ( "FP32", "FP64" ):
            for weights in ( None, w ):
                for acc in ( "plain", None ):
                    pd = PowerDiagram( pos, weights = weights, boundaries = box, accelerator = acc, kernel_dtype = kernel )
                    np.asarray( pd.measures.tensor )
                    pd.moments
                    pd.hessian_rows()
                    pd.cells
                    # les dérivées des mesures : par rapport aux positions, puis aux poids
                    _vjp( lambda p: PowerDiagram( p, weights = weights, boundaries = box, accelerator = acc, kernel_dtype = kernel ).measures, pos )
                    if weights is not None:
                        _vjp( lambda q: PowerDiagram( pos, weights = q, boundaries = box, accelerator = acc, kernel_dtype = kernel ).measures, w )
        # une densité image, et une somme de gaussiennes
        img = Image( values = rng.uniform( 0.5, 1.5, size = ( 16, ) * d ) )
        pd = PowerDiagram( pos * 16, boundaries = box_half_spaces( [ 0 ] * d, [ 16 ] * d ), distribution = img )
        np.asarray( pd.measures.tensor )
        pd.moments
        if d == 2:
            sog = SumOfGaussians( positions = pos[ :4 ], sigmas = [ 0.1 ] * 4, weights = [ 1.0 ] * 4 )
            pd = PowerDiagram( pos, boundaries = box, distribution = sog )
            np.asarray( pd.measures.tensor )
            _vjp( lambda p: PowerDiagram( p, boundaries = box, distribution = sog ).measures, pos )

    # la cellule seule, ce que `Cell` expose
    for d in ( 2, 3 ):
        c = Cell.make_hypercube( d, [ 0 ] * d, np.eye( d ).tolist() )
        c.cut( [ 1.0 ] * d, 1.0 )
        np.asarray( c.measure.tensor )


def _vjp( f, x ):
    """Une passe arrière, pour provoquer le noyau `bwd` de `f` (même chemin que `check_grad`)."""
    from loom import driver
    probe = f( x )
    dense_shape = tuple( probe.shape )
    crop = lambda t: t.raw[ tuple( slice( 0, s ) for s in dense_shape ) ]
    out, pullback = driver.vjp( lambda v: crop( f( v ) ), x )
    pullback( driver.random( out.shape, seed = 0 ) )


def main( argv = None ):
    p = argparse.ArgumentParser( description = __doc__.splitlines()[ 0 ] )
    p.add_argument( "--device", default = None )
    a = p.parse_args( argv )
    provoke( a.device )
    print( "catalogue: provoqué" )


if __name__ == "__main__":
    main()

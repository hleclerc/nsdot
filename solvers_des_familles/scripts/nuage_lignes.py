#!/usr/bin/env python
"""Le nuage de lignes de `2d_des_familles/cases/gen_cases.py`, a epaisseur reglable, SANS le
clip : un germe tire hors du carre est retire, pas ramene sur le bord ( le clip fabrique des
germes confondus dans les coins des qu'on epaissit les lignes, et un germe confondu a une cellule
de Voronoi vide -- Newton n'y est plus defini ). Poids nuls seulement.

    python scripts/nuage_lignes.py -n 100000 --sigma 0.05
"""
import argparse, os, sys
import numpy as np

CASES = os.path.join( os.path.dirname( os.path.abspath( __file__ ) ), "..", "..", "2d_des_familles", "cases" )
sys.path.insert( 0, CASES )
import gen_cases as g

p = argparse.ArgumentParser()
p.add_argument( "-n", type = int, default = 100000 )
p.add_argument( "--lines", type = int, default = 5 )
p.add_argument( "--sigma", type = float, default = 0.05 )
p.add_argument( "--seed", type = int, default = 0 )
a = p.parse_args()

rng = np.random.default_rng( a.seed )
lines = g.make_lines( a.lines, rng )
pos = np.empty( ( 0, 2 ) )
while len( pos ) < a.n:
    P = g.make_cloud( 2 * a.n, lines, a.sigma, rng )     # `make_cloud` clippe : on retire ce qu'il a clippe
    P = P[ ( P[ :, 0 ] > g.EPS ) & ( P[ :, 0 ] < 1 - g.EPS ) & ( P[ :, 1 ] > g.EPS ) & ( P[ :, 1 ] < 1 - g.EPS ) ]
    pos = np.vstack( [ pos, P ] )
pos = pos[ : a.n ]
pos = np.unique( pos, axis = 0 )
assert len( pos ) == a.n, "germes confondus"
base = "lines%d_n%d_s%g" % ( a.lines, a.n, a.sigma )
fn = os.path.join( CASES, base + "_voronoi.txt" )
g.save( fn, pos, np.zeros( a.n ), [ "nuage : %d diracs autour de %d lignes, sigma = %g, graine = %d, SANS clip ( scripts/nuage_lignes.py )" % ( a.n, a.lines, a.sigma, a.seed ),
                                    "format : n, puis n lignes « x y w »", "poids : NULS (diagramme de Voronoi)" ] )
print( "->", fn )

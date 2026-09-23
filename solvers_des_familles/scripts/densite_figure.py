#!/usr/bin/env python3
"""Les cellules autour d'un pic, aux etapes de la continuation en largeur : depuis les trames de
`densite --dump` ( une trame par etape, le diagramme converge ).

    python3 scripts/densite_figure.py trames.jsonl figures/densite_pic.png --centre 0.72,0.26 --rayon 0.12
"""
import sys, json, argparse
import numpy as np
import matplotlib
matplotlib.use( "Agg" )
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

ap = argparse.ArgumentParser()
ap.add_argument( "trames" )
ap.add_argument( "sortie" )
ap.add_argument( "--centre", default = "0.72,0.26" )
ap.add_argument( "--rayon", type = float, default = 0.12 )
ap.add_argument( "--etapes", default = "", help = "indices des trames a montrer ( defaut : 4 reparties, la derniere comprise )" )
a = ap.parse_args()
cx, cy = map( float, a.centre.split( "," ) )
trames = [ json.loads( l ) for l in open( a.trames ) if l.strip() ]
if a.etapes:
    idx = [ int( v ) for v in a.etapes.split( "," ) ]
else:
    idx = sorted( set( [ 0, len( trames ) // 3, 2 * len( trames ) // 3, len( trames ) - 1 ] ) )

fig, axes = plt.subplots( 1, len( idx ), figsize = ( 4.2 * len( idx ), 4.4 ) )
if len( idx ) == 1: axes = [ axes ]
for ax, k in zip( axes, idx ):
    tr = trames[ k ]
    polys, cols = [], []
    for cel, r in zip( tr[ "cellules" ], tr[ "aires" ] ):
        if len( cel ) < 6: continue
        P = np.array( cel ).reshape( -1, 2 )
        if P[ :, 0 ].max() < cx - a.rayon or P[ :, 0 ].min() > cx + a.rayon: continue
        if P[ :, 1 ].max() < cy - a.rayon or P[ :, 1 ].min() > cy + a.rayon: continue
        polys.append( P )
        cols.append( np.log10( max( r, 1e-3 ) ) )       # l'aire de la cellule, relative a la masse cible
    pc = PolyCollection( polys, array = np.array( cols ), cmap = "viridis", edgecolors = "k", linewidths = 0.15 )
    pc.set_clim( -1, 2 )
    ax.add_collection( pc )
    ax.set_xlim( cx - a.rayon, cx + a.rayon ); ax.set_ylim( cy - a.rayon, cy + a.rayon )
    ax.set_aspect( "equal" ); ax.set_xticks( [] ); ax.set_yticks( [] )
    ax.set_title( "s = %g  ( %d cellules vues )" % ( tr[ "s" ], len( polys ) ), fontsize = 10 )
fig.colorbar( pc, ax = axes, fraction = 0.02, pad = 0.01, label = "log10( aire / aire moyenne )" )
fig.suptitle( "les cellules autour du pic ( %.2f, %.2f ) le long de la continuation" % ( cx, cy ), fontsize = 11 )
fig.savefig( a.sortie, dpi = 130, bbox_inches = "tight" )
print( "ecrit", a.sortie )

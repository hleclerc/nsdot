#!/usr/bin/env python3
"""Les figures du banc `ecrasement` : depuis PREFIX_grille.csv et PREFIX_cellules.csv.

    python3 scripts/ecrasement_plot.py /tmp/lignes100000 "lignes, n = 1e5" figure.png
"""
import sys, csv
import matplotlib
matplotlib.use( "Agg" )
import matplotlib.pyplot as plt

C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"   # bleu, orange, aqua, violet
INK, MUTED, GRID = "#1f2937", "#6b7280", "#e5e7eb"

def lit( f ):
    with open( f ) as h:
        r = list( csv.DictReader( h ) )
    return { k: [ float( x[ k ] ) for x in r ] for k in r[ 0 ] }

def axe( ax, titre, ylab ):
    ax.set_title( titre, loc = "left", color = INK, fontsize = 10 )
    ax.set_ylabel( ylab, color = MUTED ); ax.set_xlabel( "alpha", color = MUTED )
    ax.set_xscale( "log" ); ax.grid( True, color = GRID, lw = 0.6 )
    for s in ( "top", "right" ): ax.spines[ s ].set_visible( False )
    ax.tick_params( colors = MUTED )

def main( prefix, nom, out ):
    g = lit( prefix + "_grille.csv" ); c = lit( prefix + "_cellules.csv" )
    al = g[ "alpha" ]
    fig, axs = plt.subplots( 2, 2, figsize = ( 12, 8 ) )
    fig.suptitle( "Ecrasement le long de la direction de Newton -- " + nom, x = 0.01, ha = "left", color = INK )

    # 1. la plus petite aire, exacte et polynomiale
    ax = axs[ 0, 0 ]
    nu = None
    ax.plot( al, g[ "min_exact" ], color = C1, lw = 2, label = "min exact ( diagramme )" )
    ax.plot( al, g[ "min_poly" ], color = C2, lw = 2, ls = "--", label = "min des polynomes" )
    ax.axhline( 0, color = MUTED, lw = 0.8 )
    axe( ax, "la plus petite aire", "aire" )
    lo = min( x for x in g[ "min_exact" ] if x > 0 ) / 3
    ax.set_yscale( "symlog", linthresh = lo ); ax.legend( frameon = False, fontsize = 8 )

    # 2. les cellules vides : exactes, predites, communes
    ax = axs[ 0, 1 ]
    ax.plot( al, g[ "vides_exact" ], color = C1, lw = 2, label = "vides ( diagramme )" )
    ax.plot( al, g[ "vides_poly" ], color = C2, lw = 2, ls = "--", label = "vides ( polynomes )" )
    ax.plot( al, g[ "accord" ], color = C3, lw = 2, ls = ":", label = "les deux" )
    axe( ax, "cellules vides", "nombre" ); ax.set_yscale( "symlog", linthresh = 1 ); ax.set_ylim( bottom = 0 )
    ax.legend( frameon = False, fontsize = 8 )

    # 3. l'accord cellule par cellule
    ax = axs[ 1, 0 ]
    ax.plot( al, g[ "f_1e8" ], color = C1, lw = 2, label = "|exact - poly| < 1e-8 nu" )
    ax.plot( al, g[ "f_1e4" ], color = C2, lw = 2, label = "< 1e-4 nu" )
    ax.plot( al, g[ "f_1e2" ], color = C3, lw = 2, label = "< 1e-2 nu" )
    axe( ax, "fraction des cellules dont le polynome colle a l'aire exacte", "fraction" )
    ax.set_ylim( 0, 1.02 ); ax.legend( frameon = False, fontsize = 8 )

    # 4. les cellules suivies ( les trois premieres racines predites )
    ax = axs[ 1, 1 ]
    ids = [ k[ 6: ] for k in c if k.startswith( "exact_" ) ][ :3 ]
    for i, col in zip( ids, ( C1, C2, C3 ) ):
        ax.plot( al, c[ "exact_" + i ], color = col, lw = 2, label = "cellule " + i + " exacte" )
        ax.plot( al, c[ "poly_" + i ], color = col, lw = 1.2, ls = "--", label = "cellule " + i + " polynome" )
    ax.axhline( 0, color = MUTED, lw = 0.8 )
    axe( ax, "les premieres a se vider ( plein : exact, tirets : polynome )", "aire" )
    ax.set_yscale( "symlog", linthresh = lo ); ax.legend( frameon = False, fontsize = 7, ncol = 2 )
    ax.set_ylim( -5 * max( c[ "exact_" + ids[ 0 ] ] ), 3 * max( c[ "exact_" + ids[ 0 ] ] ) )

    fig.tight_layout( rect = ( 0, 0, 1, 0.97 ) )
    fig.savefig( out, dpi = 110 )
    print( "ecrit", out )

if __name__ == "__main__":
    main( sys.argv[ 1 ], sys.argv[ 2 ], sys.argv[ 3 ] )

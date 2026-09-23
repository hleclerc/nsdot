#!/usr/bin/env python3
"""Newton contre le premier ordre : le residu `max |a - nu| / nu` apres chaque pas, contre le nombre
de diagrammes ( le cout partage par toutes les methodes ), un cadre par cas. Depuis les CSV de
`newton --courbe` et `densite --courbe` ( `methode;cas;dim;it;diagrammes;temps;reste;|r|_2` ).

    python3 scripts/methodes_plot.py figures/methodes.png courbes_2d.csv courbes_3d.csv courbes_densite.csv
"""
import sys, csv, collections
import numpy as np
import matplotlib
matplotlib.use( "Agg" )
import matplotlib.pyplot as plt

sortie, fichiers = sys.argv[ 1 ], sys.argv[ 2: ]
MONTREES = [ "newton", "newton essai-limites", "lbfgs L0 taux 0.5", "lbfgs L0", "lbfgs L0 taux 0.5 +newton", "cg L0 taux 0.5", "lbfgs L0 refacto 1 taux 0.5", "lbfgs jacobi", "lbfgs L0 sans plancher" ]
courbes = collections.OrderedDict()                     # ( cas, methode ) -> [ ( diag, temps, reste ) ]
etapes = collections.OrderedDict()                      # densite : ( cas, methode ) -> { s : ( diag cumules, temps ) a la fin de l'etape }
vus = set()
for f in fichiers:
    bloc, dernier = None, None
    for row in csv.reader( open( f ), delimiter = ";" ):
        if len( row ) < 8: continue
        meth, cas, dim, it, diag, temps, reste, r2 = row[ :8 ]
        cas = cas.split( "/" )[ -1 ].replace( "_voronoi.txt", "" ).replace( "n=100000", "" ).strip()
        cas = { "lines5_n100000_s0.005": "lignes", "planes4_n100000_s0.02": "plans" }.get( cas, cas )
        if meth not in MONTREES: continue
        if cas.startswith( "densite" ):                  # une continuation : ce qu'on lit, c'est le cout cumule par etape
            cas, s = cas.split( " s " )
            cle = ( dim + " " + cas, meth, float( s ) )
            if cle != bloc:                              # un bloc = une etape d'un lancement ; seul le premier compte
                if bloc is not None and bloc not in vus: etapes.setdefault( bloc[ :2 ], collections.OrderedDict() )[ bloc[ 2 ] ] = dernier; vus.add( bloc )
                bloc = cle
            dernier = ( int( diag ), float( temps ) )
            continue
        courbes.setdefault( ( dim + " " + cas, meth ), [] ).append( ( int( diag ), float( temps ), float( reste ) ) )
    if bloc is not None and bloc not in vus: etapes.setdefault( bloc[ :2 ], collections.OrderedDict() )[ bloc[ 2 ] ] = dernier; vus.add( bloc )
for cle in list( etapes ):                              # un lancement direct ( une seule etape ) n'est pas une continuation
    if len( etapes[ cle ] ) < 2: del etapes[ cle ]

cas_liste = list( collections.OrderedDict.fromkeys( [ c for c, _ in courbes ] + [ c for c, _ in etapes ] ) )
couleurs = { "newton": "#1f77b4", "newton essai-limites": "#17becf", "lbfgs L0 taux 0.5": "#d62728", "lbfgs L0": "#e377c2",
             "lbfgs L0 taux 0.5 +newton": "#ff7f0e", "cg L0 taux 0.5": "#2ca02c", "lbfgs L0 refacto 1 taux 0.5": "#9467bd",
             "lbfgs jacobi": "#8c564b", "lbfgs L0 sans plancher": "#7f7f7f" }
ncol = min( 3, len( cas_liste ) ); nlig = ( len( cas_liste ) + ncol - 1 ) // ncol
fig, axes = plt.subplots( nlig, ncol, figsize = ( 5.2 * ncol, 4.2 * nlig ), squeeze = False )
for ax in axes.flat[ len( cas_liste ): ]: ax.axis( "off" )
for ax, cas in zip( axes.flat, cas_liste ):
    if cas.startswith( "2D densite" ):
        for meth in MONTREES:
            if ( cas, meth ) not in etapes: continue
            e = etapes[ ( cas, meth ) ]
            ss = [ max( v, 4e-3 ) for v in e ]           # s = 0, la derniere etape, posee sous la plus petite largeur
            ax.semilogx( ss, [ v[ 0 ] for v in e.values() ], lw = 1.4, marker = ".", color = couleurs.get( meth, None ),
                         label = "%s  ( %d diag, %.0f s )" % ( meth, list( e.values() )[ -1 ][ 0 ], list( e.values() )[ -1 ][ 1 ] ) )
        ax.invert_xaxis()
        ax.set_title( cas + " : la continuation en largeur, diagrammes cumules", fontsize = 10 )
        ax.set_xlabel( "s ( la largeur de convolution, jusqu'a 0 )" )
        ax.set_ylabel( "diagrammes cumules" )
        ax.grid( True, which = "both", lw = 0.3, alpha = 0.5 )
        ax.legend( fontsize = 6.5, loc = "upper left" )
        continue
    for meth in MONTREES:
        if ( cas, meth ) not in courbes: continue
        pts = np.array( courbes[ ( cas, meth ) ] )
        # une reprise d'etape ( densite ) repart de la derniere valeur : on garde l'ordre d'ecriture
        ax.semilogy( pts[ :, 0 ], np.maximum( pts[ :, 2 ], 1e-12 ), lw = 1.4, color = couleurs.get( meth, None ),
                     label = "%s  ( %d diag, %.0f s )" % ( meth, pts[ -1, 0 ], pts[ -1, 1 ] ) )
    ax.set_title( cas, fontsize = 10 )
    ax.set_xlabel( "diagrammes" )
    ax.set_ylabel( "max |a - nu| / nu" )
    ax.grid( True, which = "both", lw = 0.3, alpha = 0.5 )
    ax.legend( fontsize = 6.5, loc = "upper right" )
fig.suptitle( "Newton amorti contre L-BFGS et gradient conjugue ( n = 100 000 )", fontsize = 11 )
fig.tight_layout()
fig.savefig( sortie, dpi = 130 )
print( "ecrit", sortie )

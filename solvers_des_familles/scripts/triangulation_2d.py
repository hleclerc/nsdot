#!/usr/bin/env python
"""LA PROLONGATION PAR LA TRIANGULATION GROSSIERE, EN 2D : ecrire le depart, pour le faire juger par
Newton ( README § 8.6 ).

Le § 8.6.5 avait juge la construction sur `c*` contre le PLANCHER DU BANC ( `0.5 min( nu_i, aire min de
Voronoi )` ) et l'avait declaree inutilisable. C'etait le mauvais critere : le plancher d'amortissement
de Newton est `eps = 0.5 min( min nu, min a_depart )` -- il S'ADAPTE au depart ( `Newton.h` ). Newton
n'exige donc pas des cellules grosses, seulement qu'AUCUNE ne soit vide, ce que la construction garantit
par `0 < c <= c*`. La seule question qui compte est : COMBIEN D'ITERATIONS RESTE-T-IL ?

Ce script ecrit le depart dans un fichier de cas ( `x y w` ), qui se fait ensuite juger par

    multiechelle --load DEPART.txt --2d --threads 8 --lisse-solution 0 [--corr releve]

( `--lisse-solution 0` = les poids du fichier, zero balayage ).

    scripts/triangulation_2d.py --grossier NIV.txt --fin CAS.txt --c cstar --out DEPART.txt
"""
import argparse, os, sys
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert( 0, os.path.dirname( os.path.abspath( __file__ ) ) )
from cstar_2d import lit_niveau, triangulation_reguliere, gradients, plis


def lit_cas( chemin, D = 2 ):
    """le format du banc : commentaires `#`, puis `n`, puis `n` lignes `x.. w`."""
    vals = []
    with open( chemin ) as f:
        for ligne in f:
            ligne = ligne.split( "#" )[ 0 ].strip()
            if ligne: vals.extend( float( v ) for v in ligne.split() )
    n = int( vals[ 0 ] )
    a = np.array( vals[ 1 : 1 + n * ( D + 1 ) ] ).reshape( n, D + 1 )
    return a[ :, :D ], a[ :, D ]


def ecrit_cas( chemin, P, W, entete ):
    with open( chemin, "w" ) as f:
        f.write( entete )
        f.write( "%d\n" % len( P ) )
        for i in range( len( P ) ):
            f.write( " ".join( "%.17g" % v for v in P[ i ] ) + " %.17g\n" % W[ i ] )


def localise( p, Pc, simplexes, val, k = 96 ):
    """le simplexe de chaque germe fin, par `Uh( p ) = max_T A_T( p )` ( `Uh` est CONVEXE, donc egal au
    max de ses morceaux affines, et l'argmax EST le simplexe qui contient `p` ). Rend l'indice du
    simplexe et les coordonnees barycentriques dedans ( negatives hors de l'enveloppe grossiere )."""
    g = gradients( Pc, val, simplexes )                                   # ( m, 2 )
    z0 = val[ simplexes[ :, 0 ] ]
    p0 = Pc[ simplexes[ :, 0 ] ]
    cen = Pc[ simplexes ].mean( axis = 1 )
    _, cand = cKDTree( cen ).query( p, k = min( k, len( cen ) ) )         # ( n, k )
    A = z0[ cand ] + np.einsum( "nkd,nkd->nk", g[ cand ], p[ :, None, : ] - p0[ cand ] )
    j = A.argmax( axis = 1 )
    t = cand[ np.arange( len( p ) ), j ]
    # les barycentriques dans le simplexe retenu
    V = Pc[ simplexes[ t ] ]                                             # ( n, 3, 2 )
    M = np.stack( [ V[ :, 1 ] - V[ :, 0 ], V[ :, 2 ] - V[ :, 0 ] ], axis = 2 )
    lam12 = np.linalg.solve( M, ( p - V[ :, 0 ] )[ :, :, None ] )[ :, :, 0 ]
    lam = np.column_stack( [ 1 - lam12.sum( axis = 1 ), lam12 ] )
    au_bord = j == A.shape[ 1 ] - 1
    return t, lam, au_bord


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument( "--grossier", required = True )
    ap.add_argument( "--fin", required = True )
    ap.add_argument( "--c", default = "cstar", help = "`cstar`, `K*cstar` ( ex 10*cstar ), ou une valeur" )
    ap.add_argument( "--out", required = True )
    o = ap.parse_args()

    Pc, Wc, nuc = lit_niveau( o.grossier )
    Pf, _ = lit_cas( o.fin )
    simplexes, _, _ = triangulation_reguliere( Pc, Wc )
    kU, kQ, _ = plis( Pc, Wc, simplexes )
    contraint = kQ > 0
    cstar = float( ( kU[ contraint ] / kQ[ contraint ] ).min() )

    c = cstar * float( o.c.split( "*" )[ 0 ] ) if o.c.endswith( "cstar" ) and "*" in o.c \
        else cstar if o.c == "cstar" else float( o.c )

    U = ( Pc * Pc ).sum( axis = 1 ) - Wc
    Q = ( Pc * Pc ).sum( axis = 1 )
    t, lam, au_bord = localise( Pf, Pc, simplexes, U )
    Uh = np.einsum( "nv,nv->n", lam, U[ simplexes[ t ] ] )
    Qh = np.einsum( "nv,nv->n", lam, Q[ simplexes[ t ] ] )
    W = ( 1 - c ) * ( Pf * Pf ).sum( axis = 1 ) - Uh + c * Qh

    hors = int( ( lam.min( axis = 1 ) < -1e-12 ).sum() )
    entete = ( "# prolongation par la triangulation grossiere ( README § 8.6, scripts/triangulation_2d.py )\n"
               "# grossier : %s ( n = %d ) ; c* = %.6e ; c = %.6e\n"
               "# format : n, puis n lignes « x y w »\n" % ( os.path.basename( o.grossier ), len( Pc ), cstar, c ) )
    ecrit_cas( o.out, Pf, W, entete )
    print( "c* = %.3e ; c = %.3e ( %.3g c* ) ; %d germes fins sur %d hors de l'enveloppe grossiere ( %.2f %% )"
           % ( cstar, c, c / cstar, hors, len( Pf ), 100 * hors / len( Pf ) ) )
    if au_bord.any():
        print( "  ATTENTION : %d germes dont l'argmax tombe sur le dernier candidat ( augmenter k )" % int( au_bord.sum() ) )
    print( "  -> %s" % o.out )


if __name__ == "__main__":
    main()

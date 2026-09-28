#!/usr/bin/env python
"""LA PROCEDURE DU § 8.7, DESSINEE EN 1D : le relevement minimal comme PROJECTION sur une enveloppe
convexe inferieure ( ce que fait `scripts/enveloppe_2d.py`, en 2D et sans figure ).

Quatre panneaux :
  1. LES CELLULES avant / apres. Le depart est la solution lissee d'un balayage de Jacobi ( la borne
     du § 8.2 ) : des cellules y sont VIDES, et elles se remplissent apres projection.
  2. LE RELEVE, qui est l'objet vrai. `psi_i = ( 1 - eps )|p_i|^2 - w_i`, trace MOINS son enveloppe
     convexe inferieure `H` pour que l'ecart soit visible ( `psi` vaut ~0.5, l'ecart ~1e-5 ). Un germe
     a `0` est un SOMMET de l'enveloppe, sa cellule est non vide, son poids ne bouge PAS. Un germe
     au-dessus de `0` a une cellule VIDE : la fleche est sa remontee, `w_i <- ( 1 - eps )|p_i|^2 - H`,
     la plus petite qui le pose sur l'enveloppe. Tout est calcule contre la MEME enveloppe, donc
     simultanement : aucune remontee ne se mesure sur le diagramme qu'une autre a modifie.
  3. LA MARGE `m_i = a_i / |Vor_i|` avant / apres. Negative = cellule vide. Apres projection, tout
     germe est au-dessus de `0`, a `~eps` pres.
  4. POURQUOI `eps > 0` EST OBLIGATOIRE. A `eps = 0` les germes remontes atterrissent exactement SUR
     l'enveloppe, donc ALIGNES dans le releve : ils ne sont pas des sommets, et leurs cellules sont
     degenerees. C'est le terme `eps |p|^2` qui les rend strictement extremaux, dans le vrai releve
     `|p|^2 - w = eps |p|^2 + H`.

Une difference avec la 2D, a garder en tete : ici l'enveloppe est prise SANS les bords du domaine,
comme en 2D. Les cellules nees hors de `[0,1]` restent donc a finir -- c'est ce que les MIROIRS font
en 2D ( § 8.7.3 ).

Usage : scripts/enveloppe_figure_1d.py [-n 400 --sigma 0.005 --eps 0.1]
"""
import argparse, os, sys
import numpy as np
import matplotlib
matplotlib.use( "Agg" )
import matplotlib.pyplot as plt

sys.path.insert( 0, os.path.dirname( os.path.abspath( __file__ ) ) )
import multiechelle_1d as me
from adoucissement_1d import _hull_sans_domaine


def projette( p, w, eps ):
    """LA PROJECTION, en quatre lignes. Rend `w1`, l'enveloppe `H`, `psi`, et les germes remontes."""
    psi  = ( 1 - eps ) * p * p - w
    hull = _hull_sans_domaine( p, psi )                  # les sommets de l'enveloppe inferieure
    H    = np.interp( p, p[ hull ], psi[ hull ] )        # `H` etant convexe affine par morceaux
    w1   = ( 1 - eps ) * p * p - H                       # les sommets ont `H = psi` : poids INCHANGE
    return w1, H, psi, np.nonzero( w1 > w + 1e-18 )[ 0 ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument( "-n", type = int, default = 400 )
    ap.add_argument( "--sigma", type = float, default = 0.005 )
    ap.add_argument( "--eps", type = float, default = 0.1 )
    ap.add_argument( "--graine", type = int, default = 0 )
    ap.add_argument( "--balayages", type = int, default = 4, help = "balayages de Jacobi qui fabriquent le mauvais depart" )
    ap.add_argument( "--out", default = os.path.join( os.path.dirname( os.path.abspath( __file__ ) ), "..", "figures" ) )
    o = ap.parse_args()
    os.makedirs( o.out, exist_ok = True )

    n = o.n
    p, _ = me.nuage( n, o.sigma, o.graine )
    nu = np.full( n, 1 / n )
    w_sol, _, _ = me.newton( p, nu, np.zeros( n ) )
    w0 = me.jacobi( p, w_sol, o.balayages )              # LE DEPART : la solution lissee
    a0, _, b0 = me.cellules( p, w0 )
    w1, H, psi, remontes = projette( p, w0, o.eps )
    a1, _, b1 = me.cellules( p, w1 )
    vides0, vides1 = np.nonzero( a0 <= 0 )[ 0 ], np.nonzero( a1 <= 0 )[ 0 ]
    print( "n = %d, sigma = %g, eps = %g" % ( n, o.sigma, o.eps ) )
    print( "  depart ( solution lissee 1 balayage ) : %d cellules vides" % len( vides0 ) )
    print( "  apres projection : %d vides, %d germes remontes ( %.1f %% ), les autres INCHANGES"
           % ( len( vides1 ), len( remontes ), 100 * len( remontes ) / n ) )

    # LA FENETRE : le plus grand TROU de l'enveloppe, c'est-a-dire le plus long intervalle entre deux
    # sommets consecutifs. Tout ce qui est dedans est remonte, et `H` y est UNE seule facette affine
    # -- c'est la que le panneau 4 se lit : a `eps = 0` les remontes y sont exactement alignes.
    hull0 = _hull_sans_domaine( p, p * p - w0 )          # a eps = 0 : le VRAI releve, dont les non-sommets
                                                         # sont exactement les cellules vides
    trou = int( np.argmax( np.diff( hull0 ) ) )
    g, dr = int( hull0[ trou ] ), int( hull0[ trou + 1 ] )   # les deux sommets qui bordent le trou
    marge_vue = max( 2, ( dr - g ) // 3 )
    sl = slice( max( 0, g - marge_vue ), min( n, dr + marge_vue + 1 ) )
    x0, x1 = p[ sl ][ 0 ], p[ sl ][ -1 ]
    print( "  fenetre : le trou entre les sommets %d et %d ( %d germes remontes dedans )" % ( g, dr, dr - g - 1 ) )

    fig, ax = plt.subplots( 2, 2, figsize = ( 15, 9 ) )
    fig.suptitle( "Le relèvement minimal comme PROJECTION sur l'enveloppe convexe inférieure "
                  "( n = %d, σ = %g, ε = %g ) — la version 1D de `scripts/enveloppe_2d.py`" % ( n, o.sigma, o.eps ) )

    # ---- 1. les cellules
    A = ax[ 0, 0 ]
    for y, b, a, nom in [ ( 1, b0, a0, "départ : solution lissée de %d balayages" % o.balayages ),
                          ( 0, b1, a1, "après projection" ) ]:
        for q in range( len( b ) - 1 ):
            if b[ q + 1 ] < x0 or b[ q ] > x1: continue
            A.plot( [ b[ q ], b[ q + 1 ] ], [ y, y ], lw = 9, solid_capstyle = "butt",
                    color = "C0" if q % 2 else "C1" )
        A.text( x0, y + 0.13, nom, fontsize = 9 )
    for i in vides0:
        if x0 <= p[ i ] <= x1: A.plot( [ p[ i ] ], [ 1 ], "v", color = "C3", ms = 7, zorder = 5 )
    A.plot( p[ sl ], np.full( sl.stop - sl.start, -0.45 ), "|", color = "0.3", ms = 9 )
    A.text( x0, -0.32, "les germes", fontsize = 9, color = "0.3" )
    A.set_ylim( -0.6, 1.45 ); A.set_xlim( x0, x1 ); A.set_yticks( [] )
    A.set_title( "1. Les cellules, au zoom — ▼ = cellule VIDE (%d au départ, %d après)" % ( len( vides0 ), len( vides1 ) ) )

    # ---- 2. LE RELEVE : psi - H, et les remontees
    A = ax[ 0, 1 ]
    d = psi - H
    dans = ( p >= x0 ) & ( p <= x1 )
    A.axhline( 0, color = "C2", lw = 2, label = "l'enveloppe convexe inférieure $H$" )
    som = dans & ( d <= 1e-18 )
    hau = dans & ( d > 1e-18 )
    A.plot( p[ som ], d[ som ], "o", ms = 7, color = "C0", label = "sommet de $H$ → cellule non vide, poids INCHANGÉ" )
    A.plot( p[ hau ], d[ hau ], "o", ms = 7, color = "C3", label = "au-dessus de $H$ → cellule VIDE" )
    for i in np.nonzero( hau )[ 0 ]:
        A.annotate( "", xy = ( p[ i ], 0 ), xytext = ( p[ i ], d[ i ] ),
                    arrowprops = dict( arrowstyle = "->", color = "C3", lw = 1.2 ) )
    A.set_xlim( x0, x1 )
    A.set_title( r"2. Le relevé : $\psi_i = (1-\varepsilon)p_i^2 - w_i$, moins son enveloppe $H$" )
    A.set_ylabel( r"$\psi_i - H(p_i)$" )
    A.legend( fontsize = 8, loc = "upper right" )
    A.text( 0.02, 0.04, "la flèche = la remontée $w_i \\leftarrow (1-\\varepsilon)p_i^2 - H(p_i)$,\n"
                        "la PLUS PETITE qui pose le point sur $H$", transform = A.transAxes, fontsize = 8 )

    # ---- 3. la marge, avant / apres
    A = ax[ 1, 0 ]
    m0, m1 = me.marge( p, w0 ), me.marge( p, w1 )
    A.axhline( 0, color = "C3", lw = 1, ls = "--" )
    A.axhline( o.eps, color = "C2", lw = 1, ls = ":", label = "ε = %g" % o.eps )
    A.semilogy( p, np.where( m0 > 0, m0, np.nan ), ".", ms = 4, color = "0.6", label = "départ ( marge > 0 )" )
    A.plot( p[ m0 <= 0 ], np.full( int( ( m0 <= 0 ).sum() ), 1e-3 ), "v", ms = 6, color = "C3",
            label = "départ : cellule VIDE (%d)" % len( vides0 ) )
    A.plot( p, np.where( m1 > 0, m1, np.nan ), ".", ms = 4, color = "C0", label = "après projection" )
    A.set_ylim( 5e-4, 2e2 ); A.set_xlabel( "p" ); A.set_ylabel( r"marge $m_i = a_i / |Vor_i|$" )
    A.set_title( "3. La marge : négative = vide. Après projection, tout germe est au-dessus, à ~ε près" )
    A.legend( fontsize = 8, loc = "upper right" )

    # ---- 4. pourquoi eps > 0
    A = ax[ 1, 1 ]
    st = slice( g, dr + 1 )                              # le trou seul, bornes comprises
    for eps, coul, mk, nom in [ ( 0.0, "C3", "s", "ε = 0 : les remontés atterrissent SUR $H$, donc ALIGNÉS\n"
                                                  "→ aucun n'est un sommet, cellules dégénérées" ),
                                ( o.eps, "C0", "o", "ε = %g : le vrai relevé vaut $\\varepsilon p^2 + H$,\n"
                                                    "strictement convexe → TOUT germe est un sommet" % o.eps ) ]:
        we, _, _, _ = projette( p, w0, eps )
        u = p * p - we                                   # LE VRAI releve
        cor = np.interp( p[ st ], [ p[ g ], p[ dr ] ], [ u[ g ], u[ dr ] ] )   # la corde du trou
        A.plot( p[ st ], u[ st ] - cor, mk + "-", ms = 6, lw = 1.2, color = coul, label = nom )
    A.axhline( 0, color = "0.7", lw = 0.8, zorder = 0 )
    A.set_xlabel( "p" ); A.set_ylabel( r"$|p|^2 - w$, moins la corde du trou" )
    A.set_title( "4. Pourquoi ε > 0 est obligatoire ( le trou, %d germes remontés )" % ( dr - g - 1 ) )
    A.legend( fontsize = 8, loc = "lower center" )

    fig.tight_layout( rect = [ 0, 0, 1, 0.96 ] )
    fn = os.path.join( o.out, "enveloppe_1d.png" )
    fig.savefig( fn, dpi = 130 ); plt.close( fig )
    print( "->", fn )


if __name__ == "__main__":
    main()

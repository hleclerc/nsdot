#!/usr/bin/env python
"""LA PARABOLE ABAISSEE, EN 2D : le diagramme de puissance grossier, et le meme GROSSI.

A gauche, le diagramme de puissance des `|S| = n / rho` germes retenus. C'en est un vrai : les
cellules PAVENT le carre. A droite, exactement le meme calcul, sauf que la cellule de `k` est
construite avec le poids `w_k + delta_k` -- et `k` SEUL : les autres gardent leur poids. Ce n'est
donc plus un diagramme de puissance mais `|S|` convexes qui se RECOUVRENT, chacun contenant tout
son paquet.

    delta_k = max_{ m du paquet } max_{ x de E_m } ( h_k - h_m )( x )

`h_k - h_m` est affine (le `|x|^2` est commun aux deux paraboles), donc ce max est atteint a un
SOMMET de `E_m` -- l'enclos de `m` contre `S` seul.

Ce que la troisieme vignette montre : POURQUOI ca deborde. `delta_k` est fixe par un seul membre du
paquet, et lu sur `E_m` qui est bien plus grand que la vraie cellule `Lag( m )`.
"""

import argparse
import os
import sys

import numpy as np

from sub1d_baisse import _style, BLANC, ORANGE, CYAN, ROSE, VERT

SQ = [ ( 0.0, 0.0 ), ( 1.0, 0.0 ), ( 1.0, 1.0 ), ( 0.0, 1.0 ) ]


# ------------------------------------------------------------------- la geometrie, sans dependance

def clip( poly, ids, nx, ny, off, cid ):
    """`poly` intersecte `nx x + ny y <= off`. `ids[ i ]` porte l'arete `i -> i+1` -- le meme
    invariant que `Cell::cut`, et c'est lui qui donne l'ADJACENCE sans la chercher.

    L'interpolation est ancree sur le sommet DEDANS : la forme qui rend exactement le sommet quand
    il est sur le plan."""
    n = len( poly )
    if n == 0:
        return poly, ids
    s = [ nx * p[ 0 ] + ny * p[ 1 ] - off for p in poly ]
    out, oid = [], []
    for i in range( n ):
        j = ( i + 1 ) % n
        if s[ i ] <= 0:
            out.append( poly[ i ] ); oid.append( ids[ i ] )
            if s[ j ] > 0:                              # on sort : l'arete du plan COURANT commence
                t = s[ i ] / ( s[ i ] - s[ j ] )
                out.append( ( poly[ i ][ 0 ] + ( poly[ j ][ 0 ] - poly[ i ][ 0 ] ) * t,
                              poly[ i ][ 1 ] + ( poly[ j ][ 1 ] - poly[ i ][ 1 ] ) * t ) )
                oid.append( cid )
        elif s[ j ] <= 0:                               # on rentre : le reste de l'arete est a `ids`
            t = s[ j ] / ( s[ j ] - s[ i ] )
            out.append( ( poly[ j ][ 0 ] + ( poly[ i ][ 0 ] - poly[ j ][ 0 ] ) * t,
                          poly[ j ][ 1 ] + ( poly[ i ][ 1 ] - poly[ j ][ 1 ] ) * t ) )
            oid.append( ids[ i ] )
    return out, oid


def cellule( i, P, W, J, wi = None ):
    """La cellule de `i` contre les germes `J`, avec eventuellement un poids MODIFIE pour `i` seul."""
    px, py = P[ i ]
    w0 = W[ i ] if wi is None else wi
    poly, ids = list( SQ ), [ -1 ] * 4
    for j in J:
        if j == i:
            continue
        dx, dy = P[ j ][ 0 ] - px, P[ j ][ 1 ] - py
        off = dx * ( px + P[ j ][ 0 ] ) / 2 + dy * ( py + P[ j ][ 1 ] ) / 2 + ( w0 - W[ j ] ) / 2
        poly, ids = clip( poly, ids, dx, dy, off, j )
        if not poly:
            break
    return poly


def diagramme( P, W, ordre, wmax ):
    """TOUTES les cellules du nuage, et pour chacune ses VOISINS. Le rayon de securite arrete le
    balayage : au-dela de `R + sqrt( R^2 + wmax - w_i )`, plus aucun germe ne peut couper."""
    cells, vois = [], []
    for i in range( len( P ) ):
        px, py = P[ i ]
        poly, ids = list( SQ ), [ -1 ] * 4
        for j in ordre[ i ]:
            if j == i:
                continue
            R = max( np.hypot( v[ 0 ] - px, v[ 1 ] - py ) for v in poly )
            d = np.hypot( P[ j ][ 0 ] - px, P[ j ][ 1 ] - py )
            if d >= R + np.sqrt( max( 0.0, R * R + wmax - W[ i ] ) ):
                break
            dx, dy = P[ j ][ 0 ] - px, P[ j ][ 1 ] - py
            off = dx * ( px + P[ j ][ 0 ] ) / 2 + dy * ( py + P[ j ][ 1 ] ) / 2 + ( W[ i ] - W[ j ] ) / 2
            poly, ids = clip( poly, ids, dx, dy, off, j )
            if not poly:
                break
        cells.append( poly )
        vois.append( { u for u in ids if u >= 0 } )
    return cells, vois


def aire( poly ):
    if len( poly ) < 3:
        return 0.0
    a = 0.0
    for i in range( len( poly ) ):
        j = ( i + 1 ) % len( poly )
        a += poly[ i ][ 0 ] * poly[ j ][ 1 ] - poly[ j ][ 0 ] * poly[ i ][ 1 ]
    return abs( a ) / 2


# ---------------------------------------------------------------------------------- les deux nuages

def morton( P ):
    """Un ordre spatial bon marche, pour que `S` soit STRATIFIE (un germe par region) et non tire
    au hasard -- c'est ce que donne gratuitement un sous-arbre du Bsp dans le banc C++."""
    q = np.clip( ( P * 65535 ).astype( np.int64 ), 0, 65535 )
    def part( v ):
        v = ( v | ( v << 8 ) ) & 0x00FF00FF
        v = ( v | ( v << 4 ) ) & 0x0F0F0F0F
        v = ( v | ( v << 2 ) ) & 0x33333333
        v = ( v | ( v << 1 ) ) & 0x55555555
        return v
    return np.argsort( part( q[ :, 0 ] ) | ( part( q[ :, 1 ] ) << 1 ) )


def nuage( nom, n, rng ):
    if nom == "uniforme":
        return rng.uniform( 0.01, 0.99, ( n, 2 ) )
    # l'analogue du nuage du banc : des germes serres le long de quelques segments, et du vide.
    a = rng.integers( 0, 5, n )
    t = rng.uniform( 0, 1, n )
    ang = 0.3 + a * 1.1
    ctr = np.stack( [ 0.25 + 0.12 * a, 0.15 + 0.16 * a ], axis = 1 )
    P = ctr + ( t[ :, None ] - 0.5 ) * 0.8 * np.stack( [ np.cos( ang ), np.sin( ang ) ], axis = 1 )
    return np.clip( P + rng.normal( 0, 0.006, ( n, 2 ) ), 0.01, 0.99 )


# --------------------------------------------------------------------------------------------- visu

def figure( path, nom, P, W, rho ):
    plt = _style()
    from matplotlib.collections import PolyCollection

    n = len( P )
    sub = np.sort( morton( P )[ :: rho ] )
    m = len( sub )
    wmax = float( W.max() )

    # les paquets : le germe de `S` le plus proche.
    d2 = ( ( P[ :, None, : ] - P[ None, sub, : ] ) ** 2 ).sum( axis = 2 )
    pack = d2.argmin( axis = 1 )

    ordre = np.argsort( ( ( P[ :, None, : ] - P[ None, :, : ] ) ** 2 ).sum( axis = 2 ), axis = 1 )
    fine, vois = diagramme( P, W, ordre, wmax )              # les VRAIES cellules
    E = [ cellule( i, P, W, sub ) for i in range( n ) ]      # `E_m`, l'enclos contre `S` seul

    # l'abaissement, et QUI le fixe.
    delta = np.zeros( m )
    qui = np.full( m, -1 )
    som = [ None ] * m
    for i in range( n ):
        k = pack[ i ]
        rx, ry = P[ sub[ k ] ]
        mx, my = P[ i ]
        dx, dy = rx - mx, ry - my
        e = dx * ( rx + mx ) + dy * ( ry + my ) - W[ sub[ k ] ] + W[ i ]
        for v in E[ i ]:
            s = e - 2 * ( dx * v[ 0 ] + dy * v[ 1 ] )
            if s > delta[ k ]:
                delta[ k ], qui[ k ], som[ k ] = s, i, v

    gros = [ cellule( sub[ k ], P, W, sub ) for k in range( m ) ]            # `C_S( k )`, le pavage
    enfl = [ cellule( sub[ k ], P, W, sub, W[ sub[ k ] ] + delta[ k ] ) for k in range( m ) ]

    # la REFERENCE : combien de paquets sont VRAIMENT adjacents. Sans elle, « le recouvrement vaut
    # 13 » ne dit pas si c'est le critere qui est large ou le regroupement qui est cher.
    adj = { ( pack[ i ], pack[ j ] ) for i in range( n ) for j in vois[ i ] }
    vraiadj = len( adj ) / m

    apaq = np.zeros( m )
    for i in range( n ):
        apaq[ pack[ i ] ] += aire( fine[ i ] )
    ratio = np.array( [ aire( enfl[ k ] ) / max( apaq[ k ], 1e-30 ) for k in range( m ) ] )

    nov = np.mean( [ sum( sat( enfl[ k ], enfl[ l ] ) for l in range( m )
                          if len( enfl[ l ] ) > 2 ) for k in range( m ) if len( enfl[ k ] ) > 2 ] )

    coul = plt.get_cmap( "turbo" )( ( np.arange( m ) * 7919 % m ) / max( m - 1, 1 ) )

    fig, ax = plt.subplots( 2, 2, figsize = ( 13.6, 13.4 ), dpi = 140 )

    # --- 1 : le diagramme de puissance grossier. UN PAVAGE.
    a = ax[ 0, 0 ]
    a.add_collection( PolyCollection( [ np.array( g ) for g in gros if len( g ) > 2 ],
                                      facecolors = coul, edgecolors = "#101216", lw = 0.6,
                                      alpha = 0.85 ) )
    a.plot( P[ :, 0 ], P[ :, 1 ], ".", ms = 1.6, color = "#20242c" )
    a.plot( P[ sub, 0 ], P[ sub, 1 ], "o", ms = 4, color = BLANC )
    a.set_title( u"1. le diagramme de puissance des $|S| = n/%d$ germes retenus\n"
                 u"les cellules PAVENT : $C_S(k)$" % rho, fontsize = 11 )

    # --- 2 : le meme, chaque cellule calculee avec SON poids augmente. PLUS UN PAVAGE.
    a = ax[ 0, 1 ]
    a.add_collection( PolyCollection( [ np.array( g ) for g in gros if len( g ) > 2 ],
                                      facecolors = "#1b1f27", edgecolors = "#454d5e", lw = 0.5 ) )
    vus = np.argsort( ratio )[ [ int( q * m ) for q in ( .3, .45, .6, .7, .8, .9 ) ] ]
    a.add_collection( PolyCollection( [ np.array( enfl[ k ] ) for k in range( m )
                                        if len( enfl[ k ] ) > 2 and k not in vus ],
                                      facecolors = "none", edgecolors = "#7f8798", lw = 0.5 ) )
    a.add_collection( PolyCollection( [ np.array( enfl[ k ] ) for k in vus ],
                                      facecolors = coul[ vus ], alpha = 0.30,
                                      edgecolors = coul[ vus ], lw = 2.0 ) )
    a.plot( P[ sub[ vus ], 0 ], P[ sub[ vus ], 1 ], "o", ms = 6, color = BLANC )
    a.set_title( r"2. le MEME diagramme, la cellule de $k$ calculee avec $w_k + \delta_k$" "\n"
                 "($k$ seul : les autres gardent leur poids) : chaque $I_k$ en chevauche %.1f,\n"
                 "alors que %.1f paquets seulement lui sont ADJACENTS" % ( nov, vraiadj ),
                 fontsize = 11 )

    # --- 3 : le RECOUVREMENT, et c'est le chiffre qui compte. `x` est dans `I_k` ssi
    #         `h_k( x ) - delta_k <= psi_S( x )` : une seule carte a calculer, puis `m` comparaisons.
    g = 420
    gx, gy = np.meshgrid( np.linspace( 0, 1, g ), np.linspace( 0, 1, g ) )
    H = np.stack( [ ( gx - P[ j ][ 0 ] ) ** 2 + ( gy - P[ j ][ 1 ] ) ** 2 - W[ j ] for j in sub ] )
    psiS = H.min( axis = 0 )
    rec = ( H - delta[ :, None, None ] <= psiS[ None ] ).sum( axis = 0 )
    a = ax[ 1, 0 ]
    im = a.imshow( rec, origin = "lower", extent = ( 0, 1, 0, 1 ), cmap = "magma",
                   interpolation = "nearest" )
    fig.colorbar( im, ax = a, fraction = 0.046, pad = 0.03 )
    a.add_collection( PolyCollection( [ np.array( g2 ) for g2 in gros if len( g2 ) > 2 ],
                                      facecolors = "none", edgecolors = "#ffffff", lw = 0.35,
                                      alpha = 0.35 ) )
    a.set_title( "3. combien de $I_k$ couvrent chaque point : mediane %d, max %d\n"
                 "(le diagramme grossier en donnerait 1 partout -- c'est un pavage)"
                 % ( np.median( rec ), rec.max() ), fontsize = 11 )

    # --- 4 : POURQUOI. un paquet, et le sommet de `E_m` qui fixe `delta`.
    k = int( np.argsort( ratio )[ int( 0.75 * m ) ] )
    mem = np.flatnonzero( pack == k )
    a = ax[ 1, 1 ]
    a.add_collection( PolyCollection( [ np.array( enfl[ k ] ) ], facecolors = CYAN, alpha = 0.16,
                                      edgecolors = CYAN, lw = 2.4 ) )
    a.add_collection( PolyCollection( [ np.array( fine[ i ] ) for i in mem if len( fine[ i ] ) > 2 ],
                                      facecolors = ROSE, alpha = 0.55, edgecolors = ROSE, lw = 0.8 ) )
    a.add_collection( PolyCollection( [ np.array( gros[ k ] ) ], facecolors = "none",
                                      edgecolors = BLANC, lw = 1.8, linestyles = "--" ) )
    j = int( qui[ k ] )
    a.add_collection( PolyCollection( [ np.array( E[ j ] ) ], facecolors = "none",
                                      edgecolors = ORANGE, lw = 1.8, linestyles = ":" ) )
    a.add_collection( PolyCollection( [ np.array( fine[ j ] ) ], facecolors = "none",
                                      edgecolors = ORANGE, lw = 1.8 ) )
    a.plot( P[ mem, 0 ], P[ mem, 1 ], ".", ms = 4, color = BLANC )
    a.plot( [ P[ sub[ k ], 0 ] ], [ P[ sub[ k ], 1 ] ], "o", ms = 7, color = BLANC )
    a.plot( [ som[ k ][ 0 ] ], [ som[ k ][ 1 ] ], "X", ms = 13, color = VERT )
    for lab, c, ls in ( ( r"$I_k$", CYAN, "-" ), ( u"les vraies cellules du paquet", ROSE, "-" ),
                        ( r"$C_S(k)$", BLANC, "--" ),
                        ( r"$Lag(m^*)$", ORANGE, "-" ),
                        ( "$E_{m^*}$, ou $" "\\delta_k$ est lu", ORANGE, ":" ) ):
        a.plot( [], [], ls, color = c, lw = 2, label = lab )
    a.plot( [], [], "X", color = VERT, ms = 10, label = "le sommet qui FIXE $" "\\delta_k$" )
    b = np.array( enfl[ k ] )
    pad = 0.12 * max( np.ptp( b[ :, 0 ] ), np.ptp( b[ :, 1 ] ) )
    a.set_xlim( b[ :, 0 ].min() - pad, b[ :, 0 ].max() + pad )
    a.set_ylim( b[ :, 1 ].min() - pad, b[ :, 1 ].max() + pad )
    a.set_title( "4. un paquet de %d germes : $\\delta_k$ est fixe par UN membre $m^*$,\n"
                 "et lu sur $E_{m^*}$ -- bien plus grand que sa vraie cellule" % len( mem ),
                 fontsize = 11 )
    a.legend( fontsize = 9, loc = "upper right" )

    for i, a in enumerate( ax.ravel() ):
        a.set_aspect( "equal" )
        if i < 3:
            a.set_xlim( 0, 1 ); a.set_ylim( 0, 1 )
    fig.suptitle( r"%s -- $n = %d$, $\rho = %d$" % ( nom, n, rho ), fontsize = 13 )
    fig.tight_layout()
    fig.savefig( path, bbox_inches = "tight" )
    plt.close( fig )
    print( "    -> %s\n       aire I/paquet %.2f (dec.9 %.2f) ; %.1f paquets chevauches contre "
           "%.1f adjacents ; recouvrement median %d"
           % ( path, np.median( ratio ), np.quantile( ratio, 0.9 ), nov, vraiadj,
               np.median( rec ) ), flush = True )



def cellule_q( q, om, P, W, J ):
    """La cellule d'un germe VIRTUEL `q` de poids `om` contre les germes `J` -- rien ne suppose
    qu'il appartienne au nuage."""
    poly, ids = list( SQ ), [ -1 ] * 4
    for j in J:
        dx, dy = P[ j ][ 0 ] - q[ 0 ], P[ j ][ 1 ] - q[ 1 ]
        off = dx * ( q[ 0 ] + P[ j ][ 0 ] ) / 2 + dy * ( q[ 1 ] + P[ j ][ 1 ] ) / 2 + ( om - W[ j ] ) / 2
        poly, ids = clip( poly, ids, dx, dy, off, j )
        if not poly:
            break
    return poly


def compare( nom, P, W, rho ):
    """LE DECALAGE AFFINE. Abaisser `w` seul, c'est choisir `b` dans

        h_q( x ) - omega = |x|^2 + a . x + b        avec   q = -a/2,  omega = |q|^2 - b

    en laissant `a = -2 p_k` impose par le representant. Deplacer AUSSI la graine, c'est choisir
    `a` : on cherche alors le meilleur MINORANT AFFINE de `min_m L_m` sur les enclos du paquet, ou
    `L_m( x ) = -2 p_m . x + |p_m|^2 - w_m`. C'est un programme lineaire a TROIS inconnues et une
    contrainte par sommet d'enclos -- exactement soluble, ici par `linprog`.

    Trois variantes, meme certificat (la demonstration n'exige `h_q - delta <= h_m` que sur `B_m`,
    quel que soit `q`) :
      1. `q` = le germe de `S` le plus proche          -- ce qui est mesure dans le banc C++
      2. `q` = le barycentre du paquet                 -- la pente corrigee a bon marche
      3. `( a, b )` optimal                            -- le plafond
    """
    from scipy.optimize import linprog

    n = len( P )
    sub = np.sort( morton( P )[ :: rho ] )
    m = len( sub )
    wmax = float( W.max() )
    d2 = ( ( P[ :, None, : ] - P[ None, sub, : ] ) ** 2 ).sum( axis = 2 )
    pack = d2.argmin( axis = 1 )

    ordre = np.argsort( ( ( P[ :, None, : ] - P[ None, :, : ] ) ** 2 ).sum( axis = 2 ), axis = 1 )
    fine, vois = diagramme( P, W, ordre, wmax )
    E = [ cellule( i, P, W, sub ) for i in range( n ) ]

    apaq = np.zeros( m )
    for i in range( n ):
        apaq[ pack[ i ] ] += aire( fine[ i ] )
    vraiadj = len( { ( pack[ i ], pack[ j ] ) for i in range( n ) for j in vois[ i ] } ) / m

    out = []
    for var in ( "plus proche", "barycentre", "affine (LP)" ):
        cel = []
        for k in range( m ):
            mem = np.flatnonzero( pack == k )
            # `A ( a0, a1, b ) <= c` : une ligne par sommet d'enclos.
            A, c = [], []
            for i in mem:
                Lm = lambda v, i = i: ( -2 * P[ i ][ 0 ] * v[ 0 ] - 2 * P[ i ][ 1 ] * v[ 1 ]
                                        + P[ i ] @ P[ i ] - W[ i ] )
                for v in E[ i ]:
                    A.append( [ v[ 0 ], v[ 1 ], 1.0 ] ); c.append( Lm( v ) )
            if not A:
                cel.append( [] ); continue
            A, c = np.array( A ), np.array( c )
            ctr = P[ mem ].mean( axis = 0 )
            if var == "affine (LP)":
                r = linprog( -np.array( [ ctr[ 0 ], ctr[ 1 ], 1.0 ] ), A_ub = A, b_ub = c,
                             bounds = [ ( None, None ) ] * 3, method = "highs" )
                a0, a1, b = r.x
            else:
                q = P[ sub[ k ] ] if var == "plus proche" else ctr
                a0, a1 = -2 * q[ 0 ], -2 * q[ 1 ]
                b = float( ( c - A[ :, 0 ] * a0 - A[ :, 1 ] * a1 ).min() )
            q = np.array( [ -a0 / 2, -a1 / 2 ] )
            cel.append( cellule_q( q, q @ q - b, P, W, sub ) )
        rat = np.array( [ aire( cel[ k ] ) / max( apaq[ k ], 1e-30 ) for k in range( m ) ] )
        nov = np.mean( [ sum( sat( cel[ k ], cel[ l ] ) for l in range( m ) if len( cel[ l ] ) > 2 )
                         for k in range( m ) if len( cel[ k ] ) > 2 ] )
        out.append( ( var, np.median( rat ), np.quantile( rat, .9 ), nov ) )

    print( "  %s, n = %d, rho = %d  (%.1f paquets VRAIMENT adjacents)" % ( nom, n, rho, vraiadj ) )
    print( "  %-14s %10s %10s %10s" % ( "germe q", "aire/paq", "dec. 9", "paq/paq" ) )
    for v, a, b, c in out:
        print( "  %-14s %10.2f %10.2f %10.1f" % ( v, a, b, c ) )
    return out


def sat( A, B ):
    """Deux convexes sont disjoints ssi une normale d'arete de l'un les separe."""
    if len( A ) < 3 or len( B ) < 3:
        return False
    for U, V in ( ( A, B ), ( B, A ) ):
        for i in range( len( U ) ):
            j = ( i + 1 ) % len( U )
            nx, ny = U[ j ][ 1 ] - U[ i ][ 1 ], U[ i ][ 0 ] - U[ j ][ 0 ]
            a = [ nx * p[ 0 ] + ny * p[ 1 ] for p in U ]
            b = [ nx * p[ 0 ] + ny * p[ 1 ] for p in V ]
            e = 1e-12 * np.hypot( nx, ny )
            if min( b ) > max( a ) + e or min( a ) > max( b ) + e:
                return False
    return True


def main():
    a = argparse.ArgumentParser( description = __doc__,
                                 formatter_class = argparse.RawDescriptionHelpFormatter )
    a.add_argument( "-n", type = int, default = 1200 )
    a.add_argument( "--rho", type = int, default = 12 )
    a.add_argument( "--seed", type = int, default = 0 )
    a.add_argument( "--affine", action = "store_true",
                    help = "comparer delta seul / barycentre / affine optimal" )
    a.add_argument( "--out", default = os.path.dirname( os.path.abspath( __file__ ) ) )
    a = a.parse_args()

    for nom in ( "uniforme", "groupe" ):
        rng = np.random.default_rng( a.seed )
        P = nuage( nom, a.n, rng )
        W = np.zeros( a.n )
        if a.affine:
            for rho in ( 4, 12, 32 ):
                compare( nom, P, W, rho )
            continue
        print( "%s (n = %d, rho = %d)" % ( nom, a.n, a.rho ), flush = True )
        figure( os.path.join( a.out, "baisse2d_%s.png" % nom ), nom, P, W, a.rho )
    return 0


if __name__ == "__main__":
    sys.exit( main() )

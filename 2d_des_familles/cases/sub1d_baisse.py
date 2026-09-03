#!/usr/bin/env python
"""LA PARABOLE ABAISSEE : un enclos par PAQUET, et non plus par germe.

L'idee (1D, sequentiel, tout est exact et visible) :

  1. on prend un sous-ensemble `S` de `|S| = n / rho` germes, et on calcule leur diagramme ;
  2. chaque germe `m` est rattache au `k` de `S` le plus proche -- c'est le PAQUET de `k` ;
  3. on ABAISSE la parabole de `k` (son minimum reste sur `p_k`, on ne fait qu'augmenter `w_k`)
     juste assez pour qu'elle minore toutes les paraboles de son paquet ;
  4. l'INTERVALLE de `k` est l'endroit ou la parabole abaissee passe encore sous l'enveloppe `psi_S`.

Pourquoi ca marche -- c'est le seul point qui compte. On note `h_i( x ) = ( x - p_i )^2 - w_i`.
Comme le terme `x^2` est commun, la difference de deux paraboles est AFFINE :

    h_k( x ) - h_m( x ) = -2 ( p_k - p_m ) x + ( p_k^2 - w_k ) - ( p_m^2 - w_m )

donc son maximum sur un intervalle est atteint a un BOUT, et l'abaissement necessaire

    delta_k = max_{ m du paquet }  max_{ x de B_m }  ( h_k - h_m )( x )      ( >= 0, car m = k donne 0 )

est exact, pas une majoration. Une fois `h_k - delta_k <= h_m` acquis sur `B_m` :

    x dans Lag( m )  ==>  h_m( x ) <= h_j( x ) pour tout j de S        ( a fortiori )
                     ==>  h_k( x ) - delta_k <= h_j( x ) pour tout j de S
                     ==>  x dans I_k := { x : h_k - delta_k <= h_j , j de S }

et donc `I_k` CONTIENT toutes les cellules de son paquet. Deux germes ne peuvent se couper que si
leurs cellules se touchent, donc que si les deux intervalles se CHEVAUCHENT : la liste de candidats
d'un germe est l'union des paquets dont l'intervalle rencontre le sien. C'est un certificat, pas une
heuristique -- et le script le verifie a chaque tirage.

Le choix de `B_m` (la zone ou l'on exige `h_k - delta_k <= h_m`) fait tout le reglage :

  - GROSSIER : `B_m = dom`. Rien a calculer, mais on paie la variation de la difference affine sur
    tout le domaine.
  - SERRE : `B_m = E_m`, l'enclos de `m` contre `S` seul (`E_m ` contient `Lag( m )` parce que
    `psi_S >= psi`, cf. sub1d.py). C'est deja disponible, et ca borne la variation par la LARGEUR
    de l'enclos au lieu de celle du domaine.
"""

import argparse
import os
import sys

import numpy as np

from sub1d import laguerre_1d, enclos, cloud_1d, weights_equal


# ------------------------------------------------------------------------- les paquets et l'abaissement

def demi_plans( mk, ck, mS, cS, dom ):
    """`{ x de dom : mk x + ck <= mS_j x + cS_j pour tout j }`, en bloc. C'est `enclos`, mais avec
    des ordonnees `ck` LIBRES -- c'est ce qui permet d'y injecter la parabole abaissee."""
    dm = mk[ :, None ] - mS[ None, : ]
    with np.errstate( divide = "ignore", invalid = "ignore" ):
        x = ( cS[ None, : ] - ck[ :, None ] ) / dm
    lo = np.where( dm < 0, x, -np.inf ).max( axis = 1 )
    hi = np.where( dm > 0, x,  np.inf ).min( axis = 1 )
    mort = ( ( dm == 0 ) & ( ck[ :, None ] > cS[ None, : ] ) ).any( axis = 1 )
    lo = np.where( mort, dom[ 1 ], lo )
    hi = np.where( mort, dom[ 0 ], hi )
    return np.clip( lo, *dom ), np.clip( hi, *dom )


def paquets( p, sub ):
    """Le `k` de `S` le plus proche, pour chaque germe. Rend l'indice DANS `sub`."""
    ps = p[ sub ]
    j = np.clip( np.searchsorted( ps, p ), 1, len( ps ) - 1 )
    return np.where( np.abs( p - ps[ j - 1 ] ) <= np.abs( ps[ j ] - p ), j - 1, j )


def intervalles( p, w, sub, dom = ( 0.0, 1.0 ), serre = True ):
    """Rend `( paquet, delta, Lo, Hi )` : l'abaissement de chaque `k` et son intervalle."""
    m = -2.0 * p
    c = p * p - w
    g = paquets( p, sub )

    if serre:
        A, B = enclos( p, w, sub, dom )          # `E_m`, deja un enclos de `Lag( m )`
    else:
        A = np.full( len( p ), dom[ 0 ] )        # `B_m = dom` : la version sans rien calculer
        B = np.full( len( p ), dom[ 1 ] )
    vif = A <= B                                 # `E_m` vide == germe cache == cellule vide

    # la difference `h_k - h_m` est AFFINE : son max sur `[ A_m, B_m ]` est a un bout.
    dm = m[ sub ][ g ] - m
    dc = c[ sub ][ g ] - c
    v = np.where( vif, np.maximum( dm * A + dc, dm * B + dc ), -np.inf )

    delta = np.zeros( len( sub ) )               # `>= 0` : `m = k` donne toujours 0
    np.maximum.at( delta, g, v )

    Lo, Hi = demi_plans( m[ sub ], c[ sub ] - delta, m[ sub ], c[ sub ], dom )
    return g, delta, Lo, Hi


def mesure( p, w, sub, dom = ( 0.0, 1.0 ), serre = True ):
    """Verifie l'inclusion, puis rend ( largeur intervalle / etendue vraie, candidats par germe )."""
    lo, hi = laguerre_1d( p, w, dom )
    g, delta, Lo, Hi = intervalles( p, w, sub, dom, serre )

    vif = hi > lo
    assert ( Lo[ g[ vif ] ] <= lo[ vif ] + 1e-11 ).all(), "l'intervalle ne contient pas !"
    assert ( Hi[ g[ vif ] ] >= hi[ vif ] - 1e-11 ).all(), "l'intervalle ne contient pas !"

    K = len( sub )
    ext_lo = np.full( K,  np.inf );  np.minimum.at( ext_lo, g[ vif ], lo[ vif ] )
    ext_hi = np.full( K, -np.inf );  np.maximum.at( ext_hi, g[ vif ], hi[ vif ] )
    plein = ext_hi > ext_lo
    ratio = ( Hi - Lo )[ plein ] / ( ext_hi - ext_lo )[ plein ]

    # chevauchement des intervalles == « ces deux paquets peuvent se couper »
    ok = Hi > Lo
    ov = ( Lo[ :, None ] <= Hi[ None, : ] ) & ( Lo[ None, : ] <= Hi[ :, None ] ) & ok[ :, None ] & ok[ None, : ]
    taille = np.bincount( g, minlength = K )
    cand = ( ov * taille[ None, : ] ).sum( axis = 1 )[ g ][ vif ]
    return ratio, cand, delta, ( Lo, Hi )


# ------------------------------------------------------------------------------------------ visu

SOMBRE = dict( figure_facecolor = "#101216", axes_facecolor = "#101216" )
BLANC, ORANGE, CYAN, ROSE, VERT = "#f0f0f0", "#ffa53a", "#4fd8e8", "#ff5f8d", "#8ee06a"


def _style():
    import matplotlib
    matplotlib.use( "Agg" )
    import matplotlib.pyplot as plt
    plt.style.use( "dark_background" )
    plt.rcParams.update( { "figure.facecolor": "#101216", "axes.facecolor": "#101216",
                           "savefig.facecolor": "#101216", "grid.color": "#303540",
                           "axes.edgecolor": "#5a6070", "text.color": BLANC,
                           "axes.labelcolor": BLANC, "xtick.color": "#b0b6c0",
                           "ytick.color": "#b0b6c0", "legend.framealpha": 0.0 } )
    return plt


def fig_principe( path, p, w, sub, dom = ( 0.0, 1.0 ) ):
    plt = _style()
    xs = np.linspace( *dom, 4000 )
    H  = ( xs[ :, None ] - p[ None, : ] ) ** 2 - w[ None, : ]
    psi, psiS = H.min( axis = 1 ), H[ :, sub ].min( axis = 1 )
    lo, hi = laguerre_1d( p, w, dom )
    g, delta, Lo, Hi = intervalles( p, w, sub, dom )
    K = len( sub )

    fig, ax = plt.subplots( 3, 1, figsize = ( 12.0, 12.6 ), dpi = 140,
                            gridspec_kw = dict( height_ratios = [ 1.0, 1.15, 0.85 ] ) )

    # --- 1 : le decor. les n paraboles, le sous-ensemble, et les deux enveloppes.
    a = ax[ 0 ]
    a.plot( xs, H, lw = 0.5, color = "#4a5160" )
    a.plot( xs, H[ :, sub ], lw = 1.0, color = ORANGE, alpha = 0.5 )
    a.plot( xs, psi,  lw = 2.2, color = BLANC,  label = r"$\psi$  (les $n$ germes)" )
    a.plot( xs, psiS, lw = 2.2, color = ORANGE, label = r"$\psi_S$  (le sous-ensemble seul)" )
    a.plot( p, -w, ".", ms = 4, color = "#8a90a0" )
    a.plot( p[ sub ], -w[ sub ], "o", ms = 8, mfc = "none", mew = 1.6, color = ORANGE )
    a.set_title( u"1. le decor : $n = %d$ paraboles, dont $|S| = %d$ retenues"
                 % ( len( p ), K ), fontsize = 11 )
    y0, y1 = psi.min(), psiS.max()
    a.set_ylim( y0 - .08 * ( y1 - y0 ), y1 + .08 * ( y1 - y0 ) )
    a.legend( fontsize = 10, loc = "upper right" )

    # --- 2 : UN paquet, en detail. c'est le coeur de l'idee.
    k = K // 2
    mem = np.flatnonzero( g == k )
    a = ax[ 1 ]
    a.plot( xs, H[ :, mem ], lw = 1.1, color = ROSE, alpha = 0.75 )
    a.plot( [], [], lw = 1.1, color = ROSE, label = r"les $h_m$ du paquet de $k$" )
    a.plot( xs, psiS, lw = 1.8, color = ORANGE, alpha = 0.85, label = r"$\psi_S$" )
    a.plot( xs, H[ :, sub[ k ] ], lw = 2.4, color = BLANC, label = r"$h_k$" )
    a.plot( xs, H[ :, sub[ k ] ] - delta[ k ], lw = 2.4, color = CYAN, ls = "--",
            label = r"$h_k - \delta_k$  (la parabole ABAISSEE)" )
    a.fill_between( xs, H[ :, sub[ k ] ] - delta[ k ], H[ :, sub[ k ] ], color = CYAN, alpha = 0.12 )

    xm = 0.5 * ( Lo[ k ] + Hi[ k ] )
    ym = ( xm - p[ sub[ k ] ] ) ** 2 - w[ sub[ k ] ]
    a.annotate( "", xy = ( xm, ym - delta[ k ] ), xytext = ( xm, ym ),
                arrowprops = dict( arrowstyle = "->", color = CYAN, lw = 1.6 ) )
    a.text( xm, ym - 0.5 * delta[ k ], r"  $\delta_k$", color = CYAN, fontsize = 12, va = "center" )

    yl = H[ :, sub[ k ] ].min() - delta[ k ]
    yh = H[ ( xs > Lo[ k ] ) & ( xs < Hi[ k ] ) ][ :, mem ].max()
    pad = 0.30 * ( yh - yl ) + 1e-9
    a.set_ylim( yl - 0.28 * pad, yh + 0.35 * pad )
    a.set_xlim( max( dom[ 0 ], Lo[ k ] - 3.0 * ( Hi[ k ] - Lo[ k ] ) ),
                min( dom[ 1 ], Hi[ k ] + 3.0 * ( Hi[ k ] - Lo[ k ] ) ) )
    a.axvspan( Lo[ k ], Hi[ k ], color = CYAN, alpha = 0.10 )
    a.axvline( Lo[ k ], color = CYAN, lw = 1.0, ls = ":" )
    a.axvline( Hi[ k ], color = CYAN, lw = 1.0, ls = ":" )
    for m in mem:
        if hi[ m ] > lo[ m ]:
            a.plot( [ lo[ m ], hi[ m ] ], [ yl - 0.16 * pad ] * 2, lw = 3.0, color = ROSE,
                    solid_capstyle = "butt" )
    a.plot( [ Lo[ k ], Hi[ k ] ], [ yl - 0.22 * pad ] * 2, lw = 3.0, color = CYAN,
            solid_capstyle = "butt" )
    a.set_title( u"2. un paquet : on abaisse $h_k$ de $\\delta_k$ jusqu'a ce qu'elle MINORE tout son "
                 u"paquet.\n$I_k = \\{ h_k - \\delta_k \\leq \\psi_S \\}$ (bande) contient alors "
                 u"toutes ses cellules (barres roses)", fontsize = 11 )
    a.legend( fontsize = 10, loc = "upper right" )

    # --- 3 : tous les intervalles, et ce qu'ils enferment.
    a = ax[ 2 ]
    for k in range( K ):
        mem = np.flatnonzero( ( g == k ) & ( hi > lo ) )
        a.plot( [ Lo[ k ], Hi[ k ] ], [ k, k ], lw = 5.0, color = CYAN, alpha = 0.30,
                solid_capstyle = "butt" )
        for m in mem:
            a.plot( [ lo[ m ], hi[ m ] ], [ k, k ], lw = 2.0, color = BLANC, solid_capstyle = "butt" )
        a.plot( p[ sub[ k ] ], k, "o", ms = 5, color = ORANGE )
    a.plot( [], [], lw = 5.0, color = CYAN, alpha = 0.30, label = r"$I_k$ (intervalle du paquet)" )
    a.plot( [], [], lw = 2.0, color = BLANC, label = "les vraies cellules du paquet" )
    a.plot( [], [], "o", ms = 5, color = ORANGE, label = r"$p_k$" )
    a.set_xlim( *dom ); a.set_ylim( -1, K )
    a.set_ylabel( "paquet $k$" ); a.set_xlabel( "$x$" )
    a.set_title( u"3. les intervalles se CHEVAUCHENT (les cellules, non). deux paquets qui ne se "
                 u"chevauchent pas ne peuvent pas se couper.", fontsize = 11 )
    a.legend( fontsize = 10, loc = "lower right" )

    fig.tight_layout()
    fig.savefig( path, bbox_inches = "tight" )
    plt.close( fig )
    print( "    -> %s" % path, flush = True )


def fig_chevauchement( path, cas, sub, dom = ( 0.0, 1.0 ) ):
    plt = _style()
    fig, ax = plt.subplots( 2, len( cas ), figsize = ( 7.4 * len( cas ), 9.0 ), dpi = 140,
                            gridspec_kw = dict( height_ratios = [ 1.0, 1.25 ] ) )
    for col, ( nom, p, w ) in enumerate( cas ):
        lo, hi = laguerre_1d( p, w, dom )
        g, delta, Lo, Hi = intervalles( p, w, sub, dom )
        K = len( sub )
        ok = Hi > Lo
        ov = ( Lo[ :, None ] <= Hi[ None, : ] ) & ( Lo[ None, : ] <= Hi[ :, None ] ) \
             & ok[ :, None ] & ok[ None, : ]

        a = ax[ 0, col ]
        for k in range( K ):
            a.plot( [ Lo[ k ], Hi[ k ] ], [ k, k ], lw = 3.6, color = CYAN, alpha = 0.35,
                    solid_capstyle = "butt" )
            mem = np.flatnonzero( ( g == k ) & ( hi > lo ) )
            if len( mem ):
                a.plot( [ lo[ mem ].min(), hi[ mem ].max() ], [ k, k ], lw = 1.6, color = BLANC,
                        solid_capstyle = "butt" )
        a.set_xlim( *dom ); a.set_ylim( -1, K )
        a.set_xlabel( "$x$" ); a.set_ylabel( "paquet $k$" )
        a.set_title( u"%s -- $I_k$ (cyan) contre l'etendue vraie du paquet (blanc)" % nom,
                     fontsize = 11 )

        a = ax[ 1, col ]
        vrai = np.zeros( ( K, K ), bool )
        o = np.argsort( p )
        for i, j in zip( o[ :-1 ], o[ 1: ] ):
            if hi[ i ] > lo[ i ] and hi[ j ] > lo[ j ]:
                vrai[ g[ i ], g[ j ] ] = vrai[ g[ j ], g[ i ] ] = True
        img = np.zeros( ( K, K, 3 ) )
        img[ ov ] = ( 0.31, 0.85, 0.91 )                       # retenu par le critere
        img[ vrai ] = ( 1.00, 0.37, 0.55 )                     # vraiment adjacent
        assert ( vrai & ~ov ).sum() == 0, "un couple adjacent a ete manque !"
        a.imshow( img, origin = "lower", interpolation = "nearest" )
        a.set_xlabel( "paquet" ); a.set_ylabel( "paquet" )
        a.set_title( u"%s -- retenu (cyan, %.1f/paquet) contre necessaire (rose, %.1f). "
                     u"aucun manque." % ( nom, ov.sum() / K, vrai.sum() / K ), fontsize = 11 )

    fig.tight_layout()
    fig.savefig( path, bbox_inches = "tight" )
    plt.close( fig )
    print( "    -> %s" % path, flush = True )


def fig_gain( path, cas, rhos, n ):
    plt = _style()
    fig, ax = plt.subplots( 1, 2, figsize = ( 14.6, 5.6 ), dpi = 140 )
    coul = { "voronoi": CYAN, "aires egales": ROSE }

    for nom, p, w in cas:
        for serre, ls, mk in ( ( True, "-", "o" ), ( False, "--", "s" ) ):
            med, cnd = [], []
            for r in rhos:
                sub = np.arange( 0, n, r )
                ratio, cand, _, _ = mesure( p, w, sub, serre = serre )
                med.append( np.median( ratio ) )
                cnd.append( np.median( cand ) )
            lab = "%s -- %s" % ( nom, "serre ($B_m = E_m$)" if serre else "grossier ($B_m$ = dom)" )
            ax[ 0 ].plot( rhos, med, mk + ls, color = coul[ nom ], alpha = 1.0 if serre else 0.45,
                          label = lab )
            ax[ 1 ].plot( rhos, cnd, mk + ls, color = coul[ nom ], alpha = 1.0 if serre else 0.45,
                          label = lab )

    a = ax[ 0 ]
    a.set_xscale( "log", base = 2 ); a.set_yscale( "log", base = 2 )
    a.set_xlabel( r"$\rho = n / |S|$" )
    a.set_ylabel( r"largeur $I_k$ / etendue vraie du paquet" )
    a.axhline( 1, color = "#707888", lw = 1, ls = ":" )
    a.set_title( u"de combien l'intervalle deborde son paquet (mediane)", fontsize = 11 )
    a.grid( True, which = "both", alpha = 0.18 ); a.legend( fontsize = 9 )

    a = ax[ 1 ]
    a.plot( rhos, [ 3 * r for r in rhos ], "-", color = "#8892a8", lw = 7, alpha = 0.55,
            zorder = 0, solid_capstyle = "round",
            label = r"$3\rho$ (le paquet et ses deux voisins) -- la BORNE BASSE" )
    a.axhline( 3, color = VERT, lw = 1, ls = ":", label = "3 (l'ideal en 1D)" )
    a.set_xscale( "log", base = 2 ); a.set_yscale( "log", base = 2 )
    a.set_xlabel( r"$\rho = n / |S|$" )
    a.set_ylabel( "candidats par germe (mediane)" )
    a.set_title( u"ce que la liste de candidats coute vraiment, $n = %d$" % n, fontsize = 11 )
    a.grid( True, which = "both", alpha = 0.18 ); a.legend( fontsize = 9 )

    fig.tight_layout()
    fig.savefig( path, bbox_inches = "tight" )
    plt.close( fig )
    print( "    -> %s" % path, flush = True )


# ------------------------------------------------------------------------------------- principal

def main():
    a = argparse.ArgumentParser( description = __doc__,
                                 formatter_class = argparse.RawDescriptionHelpFormatter )
    a.add_argument( "-n", type = int, default = 2048 )
    a.add_argument( "--small", type = int, default = 64, help = "le n de la figure de principe" )
    a.add_argument( "--mid", type = int, default = 256, help = "le n de la figure de chevauchement" )
    a.add_argument( "--rho", type = int, default = 8 )
    a.add_argument( "--seed", type = int, default = 0 )
    a.add_argument( "--out", default = os.path.dirname( os.path.abspath( __file__ ) ) )
    a = a.parse_args()

    def deux( n ):
        p = cloud_1d( n, 4, 0.012, np.random.default_rng( a.seed ) )
        return [ ( "voronoi", p, np.zeros( n ) ), ( "aires egales", p, weights_equal( p ) ) ]

    print( "figure de principe (n = %d, rho = %d)" % ( a.small, a.rho ), flush = True )
    nom, p, w = deux( a.small )[ 1 ]
    fig_principe( os.path.join( a.out, "baisse1d_principe.png" ), p, w,
                  np.arange( 0, a.small, a.rho ) )

    print( "figure de chevauchement (n = %d, rho = %d)" % ( a.mid, a.rho ), flush = True )
    fig_chevauchement( os.path.join( a.out, "baisse1d_chevauchement.png" ), deux( a.mid ),
                       np.arange( 0, a.mid, a.rho ) )

    rhos = [ 2, 4, 8, 16, 32, 64 ]
    print( "figure de gain (n = %d)" % a.n, flush = True )
    cas = deux( a.n )
    fig_gain( os.path.join( a.out, "baisse1d_gain.png" ), cas, rhos, a.n )

    print( "\n  n = %d.  « deborde » = largeur de I_k / etendue vraie du paquet." % a.n )
    for nom, p, w in cas:
        print( "  --- %s" % nom )
        print( "  %-6s %26s %26s" % ( "", "SERRE ( B_m = E_m )", "GROSSIER ( B_m = dom )" ) )
        print( "  %-6s %8s %8s %8s   %8s %8s %8s"
               % ( "rho", "deborde", "cand.", "cd. max", "deborde", "cand.", "cd. max" ) )
        for r in rhos:
            sub = np.arange( 0, a.n, r )
            out = []
            for serre in ( True, False ):
                ratio, cand, _, _ = mesure( p, w, sub, serre = serre )
                out += [ np.median( ratio ), np.median( cand ), cand.max() ]
            print( "  %-6d %8.1f %8.0f %8.0f   %8.1f %8.0f %8.0f" % ( r, *out ) )
    return 0


if __name__ == "__main__":
    sys.exit( main() )

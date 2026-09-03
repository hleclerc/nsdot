#!/usr/bin/env python
"""L'ENCLOS PAR SOUS-ECHANTILLON, en 1D, ou tout est exact et visible.

L'idee a tester : pour borner la cellule d'un germe sans la calculer, on calcule d'abord le
diagramme d'un SOUS-ENSEMBLE `S` des germes, et on s'en sert comme majorant.

Pourquoi c'est gratuit -- et c'est le seul point qui compte :

    psi( x ) = min_i ( |x - p_i|^2 - w_i )

est un MINIMUM. Retirer des germes ne peut que le REMONTER, donc `psi_S >= psi` partout, sans
correction ni marge. La cellule de `k` calculee contre `S` seul,

    C_S( k ) = { x : h_k( x ) <= h_j( x ) pour tout j de S }

a MOINS de contraintes que la vraie, donc `C_S( k ) contient C( k )` -- pour tout `k`, qu'il soit
dans `S` ou non. Les germes omis ne feront que la rogner ensuite. C'est un enclos, et il est valide
par construction.

Ce que ce script mesure : de COMBIEN l'enclos est trop large. C'est le seul chiffre qui decide, car
il fixe le rayon initial du parcours en 2D. Aujourd'hui ce rayon est le domaine entier.

Pourquoi la 1D : l'enveloppe inferieure y est exacte et se calcule en O(n log n) par la pile
monotone (le « convex hull trick » -- c'est l'analogue 1D de la coque convexe relevee dont la
projection est, en 2D, la triangulation reguliere). Aucune approximation ne vient donc polluer le
chiffre, et les deux regimes de poids du banc 2D ont ici une forme CLOSE.
"""

import argparse
import os
import sys

import numpy as np


# ------------------------------------------------------------------- le diagramme 1D, exactement

def _bp( m, c, a, b ):
    """L'abscisse ou les droites `a` et `b` se croisent."""
    return ( c[ b ] - c[ a ] ) / ( m[ a ] - m[ b ] )


def laguerre_1d( p, w, dom = ( 0.0, 1.0 ) ):
    """Les cellules de Laguerre sur `dom`, exactes. Rend `( lo, hi )` ; `lo >= hi` = cellule VIDE.

    On ecrit `h_i( x ) = ( x - p_i )^2 - w_i = x^2 + ( -2 p_i x + p_i^2 - w_i )`. Le terme `x^2` est
    COMMUN a tous les germes, donc l'argmin ne depend que des DROITES `L_i( x ) = m_i x + c_i` avec
    `m_i = -2 p_i` et `c_i = p_i^2 - w_i`. Le diagramme 1D est l'enveloppe inferieure d'un paquet de
    droites -- et un germe CACHE (poids trop bas) est exactement une droite qui n'y apparait pas,
    reconnue sans etre testee contre les autres.
    """
    m = -2.0 * p
    c = p * p - w
    n = len( p )

    # tri par pente DECROISSANTE : la plus grande pente gagne en `-inf`, et les points de rupture
    # sortent alors croissants. A pente egale, seule la plus petite ordonnee compte.
    order = np.lexsort( ( c, -m ) )

    st = []
    for i in order:
        if st and m[ i ] == m[ st[ -1 ] ]:
            continue
        while len( st ) >= 2 and _bp( m, c, st[ -1 ], i ) <= _bp( m, c, st[ -2 ], st[ -1 ] ):
            st.pop()
        st.append( int( i ) )

    lo = np.full( n, dom[ 1 ] )   # par defaut : vide
    hi = np.full( n, dom[ 0 ] )
    lo[ st[ 0 ] ] = -np.inf
    hi[ st[ -1 ] ] = np.inf
    for k in range( len( st ) - 1 ):
        b = _bp( m, c, st[ k ], st[ k + 1 ] )
        hi[ st[ k ] ] = b
        lo[ st[ k + 1 ] ] = b

    return np.clip( lo, *dom ), np.clip( hi, *dom )


def enclos( p, w, sub, dom = ( 0.0, 1.0 ) ):
    """L'ENCLOS de chaque cellule, calcule contre le seul sous-ensemble `sub`.

    `C_S( k ) = intersection_{j de S} { L_k <= L_j }` : chaque `j` donne un demi-axe, et leur
    intersection est un intervalle. On l'obtient d'un bloc, sans reconstruire d'enveloppe.
    """
    m = -2.0 * p
    c = p * p - w
    mS, cS = m[ sub ], c[ sub ]

    dm = m[ :, None ] - mS[ None, : ]                 # ( n, s )
    with np.errstate( divide = "ignore", invalid = "ignore" ):
        x = ( cS[ None, : ] - c[ :, None ] ) / dm
    # `dm > 0` (pente de k plus grande) : la contrainte est `x <= x_croisement`, donc un plafond.
    lo = np.where( dm < 0, x, -np.inf ).max( axis = 1 )
    hi = np.where( dm > 0, x,  np.inf ).min( axis = 1 )
    # pentes egales : la contrainte est constante. `j == k` ne contraint rien ; sinon, si `c_k > c_j`
    # le germe est cache par `j` seul.
    mort = ( ( dm == 0 ) & ( c[ :, None ] > cS[ None, : ] ) ).any( axis = 1 )
    lo = np.where( mort, dom[ 1 ], lo )
    hi = np.where( mort, dom[ 0 ], hi )
    return np.clip( lo, *dom ), np.clip( hi, *dom )


# ------------------------------------------------------------------------------- les deux regimes

def cloud_1d( n, nb_amas, sigma, rng ):
    """L'analogue 1D du nuage de lignes : `nb_amas` paquets serres, et du vide entre eux."""
    # ctr = np.sort( rng.uniform( 0.08, 0.92, nb_amas ) )
    # p = ctr[ rng.integers( nb_amas, size = n ) ] + rng.normal( 0, sigma, n )
    # return np.sort( np.clip( p, 1e-4, 1 - 1e-4 ) )
    # return np.linspace( 0.08, 0.92, n )
    return np.sort( rng.uniform( 0.08, 0.9, n ) )


def weights_equal( p ):
    """Les poids qui donnent a CHAQUE cellule la longueur `1/n`. En 1D c'est une formule, pas un
    solveur : le transport optimal est croissant, donc le `i`-eme germe trie recoit `[i/n,(i+1)/n]`.
    Les points de rupture sont imposes, et

        t_i = ( p_i + p_i+1 ) / 2 + ( w_i - w_i+1 ) / ( 2 ( p_i+1 - p_i ) )

    se renverse en une recurrence. C'est le pendant exact du cas `equal` du banc 2D -- celui ou une
    cellule peut etre LOIN de son germe -- mais obtenu en une ligne au lieu d'un Newton amorti.
    """
    n = len( p )
    t = np.arange( 1, n ) / n
    d = np.diff( p )
    w = np.zeros( n )
    w[ 1: ] = np.cumsum( -2.0 * d * ( t - 0.5 * ( p[ :-1 ] + p[ 1: ] ) ) )
    return w - w.mean()


# ------------------------------------------------------------------------------------------ visu

def fig_principe( path, cas, rate ):
    import matplotlib
    matplotlib.use( "Agg" )
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots( 2, len( cas ), figsize = ( 7.6 * len( cas ), 8.4 ), dpi = 140,
                              gridspec_kw = dict( height_ratios = [ 1.15, 1.0 ] ) )
    xs = np.linspace( 0, 1, 3000 )

    for col, ( nom, p, w ) in enumerate( cas ):
        n = len( p )
        sub = np.arange( 0, n, max( 1, int( round( 1 / rate ) ) ) )
        lo, hi = laguerre_1d( p, w )
        Lo, Hi = enclos( p, w, sub )
        assert ( Lo <= lo + 1e-12 ).all() and ( Hi >= hi - 1e-12 ).all(), "l'enclos ne contient pas !"

        H  = ( xs[ :, None ] - p[ None, : ] ) ** 2 - w[ None, : ]
        psi, psiS = H.min( axis = 1 ), H[ :, sub ].min( axis = 1 )
        ax = axes[ 0, col ]
        ax.plot( xs, H, lw = 0.5, color = "0.82" )
        ax.plot( xs, H[ :, sub ], lw = 0.8, color = "darkorange", alpha = 0.55 )
        ax.plot( xs, psi,  lw = 2.0, color = "black", label = r"$\psi$ (les $n$ germes)" )
        ax.plot( xs, psiS, lw = 2.0, color = "darkorange",
                 label = r"$\psi_S$ ($|S| = n/%d$)" % round( 1 / rate ) )
        ax.fill_between( xs, psi, psiS, color = "darkorange", alpha = 0.18 )
        # le cadrage suit les DEUX enveloppes : sur le cas a aires egales `psi` balaie tout un
        # ordre de grandeur, et un cadrage fixe cacherait la moitie du domaine.
        y0, y1 = psi.min(), psiS.max()
        pad = 0.10 * ( y1 - y0 )
        ax.set_ylim( y0 - pad, y1 + pad )
        ax.set_xlim( 0, 1 )
        ax.set_title( "%s  --  $\\psi_S \\geq \\psi$ PARTOUT : le sous-ensemble MAJORE" % nom,
                      fontsize = 10 )
        ax.legend( fontsize = 9, loc = "upper right" )

        ax = axes[ 1, col ]
        y = np.arange( n )
        ax.hlines( y, Lo, Hi, color = "darkorange", lw = 3.2, alpha = 0.75,
                   label = "enclos $C_S$ (contre $S$ seul)" )
        ax.hlines( y, lo, hi, color = "black", lw = 1.4, label = "vraie cellule $C$" )
        ax.plot( p, y, ".", ms = 3.5, color = "crimson", label = "le germe $p_i$" )
        ax.plot( p[ sub ], y[ sub ], "o", ms = 6, mfc = "none", color = "darkorange", mew = 1.2 )
        r = ( Hi - Lo ) / np.maximum( hi - lo, 1e-300 )
        ax.set_xlim( 0, 1 ); ax.set_ylim( -1, n )
        ax.set_ylabel( "germe (trie)" )
        ax.set_title( "$C \\subset C_S$ toujours -- largeur mediane $\\times$ %.1f" % np.median( r ),
                      fontsize = 10 )
        ax.legend( fontsize = 9, loc = "lower right" )

    fig.tight_layout()
    fig.savefig( path, bbox_inches = "tight" )
    plt.close( fig )
    print( "    -> %s" % path, flush = True )


def fig_gain( path, cas, rates ):
    import matplotlib
    matplotlib.use( "Agg" )
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots( 1, 2, figsize = ( 14.5, 5.4 ), dpi = 140 )
    coul = { "voronoi": "tab:blue", "aires egales": "tab:red" }

    for nom, p, w in cas:
        n = len( p )
        lo, hi = laguerre_1d( p, w )
        wid = hi - lo
        vif = wid > 0
        med, q9 = [], []
        for r in rates:
            sub = np.arange( 0, n, max( 1, int( round( 1 / r ) ) ) )
            Lo, Hi = enclos( p, w, sub )
            assert ( Lo <= lo + 1e-12 ).all() and ( Hi >= hi - 1e-12 ).all()
            ratio = ( Hi - Lo )[ vif ] / wid[ vif ]
            med.append( np.median( ratio ) )
            q9.append( np.quantile( ratio, 0.9 ) )
            if abs( r - 1 / 16 ) < 1e-12:
                axes[ 0 ].hist( np.log10( ratio ), bins = 60, alpha = 0.55,
                                color = coul[ nom ], label = "%s (mediane %.1f)" % ( nom, med[ -1 ] ) )
        c = coul[ nom ]
        axes[ 1 ].plot( 1 / np.array( rates ), med, "o-", color = c, label = "%s -- mediane" % nom )
        axes[ 1 ].plot( 1 / np.array( rates ), q9, "^--", color = c, alpha = 0.5,
                        label = "%s -- decile 9" % nom )

    ax = axes[ 0 ]
    ax.axvline( 0, color = "black", lw = 1 )
    ax.set_xlabel( r"$\log_{10}$ ( largeur de l'enclos / largeur de la vraie cellule ),  $|S| = n/16$" )
    ax.set_ylabel( "germes" )
    ax.set_title( "de combien l'enclos est trop large", fontsize = 10 )
    ax.legend( fontsize = 9 )

    ax = axes[ 1 ]
    f = np.array( [ 1 / r for r in rates ] )
    ax.plot( f, f, ":", color = "0.4", label = r"$n / |S|$ (la pente attendue en 1D)" )
    ax.set_xscale( "log", base = 2 ); ax.set_yscale( "log", base = 2 )
    ax.set_xlabel( r"$n / |S|$  (taux de sous-echantillonnage)" )
    ax.set_ylabel( "largeur enclos / largeur vraie" )
    ax.set_title( "le reglage : ce qu'on paie en largeur pour ce qu'on economise en germes",
                  fontsize = 10 )
    ax.grid( True, which = "both", alpha = 0.25 )
    ax.legend( fontsize = 9 )

    fig.tight_layout()
    fig.savefig( path, bbox_inches = "tight" )
    plt.close( fig )
    print( "    -> %s" % path, flush = True )


# ------------------------------------------------------------------------------------- principal

def main():
    a = argparse.ArgumentParser( description = __doc__,
                                 formatter_class = argparse.RawDescriptionHelpFormatter )
    a.add_argument( "-n", type = int, default = 4096 // 16 )
    a.add_argument( "--small", type = int, default = 48, help = "le n de la figure de principe" )
    a.add_argument( "--amas", type = int, default = 4 )
    a.add_argument( "--sigma", type = float, default = 0.012 )
    a.add_argument( "--seed", type = int, default = 0 )
    a.add_argument( "--out", default = os.path.dirname( os.path.abspath( __file__ ) ) )
    a = a.parse_args()

    def deux( n, sigma ):
        p = cloud_1d( n, a.amas, sigma, np.random.default_rng( a.seed ) )
        return [ ( "voronoi", p, np.zeros( n ) ), ( "aires egales", p, weights_equal( p ) ) ]

    print( "figure de principe (n = %d)" % a.small, flush = True )
    fig_principe( os.path.join( a.out, "sub1d_principe.png" ), deux( a.small, 0.03 ), 1 / 8 )

    print( "figure de gain (n = %d)" % a.n, flush = True )
    cas = deux( a.n, a.sigma )
    rates = [ 1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 32, 1 / 64, 1 / 128 ]
    fig_gain( os.path.join( a.out, "sub1d_gain.png" ), cas, rates )

    # STRATIFIE contre ALEATOIRE. Ce n'est pas une nuance de gout : en 2D le sous-ensemble
    # stratifie est gratuit (un germe par sous-arbre du Bsp, a une profondeur fixee) alors que le
    # tirage aleatoire est ce qu'on ferait sans y penser. Le tableau dit lequel des deux vaut la
    # peine -- et l'ecart se lit sur la QUEUE, pas sur la mediane.
    rng = np.random.default_rng( a.seed + 1 )
    print( "\n  largeur de l'enclos / largeur de la vraie cellule  (n = %d)" % a.n )
    for nom, p, w in cas:
        lo, hi = laguerre_1d( p, w )
        wid = hi - lo
        vif = wid > 0
        print( "  --- %s : %d germes vivants, largeur mediane %.3e (domaine / vraie = %.0f)"
               % ( nom, vif.sum(), np.median( wid[ vif ] ), 1.0 / np.median( wid[ vif ] ) ) )
        print( "  %-7s %23s %23s" % ( "", "STRATIFIE (1 par paquet)", "ALEATOIRE" ) )
        print( "  %-7s %7s %7s %7s   %7s %7s %7s"
               % ( "n/|S|", "median", "dec. 9", "max", "median", "dec. 9", "max" ) )
        for r in rates:
            k = max( 1, int( round( 1 / r ) ) )
            out = []
            for sub in ( np.arange( 0, a.n, k ),
                         np.sort( rng.choice( a.n, size = a.n // k, replace = False ) ) ):
                Lo, Hi = enclos( p, w, sub )
                assert ( Lo <= lo + 1e-12 ).all() and ( Hi >= hi - 1e-12 ).all()
                q = ( Hi - Lo )[ vif ] / wid[ vif ]
                out += [ np.median( q ), np.quantile( q, 0.9 ), q.max() ]
            print( "  %-7d %7.1f %7.1f %7.1f   %7.1f %7.1f %7.1f" % ( k, *out ) )
    return 0


if __name__ == "__main__":
    sys.exit( main() )

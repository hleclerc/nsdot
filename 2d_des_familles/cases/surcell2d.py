#!/usr/bin/env python
"""LES SUR-CELLULES D'AGREGAT, EN 2D : de combien elles debordent.

Un agregat `A` est un paquet de germes (Voronoi grossier, RESSERRE sur ses membres : on prend
l'enveloppe convexe `K` des germes qui lui sont assignes, pas la cellule grossiere). Il est
represente CONTINUMENT par `K` et par un MAJORANT AFFINE de ses poids `w( p ) = a.p + b`, ce qui
donne le germe-polytope

    h_A( x ) = min_{ p in K } ( |x - p|^2 - a.p - b )
             = d( x + a/2, K )^2 - a.x - |a|^2/4 - b

Comme `w( p_i ) >= w_i` pour tout membre, `h_A <= h_i` partout, donc la SUR-CELLULE

    C_A = { x : h_A( x ) <= psi( x ) }          psi = min_j h_j, l'enveloppe VRAIE

contient la reunion des vraies cellules des membres de `A`. C'est exactement ce qu'on trace.

DEUX PRECAUTIONS DE LECTURE :

  * `psi` est calcule sur TOUS les germes. C'est donc la sur-cellule la PLUS SERREE qui soit --
    une selection de points en face, comme un pavage grossier, ne peut que l'AGRANDIR. Ce qu'on
    voit est un plancher, pas une estimation.
  * les bords sont courbes et rentrants : `h_A - h_j` est concave, donc `{ h_A <= h_j }` est le
    complementaire d'un convexe. La sur-cellule n'est pas convexe, et l'echantillonnage le montre.
"""

import argparse, os, sys
import numpy as np
import matplotlib
matplotlib.use( "Agg" )
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

# ------------------------------------------------------------------ les nuages

def charge( path ):
    xs, ys, ws, n = [], [], [], None
    with open( path ) as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith( "#" ):
                continue
            t = ln.split()
            if n is None and len( t ) == 1:
                n = int( t[ 0 ] ); continue
            xs.append( float( t[ 0 ] ) ); ys.append( float( t[ 1 ] ) ); ws.append( float( t[ 2 ] ) )
    return np.array( [ xs, ys ] ).T, np.array( ws )

def uniforme( n, graine, wscale ):
    rng = np.random.default_rng( graine )
    P = rng.random( ( n, 2 ) )
    h2 = 1.0 / n
    W = wscale * h2 * rng.random( n )
    return P, W

# ------------------------------------------------ l'ordre spatial et les agregats

def ordre_median( P ):
    """Une descente BSP a coupes medianes : rend une permutation ou les voisins sont proches.
    C'est l'ordre que `AaBsp` fabrique, et dont `FrontPd` prend un germe sur `rho`."""
    ordre = np.empty( len( P ), dtype = np.int64 )
    pile, at = [ np.arange( len( P ) ) ], 0
    while pile:
        idx = pile.pop()
        if len( idx ) <= 8:
            ordre[ at : at + len( idx ) ] = idx; at += len( idx ); continue
        Q = P[ idx ]
        ax = int( np.argmax( Q.max( 0 ) - Q.min( 0 ) ) )
        m = len( idx ) // 2
        k = np.argpartition( Q[ :, ax ], m )
        pile.append( idx[ k[ m : ] ] ); pile.append( idx[ k[ : m ] ] )
    return ordre

def agregats( P, S ):
    """Voronoi grossier : un germe sur S sert de site, chaque germe va au site le plus proche.
    Les sites sont de VRAIS germes : ils serviront aussi de representants cote monde."""
    o = ordre_median( P )
    sites = o[ :: S ]
    C = P[ sites ]
    lab = np.empty( len( P ), dtype = np.int64 )
    pas = 4096
    for i in range( 0, len( P ), pas ):
        d = ( ( P[ i : i + pas, None, : ] - C[ None, :, : ] ) ** 2 ).sum( -1 )
        lab[ i : i + pas ] = d.argmin( 1 )
    return lab, sites

# ------------------------------------------------------- le germe-polytope continu

def majorant_affine( Q, w ):
    """`a, b` tels que `a.p_i + b >= w_i` pour tout membre. Moindres carres, puis on releve."""
    if len( Q ) < 3:
        return np.zeros( 2 ), float( w.max() )
    M = np.hstack( [ Q, np.ones( ( len( Q ), 1 ) ) ] )
    sol, *_ = np.linalg.lstsq( M, w, rcond = None )
    a, b = sol[ :2 ], sol[ 2 ]
    return a, float( b + np.max( w - ( Q @ a + b ) ) )

def dist2_polygone( X, V ):
    """Distance au carre d'un nuage de points a un polygone convexe donne en sommets CCW."""
    dedans = np.ones( len( X ), dtype = bool )
    best = np.full( len( X ), np.inf )
    for i in range( len( V ) ):
        A = V[ i ]; e = V[ ( i + 1 ) % len( V ) ] - A
        d = X - A
        dedans &= ( e[ 0 ] * d[ :, 1 ] - e[ 1 ] * d[ :, 0 ] ) >= 0
        ee = e @ e
        t = np.clip( ( d @ e ) / ee, 0, 1 ) if ee > 0 else np.zeros( len( X ) )
        pr = d - t[ :, None ] * e
        best = np.minimum( best, np.einsum( "ij,ij->i", pr, pr ) )
    return np.where( dedans, 0.0, best )

def h_agregat( X, Q, w ):
    """`h_A` sur les points `X`, pour l'agregat de germes `Q` et poids `w`."""
    a, b = majorant_affine( Q, w )
    if len( Q ) >= 3:
        try:
            V = Q[ ConvexHull( Q ).vertices ]
        except Exception:
            V = Q
    else:
        V = Q
    if len( V ) < 3:                       # segment ou point : distance directe aux membres
        d2 = ( ( X[ :, None, : ] - V[ None ] ) ** 2 ).sum( -1 ).min( 1 )
    else:
        d2 = dist2_polygone( X + a / 2, V )
    return d2 - X @ a - ( a @ a ) / 4 - b

def psi_sel( X, P, W, sel, pas = 512 ):
    """L'enveloppe d'une SELECTION de germes reels -- ce que l'algorithme aurait vraiment, un
    representant par agregat. Elle MAJORE `psi`, donc elle AGRANDIT la sur-cellule."""
    return psi( X, P[ sel ], W[ sel ], pas )[ 0 ]

def psi( X, P, W, pas = 512 ):
    """L'enveloppe VRAIE `min_j ( |x-p_j|^2 - w_j )`, et le gagnant."""
    v = np.full( len( X ), np.inf ); g = np.zeros( len( X ), dtype = np.int64 )
    for j in range( 0, len( P ), pas ):
        d = ( ( X[ :, None, : ] - P[ None, j : j + pas, : ] ) ** 2 ).sum( -1 ) - W[ None, j : j + pas ]
        k = d.argmin( 1 ); m = d[ np.arange( len( X ) ), k ]
        pris = m < v
        v = np.where( pris, m, v ); g = np.where( pris, j + k, g )
    return v, g

# ------------------------------------------------------------------------ la mesure

def mesure( P, W, lab, ns, Xg, psig, psis, gag, nech, rng ):
    """Par agregat echantillonne, quatre rapports. AIRE = aire de la sur-cellule / aire de la
    reunion des vraies cellules des membres. CANDIDATS = nombre de VRAIES CELLULES que la
    sur-cellule rencontre / nombre de membres -- c'est ca, la longueur de la liste a couper.
    Chaque rapport est donne avec `psi` EXACT ( le plancher ) et avec `psi_S`, l'enveloppe des
    representants ( ce que l'algorithme aurait )."""
    res = { k : [] for k in ( "aire", "aireS", "cand", "candS" ) }
    ech = set( rng.permutation( ns )[ : nech ].tolist() )
    for c in sorted( ech ):
        m = np.flatnonzero( lab == c )
        if len( m ) < 2:
            continue
        hg = h_agregat( Xg, P[ m ], W[ m ] )
        vrai = np.isin( gag, m )
        a_vrai = int( vrai.sum() )
        if a_vrai == 0:
            continue
        for cle_a, cle_c, env in ( ( "aire", "cand", psig ), ( "aireS", "candS", psis ) ):
            dedans = hg <= env + 1e-12
            res[ cle_a ].append( int( dedans.sum() ) / a_vrai )
            res[ cle_c ].append( len( np.unique( gag[ dedans ] ) ) / len( m ) )
    return { k : np.array( v ) for k, v in res.items() }

# ------------------------------------------------------------------------ le dessin

def panneau( ax, P, W, S, nech, graine, fin = 420, gros = 200 ):
    lab, sites = agregats( P, S )
    lo = P.min( 0 ) - 0.02; hi = P.max( 0 ) + 0.02
    def grille( g ):
        xs = np.linspace( lo[ 0 ], hi[ 0 ], g ); ys = np.linspace( lo[ 1 ], hi[ 1 ], g )
        Xg, Yg = np.meshgrid( xs, ys )
        return np.column_stack( [ Xg.ravel(), Yg.ravel() ] ), xs, ys, g

    ns = len( sites )
    Xc, _, _, _ = grille( gros )
    psic, gagc = psi( Xc, P, W )
    psicS = psi_sel( Xc, P, W, sites )
    rng = np.random.default_rng( graine )
    r = mesure( P, W, lab, ns, Xc, psic, psicS, gagc, nech, rng )

    Xf, xs, ys, g = grille( fin )
    psif, gagf = psi( Xf, P, W )
    psifS = psi_sel( Xf, P, W, sites )

    ax.set_xlim( lo[ 0 ], hi[ 0 ] ); ax.set_ylim( lo[ 1 ], hi[ 1 ] ); ax.set_aspect( "equal" )
    ax.set_xticks( [] ); ax.set_yticks( [] )
    ax.scatter( P[ :, 0 ], P[ :, 1 ], s = 0.7, c = "0.75", linewidths = 0 )

    # quelques agregats, tires au hasard parmi ceux qui sont bien garnis
    tailles = np.bincount( lab, minlength = ns )
    cands = np.flatnonzero( tailles >= max( 2, S // 2 ) )
    choix = np.random.default_rng( graine + 1 ).permutation( cands )[ : 5 ]
    cols = [ "#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e" ]
    for c, col in zip( choix, cols ):
        m = np.flatnonzero( lab == c )
        vrai = np.isin( gagf, m ).reshape( g, g )
        ax.contourf( xs, ys, vrai.astype( float ), levels = [ 0.5, 1.5 ], colors = [ col ], alpha = 0.75 )
        hg = h_agregat( Xf, P[ m ], W[ m ] ).reshape( g, g )
        ax.contour( xs, ys, hg - psif.reshape( g, g ), levels = [ 0.0 ], colors = [ col ],
                    linewidths = 1.7 )
        ax.contour( xs, ys, hg - psifS.reshape( g, g ), levels = [ 0.0 ], colors = [ col ],
                    linewidths = 1.1, linestyles = "dashed" )
        ax.scatter( P[ m, 0 ], P[ m, 1 ], s = 3.5, c = col, linewidths = 0 )
    return r, ns

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument( "--n", type = int, default = 2000 )
    ap.add_argument( "--graine", type = int, default = 0 )
    ap.add_argument( "--nech", type = int, default = 40 )
    ap.add_argument( "--out", default = "cases/surcell2d.png" )
    a = ap.parse_args()

    ici = os.path.dirname( os.path.abspath( __file__ ) )
    Pu, Wu = uniforme( a.n, a.graine, 0.0 )
    Pw, Ww = uniforme( a.n, a.graine, 1.0 )
    Pe, We = charge( os.path.join( ici, "lines5_n2000_s0.005_equal.txt" ) )
    cas = [ ( "uniforme, Voronoi ( w = 0 )", Pu, Wu ),
            ( "uniforme, poids aleatoires", Pw, Ww ),
            ( "5 lignes, aires egales ( le nuage dur )", Pe, We ) ]
    esses = [ 4, 16, 64 ]

    fig, axs = plt.subplots( len( cas ), len( esses ), figsize = ( 4.1 * len( esses ), 4.35 * len( cas ) ) )
    print( f"{'cas':38s} {'S':>3s} {'|A|':>4s} | {'aire':>6s} {'aire p90':>8s} {'cand':>6s} {'cand p90':>8s}"
           f" | {'aireS':>6s} {'aireS p90':>9s} {'candS':>6s} {'candS p90':>9s}" )
    for i, ( nom, P, W ) in enumerate( cas ):
        for j, S in enumerate( esses ):
            ax = axs[ i ][ j ]
            r, ns = panneau( ax, P, W, S, a.nech, a.graine )
            q = lambda k: ( np.median( r[ k ] ), np.percentile( r[ k ], 90 ) )
            ( ma, pa ), ( mc, pc ) = q( "aire" ), q( "cand" )
            ( mA, pA ), ( mC, pC ) = q( "aireS" ), q( "candS" )
            print( f"{nom:38s} {S:3d} {len(P)//ns:4d} | {ma:6.2f} {pa:8.2f} {mc:6.2f} {pc:8.2f}"
                   f" | {mA:6.2f} {pA:9.2f} {mC:6.2f} {pC:9.2f}" )
            ax.set_title( f"S = {S}   aire x{ma:.2f} / x{mA:.2f}   candidats x{mc:.1f} / x{mC:.1f}",
                          fontsize = 9 )
            if j == 0:
                ax.set_ylabel( nom, fontsize = 10 )
    fig.suptitle( "SUR-CELLULES D'AGREGAT   |   plein : les vraies cellules des membres"
                  "   |   trait continu : sur-cellule contre psi EXACT   |   pointille : contre les"
                  " REPRESENTANTS ( un par agregat )", fontsize = 11 )
    fig.tight_layout( rect = [ 0, 0, 1, 0.975 ] )
    out = a.out if os.path.isabs( a.out ) else os.path.join( os.path.dirname( ici ), a.out )
    fig.savefig( out, dpi = 130 )
    print( "->", out )

if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""LE MULTI-ECHELLE EN 1D, pour se faire une intuition ( la version 2D est `xmake run multiechelle` ).

En 1D tout est exact et se dessine :
  * la cellule de Laguerre de `i` est un intervalle ; `i` a une cellule non vide ssi son point RELEVE
    `( p_i, p_i^2 - w_i )` est un sommet de l'enveloppe convexe INFERIEURE des points releves ;
  * la condition locale est exacte : avec `h_l = p_i - p_{i-1}`, `h_r = p_{i+1} - p_i` et `w_lin` la
    valeur en `p_i` de la corde de `w` entre les deux voisins,
        cellule non vide  <=>  w_i - w_lin >= -h_l h_r
    On appelle MARGE `m_i = 1 + ( w_i - w_lin ) / ( h_l h_r )` : Voronoi ( w = 0 ) est a `m = 1`,
    la cellule se vide en `m = 0`. Et un calcul de trois lignes donne `m_i = a_i / |Vor_i|` EXACTEMENT
    ( tant que les voisins de Laguerre sont les voisins consecutifs ) : la marge d'une cellule est son
    taux de compression par rapport a Voronoi. Une perturbation de `w` de courbure locale `kappa`
    deplace `m` de `-kappa / 2` ;
  * Newton est EXACT en un pas depuis tout depart admissible : `a( w )` est lineaire tant qu'aucune
    cellule ne se vide, et sur le segment vers la solution `a = ( 1 - t ) a_0 + t nu > 0`. En 1D la
    question du multi-echelle se reduit donc a l'ADMISSIBILITE de la prolongation -- ce qui est
    justement la moitie de l'histoire que la 2D ne laisse pas voir ( `README.md`, § 8 ) ;
  * l'extension harmonique du graphe de Voronoi ( `c_ij = 1 / 2h_ij` ) est l'interpolation LINEAIRE
    en `p` entre les representants.

Le cas : `n` germes, la moitie en amas gaussien de largeur `sigma` autour de 1/2, l'autre uniforme
sur [0,1] ; cible 1/n partout. Les cellules de l'amas doivent s'ETIRER d'un facteur `S ~ 1/(8 sigma)`
( marge grande ), celles du fond se COMPRIMER de moitie et plus ( marge petite ) : ce sont elles
qui se vident.

    python scripts/multiechelle_1d.py            # ecrit figures/multiechelle_1d_*.png
    python scripts/multiechelle_1d.py --sigma 0.002 -n 800 -R 8
"""
import argparse, os
import numpy as np
import matplotlib
matplotlib.use( "Agg" )
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# ------------------------------------------------------------------------------------ la geometrie

def enveloppe_inf( p, phi, L = 1e6 ):
    """les indices des sommets de l'enveloppe convexe INFERIEURE de `( p, phi )`, LE DOMAINE [0,1]
    COMPRIS : deux points virtuels tres loin, l'un a gauche a la hauteur du minimum ( pente 0 -- le
    plan du bord x = 0 ), l'autre a droite sur la pente 2 ( le bord x = 1 ). Un germe dont la
    cellule serait entierement hors de [0,1] n'est pas un sommet de cette enveloppe-la."""
    P = np.concatenate( [ [ -L ], p, [ 1 + L ] ] )
    F = np.concatenate( [ [ phi.min() ], phi, [ np.max( phi - 2 * p ) + 2 * ( 1 + L ) ] ] )
    hull = []
    for i in range( len( P ) ):
        while len( hull ) >= 2:
            j, k = hull[ -2 ], hull[ -1 ]
            if ( P[ k ] - P[ j ] ) * ( F[ i ] - F[ j ] ) - ( F[ k ] - F[ j ] ) * ( P[ i ] - P[ j ] ) <= 0:
                hull.pop()
            else:
                break
        hull.append( i )
    return np.array( hull[ 1:-1 ] ) - 1


def cellules( p, w ):
    """les longueurs des cellules ( 0 si vide ), les indices des sommets de l'enveloppe, les bornes."""
    n = len( p )
    hull = enveloppe_inf( p, p * p - w )
    x = ( p[ hull[ :-1 ] ] + p[ hull[ 1: ] ] ) / 2 + ( w[ hull[ :-1 ] ] - w[ hull[ 1: ] ] ) / ( 2 * ( p[ hull[ 1: ] ] - p[ hull[ :-1 ] ] ) )
    b = np.concatenate( [ [ 0.0 ], np.clip( x, 0, 1 ), [ 1.0 ] ] )
    a = np.zeros( n )
    a[ hull ] = np.maximum( b[ 1: ] - b[ :-1 ], 0 )
    return a, hull, b


def marge( p, w ):
    """`m_i` pour les germes interieurs ( nan aux deux bouts )."""
    m = np.full( len( p ), np.nan )
    hl, hr = p[ 1:-1 ] - p[ :-2 ], p[ 2: ] - p[ 1:-1 ]
    lam = hr / ( hl + hr )
    wlin = lam * w[ :-2 ] + ( 1 - lam ) * w[ 2: ]
    m[ 1:-1 ] = 1 + ( w[ 1:-1 ] - wlin ) / ( hl * hr )
    return m


def laplacien( p, hull ):
    """le laplacien du graphe de Laguerre ( les cellules non vides, consecutives ) : `c = 1 / 2h`."""
    n = len( p )
    L = np.zeros( ( n, n ) )
    for i, j in zip( hull[ :-1 ], hull[ 1: ] ):
        c = 1 / ( 2 * ( p[ j ] - p[ i ] ) )
        L[ i, i ] += c; L[ j, j ] += c; L[ i, j ] -= c; L[ j, i ] -= c
    for i in range( n ):
        if L[ i, i ] == 0: L[ i, i ] = 1
    return L


def newton( p, nu, w0, tol = 1e-7, maxit = 200 ):
    """Newton amorti ( KMT ) depuis `w0`, qui doit etre admissible. Rend w, iterations, diagrammes."""
    w = w0 - w0[ 0 ]
    a, hull, _ = cellules( p, w )
    assert a.min() > 0, "depart inadmissible"
    eps = 0.5 * min( nu.min(), a.min() )
    diag = 1
    for it in range( maxit ):
        r = a - nu
        if np.max( np.abs( r ) / nu ) <= tol:
            return w, it, diag
        L = laplacien( p, hull )
        d = np.zeros( len( p ) )
        d[ 1: ] = np.linalg.solve( L[ 1:, 1: ], -r[ 1: ] )
        t = 1.0
        while True:
            w2 = w + t * d
            a2, hull2, _ = cellules( p, w2 )
            diag += 1
            if a2.min() >= eps and np.linalg.norm( a2 - nu ) <= ( 1 - t / 2 ) * np.linalg.norm( r ):
                break
            t /= 2
            if t < 1e-12:
                return w, it, diag
        w, a, hull = w2, a2, hull2
    return w, maxit, diag


def jacobi( p, w, k ):
    """`k` balayages de Jacobi amorti sur le graphe de Voronoi ( les voisins consecutifs )."""
    w = w.copy()
    h = np.diff( p )
    c = 1 / ( 2 * h )
    for _ in range( k ):
        Lw = np.zeros_like( w ); D = np.zeros_like( w )
        Lw[ :-1 ] += c * ( w[ :-1 ] - w[ 1: ] ); Lw[ 1: ] += c * ( w[ 1: ] - w[ :-1 ] )
        D[ :-1 ] += c; D[ 1: ] += c
        w = w - ( 2 / 3 ) * Lw / D
    return w

# ------------------------------------------------------------------------------------ le cas

def nuage( n, sigma, graine ):
    rng = np.random.default_rng( graine )
    amas = np.clip( rng.normal( 0.5, sigma, n // 2 ), 0.02, 0.98 )
    fond = rng.uniform( 0.001, 0.999, n - n // 2 )
    p = np.sort( np.concatenate( [ amas, fond ] ) )
    return p, np.sort( amas )

# ------------------------------------------------------------------------------------ les prolongations

def paquets( n, R ):
    """paquets consecutifs de `R`, representant au milieu."""
    k = np.arange( n ) // R
    reps = np.array( [ int( np.mean( np.nonzero( k == q )[ 0 ] ) ) for q in range( k.max() + 1 ) ] )
    return k, reps


def prolonge( nom, p, pc, wc, k ):
    if nom == "copie":
        return wc[ k ]
    if nom == "harmonique":                                # = lineaire en p en 1D ( flux constant )
        return np.interp( p, pc, wc )
    if nom == "spline":
        return CubicSpline( pc, wc, bc_type = "natural" )( p )
    if nom == "ctransf":
        return np.max( wc[ None, : ] - ( p[ :, None ] - pc[ None, : ] ) ** 2, axis = 1 )
    raise ValueError( nom )


def admissible( p, w, nu, avor ):
    a, _, _ = cellules( p, w )
    return a.min() >= 0.5 * min( nu.min(), avor.min() ), int( np.sum( a <= 0 ) )

# ------------------------------------------------------------------------------------ les figures

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument( "-n", type = int, default = 400 )
    ap.add_argument( "--sigma", type = float, default = 0.005 )
    ap.add_argument( "-R", type = int, default = 8 )
    ap.add_argument( "--graine", type = int, default = 0 )
    ap.add_argument( "--out", default = os.path.join( os.path.dirname( os.path.abspath( __file__ ) ), "..", "figures" ) )
    o = ap.parse_args()
    os.makedirs( o.out, exist_ok = True )
    def sauve( fig, nom ):
        fn = os.path.join( o.out, "multiechelle_1d_%s.png" % nom )
        fig.savefig( fn, dpi = 130 ); plt.close( fig ); print( "->", fn )

    n, R = o.n, o.R
    p, amas = nuage( n, o.sigma, o.graine )
    nu = np.full( n, 1 / n )
    avor, _, bvor = cellules( p, np.zeros( n ) )
    S = ( 1 / n ) / np.median( avor[ ( p > 0.5 - o.sigma ) & ( p < 0.5 + o.sigma ) ] )
    w, its0, diag0 = newton( p, nu, np.zeros( n ) )
    a, hull, b = cellules( p, w )
    print( "n = %d, sigma = %g : etirement S = %.0f dans l'amas ; Newton depuis Voronoi : %d iterations, %d diagrammes, max|a-nu|/nu = %.1e"
           % ( n, o.sigma, S, its0, diag0, np.max( np.abs( a - nu ) / nu ) ) )
    zoom = ( 0.5 - 4 * o.sigma, 0.5 + 4 * o.sigma )
    dans = ( p > zoom[ 0 ] ) & ( p < zoom[ 1 ] )

    # ---- 1. le probleme : les cellules, les poids, le releve
    fig, ax = plt.subplots( 3, 2, figsize = ( 14, 10 ) )
    fig.suptitle( "1D : n = %d, la moitie en amas sigma = %g ( etirement S ~ %.0f ) -- Voronoi contre la solution" % ( n, o.sigma, S ) )
    for col, ( x0, x1 ) in enumerate( [ ( 0, 1 ), zoom ] ):
        A = ax[ 0, col ]
        for y, bb, nom in [ ( 1, bvor, "Voronoi" ), ( 0, b, "solution" ) ]:
            for q in range( len( bb ) - 1 ):
                A.plot( [ bb[ q ], bb[ q + 1 ] ], [ y, y ], lw = 6, color = "C0" if q % 2 else "C1", solid_capstyle = "butt" )
            A.text( x0, y + 0.15, nom )
        A.plot( p, np.full( n, -0.6 ), "|", color = "k", ms = 12 ); A.text( x0, -0.45, "germes" )
        A.set_xlim( x0, x1 ); A.set_ylim( -0.9, 1.5 ); A.set_yticks( [] )
        A.set_title( "les cellules" + ( " ( zoom sur l'amas )" if col else "" ) )
        A = ax[ 1, col ]
        A.plot( p, w, ".-", ms = 3, lw = 0.5, label = "w ( solution )" )
        A.set_xlim( x0, x1 ); A.set_title( "les poids de la solution" ); A.legend()
        if col: A.set_ylim( w[ dans ].min() - 1e-3, w[ dans ].max() + 1e-3 )
        A = ax[ 2, col ]
        A.plot( p, p * p, ".", ms = 3, color = "C0", label = "Voronoi : p^2 ( tous sommets )" )
        A.plot( p, p * p - w, ".", ms = 3, color = "C1", label = "solution : p^2 - w" )
        A.plot( p[ hull ], ( p * p - w )[ hull ], "-", lw = 0.6, color = "C1" )
        A.set_xlim( x0, x1 ); A.set_title( "les points releves et l'enveloppe inferieure" ); A.legend()
        if col:
            y = ( p * p - w )[ dans ]; A.set_ylim( y.min() - 1e-4, y.max() + 1e-4 )
    sauve( fig, "1_probleme" )

    # ---- 2. la marge, et la solution lissee
    m_sol = marge( p, w )
    fig, ax = plt.subplots( 2, 1, figsize = ( 14, 8 ) )
    fig.suptitle( "la MARGE m_i = 1 + ( w_i - w_lin ) / ( h_l h_r ) = a_i / |Vor_i| : Voronoi = 1, vide en 0" )
    A = ax[ 0 ]
    A.axhline( 1, ls = "--", color = "C0", label = "Voronoi" )
    A.semilogy( p, m_sol, ".", ms = 5, color = "C1", label = "solution : la marge m_i" )
    A.semilogy( p, a / avor, "+", ms = 5, color = "k", alpha = 0.5, label = "solution : a_i / |Vor_i| ( les memes points )" )
    A.axvspan( *zoom, color = "k", alpha = 0.05 )
    A.set_ylim( 1e-2, 1e2 ); A.legend( loc = "upper right" )
    A.set_title( "la solution : marge = compression. Grande dans l'amas ( S ~ %.0f ), PETITE pour les germes du fond, dont la cellule doit retrecir" % S )
    A = ax[ 1 ]
    lignes = []
    for k, c in [ ( 1, "C2" ), ( 2, "C3" ), ( 8, "C4" ) ]:
        wk = jacobi( p, w, k )
        mk = marge( p, wk ); ak, _, _ = cellules( p, wk )
        vides = int( np.sum( ak <= 0 ) )
        A.plot( p, mk, ".", ms = 4, color = c, label = "solution lissee par %d balayage(s) de Jacobi : %d cellules vides" % ( k, vides ) )
        A.plot( p[ ak <= 0 ], mk[ ak <= 0 ], "x", ms = 7, color = c )
        lignes.append( ( k, vides ) )
    A.plot( p, m_sol, ".", ms = 3, color = "C1", alpha = 0.6, label = "solution" )
    A.axhline( 0, color = "k", lw = 0.8 ); A.axhline( 1, color = "C0", ls = "--", lw = 0.8 )
    A.set_yscale( "symlog", linthresh = 1 ); A.set_ylim( -1e3, 1e3 ); A.legend( loc = "upper right", fontsize = 8 )
    A.set_title( "la solution LISSEE : le lissage deplace la courbure de w ( -78 dans l'amas, +1.4 au fond ) vers les voisins, et une courbure kappa coute kappa / 2 de marge" )
    sauve( fig, "2_marge" )
    print( "solution lissee : " + ", ".join( "%d balayage(s) -> %d vides" % kv for kv in lignes ) )

    # ---- 3. le multi-echelle : niveau grossier, prolongations
    k, reps = paquets( n, R )
    pc = p[ reps ]; nc = len( reps )
    nuc = np.array( [ np.sum( nu[ k == q ] ) for q in range( nc ) ] )
    wc, itsc, _ = newton( pc, nuc, np.zeros( nc ) )
    wc = wc - np.mean( wc - w[ reps ] )                   # la jauge : celle de la solution fine, pour les dessins
    print( "niveau grossier : %d representants ( R = %d ), Newton %d iterations" % ( nc, R, itsc ) )
    zoom3 = ( 0.5 - 8 * o.sigma, 0.5 + 8 * o.sigma )
    dans3 = ( p > zoom3[ 0 ] ) & ( p < zoom3[ 1 ] )
    noms = [ "copie", "harmonique", "spline", "ctransf" ]
    fig, ax = plt.subplots( 3, len( noms ), figsize = ( 5 * len( noms ), 11 ) )
    fig.suptitle( "les prolongations depuis %d representants ( R = %d ) -- deux lignes zoomees sur l'amas et ses flancs, la marge sur tout [0,1]" % ( nc, R ) )
    bilan = {}
    for col, nom in enumerate( noms ):
        wp = prolonge( nom, p, pc, wc, k )
        ap_, hp, _ = cellules( p, wp )
        mp = marge( p, wp )
        vides = int( np.sum( ap_ <= 0 ) )
        A = ax[ 0, col ]
        A.plot( p, w, "-", lw = 0.8, color = "C1", label = "solution fine" )
        A.plot( pc, wc, "o", ms = 5, color = "k", label = "representants" )
        A.plot( p, wp, ".", ms = 3, color = "C2", label = nom )
        A.set_title( "%s : %d cellules vides sur %d" % ( nom, vides, n ) )
        A.set_xlim( *zoom3 ); A.set_ylim( w[ dans3 ].min() - 2e-3, w[ dans3 ].max() + 2e-3 ); A.legend( fontsize = 8 )
        A = ax[ 1, col ]
        phi = p * p - wp
        A.plot( p, phi, ".", ms = 3, color = "C2" ); A.plot( p[ hp ], phi[ hp ], "-", lw = 0.6, color = "C2" )
        A.plot( pc, pc * pc - wc, "o", ms = 5, color = "k" )
        vid = ap_ <= 0
        A.plot( p[ vid ], phi[ vid ], "x", ms = 6, color = "C3", label = "vides ( au-dessus de l'enveloppe )" )
        y = phi[ dans3 ]; A.set_xlim( *zoom3 ); A.set_ylim( y.min() - 2e-4, y.max() + 2e-4 ); A.legend( fontsize = 8 )
        A.set_title( "releve p^2 - w et enveloppe" )
        A = ax[ 2, col ]
        A.plot( p, mp, ".", ms = 3, color = "C2", label = nom ); A.plot( p[ vid ], mp[ vid ], "x", ms = 6, color = "C3", label = "vides" )
        A.plot( p, m_sol, ".", ms = 2, color = "C1", alpha = 0.5, label = "solution" )
        A.axhline( 0, color = "k", lw = 0.8 ); A.axhline( 1, color = "C0", ls = "--", lw = 0.8 )
        A.set_yscale( "symlog", linthresh = 1 ); A.set_ylim( -1e6, 1e6 ); A.set_title( "marge" ); A.legend( fontsize = 8, loc = "lower right" )
        # le retrait t.w, puis Newton
        t = 1.0; essais = 0
        while True:
            ok, _ = admissible( p, t * wp, nu, avor ); essais += 1
            if ok: break
            t /= 2
        _, its, diag = newton( p, nu, t * wp )
        bilan[ nom ] = ( vides, t, its, diag )
    sauve( fig, "3_prolongations" )
    print( "Newton depuis Voronoi : %d iterations, %d diagrammes" % ( its0, diag0 ) )
    for nom, ( vides, t, its, diag ) in bilan.items():
        print( "  %-10s : %3d vides a t = 1 ; passe a t = %-8.3g ; Newton depuis la : %d iterations, %d diagrammes" % ( nom, vides, t, its, diag ) )

    # ---- 4. le retrait, et qui se vide
    fig, ax = plt.subplots( 1, 2, figsize = ( 14, 5 ) )
    ts = 2.0 ** -np.arange( 0, 14 )
    for nom in noms:
        wp = prolonge( nom, p, pc, wc, k )
        vides = [ int( np.sum( cellules( p, t * wp )[ 0 ] <= 0 ) ) for t in ts ]
        ax[ 0 ].semilogx( ts, vides, "o-", label = nom )
    ax[ 0 ].set_xlabel( "t ( w = t x prolongation )" ); ax[ 0 ].set_ylabel( "cellules vides" ); ax[ 0 ].legend()
    ax[ 0 ].set_title( "le retrait : cellules vides en t x w\n( des que c'est 0, Newton converge en UN pas -- en 1D )" )
    A = ax[ 1 ]
    est_rep = np.zeros( n, bool ); est_rep[ reps ] = True
    for nom, c in [ ( "harmonique", "C1" ), ( "spline", "C2" ) ]:
        mp = marge( p, prolonge( nom, p, pc, wc, k ) )
        A.plot( m_sol[ ~est_rep ], mp[ ~est_rep ], ".", ms = 4, color = c, label = "%s, germes libres" % nom )
        A.plot( m_sol[ est_rep ], mp[ est_rep ], "s", ms = 5, color = c, mfc = "none", label = "%s, representants" % nom )
    A.axhline( 0, color = "k", lw = 0.8 ); A.axhline( 1, color = "C0", ls = "--", lw = 0.8 )
    A.set_xscale( "log" ); A.set_yscale( "symlog", linthresh = 1 ); A.set_ylim( -1e4, 1e4 )
    A.set_xlabel( "marge de la SOLUTION ( = compression a_i / |Vor_i| )" ); A.set_ylabel( "marge de la PROLONGATION ( t = 1 )" )
    A.set_title( "qui se vide ( sous 0 ) : les representants de l'harmonique la ou w est convexe\n( compression < 1 ) -- le pli vaut ( H_c / h ) x la courbure ; la spline suit la solution" )
    A.legend( fontsize = 8, loc = "lower right" )
    sauve( fig, "4_retrait" )


if __name__ == "__main__":
    main()

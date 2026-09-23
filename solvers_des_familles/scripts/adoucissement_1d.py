#!/usr/bin/env python
"""ADOUCIR UNE PROLONGATION ( 1D ) : plutot que de l'aplatir vers Voronoi ( `t w` ), la rendre
admissible en touchant le moins possible a ce qu'elle sait.

Ce qui rend une `w` LISSE admissible, en 1D, tient en une ligne : la marge `m = 1 - w'' / 2`, donc
    admissible  <=>  w'' < 2  partout        ( 2 = la courbure de p^2 )
C'est une borne A SENS UNIQUE sur la courbure. La solution a `w'' = 2 ( 1 - S(p) )`, `S` la compression
locale ( petite au fond ), donc elle frole 2 la ou les cellules doivent retrecir, et une prolongation
qui se trompe de `2 S` sur la courbure y vide des cellules.

Cinq facons d'adoucir une prolongation de base ( la spline, l'harmonique ) -- les quatre premieres
sont des lissages a parametre croissant, la derniere une PROJECTION exacte :
  FILTRE      convolution gaussienne de largeur `lambda` ( passe-bas, sur une grille fine ) ;
  NOYAU-d     regression polynomiale LOCALE de degre d ( 0 = Nadaraya-Watson, 1, 2 = MLS ) des poids
              des representants, poids gaussiens de largeur `lambda` ;
  KRR         regression ridge a noyau gaussien ( largeur `lambda`, regularisation `alpha` ) : la
              « meilleure » approximation a noyaux au sens des moindres carres regularises ;
  LISSANTE    la spline de lissage ( `make_smoothing_spline`, parametre `lam` ) ;
  ENVELOPPE   la projection sur l'admissible a marge `eps` : `psi = ( 1 - eps ) p^2 - w`, son
              enveloppe convexe inferieure `H`, et `w <- ( 1 - eps ) p^2 - H( p_i )`. Ne touche QUE les
              germes dont la cellule est vide ( ou de marge < eps ), les remonte du minimum, en une
              passe, sans cascade -- c'est le relevement MINIMAL ( le rattrapage a `-psi( p_i )`
              exigeait `p_i` DANS sa cellule, bien plus fort, d'ou la cascade ).

Ce qu'on mesure : le nombre de cellules vides, et, pour le premier parametre admissible, le RESIDU du
depart `max |a - nu| / nu` et `|a - nu|_2` -- ce qui reste a faire a Newton ( en 1D il le fait en un
pas ; en 2D c'est ce qui compterait ). Voronoi est la reference.

    python scripts/adoucissement_1d.py           # figures/adoucissement_1d_*.png
"""
import argparse, os, sys
import numpy as np
import matplotlib
matplotlib.use( "Agg" )
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, make_smoothing_spline
from scipy.ndimage import gaussian_filter1d

sys.path.insert( 0, os.path.dirname( os.path.abspath( __file__ ) ) )
import multiechelle_1d as me

# ------------------------------------------------------------------------------------ les adoucissements

def filtre( p, wb, lam ):
    """convolution gaussienne de `wb` ( donnee aux germes ) de largeur `lam`, via une grille fine."""
    g = np.linspace( 0, 1, 8001 )
    wg = np.interp( g, p, wb )
    if lam > 0:
        wg = gaussian_filter1d( wg, lam / ( g[ 1 ] - g[ 0 ] ), mode = "nearest" )
    return np.interp( p, g, wg )


def noyau( p, pc, wc, lam, deg ):
    """regression polynomiale locale de degre `deg`, poids gaussiens de largeur `lam`."""
    w = np.empty( len( p ) )
    for i, x in enumerate( p ):
        u = ( pc - x ) / lam
        om = np.exp( -u * u / 2 )
        V = np.vander( u, deg + 1, increasing = True )
        A = ( V * om[ :, None ] ).T @ V
        b = ( V * om[ :, None ] ).T @ wc
        try:
            w[ i ] = np.linalg.solve( A + 1e-14 * np.trace( A ) * np.eye( deg + 1 ), b )[ 0 ]
        except np.linalg.LinAlgError:
            w[ i ] = wc[ np.argmin( np.abs( pc - x ) ) ]
    return w


def krr( p, pc, wc, lam, alpha ):
    """regression ridge a noyau gaussien."""
    K = np.exp( -( pc[ :, None ] - pc[ None, : ] ) ** 2 / ( 2 * lam * lam ) )
    c = np.linalg.solve( K + alpha * np.eye( len( pc ) ), wc )
    return np.exp( -( p[ :, None ] - pc[ None, : ] ) ** 2 / ( 2 * lam * lam ) ) @ c


def lissante( p, pc, wc, lam ):
    return make_smoothing_spline( pc, wc, lam = lam )( p )


def releve_minimal( p, w, eps, passes = 10, seules = None ):
    """LE RELEVEMENT MINIMAL, une cellule vide a la fois. Une cellule vide NAIT en un sommet du
    diagramme des autres ( en 1D : une borne entre deux cellules, ou un bout du domaine ), et le poids
    qui l'y fait naitre est
        w_i = min_v ( |p_i - v|^2 - psi( v ) ),   psi( v ) = min_j ( |v - p_j|^2 - w_j )
    -- la hauteur de l'enveloppe convexe inferieure des points releves en `p_i` ( `H( p ) = max_v
    [ 2 v p - v^2 + psi( v ) ]` ), domaine compris. On y ajoute `eps h_l h_r` : la cellule nait avec
    une marge `eps`. C'est la forme qui se porte en 2D ( les sommets des cellules voisines ). Elle
    peut faire du ping-pong a l'echelle de `eps` entre deux vides adjacentes : `passes` le borne.
    Rend `w` et le nombre de relevements."""
    w = w.copy()
    n = len( p )
    hl = np.empty( n ); hr = np.empty( n )
    hl[ 1: ] = np.diff( p ); hl[ 0 ] = hl[ 1 ]
    hr[ :-1 ] = np.diff( p ); hr[ -1 ] = hr[ -2 ]
    remontes = 0
    for _ in range( passes ):
        a, hull, b = me.cellules( p, w )
        vides = np.nonzero( a <= 0 )[ 0 ]
        if seules is not None:
            vides = np.array( [ i for i in vides if i in seules ], int )
        if len( vides ) == 0:
            break
        for i in vides:
            a, hull, b = me.cellules( p, w )
            if a[ i ] > 0:
                continue
            psi_v = np.min( ( b[ :, None ] - p[ None, hull ] ) ** 2 - w[ None, hull ], axis = 1 )
            w[ i ] = np.min( ( p[ i ] - b ) ** 2 - psi_v ) + eps * hl[ i ] * hr[ i ]
            remontes += 1
    return w, remontes


def enveloppe( p, w, eps ):
    """LA PROJECTION sur l'admissible a marge `eps`, d'un coup : `psi = ( 1 - eps ) p^2 - w`, son
    enveloppe convexe inferieure `H`, `w <- ( 1 - eps ) p^2 - H( p_i )`. Tout point au-dessus de
    l'enveloppe descend dessus, les autres ne bougent pas ; dans le releve vrai `p^2 - w = eps p^2 +
    H` est strictement convexe, donc TOUT germe est un sommet, de marge >= eps -- pas de cascade, pas
    de ping-pong. Reste le DOMAINE : les cellules nees hors de [0,1] ( les bouts ) sont ensuite
    relevees une a une par `releve_minimal`, qui connait les bords."""
    psi = ( 1 - eps ) * p * p - w
    hull = _hull_sans_domaine( p, psi )
    H = np.interp( p, p[ hull ], psi[ hull ] )
    w1 = ( 1 - eps ) * p * p - H
    a, _, _ = me.cellules( p, w1 )
    bords = set( np.nonzero( a <= 0 )[ 0 ] )
    w2, nb = releve_minimal( p, w1, eps, seules = bords )
    return w2, int( np.sum( w1 > w + 1e-15 ) ) + nb


def _hull_sans_domaine( p, phi ):
    hull = []
    for i in range( len( p ) ):
        while len( hull ) >= 2:
            j, k = hull[ -2 ], hull[ -1 ]
            if ( p[ k ] - p[ j ] ) * ( phi[ i ] - phi[ j ] ) - ( phi[ k ] - phi[ j ] ) * ( p[ i ] - p[ j ] ) <= 0:
                hull.pop()
            else:
                break
        hull.append( i )
    return np.array( hull )

# ------------------------------------------------------------------------------------ la mesure

def bilan( p, w, nu ):
    a, _, _ = me.cellules( p, w )
    r = a - nu
    return int( np.sum( a <= 0 ) ), np.max( np.abs( r ) / nu ), np.linalg.norm( r ) / np.linalg.norm( nu )


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
        fn = os.path.join( o.out, "adoucissement_1d_%s.png" % nom )
        fig.savefig( fn, dpi = 130 ); plt.close( fig ); print( "->", fn )

    n, R = o.n, o.R
    p, _ = me.nuage( n, o.sigma, o.graine )
    nu = np.full( n, 1 / n )
    avor, _, _ = me.cellules( p, np.zeros( n ) )
    w, _, _ = me.newton( p, nu, np.zeros( n ) )
    k, reps = me.paquets( n, R )
    pc = p[ reps ]
    nuc = np.array( [ np.sum( nu[ k == q ] ) for q in range( len( reps ) ) ] )
    wc, _, _ = me.newton( pc, nuc, np.zeros( len( reps ) ) )
    wc = wc - np.mean( wc - w[ reps ] )
    m_sol = me.marge( p, w )
    v0, r0max, r02 = bilan( p, np.zeros( n ), nu )
    print( "Voronoi : max|a-nu|/nu = %.2f, |a-nu|_2/|nu|_2 = %.3f" % ( r0max, r02 ) )
    bases = { "spline": me.prolonge( "spline", p, pc, wc, k ), "harmonique": me.prolonge( "harmonique", p, pc, wc, k ) }
    for nom, wb in bases.items():
        print( "base %-10s : %d vides, residu max %.2f, l2 %.3f" % ( ( nom, ) + bilan( p, wb, nu ) ) )

    # ---- les familles : ( nom, parametres, fonction( param ) -> w ), par base quand ca depend de la base
    lams = np.geomspace( 5e-4, 0.2, 14 )
    familles = {
        "filtre ( spline )":   ( lams, lambda l: filtre( p, bases[ "spline" ], l ), "largeur lambda" ),
        "filtre ( harmonique )": ( lams, lambda l: filtre( p, bases[ "harmonique" ], l ), "largeur lambda" ),
        "noyau degre 0":       ( lams, lambda l: noyau( p, pc, wc, l, 0 ), "largeur lambda" ),
        "noyau degre 1":       ( lams, lambda l: noyau( p, pc, wc, l, 1 ), "largeur lambda" ),
        "noyau degre 2":       ( lams, lambda l: noyau( p, pc, wc, l, 2 ), "largeur lambda" ),
        "krr alpha = 1e-8":    ( lams, lambda l: krr( p, pc, wc, l, 1e-8 ), "largeur lambda" ),
        "krr lambda = 0.02":   ( np.geomspace( 1e-10, 1e-1, 14 ), lambda a: krr( p, pc, wc, 0.02, a ), "alpha" ),
        "lissante":            ( np.geomspace( 1e-12, 1e-2, 14 ), lambda l: lissante( p, pc, wc, l ), "lam" ),
        "enveloppe ( spline )": ( np.array( [ 0.01, 0.03, 0.1, 0.3, 0.5, 0.7 ] ), lambda e: enveloppe( p, bases[ "spline" ], e )[ 0 ], "marge eps" ),
        "enveloppe ( harmonique )": ( np.array( [ 0.01, 0.03, 0.1, 0.3, 0.5, 0.7 ] ), lambda e: enveloppe( p, bases[ "harmonique" ], e )[ 0 ], "marge eps" ),
    }
    resultats = {}
    for nom, ( params, f, etiq ) in familles.items():
        vides, rmax, r2 = [], [], []
        for x in params:
            try:
                wx = f( x )
                v, rm, rr = bilan( p, wx, nu )
            except Exception as e:
                v, rm, rr = -1, np.nan, np.nan
            vides.append( v ); rmax.append( rm ); r2.append( rr )
        vides, rmax, r2 = np.array( vides ), np.array( rmax ), np.array( r2 )
        adm = np.nonzero( vides == 0 )[ 0 ]
        premier = adm[ 0 ] if len( adm ) else None
        resultats[ nom ] = ( params, vides, rmax, r2, premier, etiq )
        if premier is None:
            print( "%-26s : jamais admissible ( vides : %s )" % ( nom, " ".join( "%d" % v for v in vides ) ) )
        else:
            print( "%-26s : admissible des %s = %.3g ; residu max %.2f, l2 %.3f ( vides : %s )"
                   % ( nom, etiq, params[ premier ], rmax[ premier ], r2[ premier ], " ".join( "%d" % v for v in vides ) ) )

    # ---- figure 1 : residu du depart en fonction du parametre, admissible ou pas
    noms = list( familles )
    fig, ax = plt.subplots( 2, 5, figsize = ( 22, 8 ) )
    fig.suptitle( "adoucir une prolongation : le residu l2 du depart ( trait ) et les cellules vides ( pointilles ) selon le parametre -- rond plein = admissible. Voronoi en gris" )
    for q, nom in enumerate( noms ):
        A = ax[ q // 5, q % 5 ]
        params, vides, rmax, r2, premier, etiq = resultats[ nom ]
        A.semilogx( params, r2, "-", color = "C0" )
        ok = vides == 0
        A.semilogx( params[ ok ], r2[ ok ], "o", color = "C0", label = "residu l2 ( admissible )" )
        A.semilogx( params[ ~ok ], r2[ ~ok ], "o", mfc = "none", color = "C0", label = "residu l2 ( vides )" )
        A.axhline( r02, color = "gray", ls = "--", label = "Voronoi" )
        A.set_ylim( 0, max( r02 * 1.6, np.nanmax( r2 ) * 1.05 ) ); A.set_xlabel( etiq ); A.set_title( nom )
        B = A.twinx()
        B.semilogx( params, vides, ":", color = "C3" ); B.set_ylim( 0, max( 10, vides.max() * 1.1 ) )
        B.set_ylabel( "vides", color = "C3" )
        if q == 0: A.legend( fontsize = 8 )
    fig.tight_layout( rect = ( 0, 0, 1, 0.95 ) )
    sauve( fig, "1_residu" )

    # ---- figure 2 : les marges au premier parametre admissible, pour quelques familles
    choix = [ "filtre ( spline )", "noyau degre 2", "krr alpha = 1e-8", "lissante", "enveloppe ( spline )", "enveloppe ( harmonique )" ]
    fig, ax = plt.subplots( len( choix ), 1, figsize = ( 14, 3.2 * len( choix ) ), sharex = True )
    fig.suptitle( "les marges du premier depart admissible ( vert ), la base ( gris ), la solution ( orange )" )
    for A, nom in zip( ax, choix ):
        params, vides, rmax, r2, premier, etiq = resultats[ nom ]
        base = bases[ "harmonique" if "harmonique" in nom else "spline" ]
        mb = me.marge( p, base )
        A.plot( p, mb, ".", ms = 3, color = "gray", alpha = 0.6, label = "base" )
        A.plot( p, m_sol, ".", ms = 3, color = "C1", alpha = 0.6, label = "solution" )
        if premier is not None:
            wx = familles[ nom ][ 1 ]( params[ premier ] )
            mx = me.marge( p, wx )
            A.plot( p, mx, ".", ms = 4, color = "C2", label = "%s = %.3g : residu max %.2f, l2 %.3f" % ( etiq, params[ premier ], rmax[ premier ], r2[ premier ] ) )
        A.axhline( 0, color = "k", lw = 0.8 ); A.axhline( 1, color = "C0", ls = "--", lw = 0.8 )
        A.set_yscale( "symlog", linthresh = 1 ); A.set_ylim( -1e3, 1e3 ); A.set_title( nom, fontsize = 10 ); A.legend( fontsize = 8, loc = "lower right" )
    sauve( fig, "2_marges" )

    # ---- figure 3 : l'enveloppe, vue dans le releve, autour du representant le plus enfonce
    wb = bases[ "harmonique" ]
    eps = 0.03                                             # sous la plus petite compression de la solution ( ~0.1 ) :
    we, _ = enveloppe( p, wb, eps )                        # au-dela, la projection aplatit aussi le fond
    mb = me.marge( p, wb )
    kr = reps[ np.nanargmin( mb[ reps ] ) ]                    # le representant de pire marge
    lo, hi = max( kr - 2 * R, 0 ), min( kr + 2 * R + 1, n )
    sl = slice( lo, hi )
    fig, ax = plt.subplots( 1, 3, figsize = ( 20, 5.5 ) )
    fig.suptitle( "l'ENVELOPPE sur l'harmonique, marge eps = %.2f, autour du representant le plus enfonce ( p = %.3f, marge %.0f ) -- tout est trace MOINS LA CORDE entre les deux bouts de la fenetre" % ( eps, p[ kr ], mb[ kr ] ) )
    def moins_corde( y, ref = None ):                        # la meme corde pour toutes les series d'un panneau
        r = y if ref is None else ref
        return y - np.interp( p[ sl ], [ p[ lo ], p[ hi - 1 ] ], [ r[ 0 ], r[ -1 ] ] )
    psi_b = ( 1 - eps ) * p * p - wb
    psi_e = ( 1 - eps ) * p * p - we
    hull = _hull_sans_domaine( p, psi_b )
    A = ax[ 0 ]
    yb = moins_corde( psi_b[ sl ] ); ye = moins_corde( psi_e[ sl ], psi_b[ sl ] )
    A.plot( p[ sl ], yb, "o-", ms = 4, lw = 0.6, color = "gray", label = "psi de l'harmonique" )
    dans_hull = np.isin( np.arange( lo, hi ), hull )
    A.plot( p[ sl ][ dans_hull ], yb[ dans_hull ], "-", lw = 1.2, color = "C2", label = "enveloppe convexe inferieure" )
    A.plot( p[ sl ], ye, ".", ms = 7, color = "C3", label = "apres projection" )
    A.plot( [ p[ kr ] ], [ yb[ kr - lo ] ], "s", ms = 9, mfc = "none", color = "k", label = "le representant" )
    A.legend( fontsize = 8 ); A.set_title( "le releve psi = ( 1 - eps ) p^2 - w", fontsize = 10 )
    A = ax[ 1 ]
    A.plot( p[ sl ], moins_corde( w[ sl ] ), "-", lw = 1, color = "C1", label = "solution" )
    A.plot( p[ sl ], moins_corde( wb[ sl ], w[ sl ] ), "o-", ms = 4, lw = 0.6, color = "gray", label = "harmonique" )
    A.plot( p[ sl ], moins_corde( we[ sl ], w[ sl ] ), ".", ms = 7, color = "C3", label = "harmonique + enveloppe" )
    A.plot( [ p[ kr ] ], [ moins_corde( wb[ sl ], w[ sl ] )[ kr - lo ] ], "s", ms = 9, mfc = "none", color = "k" )
    A.legend( fontsize = 8 ); A.set_title( "les poids ( moins la corde de la solution )", fontsize = 10 )
    A = ax[ 2 ]
    touche = we > wb + 1e-15
    A.semilogy( p[ touche ], ( we - wb )[ touche ] / ( np.gradient( p )[ touche ] ** 2 ), ".", ms = 5, color = "C3", label = "remontee / h^2 des %d germes touches" % int( touche.sum() ) )
    A.set_xlabel( "p" ); A.legend( fontsize = 8 ); A.set_title( "de combien on remonte ( en h^2 )", fontsize = 10 )
    sauve( fig, "3_enveloppe" )

    # ---- la borne : la solution lissee par un balayage, puis l'enveloppe
    ws = me.jacobi( p, w, 1 )
    print( "solution lissee ( 1 balayage )       : %d vides, residu max %.2f, l2 %.3f" % bilan( p, ws, nu ) )
    for eps in [ 0.03, 0.1, 0.3 ]:
        print( "  + enveloppe eps = %.2f                : %d vides, residu max %.2f, l2 %.3f" % ( ( eps, ) + bilan( p, enveloppe( p, ws, eps )[ 0 ], nu ) ) )


if __name__ == "__main__":
    main()

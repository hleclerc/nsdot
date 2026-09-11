#!/usr/bin/env python
"""LES COUPES GRISES : la sur-cellule exacte comme INTERSECTION DE DEMI-ESPACES COURBES.

Un dirac contre le reste du monde donne une coupe FRANCHE : un demi-plan. Un agregat represente
CONTINUMENT par `K` donne une coupe GRISE -- la famille des demi-plans `H_p = { l_p >= l_j }` pour
`p` parcourant `K`. La coupe qui coupe le MOINS est leur REUNION,

    R_j = U_{p in K} H_p = { x : Phi_K( x ) >= l_j( x ) }      Phi_K = max_{p in K} l_p

et la sur-cellule est `S_A = inter_j R_j`. C'est le meme objet que `surcell2d_formes.py` mesurait
par sa fonction de distance ; ici on regarde sa STRUCTURE : de quoi son bord est fait, combien il a
d'arcs, s'il a des trous, et ce que couterait de le construire coupe par coupe.

= LES TROIS REGIMES DU BORD, ET ILS SONT FERMES

`Phi_K` est le paraboloide TRONQUE a `K` et prolonge par ses PLANS TANGENTS. L'eventail normal de
`K` decoupe le plan, et dans chaque secteur `dPhi_K = l_j` est une conique :

  * dans `K`            : `Phi_K = |x|^2`          -> `|x - p_j|^2 = w_j - b`   un CERCLE
  * bande d'une ARETE   : `Phi_K = |x|^2 - s^2`    -> `( t - t_j )^2 = 2 s_j s + ...`  une PARABOLE
  * cone d'un SOMMET v  : `Phi_K = l_v`            -> une DROITE ( la mediatrice de puissance )

( `s` = distance a la droite de l'arete, `t` = abscisse le long. ) Les morceaux se raccordent en
`C^1`, le gradient de `Phi_K` valant `2 proj_K( x )`.

= POURQUOI CA NE PEUT PAS EXPLOSER

`C_j = { Phi_K < l_j }` est CONVEXE ( sous-niveau d'une fonction convexe ). Deux bords `dC_j` et
`dC_k` se croisent dans `{ l_j = l_k }`, une DROITE, ou `Phi_K - l_j` est convexe donc a au plus
DEUX zeros. Les `C_j` sont donc des pseudo-disques convexes, et la reunion de `n` pseudo-disques a
au plus `6n - 12` arcs. Ce script mesure la constante reelle.

= CE QUI EST COMPTE

  coupes   : germes distincts qui apparaissent sur le bord -- les coupes EFFECTIVES.
  arcs     : morceaux de bord a germe constant. Le majorant theorique est `6 x coupes`.
  cand     : germes dont la vraie cellule rencontre la sur-cellule -- la liste a PARCOURIR.
  morceaux : composantes connexes de la sur-cellule.   trous : composantes de son complementaire
             qui ne touchent pas le bord de la fenetre -- ce qu'un lacet ne sait pas representer.
  kaire kcand : les MEMES deux mesures sur le k-DOP a 8 directions de la sur-cellule. C'est lui
             que l'algorithme range et intersecte ( `AaBspHull` en garde huit appuis ), donc c'est
             lui qui decide -- la region courbe n'est qu'une etape pour l'obtenir.

Tout est lu sur une grille : les arcs plus courts qu'un pixel sont invisibles. `--res` compare deux
resolutions, c'est la seule facon honnete de dire si le compte tient.
"""

import argparse, os, sys
import numpy as np
from scipy.ndimage import label, binary_erosion, binary_dilation

sys.path.insert( 0, os.path.dirname( os.path.abspath( __file__ ) ) )
from surcell2d import charge, uniforme, agregats, majorant_affine
from surcell2d_formes import psi_kd, Enveloppe, KDop, Boite, ENCRE, ENCRE2, FOND

C8 = np.ones( ( 3, 3 ), bool )

# --------------------------------------------------------- les trois regimes, par point

def regime( Y, V ):
    """0 = dans `K` ( cercle ), 1 = bande d'arete ( parabole ), 2 = cone de sommet ( droite ).
    Lu sur la projection : `t` strictement dans `] 0, 1 [` -> arete, sinon sommet."""
    dedans = np.ones( len( Y ), bool )
    best = np.full( len( Y ), np.inf )
    inte = np.zeros( len( Y ), bool )
    for i in range( len( V ) ):
        A = V[ i ]; e = V[ ( i + 1 ) % len( V ) ] - A
        d = Y - A
        dedans &= ( e[ 0 ] * d[ :, 1 ] - e[ 1 ] * d[ :, 0 ] ) >= 0
        ee = float( e @ e )
        t = np.clip( ( d @ e ) / ee, 0, 1 ) if ee > 0 else np.zeros( len( Y ) )
        pr = d - t[ :, None ] * e
        d2 = np.einsum( "ij,ij->i", pr, pr )
        pris = d2 < best
        best = np.where( pris, d2, best )
        inte = np.where( pris, ( t > 1e-9 ) & ( t < 1 - 1e-9 ), inte )
    return np.where( dedans, 0, np.where( inte, 1, 2 ) )

# ------------------------------------------------------------------ la structure du bord

def structure( R, gag, g ):
    """`R` le masque de la sur-cellule, `gag` le gagnant de `psi`. Sur le bord de `R`, la
    contrainte ACTIVE est exactement `gag` -- le bord est ou `Phi_K = max_j l_j`, donc le `j` qui
    realise le max est celui qui realise le min de `h_j`, c'est-a-dire le gagnant. Rien a
    recalculer."""
    Rm = R.reshape( g, g )
    interne = np.zeros( ( g, g ), bool ); interne[ 1 : -1, 1 : -1 ] = True
    bord = Rm & ~binary_erosion( Rm, C8 ) & interne
    G = gag.reshape( g, g )
    js = np.unique( G[ bord ] )
    arcs = 0
    for j in js:
        _, k = label( bord & ( G == j ), structure = C8 )
        arcs += k
    morceaux = label( Rm, structure = C8 )[ 1 ]
    lab, k = label( ~Rm, structure = C8 )
    dehors = set( lab[ 0 ].tolist() ) | set( lab[ -1 ].tolist() ) \
           | set( lab[ :, 0 ].tolist() ) | set( lab[ :, -1 ].tolist() )
    trous = sum( 1 for c in range( 1, k + 1 ) if c not in dehors )
    return len( js ), arcs, morceaux, trous, bord

# ------------------------------------------------------------------------- la campagne

def surcellule( Xw, psw, m, P, W, cls ):
    a, b = majorant_affine( P[ m ], W[ m ] )
    K = cls( P[ m ] )
    f = K.d2( Xw + a / 2 ) - Xw @ a - ( a @ a ) / 4 - b
    return f <= psw + 1e-12, K, a, b

def campagne( nom, P, W, S, g, nech, graine, cls = Enveloppe, verbose = True ):
    lo, hi = P.min( 0 ) - 0.02, P.max( 0 ) + 0.02
    xs = np.linspace( lo[ 0 ], hi[ 0 ], g ); ys = np.linspace( lo[ 1 ], hi[ 1 ], g )
    Ag, Bg = np.meshgrid( xs, ys )
    X = np.column_stack( [ Ag.ravel(), Bg.ravel() ] )
    psw, gag = psi_kd( P, W )( X )

    lab_, sites = agregats( P, S )
    na = len( sites )
    tailles = np.bincount( lab_, minlength = na )
    ok = np.flatnonzero( tailles >= max( 2, S // 2 ) )
    ech = np.random.default_rng( graine ).permutation( ok )[ : nech ]

    U8 = np.column_stack( [ np.cos( np.arange( 8 ) * np.pi / 4 ),
                            np.sin( np.arange( 8 ) * np.pi / 4 ) ] )
    XU = X @ U8.T
    acc = { k : [] for k in ( "aire", "coupes", "arcs", "ratio", "cand", "morceaux", "trous",
                              "kaire", "kcand" ) }
    for c in ech:
        m = np.flatnonzero( lab_ == c )
        U = np.isin( gag, m )
        if U.sum() < 4: continue
        R, K, a, b = surcellule( X, psw, m, P, W, cls )
        nj, arcs, mor, tr, _ = structure( R, gag, g )
        acc[ "aire" ].append( R.sum() / U.sum() )
        acc[ "coupes" ].append( nj )
        acc[ "arcs" ].append( arcs )
        acc[ "ratio" ].append( arcs / max( nj, 1 ) )
        acc[ "cand" ].append( len( np.unique( gag[ R ] ) ) )
        acc[ "morceaux" ].append( mor )
        acc[ "trous" ].append( tr )
        D = ( XU <= XU[ R ].max( 0 ) + 1e-12 ).all( 1 )          # le k-DOP de la sur-cellule
        acc[ "kaire" ].append( D.sum() / U.sum() )
        acc[ "kcand" ].append( len( np.unique( gag[ D ] ) ) )
    r = { k : np.array( v, float ) for k, v in acc.items() }
    if verbose:
        q = lambda k: ( np.median( r[ k ] ), np.percentile( r[ k ], 90 ), r[ k ].max() )
        print( f"\n--- {nom}   S = {S}   |A| median {int( np.median( tailles[ ok ] ) )}"
               f"   {len( ech )} agregats   grille {g}x{g}   K = {cls.nom}" )
        print( f"    {'':10s} {'med':>7s} {'p90':>7s} {'max':>7s}" )
        for k in ( "aire", "kaire", "cand", "kcand", "coupes", "arcs", "ratio",
                   "morceaux", "trous" ):
            a_, b_, c_ = q( k )
            print( f"    {k:10s} {a_:7.2f} {b_:7.2f} {c_:7.2f}" )
    return r

# --------------------------------------------------------------------------- le dessin

import matplotlib
matplotlib.use( "Agg" )
import matplotlib.pyplot as plt

REG = [ ( "cercle ( dans K )", "#0072B2" ), ( "parabole ( bande d'arete )", "#D55E00" ),
        ( "droite ( cone de sommet )", "#009E73" ) ]

def cadre( ax, xs, ys ):
    ax.set_xlim( xs[ 0 ], xs[ -1 ] ); ax.set_ylim( ys[ 0 ], ys[ -1 ] )
    ax.set_aspect( "equal" ); ax.set_xticks( [] ); ax.set_yticks( [] )
    ax.set_facecolor( FOND )
    for s in ax.spines.values(): s.set_color( "#dddddd" )

def scene( P, W, S, g, graine, cls = Enveloppe, carre = False ):
    """Un agregat ( le plus allonge ), sa fenetre, sa sur-cellule et le voisin au plus long arc."""
    lo, hi = P.min( 0 ) - 0.02, P.max( 0 ) + 0.02
    xg = np.linspace( lo[ 0 ], hi[ 0 ], 220 ); yg = np.linspace( lo[ 1 ], hi[ 1 ], 220 )
    A0, B0 = np.meshgrid( xg, yg )
    X0 = np.column_stack( [ A0.ravel(), B0.ravel() ] )
    ev = psi_kd( P, W )
    lab_, sites = agregats( P, S )
    _, g0 = ev( X0 )

    best, m = -1.0, None
    for c in range( len( sites ) ):
        mm = np.flatnonzero( lab_ == c )
        if len( mm ) < max( 3, S // 2 ): continue
        sv = np.linalg.svd( P[ mm ] - P[ mm ].mean( 0 ), compute_uv = False )
        r = sv[ 0 ] / max( sv[ 1 ], 1e-12 )
        if r > best: best, m = r, mm

    pts = X0[ np.isin( g0, m ) ]
    pts = np.vstack( [ pts, P[ m ] ] ) if len( pts ) else P[ m ]
    ce = 0.5 * ( pts.min( 0 ) + pts.max( 0 ) )
    de = np.maximum( np.ptp( pts, axis = 0 ), 1e-6 )
    de = 0.80 * ( np.full( 2, de.max() ) if carre else de )
    if carre:
        # on RECENTRE plutot que de rogner : sinon une fenetre carree redevient rectangulaire
        # des que l'agregat touche le bord du domaine, et les panneaux ne s'alignent plus. Le
        # demi-cote est borne par la moitie du PLUS PETIT cote du domaine, sinon le recentrage
        # n'a pas de solution et le cadre repart de travers.
        de = np.full( 2, min( de.max(), 0.5 * float( ( hi - lo ).min() ) ) )
        ce = np.minimum( np.maximum( ce, lo + de ), hi - de )
    lo = np.maximum( ce - de, lo ); hi = np.minimum( ce + de, hi )
    xs = np.linspace( lo[ 0 ], hi[ 0 ], g ); ys = np.linspace( lo[ 1 ], hi[ 1 ], g )
    Ag, Bg = np.meshgrid( xs, ys )
    X = np.column_stack( [ Ag.ravel(), Bg.ravel() ] )
    psw, gag = ev( X )
    R, K, a, b = surcellule( X, psw, m, P, W, cls )
    nj, arcs, mor, tr, bord = structure( R, gag, g )
    V = K.V if hasattr( K, "V" ) else P[ m ]
    reg = regime( X + a / 2, V )

    # le voisin au plus long arc : c'est lui qu'on isole pour montrer l'etalement
    hh = gag[ bord.ravel() ]
    vals, cnt = np.unique( hh, return_counts = True )
    cV = P[ m ].mean( 0 ); rV = np.linalg.norm( P[ m ] - cV, axis = 1 ).max()
    loin = np.linalg.norm( P[ vals ] - cV, axis = 1 ) > 1.2 * rV
    if loin.any(): vals, cnt = vals[ loin ], cnt[ loin ]
    j = int( vals[ np.argmax( cnt ) ] )
    return dict( P = P, W = W, m = m, xs = xs, ys = ys, g = g, X = X, psw = psw, gag = gag,
                 R = R, K = K, V = V, a = a, b = b, reg = reg, bord = bord, j = j,
                 nj = nj, arcs = arcs, mor = mor, tr = tr, aniso = best )

def trace_bord( ax, d, masque = None ):
    bo = d[ "bord" ].ravel() if masque is None else masque.ravel()
    for k, ( nom, coul ) in enumerate( REG ):
        s = bo & ( d[ "reg" ] == k )
        if s.any():
            ax.scatter( d[ "X" ][ s, 0 ], d[ "X" ][ s, 1 ], s = 2.2, c = coul, linewidths = 0,
                        zorder = 5 )

def figure_grise( P, W, S, g, graine, out, nom ):
    d = scene( P, W, S, g, graine, carre = True )
    X, xs, ys, gg, m, a, b = d[ "X" ], d[ "xs" ], d[ "ys" ], d[ "g" ], d[ "m" ], d[ "a" ], d[ "b" ]
    P, W, V, j = d[ "P" ], d[ "W" ], d[ "V" ], d[ "j" ]
    q = P[ m ].mean( 0 )
    ce = np.array( [ 0.5 * ( xs[ 0 ] + xs[ -1 ] ), 0.5 * ( ys[ 0 ] + ys[ -1 ] ) ] )
    de = 1.15 * max( xs[ -1 ] - xs[ 0 ], ys[ -1 ] - ys[ 0 ] )
    xl = np.linspace( ce[ 0 ] - de, ce[ 0 ] + de, gg )
    yl = np.linspace( ce[ 1 ] - de, ce[ 1 ] + de, gg )
    Al, Bl = np.meshgrid( xl, yl )
    XL = np.column_stack( [ Al.ravel(), Bl.ravel() ] )
    YL = XL + a / 2
    hj = ( ( XL - P[ j ] ) ** 2 ).sum( 1 ) - W[ j ]
    dec = - XL @ a - ( a @ a ) / 4 - b
    regL = regime( YL, V )
    dL = dict( X = XL, reg = regL )

    fig, axs = plt.subplots( 1, 4, figsize = ( 17.5, 4.9 ), facecolor = FOND )
    fond = lambda ax: ( ax.scatter( P[ :, 0 ], P[ :, 1 ], s = 1.0, c = "#cfcfcf", linewidths = 0 ),
                        ax.scatter( P[ m, 0 ], P[ m, 1 ], s = 7, c = ENCRE, linewidths = 0,
                                    zorder = 6 ),
                        ax.scatter( [ P[ j, 0 ] ], [ P[ j, 1 ] ], s = 46, marker = "*",
                                    c = "#b03030", linewidths = 0, zorder = 7 ) )

    # (a) UN DIRAC : coupe franche
    ax = axs[ 0 ]; cadre( ax, xl, yl ); fond( ax )
    fq = ( ( YL - q ) ** 2 ).sum( 1 ) + dec
    ax.contourf( xl, yl, ( fq <= hj ).reshape( gg, gg ).astype( float ), levels = [ 0.5, 1.5 ],
                 colors = [ "#999999" ], alpha = 0.30 )
    ax.contour( xl, yl, ( fq - hj ).reshape( gg, gg ), levels = [ 0.0 ], colors = [ ENCRE ],
                linewidths = 2.0 )
    ax.scatter( [ q[ 0 ] ], [ q[ 1 ] ], s = 40, c = ENCRE, marker = "D", zorder = 8 )
    ax.set_title( "UN DIRAC contre un voisin\ncoupe FRANCHE : un demi-plan", fontsize = 10,
                  color = ENCRE )

    # (b) LA ZONE K : coupe grise. Les H_p pour p sur dK, et la bande entre inter et union.
    ax = axs[ 1 ]; cadre( ax, xl, yl ); fond( ax )
    lv = np.stack( [ 2 * ( YL @ v ) - v @ v for v in V ] )          # l_v( y ), y = x + a/2
    fi = -lv.min( 0 ) + ( YL * YL ).sum( 1 ) + dec                  # l'INTERSECTION des H_v
    K = d[ "K" ]
    fK = K.d2( YL ) + dec
    ax.contourf( xl, yl, ( ( fi <= hj ) ).reshape( gg, gg ).astype( float ), levels = [ 0.5, 1.5 ],
                 colors = [ "#8c8c8c" ], alpha = 0.42 )
    ax.contourf( xl, yl, ( ( fK <= hj ) & ( fi > hj ) ).reshape( gg, gg ).astype( float ),
                 levels = [ 0.5, 1.5 ], colors = [ "#8c8c8c" ], alpha = 0.16 )
    ech = np.vstack( [ V[ i ] + t * ( V[ ( i + 1 ) % len( V ) ] - V[ i ] )
                       for i in range( len( V ) ) for t in np.linspace( 0, 1, 5, endpoint = False ) ] )
    for p in ech:
        fp = ( ( YL - p ) ** 2 ).sum( 1 ) + dec
        ax.contour( xl, yl, ( fp - hj ).reshape( gg, gg ), levels = [ 0.0 ], colors = [ "#7a7a7a" ],
                    linewidths = 0.5, alpha = 0.55 )
    ax.contour( xl, yl, ( fK - hj ).reshape( gg, gg ), levels = [ 0.0 ], colors = [ ENCRE ],
                linewidths = 2.0 )
    ax.fill( *np.vstack( [ V, V[ :1 ] ] ).T, facecolor = "none", edgecolor = ENCRE, lw = 1.0,
             ls = "--" )
    ax.set_title( "LA ZONE K contre le meme voisin\ncoupe GRISE : les H_p s'etalent,"
                  " on garde leur REUNION", fontsize = 10, color = ENCRE )

    # (c) la meme coupe, bord colorie par REGIME
    ax = axs[ 2 ]; cadre( ax, xl, yl ); fond( ax )
    Rj = ( fK <= hj )
    ax.contourf( xl, yl, Rj.reshape( gg, gg ).astype( float ), levels = [ 0.5, 1.5 ],
                 colors = [ "#999999" ], alpha = 0.22 )
    bj = Rj.reshape( gg, gg ) & ~binary_erosion( Rj.reshape( gg, gg ), C8 )
    bj[ 0 ] = bj[ -1 ] = False; bj[ :, 0 ] = bj[ :, -1 ] = False
    trace_bord( ax, dL, bj )
    ax.fill( *np.vstack( [ V, V[ :1 ] ] ).T, facecolor = "none", edgecolor = ENCRE, lw = 1.0,
             ls = "--" )
    ax.set_title( "le bord de CETTE coupe, par regime\nles trois morceaux sont fermes",
                  fontsize = 10, color = ENCRE )

    # (d) toutes les coupes : la sur-cellule
    ax = axs[ 3 ]; cadre( ax, xs, ys )
    ax.contourf( xs, ys, np.isin( d[ "gag" ], m ).reshape( gg, gg ).astype( float ),
                 levels = [ 0.5, 1.5 ], colors = [ "#d0d0d0" ] )
    ax.contourf( xs, ys, d[ "R" ].reshape( gg, gg ).astype( float ), levels = [ 0.5, 1.5 ],
                 colors = [ "#7fb3d5" ], alpha = 0.28 )
    fond( ax )
    trace_bord( ax, d )
    ax.fill( *np.vstack( [ V, V[ :1 ] ] ).T, facecolor = "none", edgecolor = ENCRE, lw = 1.0,
             ls = "--" )
    ax.set_title( f"TOUTES les coupes : la sur-cellule\n{d['nj']} coupes, {d['arcs']} arcs,"
                  f" {d['tr']} trou(s)", fontsize = 10, color = ENCRE )
    bb = X[ d[ "R" ] ]
    mg = 0.10 * np.ptp( bb, axis = 0 ).max()
    ax.set_xlim( bb[ :, 0 ].min() - mg, bb[ :, 0 ].max() + mg )
    ax.set_ylim( bb[ :, 1 ].min() - mg, bb[ :, 1 ].max() + mg )

    for k, ( nk, ck ) in enumerate( REG ):
        axs[ 0 ].scatter( [], [], s = 26, c = ck, label = nk, linewidths = 0 )
    fig.legend( loc = "lower center", ncol = 3, fontsize = 9, frameon = False,
                bbox_to_anchor = ( 0.5, -0.005 ) )
    fig.suptitle( f"LA COUPE GRISE  |  {nom}, S = {S}  |  gris clair : la reunion des VRAIES "
                  f"cellules  -  etoile : le voisin isole  -  tirets : l'enclos K",
                  fontsize = 11, color = ENCRE )
    fig.tight_layout( rect = [ 0, 0.055, 1, 0.92 ] )
    fig.savefig( out, dpi = 135, facecolor = FOND )
    print( "->", out )

def figure_arcs( cas, S, g, graine, out ):
    fig, axs = plt.subplots( 2, len( cas ), figsize = ( 3.5 * len( cas ), 8.4 ),
                             facecolor = FOND )
    for i, cls in enumerate( ( Enveloppe, KDop ) ):
        for k, ( nom, P, W ) in enumerate( cas ):
            d = scene( P, W, S, g, graine, cls, carre = True )
            ax = axs[ i ][ k ]; cadre( ax, d[ "xs" ], d[ "ys" ] )
            ax.contourf( d[ "xs" ], d[ "ys" ],
                         np.isin( d[ "gag" ], d[ "m" ] ).reshape( g, g ).astype( float ),
                         levels = [ 0.5, 1.5 ], colors = [ "#d0d0d0" ] )
            ax.contourf( d[ "xs" ], d[ "ys" ], d[ "R" ].reshape( g, g ).astype( float ),
                         levels = [ 0.5, 1.5 ], colors = [ "#7fb3d5" ], alpha = 0.25 )
            ax.scatter( P[ :, 0 ], P[ :, 1 ], s = 0.9, c = "#cfcfcf", linewidths = 0 )
            ax.scatter( P[ d[ "m" ], 0 ], P[ d[ "m" ], 1 ], s = 5, c = ENCRE, linewidths = 0,
                        zorder = 6 )
            trace_bord( ax, d )
            ax.set_title( f"{nom}\nK = {cls.nom} : {d['nj']} coupes, {d['arcs']} arcs"
                          f" ( x{d['arcs']/max(d['nj'],1):.1f} ), {d['tr']} trou(s)",
                          fontsize = 8.5, color = ENCRE )
    for k, ( nk, ck ) in enumerate( REG ):
        axs[ 0 ][ 0 ].scatter( [], [], s = 26, c = ck, label = nk, linewidths = 0 )
    axs[ 0 ][ 0 ].legend( loc = "upper left", fontsize = 7.5, frameon = False )
    fig.suptitle( "LE BORD DE LA SUR-CELLULE EXACTE, par regime  |  le majorant theorique est "
                  "6 arcs par coupe", fontsize = 11, color = ENCRE )
    fig.tight_layout( rect = [ 0, 0, 1, 0.94 ], h_pad = 2.2 )
    fig.savefig( out, dpi = 135, facecolor = FOND )
    print( "->", out )

# ------------------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument( "--n", type = int, default = 2000 )
    ap.add_argument( "--g", type = int, default = 500 )
    ap.add_argument( "--gd", type = int, default = 460 )
    ap.add_argument( "--nech", type = int, default = 25 )
    ap.add_argument( "--esses", type = int, nargs = "*", default = [ 8, 16, 64 ] )
    ap.add_argument( "--graine", type = int, default = 0 )
    ap.add_argument( "--sfig", type = int, default = 16 )
    ap.add_argument( "--res", action = "store_true", help = "controle de resolution" )
    ap.add_argument( "--sans-mesure", action = "store_true" )
    ap.add_argument( "--chev", action = "store_true", help = "qui deborde sur quoi" )
    ap.add_argument( "--bilan", action = "store_true", help = "psi_S, les plans, le juge" )
    ap.add_argument( "--gc", type = int, default = 220 )
    a = ap.parse_args()

    ici = os.path.dirname( os.path.abspath( __file__ ) )
    Pu, Wu = uniforme( a.n, a.graine, 0.0 )
    Pw, Ww = uniforme( a.n, a.graine, 1.0 )
    Pv, Wv = charge( os.path.join( ici, "lines5_n2000_s0.005_voronoi.txt" ) )
    Pe, We = charge( os.path.join( ici, "lines5_n2000_s0.005_equal.txt" ) )
    cas = [ ( "uniforme, Voronoi", Pu, Wu ), ( "uniforme, poids aleatoires", Pw, Ww ),
            ( "5 lignes, Voronoi", Pv, Wv ), ( "5 lignes, aires egales", Pe, We ) ]

    if not a.sans_mesure:
        for nom, P, W in cas:
            for S in a.esses:
                campagne( nom, P, W, S, a.g, a.nech, a.graine )
        if a.res:
            print( "\n=== CONTROLE DE RESOLUTION ( aire, coupes, arcs ) ===" )
            for nom, P, W in cas:
                for gg in ( 300, 500, 800 ):
                    r = campagne( nom, P, W, a.sfig, gg, 10, a.graine, verbose = False )
                    print( f"    {nom:28s} g={gg:4d}   aire {np.median(r['aire']):5.2f}"
                           f"   coupes {np.median(r['coupes']):5.1f}"
                           f"   arcs {np.median(r['arcs']):5.1f}" )

    if a.bilan:
        V = [ "courbe env", "courbe kdop", "plans kdop", "plans boite" ]
        for nom, P, W in cas:
            for S in a.esses:
                bilan( nom, P, W, S, a.gc, a.graine, V, exact = True )
                bilan( nom, P, W, S, a.gc, a.graine, V, exact = False )
        return

    if a.chev:
        for nom, P, W in cas:
            for S in a.esses:
                chevauchement( nom, P, W, S, a.gc, a.graine )
        return

    figure_grise( Pu, Wu, a.sfig, a.gd, a.graine,
                  os.path.join( ici, "surcell2d_coupe_grise.png" ), "uniforme, Voronoi" )
    figure_arcs( cas, a.sfig, a.gd, a.graine,
                 os.path.join( ici, "surcell2d_coupes_arcs.png" ) )


# ------------------------------------------------ QUI DEBORDE SUR QUOI : la vraie liste

def chevauchement( nom, P, W, S, g, graine ):
    """La longueur de liste REELLEMENT atteignable. `cand` compte les cellules que la sur-cellule
    rencontre ; ce n'est pas ce que l'algorithme obtient. Lui teste sur-cellule CONTRE sur-cellule,
    donc il herite d'un agregat ENTIER des qu'il en touche un bout :

        liste( germe de A )  =  somme des |B| sur les B dont la sur-cellule coupe celle de A

C'est cette somme qu'on mesure, contre le PLANCHER -- la meme somme sur les seuls B reellement
adjacents a A. Et on la donne pour la region COURBE et pour son k-DOP, qui est la version que
`AaBspHull` rangerait."""
    lo, hi = P.min( 0 ) - 0.02, P.max( 0 ) + 0.02
    xs = np.linspace( lo[ 0 ], hi[ 0 ], g ); ys = np.linspace( lo[ 1 ], hi[ 1 ], g )
    Ag, Bg = np.meshgrid( xs, ys )
    X = np.column_stack( [ Ag.ravel(), Bg.ravel() ] )
    psw, gag = psi_kd( P, W )( X )
    lab_, sites = agregats( P, S )
    na = len( sites )
    taille = np.bincount( lab_, minlength = na ).astype( float )

    U8 = np.column_stack( [ np.cos( np.arange( 8 ) * np.pi / 4 ),
                            np.sin( np.arange( 8 ) * np.pi / 4 ) ] )
    XU = X @ U8.T
    M = np.zeros( ( na, len( X ) ), np.float32 )
    Mk = np.zeros( ( na, len( X ) ), np.float32 )
    for c in range( na ):
        m = np.flatnonzero( lab_ == c )
        if len( m ) == 0: continue
        R, _, _, _ = surcellule( X, psw, m, P, W, Enveloppe )
        M[ c ] = R
        Mk[ c ] = ( XU <= XU[ R ].max( 0 ) + 1e-12 ).all( 1 ) if R.any() else 0

    # le PLANCHER : agregats reellement adjacents, lus sur le gagnant de psi
    # l'adjacence des CELLULES, d'ou tombent les deux planchers : celui des agregats et, pour le
    # juge de paix, le nombre de vrais voisins ETRANGERS a l'agregat du germe.
    Gc = gag.reshape( g, g )
    adjc = np.zeros( ( len( P ), len( P ) ), bool )
    for A_, B_ in ( ( Gc[ :, :-1 ], Gc[ :, 1: ] ), ( Gc[ :-1 ], Gc[ 1: ] ) ):
        d = A_ != B_
        adjc[ A_[ d ].ravel(), B_[ d ].ravel() ] = True
    adjc |= adjc.T
    etr = ( adjc & ( lab_[ None, : ] != lab_[ :, None ] ) ).sum( 1 )

    G = lab_[ gag ].reshape( g, g )
    adj = np.zeros( ( na, na ), bool )
    for A_, B_ in ( ( G[ :, :-1 ], G[ :, 1: ] ), ( G[ :-1 ], G[ 1: ] ) ):
        d = A_ != B_
        adj[ A_[ d ].ravel(), B_[ d ].ravel() ] = True
    adj |= adj.T
    np.fill_diagonal( adj, True )

    res = { }
    for nomM, MM in ( ( "region courbe", M ), ( "son k-DOP 8", Mk ) ):
        # LA TANGENCE : deux sur-cellules adjacentes se touchent en mesure nulle, et la grille les
        # separe. On dilate UN des deux cotes d'un pixel -- le test devient « a moins d'un pixel »,
        # ce qui ne peut qu'AJOUTER des voisins, donc reste un certificat. C'est le meme correctif
        # que le `delta += 1e-9` de `pack`, et sans lui le compte passe SOUS le plancher.
        MD = np.stack( [ binary_dilation( r.reshape( g, g ) > 0.5, C8 ).ravel() for r in MM ] )
        O = ( MD.astype( np.float32 ) @ MM.T ) > 0.5
        O |= O.T
        np.fill_diagonal( O, True )
        vus = MM.sum( 1 ) > 0
        res[ nomM ] = ( ( O[ vus ] ).sum( 1 ), ( O[ vus ] * taille ).sum( 1 ) )
    pl = ( adj[ taille > 0 ].sum( 1 ), ( adj[ taille > 0 ] * taille ).sum( 1 ) )

    print( f"\n--- {nom}   S = {S}   {na} agregats   grille {g}x{g}" )
    print( f"    {'':16s} {'agregats touches':>17s} {'p90':>6s} | {'LISTE / germe':>14s} {'p90':>6s}" )
    for k, ( o, l ) in list( res.items() ) + [ ( "PLANCHER ( vrais )", pl ) ]:
        print( f"    {k:16s} {np.median( o ):17.1f} {np.percentile( o, 90 ):6.1f} |"
               f" {np.median( l ):14.0f} {np.percentile( l, 90 ):6.0f}" )


# ========================================================= LE BILAN, TOUTES VARIANTES
#
# Trois questions d'un coup, sur la MEME grille et les MEMES agregats :
#
#   1. contre `psi_S` -- l'enveloppe des seuls REPRESENTANTS, ce que l'algorithme a vraiment --
#      et non contre `psi` exact, qui n'etait qu'un plancher ;
#   2. ce que coute de remplacer la PARABOLE par des PLANS ;
#   3. le JUGE DE PAIX : une fois la cellule d'un germe construite avec les diracs de son propre
#      agregat, combien de diracs reste-t-il a examiner pour la CERTIFIER -- c'est-a-dire la somme
#      des |B| sur les agregats dont la sur-cellule rencontre CETTE cellule-la.

def enclos_plans( Q, cls ):
    """L'approximation PLANE : `K` reste l'enclos, mais son bord parabolique est remplace par les
    plans tangents aux SOMMETS, releves du cran `omega = max L^2 / 4`. Sound : dans la bande d'une
    arete de longueur `L`, `max( l_v, l_v' ) + L^2/4 >= Phi_K`. L'interieur de `K` est garde en
    entier, ce qui ne coute rien -- le regime cercle ne tire jamais."""
    K = cls( Q )
    V = getattr( K, "V", None )
    if V is None or len( V ) < 3:
        V = K.trace()                       # la boite rend ses quatre coins
    if V is None or len( V ) < 3:
        V = Q                               # agregat degenere ( aligne, doublons ) : les germes
    V = np.asarray( V, float )
    L = np.linalg.norm( np.roll( V, -1, 0 ) - V, axis = 1 ) if len( V ) > 2 else np.zeros( 1 )
    return V, float( ( L * L ).max() / 4 ), K

def masque( X, ps, m, P, W, quoi ):
    """Le masque de la sur-cellule, pour une variante. Rend aussi l'aire en pixels."""
    a, b = majorant_affine( P[ m ], W[ m ] )
    Y = X + a / 2
    dec = - X @ a - ( a @ a ) / 4 - b
    if quoi in ( "courbe env", "courbe kdop" ):
        cls = Enveloppe if quoi.endswith( "env" ) else KDop
        return cls( P[ m ] ).d2( Y ) + dec <= ps + 1e-12
    V, om, K = enclos_plans( P[ m ], KDop if "kdop" in quoi else Boite )
    d2 = ( ( Y[ :, None, : ] - V[ None ] ) ** 2 ).sum( -1 ).min( 1 )      # min_v |y - v|^2
    R = ( d2 + dec - om ) <= ps + 1e-12
    return R | ( K.d2( Y ) <= 0 )                                        # plus l'interieur de K

def bilan( nom, P, W, S, g, graine, variantes, exact = False ):
    lo, hi = P.min( 0 ) - 0.02, P.max( 0 ) + 0.02
    xs = np.linspace( lo[ 0 ], hi[ 0 ], g ); ys = np.linspace( lo[ 1 ], hi[ 1 ], g )
    Ag, Bg = np.meshgrid( xs, ys )
    X = np.column_stack( [ Ag.ravel(), Bg.ravel() ] )
    ps_ex, gag = psi_kd( P, W )( X )
    lab_, sites = agregats( P, S )
    na = len( sites )
    taille = np.bincount( lab_, minlength = na ).astype( np.float32 )
    ps = ps_ex if exact else psi_kd( P[ sites ], W[ sites ] )( X )[ 0 ]
    membres = [ np.flatnonzero( lab_ == c ) for c in range( na ) ]
    aire_vraie = np.bincount( gag, minlength = len( P ) )

    # l'adjacence des CELLULES, d'ou tombent les deux planchers : celui des agregats et, pour le
    # juge de paix, le nombre de vrais voisins ETRANGERS a l'agregat du germe.
    Gc = gag.reshape( g, g )
    adjc = np.zeros( ( len( P ), len( P ) ), bool )
    for A_, B_ in ( ( Gc[ :, :-1 ], Gc[ :, 1: ] ), ( Gc[ :-1 ], Gc[ 1: ] ) ):
        d = A_ != B_
        adjc[ A_[ d ].ravel(), B_[ d ].ravel() ] = True
    adjc |= adjc.T
    etr = ( adjc & ( lab_[ None, : ] != lab_[ :, None ] ) ).sum( 1 )

    G = lab_[ gag ].reshape( g, g )
    adj = np.zeros( ( na, na ), bool )
    for A_, B_ in ( ( G[ :, :-1 ], G[ :, 1: ] ), ( G[ :-1 ], G[ 1: ] ) ):
        d = A_ != B_
        adj[ A_[ d ].ravel(), B_[ d ].ravel() ] = True
    adj |= adj.T; np.fill_diagonal( adj, True )

    print( f"\n--- {nom}   S = {S}   {na} agregats   grille {g}x{g}"
           f"   opposition : {'psi EXACT' if exact else 'psi_S ( les representants )'}" )
    print( f"    {'variante':16s} {'aire':>5s} | {'agreg.':>6s} {'p90':>5s} | {'liste/germe':>11s}"
           f" {'p90':>5s} | {'JUGE/cellule':>12s} {'p90':>5s}" )
    for quoi in variantes:
        M = np.zeros( ( na, len( X ) ), np.float32 )
        aires, juge = [], np.zeros( len( P ) )
        for c in range( na ):
            mm = membres[ c ]
            if len( mm ) == 0: continue
            R = masque( X, ps, mm, P, W, quoi )
            M[ c ] = R
            av = aire_vraie[ mm ].sum()
            if av: aires.append( R.sum() / av )
            cel = np.unique( gag[ R ] )
            juge[ cel ] += len( mm )
            juge[ mm ]  -= len( mm )        # son propre agregat est deja traite
        MD = np.stack( [ binary_dilation( r.reshape( g, g ) > 0.5, C8 ).ravel() for r in M ] )
        O = ( MD.astype( np.float32 ) @ M.T ) > 0.5
        O |= O.T; np.fill_diagonal( O, True )
        vus = M.sum( 1 ) > 0
        ag, li = O[ vus ].sum( 1 ), ( O[ vus ] * taille ).sum( 1 )
        j = juge[ aire_vraie > 0 ]
        print( f"    {quoi:16s} {np.median( aires ):5.2f} | {np.median( ag ):6.1f}"
               f" {np.percentile( ag, 90 ):5.1f} | {np.median( li ):11.0f}"
               f" {np.percentile( li, 90 ):5.0f} | {np.median( j ):12.0f}"
               f" {np.percentile( j, 90 ):5.0f}" )
    ag, li = adj.sum( 1 ), ( adj * taille ).sum( 1 )
    jv = np.zeros( len( P ) )
    for c in range( na ):
        for d_ in np.flatnonzero( adj[ c ] ):
            jv[ membres[ c ] ] += len( membres[ d_ ] )
        jv[ membres[ c ] ] -= len( membres[ c ] )
    print( f"    {'sans le juge':16s} {1.00:5.2f} | {np.median( ag ):6.1f}"
           f" {np.percentile( ag, 90 ):5.1f} | {np.median( li ):11.0f}"
           f" {np.percentile( li, 90 ):5.0f} | {np.median( jv[ aire_vraie > 0 ] ):12.0f}"
           f" {np.percentile( jv[ aire_vraie > 0 ], 90 ):5.0f}" )
    v = aire_vraie > 0
    print( f"    {'PLANCHER ( vrai )':16s} {1.00:5.2f} | {np.median( adj.sum( 1 ) ):6.1f}"
           f" {np.percentile( adj.sum( 1 ), 90 ):5.1f} | {'-':>11s} {'-':>5s} |"
           f" {np.median( etr[ v ] ):12.0f} {np.percentile( etr[ v ], 90 ):5.0f}" )

if __name__ == "__main__":
    main()

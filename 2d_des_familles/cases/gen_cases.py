#!/usr/bin/env python
"""Des nuages DURS pour le banc 2d_des_familles, et leurs poids.

Pourquoi ce fichier existe : le banc ne tirait que de l'uniforme, et l'uniforme est le seul regime
ou une grille reguliere est bonne et ou une cellule est toujours a cote de son germe. Les deux
hypotheses sont fausses en transport optimal semi-discret, et un accelerateur qui ne tient que sous
elles ne tient pas.

Le nuage : des diracs SERRES AUTOUR DE QUELQUES LIGNES qui traversent le carre. La densite varie de
plusieurs ordres de grandeur entre le voisinage d'une ligne et le vide entre deux lignes.

Deux jeux de poids sur le MEME nuage :

  * `voronoi` -- poids nuls. Les cellules pres des lignes sont des lamelles minuscules, celles du
    bord des amas avalent le vide : les aires s'etalent sur cinq ou six ordres de grandeur.

  * `equal` -- les poids qui rendent TOUTES LES AIRES EGALES a `1/n`, obtenus par L-BFGS sur le dual
    de Kantorovich. C'est le cas vraiment dur : une cellule peut alors etre LOIN de son germe (le
    germe est sur une ligne, sa cellule est dans le vide qu'elle doit remplir), ce qui casse le
    raccourci implicite de tout accelerateur -- « la cellule de `i` est autour de `p_i` ».

Le dual, tel qu'on le maximise ici :

    Phi( w ) = somme_i [ integrale_{Lag_i(w)} |x - p_i|^2 dx - w_i |Lag_i(w)| ] + somme_i w_i nu_i

concave en `w`, de gradient `nu_i - |Lag_i(w)|`. Rien d'autre n'est necessaire : ni hessienne, ni
amortissement, ni schema multi-echelle -- L-BFGS suffit, et c'est ce qui rend ce script court.
"""

import argparse
import os
import sys
import time

import numpy as np
from scipy.optimize import minimize

from pysdot import PowerDiagram
from pysdot.domain_types import ConvexPolyhedraAssembly

EPS = 1e-4   # on garde les germes STRICTEMENT dans le carre : un germe sur un bord donne une
             # cellule degeneree, qui ne mesure rien et complique la comparaison.


# ----------------------------------------------------------------------------------------- nuage

def make_lines( nb_lines, rng, min_len = 0.7 ):
    """`nb_lines` cordes du carre unite, chacune assez longue pour le TRAVERSER."""
    out = []
    while len( out ) < nb_lines:
        def on_border():
            t = rng.random()
            side = rng.integers( 4 )
            return { 0: ( t, 0.0 ), 1: ( 1.0, t ), 2: ( t, 1.0 ), 3: ( 0.0, t ) }[ int( side ) ]
        a = np.array( on_border() )
        b = np.array( on_border() )
        if np.linalg.norm( b - a ) >= min_len:
            out.append( ( a, b ) )
    return out


def make_cloud( n, lines, sigma, rng ):
    """`n` diracs, repartis sur les lignes au prorata de leur longueur, avec un ecart TRANSVERSE
    gaussien d'ecart-type `sigma`. Le long de la ligne c'est uniforme : ce qui doit etre dur est le
    contraste entre les directions, pas un amas de plus dans un amas."""
    lens = np.array( [ np.linalg.norm( b - a ) for a, b in lines ] )
    which = rng.choice( len( lines ), size = n, p = lens / lens.sum() )
    t = rng.random( n )
    e = rng.normal( 0.0, sigma, n )

    A = np.array( [ a for a, b in lines ] )[ which ]
    B = np.array( [ b for a, b in lines ] )[ which ]
    D = B - A
    L = np.linalg.norm( D, axis = 1 )[ :, None ]
    N = np.stack( [ -D[ :, 1 ], D[ :, 0 ] ], axis = 1 ) / L   # la normale unitaire a la ligne

    P = A + t[ :, None ] * D + e[ :, None ] * N
    return np.clip( P, EPS, 1 - EPS )


# ---------------------------------------------------------------------------------------- solveur

def newton( pd, nu, w0, tol = 1e-10, maxiter = 100, verbose = True ):
    """Newton AMORTI sur `nabla Phi = 0`, avec le mode constant EPINGLE.

    Pourquoi ne pas prendre celui de pysdot : la hessienne `J = d(aires)/d(poids)` est EXACTEMENT
    singuliere -- ajouter une constante a tous les poids ne change pas le diagramme, donc `J . 1 = 0`
    et, `J` etant symetrique, `1^T J = 0` aussi. `spsolve` sur un systeme singulier ne refuse pas :
    il rend une solution dont la composante le long de `1` est arbitraire et enorme, que l'appelant
    rattrape ensuite en amortissant a l'aveugle. MESURE avec pysdot a n=1e5 : 945 iterations, 19
    minutes, et un arret a `3e-6`.

    L'epinglage : on ajoute `d` a l'entree `(0,0)`. Le systeme reste EXACTEMENT le meme, et ce n'est
    pas une approximation -- le second membre `g = aires - nu` verifie `1^T g = 1 - 1 = 0`, donc de
    `( J + d e0 e0^T ) delta = g` on tire `d . delta_0 = 1^T g = 0`, soit `delta_0 = 0` et
    `J delta = g`. On a simplement choisi, parmi les solutions qui different d'une constante, celle
    qui laisse le premier poids tranquille.

    L'amortissement : le pas est divise par deux tant qu'une cellule se vide ou que le residu ne
    decroit pas. C'est la condition de Kitagawa-Merigot-Thibert, et elle garantit la convergence
    globale.
    """
    import scipy.sparse as sp
    from scipy.sparse.linalg import spsolve

    n = len( nu )
    t0 = time.time()
    w = w0.copy()
    pd.set_weights( w )
    A = pd.integrals()
    # le plancher de masse sous lequel on refuse de descendre : une cellule vide rend la hessienne
    # fausse (et le dual non strictement concave dans cette direction).
    floor = 0.5 * min( A.min(), nu.min() )
    err = np.abs( A - nu ).max() * n     # ce qu'on RAPPORTE : le pire ecart relatif d'aire
    res = np.abs( A - nu ).sum() * n     # ce qu'on fait DECROITRE : voir plus bas

    for it in range( maxiter ):
        if err <= tol:
            break
        mvs = pd.der_integrals_wrt_weights( stop_if_void = True )
        if mvs.error:
            raise RuntimeError( "cellule vide dans la hessienne a l'iteration %d" % it )
        J = sp.csr_matrix( ( mvs.m_values, mvs.m_columns, mvs.m_offsets ), shape = ( n, n ) )
        d = np.abs( J.diagonal() ).mean()
        J = J + sp.csr_matrix( ( [ d ], ( [ 0 ], [ 0 ] ) ), shape = ( n, n ) )
        step = spsolve( J.tocsc(), A - nu )

        t = 1.0
        while True:
            pd.set_weights( w - t * step )
            An = pd.integrals()
            en = np.abs( An - nu ).max() * n
            rn = np.abs( An - nu ).sum() * n
            # La decroissance est exigee sur la norme L1 du residu, PAS sur son maximum. Avec le
            # maximum, une seule cellule dont la hessienne est mal evaluee -- une arete tres courte,
            # deux germes presque confondus -- bloque le pas pour les 99 999 autres, et
            # l'amortissement descend jusqu'a l'arret. MESURE : le critere en maximum stagnait a
            # `2.4e-6` a n=1e5, celui en L1 va au bout.
            # `1 - t/4` : on n'exige pas la decroissance complete du pas de Newton, seulement une
            # fraction -- sinon un pas legitime serait rejete des que le modele quadratique est un
            # peu faux.
            if An.min() > floor and rn <= ( 1 - t / 4 ) * res:
                break
            t *= 0.5
            if t < 1e-8:
                if verbose:
                    print( "    Newton : pas amorti jusqu'a 1e-8 (residu L1 %.3e -> %.3e, "
                           "aire min %.3e), on s'arrete" % ( res, rn, An.min() ), flush = True )
                return w, err
        w = w - t * step
        A, err, res = An, en, rn
        if verbose and ( it < 3 or it % 5 == 0 ):
            print( "    it %3d  t=%.3g  err_rel %.3e  (%.1f s)"
                   % ( it, t, err, time.time() - t0 ), flush = True )

    if verbose:
        print( "    Newton : %d iterations, %.1f s, erreur relative d'aire max %.3e"
               % ( it + 1, time.time() - t0, err ), flush = True )
    return w, err


def solve_equal_areas( pos, tol = 1e-7, maxiter = 20000, verbose = True, solver = "newton",
                       w0 = None ):
    """Les poids qui egalisent les aires.

    Deux solveurs, et la difference n'est pas de gout :

      * `lbfgs` -- L-BFGS sur `-Phi`, ce qu'on ferait naturellement puisque le gradient est gratuit
        (c'est l'aire) et qu'aucune hessienne n'est requise. MESURE sur ce nuage : il stagne vers
        `3e-4` d'erreur relative d'aire. La raison est visible dans la formule -- `Phi` porte le
        terme `somme_i integrale |x - p_i|^2`, d'ordre `1`, alors que ce qui reste a gagner pres de
        l'optimum est d'ordre `1e-12` ; la decroissance disparait sous l'arrondi de `Phi` bien avant
        que le gradient ne s'annule. C'est une limite du critere, pas du reglage.

      * `newton` -- Newton amorti sur `nabla Phi = 0` (celui de pysdot). Il n'evalue JAMAIS `Phi`,
        seulement son gradient et sa hessienne, donc le plancher ci-dessus n'existe pas : on arrive
        a `1e-15`. C'est le defaut, parce qu'un cas de test dont les aires ne sont egales qu'a
        `3e-4` pres ne serait pas le cas qu'on croit tester.

    Mise a l'echelle (L-BFGS) : les aires valent `1/n`, donc le gradient brut vaut `1/n` et les
    criteres d'arret de scipy, qui sont absolus, ne voudraient rien dire. On multiplie l'objectif
    par `n` : le gradient devient l'ERREUR RELATIVE D'AIRE, et `gtol` se lit directement.
    """
    n = len( pos )
    domain = ConvexPolyhedraAssembly()
    domain.add_box( [ 0, 0 ], [ 1, 1 ] )
    pd = PowerDiagram( pos, weights = np.zeros( n ), domain = domain )
    nu = np.full( n, 1.0 / n )
    if w0 is None:
        w0 = np.zeros( n )

    if solver == "newton":
        w, err = newton( pd, nu, w0, tol = tol, verbose = verbose )
        return w - w.mean(), err

    hist = { "nfev": 0, "t0": time.time() }

    def fg( w ):
        pd.set_weights( w )
        A = pd.integrals()
        M = pd.second_order_moments()
        phi = M.sum() - np.dot( w, A ) + np.dot( w, nu )
        hist[ "nfev" ] += 1
        hist[ "err" ] = np.abs( A - nu ).max() * n
        return -phi * n, ( A - nu ) * n

    def cb( w ):
        if verbose and hist[ "nfev" ] % 200 < 2:
            print( "    nfev %6d  err_rel %.3e  (%.1f s)"
                   % ( hist[ "nfev" ], hist[ "err" ], time.time() - hist[ "t0" ] ), flush = True )

    r = minimize( fg, w0, jac = True, method = "L-BFGS-B", callback = cb,
                  options = dict( maxiter = maxiter, maxfun = 10 * maxiter, maxcor = 30,
                                  ftol = 0.0, gtol = tol ) )

    pd.set_weights( r.x )
    A = pd.integrals()
    err = np.abs( A - nu ).max() * n
    if verbose:
        print( "    L-BFGS : %d evaluations, %.1f s, erreur relative d'aire max %.3e  (%s)"
               % ( hist[ "nfev" ], time.time() - hist[ "t0" ], err, r.message ), flush = True )

    # les poids sont definis a une constante pres : on la fixe pour que les fichiers soient
    # comparables entre eux et que les nombres restent lisibles.
    return r.x - r.x.mean(), err


# ------------------------------------------------------------------------------------- entree/sortie

def save( path, pos, w, header ):
    with open( path, "w" ) as f:
        for line in header:
            f.write( "# %s\n" % line )
        f.write( "%d\n" % len( pos ) )
        # `%.17g` : la representation la PLUS COURTE qui relit exactement le meme double. Les poids
        # du cas `equal` sont le resultat d'une optimisation ; les tronquer changerait la solution.
        np.savetxt( f, np.column_stack( [ pos, w ] ), fmt = "%.17g" )


# ---------------------------------------------------------------------------------------- visu

def draw( path, pos, w, lines, title, mode, zoom = ( 0.42, 0.58, 0.42, 0.58 ) ):
    """Deux panneaux : le domaine entier, et un ZOOM. Sans le zoom on ne voit rien -- a 2000
    cellules sur une image, les aretes se touchent ; c'est pourtant la FORME des cellules qui est
    l'objet du test.

    `mode` decide de la couleur, et ce n'est pas un detail :
      * `area` -- l'aire, en echelle log. Sur le cas Voronoi elle couvre quatre ordres de grandeur.
      * `move` -- la distance du germe au CENTRE de sa cellule. Sur le cas a aires egales, l'aire
        est constante par construction et ne dit donc rien ; ce qui est remarquable est justement
        cette distance, qui mesure a quel point une cellule a QUITTE son germe.
    """
    import matplotlib
    matplotlib.use( "Agg" )
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection, LineCollection
    from matplotlib.colors import LogNorm

    domain = ConvexPolyhedraAssembly()
    domain.add_box( [ 0, 0 ], [ 1, 1 ] )
    pd = PowerDiagram( pos, weights = w, domain = domain )
    off, crd = pd.cell_polyhedra()
    polys = [ crd[ off[ i ] : off[ i + 1 ] ] for i in range( len( pos ) ) ]
    areas = pd.integrals()
    cen = pd.centroids()
    mov = np.linalg.norm( cen - pos, axis = 1 )

    if mode == "area":
        val, lab = np.maximum( areas, 1e-14 ), "aire de la cellule"
    else:
        val, lab = np.maximum( mov, 1e-6 ), "distance germe -> centre de sa cellule"

    fig, axes = plt.subplots( 1, 2, figsize = ( 15.5, 7.4 ), dpi = 140 )
    norm = LogNorm( vmin = np.quantile( val, 0.001 ), vmax = val.max() )

    for ax, box, lw in ( ( axes[ 0 ], ( 0, 1, 0, 1 ), 0.12 ),
                         ( axes[ 1 ], zoom, 0.5 ) ):
        x0, x1, y0, y1 = box
        m = 0.5 * ( x1 - x0 )
        keep = ( cen[ :, 0 ] > x0 - m ) & ( cen[ :, 0 ] < x1 + m ) \
             & ( cen[ :, 1 ] > y0 - m ) & ( cen[ :, 1 ] < y1 + m )
        idx = np.nonzero( keep )[ 0 ]
        pc = PolyCollection( [ polys[ i ] for i in idx ], array = val[ idx ], cmap = "viridis",
                             norm = norm, linewidths = lw, edgecolors = ( 0, 0, 0, 0.45 ) )
        ax.add_collection( pc )
        for u, v in lines:
            ax.plot( [ u[ 0 ], v[ 0 ] ], [ u[ 1 ], v[ 1 ] ], "-", color = "crimson", lw = 0.9, alpha = 0.55 )
        if mode == "move":
            # le segment germe -> centre, sur un ECHANTILLON : tous les tracer noircit l'image et
            # cache ce qu'ils sont censes montrer.
            sub = idx if len( idx ) < 400 else idx[ :: max( 1, len( idx ) // 400 ) ]
            ax.add_collection( LineCollection( np.stack( [ pos[ sub ], cen[ sub ] ], axis = 1 ),
                                               colors = "white", linewidths = 0.6, alpha = 0.85 ) )
        ax.plot( pos[ idx, 0 ], pos[ idx, 1 ], ".", ms = 1.2 if lw > 0.2 else 0.5,
                 color = "black", alpha = 0.6 )
        ax.set_xlim( x0, x1 ); ax.set_ylim( y0, y1 ); ax.set_aspect( "equal" )

    axes[ 0 ].set_title( title, fontsize = 10 )
    axes[ 1 ].set_title( "zoom [%g, %g] x [%g, %g]" % zoom, fontsize = 10 )
    cb = fig.colorbar( pc, ax = axes, fraction = 0.03, pad = 0.02 )
    cb.set_label( lab )
    fig.savefig( path, bbox_inches = "tight" )
    plt.close( fig )
    print( "    -> %s   (aires %.2e .. %.2e, deplacement median %.3f, max %.3f)"
           % ( path, areas.min(), areas.max(), np.median( mov ), mov.max() ), flush = True )


# ---------------------------------------------------------------------------------------- principal

def main():
    p = argparse.ArgumentParser( description = __doc__,
                                 formatter_class = argparse.RawDescriptionHelpFormatter )
    p.add_argument( "-n", type = int, default = 10000, help = "nombre de diracs" )
    p.add_argument( "--lines", type = int, default = 5, help = "nombre de lignes" )
    p.add_argument( "--sigma", type = float, default = 0.005, help = "epaisseur des lignes" )
    p.add_argument( "--seed", type = int, default = 0 )
    p.add_argument( "--tol", type = float, default = 1e-10, help = "erreur relative d'aire visee" )
    p.add_argument( "--maxiter", type = int, default = 20000 )
    p.add_argument( "--solver", default = "newton", choices = [ "newton", "lbfgs" ] )
    p.add_argument( "--no-equal", action = "store_true", help = "ne pas resoudre le cas a aires egales" )
    p.add_argument( "--no-viz", action = "store_true" )
    p.add_argument( "--viz-max", type = int, default = 40000,
                    help = "au-dela, on ne dessine pas (illisible et lent)" )
    p.add_argument( "--out", default = os.path.dirname( os.path.abspath( __file__ ) ) )
    a = p.parse_args()

    rng = np.random.default_rng( a.seed )
    lines = make_lines( a.lines, rng )
    pos = make_cloud( a.n, lines, a.sigma, rng )
    base = "lines%d_n%d_s%g" % ( a.lines, a.n, a.sigma )
    meta = [ "nuage : %d diracs autour de %d lignes, sigma = %g, graine = %d"
             % ( a.n, a.lines, a.sigma, a.seed ),
             "format : n, puis n lignes « x y w »" ]

    print( "%s : %d diracs, %d lignes" % ( base, a.n, a.lines ), flush = True )

    fn = os.path.join( a.out, base + "_voronoi.txt" )
    save( fn, pos, np.zeros( a.n ), meta + [ "poids : NULS (diagramme de Voronoi)" ] )
    print( "  voronoi -> %s" % fn, flush = True )
    if not a.no_viz and a.n <= a.viz_max:
        draw( os.path.join( a.out, base + "_voronoi.png" ), pos, np.zeros( a.n ), lines,
              "%s -- poids nuls (Voronoi)" % base, mode = "area" )

    if not a.no_equal:
        print( "  equal : resolution (%s)..." % a.solver, flush = True )
        w, err = solve_equal_areas( pos, tol = a.tol, maxiter = a.maxiter, solver = a.solver )
        fn = os.path.join( a.out, base + "_equal.txt" )
        save( fn, pos, w, meta + [ "poids : aires TOUTES egales a 1/n (%s)" % a.solver,
                                   "erreur relative d'aire max = %.3e" % err ] )
        print( "  equal   -> %s" % fn, flush = True )
        if not a.no_viz and a.n <= a.viz_max:
            draw( os.path.join( a.out, base + "_equal.png" ), pos, w, lines,
                  "%s -- aires egales (Laguerre), trait blanc = germe -> centre de sa cellule" % base,
                  mode = "move" )

    return 0


if __name__ == "__main__":
    sys.exit( main() )

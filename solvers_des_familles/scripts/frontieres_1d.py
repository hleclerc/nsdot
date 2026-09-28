#!/usr/bin/env python
"""LES FRONTIERES PLUTOT QUE LES POIDS ( 1D ) : ce que la construction vaut, et ce qu'elle devient en nD.

Posee apres le constat du § 8 ( aucune prolongation n'est admissible, et aucun lissage ne peut la
rendre admissible : `w'' < 2` est une borne A SENS UNIQUE ). L'idee : prolonger la GEOMETRIE et non
le potentiel. En 1D on se donne les frontieres `b` entre cellules fines et on en DEDUIT les poids,
exactement,
    w_{i+1} = w_i + ( p_i + p_{i+1} - 2 b_i )( p_{i+1} - p_i )
et l'admissibilite est GRATUITE des que `b` est croissante. Deux facons de se donner `b` depuis le
niveau grossier :
  * `sousdiv`     -- garder les frontieres GROSSIERES et couper chaque cellule grossiere en parts
                     egales entre ses enfants ;
  * `barycentres` -- un linspace entre les BARYCENTRES des cellules grossieres, les bords du domaine
                     en ancres supplementaires.

CE QUE LA MESURE DIT ( n = 400, sigma = 0.005, R = 8 ; residu l2 = |a-nu| / |nu|, Voronoi a 1.42 ) :

  * `sousdiv` rend la solution EXACTE ( residu 0 ), `barycentres` un residu de 0.017, zero cellule
    vide -- contre 0.70 pour le meilleur depart admissible connu ( spline + enveloppe, § 8.4 ) et 0.54
    pour la solution fine lissee d'un balayage puis reparee. Quatre-vingts fois mieux que Voronoi.

  * ET C'EST UN ACCIDENT DE DIMENSION, verifie a 1e-11 ( section `equivalence` ) : projeter une
    tessellation prescrite sur les `w` realisables, au sens des moindres carres sur les aretes
        min_w sum_ij omega_ij ( w_i - w_j - c_ij )^2,  omega_ij = |f_ij| / 2 d_ij,  c_ij = 2 d_ij s_ij
    ( `s_ij` le deplacement voulu de la facette entre `i` et `j` ), donne les equations normales
    `L w = div c` avec
        ( div c )_i = sum_j omega_ij c_ij = sum_j |f_ij| s_ij = LA VARIATION D'AIRE VOULUE
    -- c'est-a-dire UN PAS DE NEWTON, et rien d'autre : la projection ne retient de la tessellation
    prescrite que sa DIVERGENCE. En 1D il y a `n-1` aretes pour `n-1` inconnues de jauge, donc residu
    des moindres carres NUL ( la tessellation est realisee exactement ) et le pas de Newton est exact
    ( § 8.3 ) : les deux moities du miracle 1D sont le meme fait. En 2D il y a ~3n aretes pour n
    inconnues, en 3D ~15n : les 2n ( ou 14n ) autres contraintes sont jetees, et ce qui survit est la
    contrainte de MASSE, qu'on connaissait deja. Prescrire la tessellation n'apporte donc AUCUNE
    information nouvelle en nD -- c'est l'obstruction de realisabilite des figures reciproques, lue en
    comptage de degres de liberte.

  * `diag`, un pas de Jacobi sur le RESIDU ( `w_i += omega ( nu_i - a_i ) / ( da_i/dw_i )`, a ne pas
    confondre avec `lisse_jacobi` qui est un pas sur le systeme HOMOGENE et ENLEVE la haute frequence
    au lieu de l'injecter ) : l'hypothese etait qu'il reproduit la composante a l'echelle de la
    cellule, `( S_i - 1 ) h^2` avec `S_i = nu_i / |Vor_i|` la compression. ELLE EST FAUSSE : la marge
    obtenue est ANTI-correlee a `S` ( -0.77 ), parce que le pas deplace aussi les voisins, donc la
    corde `w_lin` a laquelle la marge se mesure. Ce que `diag` fait vraiment, mesure : il baisse le
    residu l2 et, amorti a `omega = 1/2`, le residu MAX ( spline 2.89 -> 2.27, l2 0.70 -> 0.53 ;
    solution lissee + enveloppe 2.23 -> 1.66, l2 0.54 -> 0.36, sans creer de vide ) ; a `omega = 1` il
    en CREE. Ce n'est pas un remede a l'admissibilite -- mais le residu max est justement ce qui
    etrangle l'amortissement juste apres un relevement ( § 8.5 ).

Usage : scripts/frontieres_1d.py [-n 400 --sigma 0.005 -R 8]
"""
import argparse, os, sys
import numpy as np

sys.path.insert( 0, os.path.dirname( os.path.abspath( __file__ ) ) )
import multiechelle_1d as me
from adoucissement_1d import enveloppe, bilan

# ------------------------------------------------------------------- les frontieres -> les poids

def poids_depuis_frontieres( p, b ):
    """`b` : les `n-1` frontieres interieures, croissantes. Rend `w` ( jauge `w_0 = 0` )."""
    d = np.diff( p )
    return np.concatenate( [ [ 0.0 ], np.cumsum( ( p[ :-1 ] + p[ 1: ] - 2 * b ) * d ) ] )


def frontieres_sousdiv( p, k, B ):
    """les frontieres grossieres gardees, chaque cellule grossiere coupee en parts egales."""
    b = []
    for q in range( k.max() + 1 ):
        R = int( np.sum( k == q ) )
        b.extend( B[ q ] + ( np.arange( 1, R ) / R ) * ( B[ q + 1 ] - B[ q ] ) )   # les interieures
        if q < k.max(): b.append( B[ q + 1 ] )                                     # la frontiere gardee
    return np.array( b )


def frontieres_barycentres( p, reps, B ):
    """un linspace entre les barycentres des cellules grossieres ( bords du domaine en ancres )."""
    n = len( p )
    c = ( B[ :-1 ] + B[ 1: ] ) / 2
    anc_i = np.concatenate( [ [ -0.5 ], reps, [ n - 0.5 ] ] )
    anc_x = np.concatenate( [ [ 0.0 ], c, [ 1.0 ] ] )
    return np.interp( np.arange( n - 1 ) + 0.5, anc_i, anc_x )

# ------------------------------------------------------------------- la correction de compression

def dadw( p, hull ):
    """`da_i / dw_i` = la diagonale du laplacien de Laguerre ( 0 pour une cellule vide )."""
    D = np.zeros( len( p ) )
    for i, j in zip( hull[ :-1 ], hull[ 1: ] ):
        c = 1 / ( 2 * ( p[ j ] - p[ i ] ) )
        D[ i ] += c; D[ j ] += c
    return D


def corr_diag( p, w, nu, omega = 1.0, nb = 1 ):
    """`nb` pas de Jacobi sur le RESIDU : `w_i += omega ( nu_i - a_i ) / ( da_i/dw_i )`."""
    w = w.copy()
    for _ in range( nb ):
        a, hull, _ = me.cellules( p, w )
        D = dadw( p, hull )
        ok = D > 0
        w[ ok ] += omega * ( nu[ ok ] - a[ ok ] ) / D[ ok ]
    return w

# ------------------------------------------------------------------- le banc

def niveau_grossier( p, k, reps ):
    n = len( p )
    pc = p[ reps ]
    nuc = np.array( [ np.sum( k == q ) for q in range( k.max() + 1 ) ] ) / n
    wc, itc, _ = me.newton( pc, nuc, np.zeros( len( pc ) ) )
    ac, hullc, B = me.cellules( pc, wc )
    assert len( hullc ) == len( pc ), "le niveau grossier a des cellules vides"
    return pc, nuc, wc, ac, B, itc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument( "-n", type = int, default = 400 )
    ap.add_argument( "--sigma", type = float, default = 0.005 )
    ap.add_argument( "-R", type = int, default = 8 )
    ap.add_argument( "--graine", type = int, default = 0 )
    ap.add_argument( "--eps", type = float, default = 0.1 )
    o = ap.parse_args()

    n, R = o.n, o.R
    p, _ = me.nuage( n, o.sigma, o.graine )
    nu = np.full( n, 1 / n )
    avor, hull_vor, _ = me.cellules( p, np.zeros( n ) )
    S = nu / avor                                       # la COMPRESSION cible, lisible sans rien resoudre
    w_sol, its0, diag0 = me.newton( p, nu, np.zeros( n ) )
    print( "n = %d, sigma = %g, R = %d" % ( n, o.sigma, R ) )
    print( "  Voronoi           : %d vides, residu max %.2f, l2 %.3f ; Newton de la : %d it, %d diag"
           % ( bilan( p, np.zeros( n ), nu ) + ( its0, diag0 ) ) )
    print( "  compression cible : min %.2e, mediane %.3f, max %.1f" % ( S.min(), np.median( S ), S.max() ) )

    k, reps = me.paquets( n, R )
    pc, nuc, wc, ac, B, itc = niveau_grossier( p, k, reps )
    print( "  niveau grossier   : n = %d, %d it, max|a-nu|/nu = %.1e" % ( len( pc ), itc, np.max( np.abs( ac - nuc ) / nuc ) ) )

    # ---- l'hypothese refutee : un pas diagonal reproduit-il la compression ?
    m1 = me.marge( p, corr_diag( p, np.zeros( n ), nu, omega = 1.0 ) )
    fini = np.isfinite( m1 )
    print( "\n-- l'hypothese ( refutee ) marge <-> compression, un pas diagonal depuis Voronoi" )
    print( "   marge obtenue contre S = nu/|Vor| : ecart relatif median %.3f, correlation %.4f"
           % ( np.median( np.abs( m1[ fini ] - S[ fini ] ) / S[ fini ] ), np.corrcoef( m1[ fini ], S[ fini ] )[ 0, 1 ] ) )

    # ---- les prolongations
    bases = { nom: me.prolonge( nom, p, pc, wc, k ) for nom in [ "copie", "harmonique", "spline", "ctransf" ] }
    b_sousdiv = frontieres_sousdiv( p, k, B )
    b_bary    = frontieres_barycentres( p, reps, B )
    bases[ "front-sousdiv" ]     = poids_depuis_frontieres( p, b_sousdiv )
    bases[ "front-barycentres" ] = poids_depuis_frontieres( p, b_bary )

    print( "\n-- les prolongations, et trois corrections ( eps = %g, omega = 1 )" % o.eps )
    print( "%-19s | %s" % ( "", " | ".join( "%-24s" % t for t in [ "brute", "+ diag", "+ enveloppe", "+ enveloppe + diag" ] ) ) )
    print( "%-19s | %s" % ( "", " | ".join( [ "%-24s" % "vides  rmax      l2" ] * 4 ) ) )
    for nom, wb in bases.items():
        lignes = []
        for corr in [ "brute", "diag", "env", "env+diag" ]:
            w = wb.copy()
            if corr in ( "env", "env+diag" ): w = enveloppe( p, w, o.eps )[ 0 ]
            if corr in ( "diag", "env+diag" ): w = corr_diag( p, w, nu, omega = 1.0 )
            lignes.append( "%5d  %7.2f  %8.3f" % bilan( p, w, nu ) )
        print( "%-19s | %s" % ( nom, " | ".join( "%-24s" % l for l in lignes ) ) )

    # ---- omega et le nombre de balayages
    print( "\n-- le pas diagonal : omega et le nombre de balayages ( apres enveloppe ) -- vides/rmax/l2" )
    for nom in [ "spline", "front-barycentres", "harmonique" ]:
        w0 = enveloppe( p, bases[ nom ], o.eps )[ 0 ]
        for omega in [ 0.5, 2 / 3, 1.0 ]:
            out = [ "nb=%d: %d/%.2f/%.3f" % ( ( nb, ) + bilan( p, corr_diag( p, w0, nu, omega, nb ), nu ) ) for nb in [ 1, 2, 4, 8 ] ]
            print( "   %-18s omega %.2f : %s" % ( nom, omega, "   ".join( out ) ) )

    # ---- l'equivalence : prescrire la tessellation = UN pas de Newton dont le second membre est
    #      la variation d'aire voulue. En 1D le residu des moindres carres est nul, donc egalite.
    a_cible = np.diff( np.concatenate( [ [ 0.0 ], b_bary, [ 1.0 ] ] ) )
    L = me.laplacien( p, hull_vor )
    dw = np.zeros( n )
    dw[ 1: ] = np.linalg.solve( L[ 1:, 1: ], ( a_cible - avor )[ 1: ] )
    wf = bases[ "front-barycentres" ]; wf = wf - wf[ 0 ]
    wn = dw - dw[ 0 ]
    print( "\n-- l'equivalence : les frontieres prescrites, projetees, SONT un pas de Newton" )
    print( "   || w_frontieres - w_newton ||_inf / || w_frontieres ||_inf = %.2e" % ( np.max( np.abs( wf - wn ) ) / np.max( np.abs( wf ) ) ) )
    print( "   aires du pas de Newton contre les aires demandees : ecart max %.2e" % np.max( np.abs( me.cellules( p, wn )[ 0 ] - a_cible ) ) )

    # ---- la borne : la solution lissee, puis reparee
    ws = me.jacobi( p, w_sol, 1 )
    print( "\n-- la borne ( la solution fine lissee d'un balayage de Jacobi )" )
    for corr, w in [ ( "brute", ws ), ( "+ enveloppe", enveloppe( p, ws, o.eps )[ 0 ] ),
                     ( "+ enveloppe + diag 1/2", corr_diag( p, enveloppe( p, ws, o.eps )[ 0 ], nu, 0.5 ) ) ]:
        print( "   %-24s : %d vides, rmax %.2f, l2 %.3f" % ( ( corr, ) + bilan( p, w, nu ) ) )


if __name__ == "__main__":
    main()

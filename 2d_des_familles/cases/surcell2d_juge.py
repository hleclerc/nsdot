#!/usr/bin/env python
"""LE JUGE DE PAIX : ce que coute la VERIFICATION d'une cellule deja construite.

L'hypothese de travail est que la majorite des cellules sera BATIE par un algorithme local
(CGAL) dans son agregat. Ce qui reste a payer, c'est de CERTIFIER qu'une cellule complete l'est
vraiment. Le critere est exact :

    si la sur-cellule `C_A` ne rencontre pas `Lag_i`, aucun membre de `A` ne peut couper `Lag_i`

parce que `h_A <= h_j` pour tout membre `j`, donc `Lag_j` est inclus dans `C_A`. Le nombre de
diracs a examiner pour certifier `Lag_i` est donc

    N_i = somme des |A| sur les agregats dont la sur-cellule DEBORDE sur `Lag_i`

C'est ca qu'on compte, et on le compare au nombre de VRAIS voisins de `Lag_i`.

TROIS FACONS DE REPRESENTER L'OPPOSITION, du plancher au realiste :

  * `psi` EXACT       -- tous les germes. Ce que rien ne peut battre.
  * `psi` EXTREMAUX   -- les SOMMETS de l'enveloppe convexe de chaque agregat. C'est la selection
                         de points extremaux dans toutes les directions a la fois, et elle sert
                         DEJA de representation cote A : un seul objet pour les deux roles.
  * `psi` REPRESENTANT-- un germe par agregat. La version paresseuse, pour l'ecart.

ET LA CONVEXIFICATION : `conv( C_A )`, mesuree en aire et en juge de paix. Elle n'a d'interet
que si les sommets sont bien moins nombreux que les membres -- on imprime le rapport `m/|A|`.
"""

import argparse, os, sys
import numpy as np
sys.path.insert( 0, os.path.dirname( os.path.abspath( __file__ ) ) )
from surcell2d import charge, uniforme, agregats, h_agregat, psi, majorant_affine
from scipy.spatial import ConvexHull

def sommets( Q ):
    if len( Q ) < 3:
        return Q
    try:
        return Q[ ConvexHull( Q ).vertices ]
    except Exception:
        return Q

def dedans_hull( X, V ):
    """Points de `X` dans l'enveloppe convexe des points `V` ( sommets CCW )."""
    if len( V ) < 3:
        return np.zeros( len( X ), dtype = bool )
    ok = np.ones( len( X ), dtype = bool )
    for i in range( len( V ) ):
        A = V[ i ]; e = V[ ( i + 1 ) % len( V ) ] - A
        d = X - A
        ok &= ( e[ 0 ] * d[ :, 1 ] - e[ 1 ] * d[ :, 0 ] ) >= -1e-12
    return ok

def voisins_vrais( gag, g ):
    """Degre de chaque cellule, lu sur la grille par contact a 4 voisins."""
    G = gag.reshape( g, g )
    paires = set()
    for A, B in ( ( G[ :, :-1 ], G[ :, 1: ] ), ( G[ :-1 ], G[ 1: ] ) ):
        m = A != B
        for a, b in zip( A[ m ].ravel(), B[ m ].ravel() ):
            paires.add( ( a, b ) if a < b else ( b, a ) )
    deg = np.zeros( int( gag.max() ) + 1, dtype = np.int64 )
    for a, b in paires:
        deg[ a ] += 1; deg[ b ] += 1
    return deg

def campagne( nom, P, W, esses, g, out ):
    lo = P.min( 0 ) - 0.02; hi = P.max( 0 ) + 0.02
    xs = np.linspace( lo[ 0 ], hi[ 0 ], g ); ys = np.linspace( lo[ 1 ], hi[ 1 ], g )
    Xg, Yg = np.meshgrid( xs, ys )
    X = np.column_stack( [ Xg.ravel(), Yg.ravel() ] )
    n = len( P )

    psi_ex, gag = psi( X, P, W )
    deg = voisins_vrais( gag, g )
    aire_vraie = np.bincount( gag, minlength = n )

    for S in esses:
        lab, sites = agregats( P, S )
        na = len( sites )
        membres = [ np.flatnonzero( lab == c ) for c in range( na ) ]
        soms    = [ sommets( P[ m ] ) for m in membres ]
        nsom    = np.array( [ len( v ) for v in soms ] )
        taille  = np.array( [ len( m ) for m in membres ] )

        # les enveloppes d'opposition
        idx_som = np.concatenate( [ m[ ConvexHull( P[ m ] ).vertices ] if len( m ) >= 3 else m
                                    for m in membres ] )
        env = { "psi exact"     : psi_ex,
                "extremaux"     : psi( X, P[ idx_som ], W[ idx_som ] )[ 0 ],
                "representant"  : psi( X, P[ sites ], W[ sites ] )[ 0 ] }

        acc = { k : { "aire" : [], "juge" : np.zeros( n ), "juge_h" : np.zeros( n ) } for k in env }
        acc_cv = { "aire" : [], "juge" : np.zeros( n ), "juge_h" : np.zeros( n ) }

        for c in range( na ):
            m = membres[ c ]
            if len( m ) < 2:
                continue
            hA = h_agregat( X, P[ m ], W[ m ] )
            av = aire_vraie[ m ].sum()
            for k, e in env.items():
                d = hA <= e + 1e-12
                if av: acc[ k ][ "aire" ].append( int( d.sum() ) / av )
                cel = np.unique( gag[ d ] )
                acc[ k ][ "juge" ][ cel ] += len( m )
                acc[ k ][ "juge_h" ][ cel ] += len( m )
                acc[ k ][ "juge_h" ][ m ]   -= len( m )
            # la CONVEXIFICATION, sur la variante realiste ( extremaux )
            d = hA <= env[ "extremaux" ] + 1e-12
            if d.sum() >= 3:
                V = sommets( X[ d ] )
                dc = dedans_hull( X, V )
                if av: acc_cv[ "aire" ].append( int( dc.sum() ) / av )
                cel = np.unique( gag[ dc ] )
                acc_cv[ "juge" ][ cel ] += len( m )
                acc_cv[ "juge_h" ][ cel ] += len( m )
                acc_cv[ "juge_h" ][ m ]   -= len( m )

        vus = aire_vraie > 0
        print( f"\n--- {nom}   S = {S}   |A| median {int(np.median(taille))}"
               f"   sommets m/|A| = {np.mean(nsom/np.maximum(taille,1)):.2f}"
               f"   ( m median {int(np.median(nsom))} )"
               f"   vrais voisins {np.median(deg[vus]):.1f}" )
        print( f"    {'opposition':16s} {'aire':>6s} | {'JUGE moy':>8s} {'med':>6s} {'p90':>6s}"
               f" | {'HORS moy':>8s} {'med':>6s} {'p90':>6s}" )
        for k in list( env ) + [ "conv( extremaux )" ]:
            a = acc_cv if k.startswith( "conv" ) else acc[ k ]
            ar = np.array( a[ "aire" ] )
            j, jh = a[ "juge" ][ vus ], a[ "juge_h" ][ vus ]
            print( f"    {k:16s} {np.median(ar):6.2f} | {j.mean():8.1f} {np.median(j):6.0f}"
                   f" {np.percentile(j,90):6.0f} | {jh.mean():8.1f} {np.median(jh):6.0f}"
                   f" {np.percentile(jh,90):6.0f}" )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument( "--n", type = int, default = 4000 )
    ap.add_argument( "--g", type = int, default = 500 )
    ap.add_argument( "--graine", type = int, default = 0 )
    a = ap.parse_args()
    ici = os.path.dirname( os.path.abspath( __file__ ) )
    Pu, Wu = uniforme( a.n, a.graine, 0.0 )
    Pw, Ww = uniforme( a.n, a.graine, 1.0 )
    Pe, We = charge( os.path.join( ici, "lines5_n2000_s0.005_equal.txt" ) )
    for nom, P, W in ( ( "uniforme, Voronoi", Pu, Wu ),
                       ( "uniforme, poids aleatoires", Pw, Ww ),
                       ( "5 lignes, aires egales", Pe, We ) ):
        campagne( nom, P, W, [ 8, 16, 32, 64 ], a.g, None )

main()

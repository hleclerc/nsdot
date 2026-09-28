#!/usr/bin/env python
"""`c*` EN 2D : la garantie de la prolongation par la triangulation grossiere, mesuree sur les niveaux
du multi-echelle ( le test decisif de l'etude 1D, `scripts/triangulation_1d.py` ).

Rappel de la construction : `u = Uh - c g`, soit `w = ( 1 - c )( |p|^2 - Uh ) + c w_tilde`, avec `Uh`
l'enveloppe convexe inferieure du releve grossier et `g = Q - |p|^2` l'ecart au paraboloide, `Q`
l'interpole affine de `|P|^2` sur la triangulation reguliere grossiere. Dans chaque simplexe `u` est
strictement convexe des que `c > 0` ; a une facette le pli de `u` vaut `pli( Uh ) - c pli( Q )`. D'ou
ZERO cellule vide pour tout

    0 < c <= c* = min over F, pli_F( Q ) > 0 of  pli_F( Uh ) / pli_F( Q )

et la marge obtenue vaut `~ c` ( verifie en 1D ). La question tranchee ici : combien vaut `c*` sur un
vrai niveau grossier 2D, et comment il bouge avec le contraste du nuage.

LA LECTURE. Le banc declare admissible « toute aire >= 0.5 * min( nu_i, aire min de Voronoi ) », qui
est le plancher de l'amortissement de Newton. Une marge `c*` ne franchit ce plancher que si
`c* >~ 0.5`. En dessous, la construction donne bien zero cellule VIDE, mais des cellules trop petites
pour que Newton amortisse -- la garantie est vraie et inutile.

LE VERDICT ( mesure sur les niveaux de `lines5_n100000`, R = 8, sigma 0.1 a 0.005 ) : `c*` vaut
**1e-3 a 4e-5**, soit 250 a 14 000 fois sous le plancher, et ce sur TOUS les cas, le plus facile
compris. Trois faits qui ferment la piste :

  * `c*` NE SUIT PAS la difficulte du cas : a n = 16384, sigma = 0.1 donne 3.53e-5 et sigma = 0.005
    donne 3.88e-5, alors que la compression grossiere minimale passe de 1.0e-2 a 2.8e-3. L'identite 1D
    `c* = min compression grossiere` est donc FAUSSE en 2D -- elle valait pour une raison de dimension,
    comme le reste de la 1D ;
  * `c*` DECROIT AVEC n, en ~1/n : 9.89e-4 ( n = 256 ), 2.45e-4 ( 2048 ), 3.53e-5 ( 16384 ). Raffiner
    degrade la garantie, exactement le contraire de ce qu'un multi-echelle demande ;
  * la MEDIANE de `c_F` vaut 1.05 : la facette TYPIQUE autoriserait `c > 1`. `c*` est un minimum sur
    ~3n facettes d'une loi qui a de la masse en 0, et ce sont les facettes PRESQUE DEGENEREES de la
    triangulation reguliere grossiere ( quatre germes grossiers presque co-spheriques au sens du
    releve, donc `pli( Uh ) -> 0` ) qui l'imposent. Elles sont partout et se multiplient avec n.

Ce qui reste utilisable : le certificat se calcule sur le niveau grossier SEUL et dit OU le niveau fin
sera malade ( au voisinage de ces facettes ), avant tout diagramme fin -- la ou `releve_minimal` les
decouvre aujourd'hui en construisant le diagramme fin et en le testant.

    scripts/cstar_2d.py /tmp/niv/s0.1_niveau1.txt [...]
"""
import argparse, os, sys
import numpy as np
from scipy.spatial import ConvexHull, cKDTree


def lit_niveau( chemin ):
    with open( chemin ) as f:
        n, D = ( int( v ) for v in f.readline().split() )
    a = np.loadtxt( chemin, skiprows = 1 )
    return a[ :, :D ], a[ :, D ], a[ :, D + 1 ]              # P, W, nu


def triangulation_reguliere( P, W ):
    """la triangulation reguliere de `( P, W )` = la projection de l'enveloppe convexe INFERIEURE des
    points releves `( P, |P|^2 - W )`. Rend les simplexes et les sommets qui y apparaissent."""
    z = ( P * P ).sum( axis = 1 ) - W
    hull = ConvexHull( np.column_stack( [ P, z ] ) )
    bas = hull.equations[ :, P.shape[ 1 ] ] < 0              # normale sortante vers le bas
    return hull.simplices[ bas ], hull, bas


def gradients( P, val, simplexes ):
    """le gradient de l'interpole affine de `val` sur chaque simplexe ( D = 2 : un plan par triangle )."""
    A = P[ simplexes ]                                      # ( m, 3, 2 )
    v = val[ simplexes ]                                     # ( m, 3 )
    M = np.stack( [ A[ :, 1 ] - A[ :, 0 ], A[ :, 2 ] - A[ :, 0 ] ], axis = 1 )   # ( m, 2, 2 )
    b = np.stack( [ v[ :, 1 ] - v[ :, 0 ], v[ :, 2 ] - v[ :, 0 ] ], axis = 1 )   # ( m, 2 )
    return np.linalg.solve( M, b[ :, :, None ] )[ :, :, 0 ]


def plis( P, W, simplexes ):
    """pour chaque facette interieure ( arete partagee par deux triangles ) : `pli( Uh )`, `pli( Q )`."""
    U = ( P * P ).sum( axis = 1 ) - W
    Q = ( P * P ).sum( axis = 1 )
    gU = gradients( P, U, simplexes )
    gQ = gradients( P, Q, simplexes )

    # les aretes, et les deux triangles de chacune
    aretes = {}
    for t, s in enumerate( simplexes ):
        for e in [ ( s[ 0 ], s[ 1 ] ), ( s[ 1 ], s[ 2 ] ), ( s[ 0 ], s[ 2 ] ) ]:
            aretes.setdefault( tuple( sorted( e ) ), [] ).append( t )
    inter = [ ( e, ts ) for e, ts in aretes.items() if len( ts ) == 2 ]

    kU = np.empty( len( inter ) ); kQ = np.empty( len( inter ) )
    for f, ( ( i, j ), ( t, tp ) ) in enumerate( inter ):
        d = P[ j ] - P[ i ]
        nrm = np.array( [ -d[ 1 ], d[ 0 ] ] ); nrm /= np.linalg.norm( nrm )
        # orienter `nrm` de `t` vers `tp` : le sommet de `tp` hors de l'arete doit etre du bon cote
        op = [ v for v in simplexes[ tp ] if v not in ( i, j ) ][ 0 ]
        if np.dot( P[ op ] - P[ i ], nrm ) < 0: nrm = -nrm
        kU[ f ] = np.dot( gU[ tp ] - gU[ t ], nrm )
        kQ[ f ] = np.dot( gQ[ tp ] - gQ[ t ], nrm )
    return kU, kQ, np.array( [ e for e, _ in inter ] )


def aires_voronoi( P, res = 2048 ):
    """les aires de Voronoi sur `[0,1]^2`, par une grille ( le clip au domaine est automatique )."""
    g = ( np.arange( res ) + 0.5 ) / res
    X, Y = np.meshgrid( g, g, indexing = "ij" )
    _, idx = cKDTree( P ).query( np.column_stack( [ X.ravel(), Y.ravel() ] ) )
    return np.bincount( idx, minlength = len( P ) ) / res ** 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument( "fichiers", nargs = "+" )
    ap.add_argument( "--res", type = int, default = 2048 )
    o = ap.parse_args()

    print( "%-30s %7s | %-28s | %-22s | %s" % ( "niveau", "n", "c* et la loi des facettes", "compression grossiere", "verdict ( plancher 0.5 )" ) )
    for chemin in o.fichiers:
        P, W, nu = lit_niveau( chemin )
        simplexes, hull, bas = triangulation_reguliere( P, W )
        dedans = np.unique( simplexes )
        kU, kQ, aretes = plis( P, W, simplexes )
        contraint = kQ > 0
        c = kU[ contraint ] / kQ[ contraint ]
        cet = c.min()
        S = nu / aires_voronoi( P, o.res )
        verdict = "PASSE" if cet >= 0.5 else "sous le plancher ( x%.0f trop petit )" % ( 0.5 / cet )
        print( "%-30s %7d | c* = %-8.2e  %5.1f %% de facettes contraintes, mediane %.3f | min %.2e, mediane %.2f | %s"
               % ( os.path.basename( chemin ), len( P ), cet, 100 * contraint.mean(), np.median( c ), S.min(), np.median( S ), verdict ) )
        hors = len( P ) - len( dedans )
        if hors: print( "%-30s   ( %d germes grossiers HORS de l'enveloppe inferieure : cellules grossieres vides )" % ( "", hors ) )
        q = np.percentile( c, [ 0, 0.1, 1, 10, 50 ] )
        print( "%-30s   quantiles de c_F : min %.2e, 0.1%% %.2e, 1%% %.2e, 10%% %.2e, 50%% %.2e ; min S = %.2e"
               % ( "", q[ 0 ], q[ 1 ], q[ 2 ], q[ 3 ], q[ 4 ], S.min() ) )


if __name__ == "__main__":
    main()

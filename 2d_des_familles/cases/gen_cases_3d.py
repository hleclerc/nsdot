#!/usr/bin/env python3
"""Le pendant 3D des nuages durs de `gen_cases.py`.

Le nuage : des diracs SERRES AUTOUR DE QUELQUES PLANS qui traversent le cube. C'est l'analogue exact
des lignes en 2D -- une variete de CODIMENSION 1 -- donc la densite s'effondre entre deux nappes, et
elle varie de plusieurs ordres de grandeur.

L'ecart-type transverse est choisi a l'echelle de l'ESPACEMENT MOYEN, `h = n^(-1/3)`, exactement
comme en 2D ou `sigma = 0.005` vaut 1.6 fois `h = n^(-1/2)`. Un `sigma` beaucoup plus petit ferait
un nuage quasi bidimensionnel, dont les cellules degenereraient pour une raison qui n'est plus celle
qu'on veut tester.

= CE QUI CHANGE PAR RAPPORT A LA 2D, ET IL FAUT LE DIRE

En 2D les poids du cas `equal` viennent de `pysdot`, donc d'un code INDEPENDANT du banc : les
comparer aux poids que `pd_newton` retrouve est une vraie verification croisee. Ici pysdot n'est pas
disponible (il est compile pour un interprete qui n'existe plus sur cette machine), et ce script
n'ecrit donc QUE le cas `voronoi`, qui ne demande aucun solveur -- les poids y sont nuls.

Le cas `equal` se fabrique avec le banc lui-meme :

    xmake run pd_newton --3d --load cases/planes4_n100000_s0.02_voronoi.txt \\
                        --ecrire cases/planes4_n100000_s0.02_equal.txt

Il reste un CAS DE CHRONOMETRAGE parfaitement valide -- c'est un nuage avec des poids qui eloignent
les cellules de leurs germes, et c'est tout ce qu'on lui demande -- mais il n'est PAS un temoin
independant pour Newton. Le fichier le dit dans son en-tete, pour qu'on ne s'y trompe pas plus tard.
"""

import argparse

import numpy as np

EPS = 1e-4      # les germes STRICTEMENT dans le cube : un germe sur une face donne une cellule
                # degeneree, qui ne mesure rien et complique la comparaison.


def make_planes( nb, rng ):
    """`nb` plans qui traversent le cube : une normale unitaire au hasard, et un decalage tel que le
    plan passe pres du centre -- un plan qui n'effleure qu'un coin ne porterait presque rien."""
    out = []
    while len( out ) < nb:
        u = rng.normal( size = 3 )
        u /= np.linalg.norm( u )
        c = u @ np.array( [ 0.5, 0.5, 0.5 ] ) + rng.uniform( -0.25, 0.25 )
        out.append( ( u, c ) )
    return out


def make_cloud( n, planes, sigma, rng ):
    """`n` diracs repartis sur les plans, avec un ecart TRANSVERSE gaussien d'ecart-type `sigma`.

    Sur le plan, la position est tiree par REJET dans le cube : projeter des points uniformes du
    cube donnerait une densite piquee au centre de la trace, ce qui ajouterait un amas dans l'amas.
    Ce qui doit etre dur ici est le contraste ENTRE les directions, pas un second regroupement.
    """
    pts = []
    per = n // len( planes ) + 1
    for u, c in planes:
        # une base orthonormee du plan
        a = np.array( [ 1.0, 0.0, 0.0 ] )
        if abs( u[ 0 ] ) > 0.9:
            a = np.array( [ 0.0, 1.0, 0.0 ] )
        v1 = np.cross( u, a )
        v1 /= np.linalg.norm( v1 )
        v2 = np.cross( u, v1 )
        got = []
        while sum( len( g ) for g in got ) < per:
            m = 4 * per
            st = rng.uniform( -1.0, 1.0, size = ( m, 2 ) ) * 1.2
            P = c * u + st[ :, 0 ][ :, None ] * v1 + st[ :, 1 ][ :, None ] * v2
            ok = np.all( ( P > 0 ) & ( P < 1 ), axis = 1 )
            got.append( P[ ok ] )
        P = np.concatenate( got )[ :per ]
        P = P + rng.normal( 0.0, sigma, size = ( len( P ), 1 ) ) * u
        pts.append( P )
    P = np.concatenate( pts )
    rng.shuffle( P )
    return np.clip( P[ :n ], EPS, 1 - EPS )


def save( path, pos, w, header ):
    with open( path, "w" ) as f:
        for line in header:
            f.write( "# %s\n" % line )
        f.write( "%d\n" % len( pos ) )
        # `%.17g` : la representation la PLUS COURTE qui relit exactement le meme double.
        np.savetxt( f, np.column_stack( [ pos, w ] ), fmt = "%.17g" )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument( "-n", type = int, default = 100000 )
    ap.add_argument( "--planes", type = int, default = 4 )
    ap.add_argument( "--sigma", type = float, default = 0.02 )
    ap.add_argument( "--seed", type = int, default = 0 )
    ap.add_argument( "--out", default = "cases" )
    a = ap.parse_args()

    rng = np.random.default_rng( a.seed )
    planes = make_planes( a.planes, rng )
    pos = make_cloud( a.n, planes, a.sigma, rng )

    base = "%s/planes%d_n%d_s%g" % ( a.out, a.planes, a.n, a.sigma )
    save( base + "_voronoi.txt", pos, np.zeros( len( pos ) ),
          [ "nuage 3D : %d diracs autour de %d plans, sigma = %g" % ( a.n, a.planes, a.sigma ),
            "poids NULS -- c'est le cas euclidien (Voronoi)",
            "graine %d" % a.seed ] )
    print( "ecrit %s_voronoi.txt (%d germes)" % ( base, len( pos ) ) )
    print( "pour le cas a volumes egaux :" )
    print( "  xmake run pd_newton --3d --load %s_voronoi.txt --ecrire %s_equal.txt"
           % ( base, base ) )


if __name__ == "__main__":
    main()

"""L'ENVELOPPE d'un nuage, approchée par l'EXTÉRIEUR : l'intersection de `K` demi-espaces qui
s'appuient sur le nuage, un par direction.

Ce dont un transport a besoin quand rien ne borne le domaine ( des gaussiennes sans `boundaries` )
n'est pas l'enveloppe convexe exacte : elle peut avoir autant d'arêtes que de germes ( des points
sur un cercle ), et chaque cellule du diagramme est coupée par TOUS les plans du domaine -- `n`
coupes par cellule, le diagramme deviendrait quadratique. Ce qu'il lui faut est un convexe BORNÉ
qui contient tous les germes, avec peu de faces : pour `K` directions `u_k`, le demi-espace
`u_k . x <= max_i u_k . p_i + marge` s'appuie sur le nuage, et leur intersection est un polyèdre
convexe qui contient l'enveloppe, de `K` faces, à `marge` près -- les directions des axes en font
partie, donc un pavé de départ en sort ( `PowerDiagram.axis_aligned_box` ), et les autres
directions ne sont que `K - 2d` coupes par cellule.

Chaque germe est dedans ( ou sur le bord ), donc sa cellule de Voronoï y a une mesure positive :
c'est ce qui rend le départ de `OtPlan` admissible ( `otplan/Solve.h` ). Ce qui déborde du polyèdre
n'a pas de masse : le transport est celui vers la densité RESTREINTE à ce domaine.

Des réductions du driver ( `max` d'un produit scalaire ), sur le device où vivent les positions.
"""

import itertools

import numpy as np

from loom.drivers.driver import driver


def hull_directions( nb_dims, nb_directions = None ):
    """`[ K, d ]` directions unitaires : en 2D `K` angles réguliers ( 16 par défaut, les axes
    compris ) ; au-delà, tous les vecteurs de `{ -1, 0, 1 }^d` à une ou deux composantes non
    nulles ( `2d + 4 C( d, 2 )` : 18 en 3D ), plus les coins en 3D ( 26 )."""
    d = int( nb_dims )
    if d == 1:
        return np.array( [ [ 1.0 ], [ -1.0 ] ] )
    if d == 2:
        K = int( nb_directions or 16 )
        a = 2 * np.pi * np.arange( K ) / K
        dirs = np.stack( [ np.cos( a ), np.sin( a ) ], axis = 1 )
        dirs[ np.abs( dirs ) < 1e-12 ] = 0.0            # les axes EXACTEMENT : ce que `axis_aligned_box` reconnaît
        return dirs
    dirs = []
    max_nz = 3 if d == 3 else 2
    for v in itertools.product( ( -1, 0, 1 ), repeat = d ):
        nz = sum( 1 for x in v if x )
        if 1 <= nz <= max_nz:
            dirs.append( v )
    dirs = np.array( dirs, dtype = float )
    return dirs / np.linalg.norm( dirs, axis = 1, keepdims = True )


def supporting_half_spaces( positions, nb_directions = None, margin = 0.0 ):
    """`( directions, offsets )`, les demi-espaces `direction . x <= offset` qui s'appuient sur le
    nuage `positions` ( `[ n, d ]`, un `Tensor` ou un tableau ), écartés de `margin`. Le format
    qu'attend `PowerDiagram( boundaries = ... )`."""
    raw = getattr( positions, "raw", positions )
    raw = raw if hasattr( raw, "shape" ) else np.asarray( positions, dtype = float )
    d = int( raw.shape[ -1 ] )
    dirs = hull_directions( d, nb_directions )
    proj = raw.reshape( -1, d ) @ driver.array( dirs.T )
    offsets = np.asarray( driver.max( proj, axis = 0 ), dtype = float ).reshape( -1 ) + float( margin )
    return dirs, offsets

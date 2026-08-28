"""Changement de représentation d'un polytope convexe : des demi-espaces vers des sommets.

`{ x : dir_i . x <= off_i }` (H-représentation) est ce qu'une cellule sait toujours donner, mais
c'est inaffichable tel quel : il faut des SOMMETS, des ARÊTES et des FACES. La page HTML fait ce
travail dans le navigateur, à chaque coupe ; ici c'est la version Python, dont les sorties fichier
(VTK) ont besoin.

Écrit en dimension QUELCONQUE, pas seulement en 3D :
- un SOMMET est l'intersection de `d` plans, admissible pour tous les autres ;
- une ARÊTE joint deux sommets qui partagent `d-1` plans actifs ;
- une FACE (au sens d'un polygone, ce que VTK sait tracer) est portée par `d-2` plans communs.
  En 3D cela redonne « une face par coupe » ; en 4D ce sont les carrés d'un tesseract ; en 2D il
  ne reste qu'un seul polygone, tout le polytope.

C'est exact pour un polytope SIMPLE (aucun sommet ne portant plus de `d` plans), ce que sont les
cellules en position générique. Un sommet dégénéré (plus de `d` plans concourants) peut faire
apparaître des arêtes en trop.

Le coût est en C(nb_plans, d) petits systèmes -- négligeable pour une cellule (quelques centaines
en 3D, quelques milliers en 4D), mais c'est bien un calcul HÔTE : le jour où ça devient le goulot,
c'est un noyau qu'il faut, pas une optimisation d'ici.
"""
from itertools import combinations

import numpy as np


def clip_planes( bounds ):
    """Les 2d demi-espaces d'une boîte `[ [lo, hi], ... ]`.

    À ajouter à un polytope NON BORNÉ : sans ça il n'a aucun sommet, donc rien à montrer.
    Sur un polytope borné qui tient dans la boîte, ces plans restent inactifs et ne changent rien.
    """
    d = len( bounds )
    dirs, offs = [], []
    for k, ( lo, hi ) in enumerate( bounds ):
        e = np.zeros( d ); e[ k ] = 1.0
        dirs.append(  e ); offs.append(  hi )
        dirs.append( -e ); offs.append( -lo )
    return np.array( dirs, np.float64 ), np.array( offs, np.float64 )


def vertices_of( dirs, offs, tol = 1e-9 ):
    """Les sommets du polytope, et pour chacun l'ensemble des plans qui le portent.

    Rend `( verts [n, d], active [n] )`, `active[ i ]` étant un `frozenset` d'indices de plans.
    """
    A = np.asarray( dirs, np.float64 )
    b = np.asarray( offs, np.float64 ).reshape( -1 )
    nrm = np.linalg.norm( A, axis = 1 )
    keep = nrm > 1e-12                       # un plan dégénéré ne porte aucun sommet
    A, b = A[ keep ] / nrm[ keep, None ], b[ keep ] / nrm[ keep ]
    n, d = A.shape
    if n < d:
        return np.zeros( ( 0, d ) ), []

    # tous les d-uplets de plans d'un coup : `det` écarte les systèmes liés, puis un seul
    # `solve` groupé. C'est ce qui rend l'énumération tenable en Python.
    combos = np.array( list( combinations( range( n ), d ) ), dtype = np.int64 )
    M = A[ combos ]                                        # [ k, d, d ]
    rhs = b[ combos ]                                      # [ k, d ]
    ok = np.abs( np.linalg.det( M ) ) > 1e-10
    if not ok.any():
        return np.zeros( ( 0, d ) ), []
    # `rhs[ ..., None ]` : depuis numpy 2, un second membre de rang 2 est lu comme UNE
    # matrice, pas comme une pile de vecteurs -- il faut donc l'axe explicite.
    X = np.linalg.solve( M[ ok ], rhs[ ok ][ ..., None ] )[ ..., 0 ]     # [ k', d ]

    scale = max( 1.0, float( np.abs( b ).max() ) )
    eps   = tol * scale
    inside = ( X @ A.T <= b + eps ).all( axis = 1 )
    X = X[ inside ]
    if len( X ) == 0:
        return np.zeros( ( 0, d ) ), []

    # sommets confondus fusionnés : deux d-uplets différents désignent le même coin dès que plus
    # de d plans y concourent.
    keys = np.round( X / ( 10 * eps ) ).astype( np.int64 )
    _, first = np.unique( keys, axis = 0, return_index = True )
    X = X[ np.sort( first ) ]

    on = np.abs( X @ A.T - b ) < 10 * eps                  # [ n_verts, n_planes ]
    active = [ frozenset( np.flatnonzero( row ).tolist() ) for row in on ]
    return X, active


def edges_of( active, nb_dims, nb_real = None ):
    """Les arêtes : deux sommets qui partagent au moins `d-1` plans actifs.

    `nb_real` marque la frontière entre les plans du polytope (les premiers) et ceux de la BOÎTE
    DE ROGNAGE (voir `clip_planes`) : une arête posée sur la boîte n'est pas une arête du
    polytope, c'est le bord du champ, et la tracer revient à dessiner la boîte.
    """
    res = []
    for i in range( len( active ) ):
        for j in range( i + 1, len( active ) ):
            shared = active[ i ] & active[ j ]
            if len( shared ) < nb_dims - 1:
                continue
            if nb_real is not None and any( p >= nb_real for p in shared ):
                continue
            res.append( ( i, j ) )
    return np.array( res, np.int64 ).reshape( -1, 2 )


def faces_of( verts, active, edges, nb_dims ):
    """Les faces POLYGONALES, en ordre cyclique -- ce que VTK (ou un rendu) sait tracer.

    Une face est portée par `d-2` plans communs. On ne les cherche pas parmi tous les sous-
    ensembles possibles : chaque ARÊTE en porte déjà `d-1`, donc les faces qui la contiennent
    s'obtiennent en lui en retirant un. Ce qui évite une combinatoire inutile.
    """
    groups = {}
    for i, j in edges:
        shared = active[ i ] & active[ j ]
        for sub in combinations( sorted( shared ), nb_dims - 2 ):
            groups.setdefault( sub, set() ).update( ( int( i ), int( j ) ) )

    res = []
    for ids in groups.values():
        ids = sorted( ids )
        if len( ids ) < 3:
            continue
        order = _cyclic_order( verts[ ids ] )
        if order is not None:
            res.append( [ ids[ k ] for k in order ] )
    return res


def _cyclic_order( pts, tol = 1e-12 ):
    """Range des points COPLANAIRES en tournant autour de leur centre.

    Le plan de la face est trouvé sur place (deux directions indépendantes prises parmi les écarts
    au centre, orthonormalisées) : ça vaut en dimension quelconque, il n'y a pas de « normale » à
    invoquer au-delà de la 3D.
    """
    ctr = pts.mean( axis = 0 )
    rel = pts - ctr
    scale = float( np.linalg.norm( rel, axis = 1 ).max() )
    if scale < tol:
        return None

    u = rel[ int( np.argmax( np.linalg.norm( rel, axis = 1 ) ) ) ]
    u = u / np.linalg.norm( u )
    perp = rel - np.outer( rel @ u, u )
    k = int( np.argmax( np.linalg.norm( perp, axis = 1 ) ) )
    if np.linalg.norm( perp[ k ] ) < tol * scale:
        return None                                        # points alignés : pas un polygone
    w = perp[ k ] / np.linalg.norm( perp[ k ] )
    return list( np.argsort( np.arctan2( rel @ w, rel @ u ) ) )


def polytope_mesh( dirs, offs, bounds = None ):
    """`( verts [n, d], edges [m, 2], faces )` d'un polytope donné en demi-espaces.

    `bounds` ajoute une boîte de rognage (voir `clip_planes`) : indispensable si le polytope peut
    être non borné, sans effet sinon.
    """
    dirs = np.asarray( dirs, np.float64 )
    offs = np.asarray( offs, np.float64 ).reshape( -1 )

    # les plans dégénérés sont écartés ICI et non dans `vertices_of` : celui-ci renumérote ce
    # qu'il garde, et la frontière `nb_real` ne s'y retrouverait plus.
    nrm = np.linalg.norm( dirs, axis = 1 )
    dirs, offs = dirs[ nrm > 1e-12 ], offs[ nrm > 1e-12 ]
    nb_real = len( dirs )

    if bounds is not None:
        cd, co = clip_planes( bounds )
        dirs, offs = np.concatenate( [ dirs, cd ] ), np.concatenate( [ offs, co ] )

    verts, active = vertices_of( dirs, offs )
    if len( verts ) == 0:
        return verts, np.zeros( ( 0, 2 ), np.int64 ), []
    d = dirs.shape[ 1 ]

    # deux listes d'arêtes, et ce n'est pas une inélégance : les FACES ont besoin de toutes (une
    # face de rognage est faite d'arêtes de rognage, et sans elle on verrait l'intérieur du
    # polytope), le TRACÉ n'a besoin que de celles du polytope.
    edges = edges_of( active, d )
    return verts, edges_of( active, d, nb_real ), faces_of( verts, active, edges, d )

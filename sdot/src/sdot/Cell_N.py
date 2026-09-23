"""Un polytope SIMPLE en dimension >= 3 : la connectivité est portée par les sommets.

Chaque sommet est sur EXACTEMENT `d` plans ( position générale ), et nomme ses `d` coupes
( `vertex_cuts`, indices dans `cut_ids`, croissants ) et ses `d` voisins ( `vertex_nbrs` ) -- le
voisin `r` étant « en face » de la coupe `r` : de l'autre côté de l'arête portée par les `d - 1`
autres coupes. C'est cette bijection qui rend les faces et les arêtes lisibles sans rien chercher.
Les plans se relisent sur les sommets de chaque face. Côté noyau : `cell/LocalN.h`.
"""

import numpy as np
from loom.tensor import Axis, CtShapeVar, IntTensor, RealTensor, ShapeVar

from .Cell import Cell, Item
from .CellScratch import words_of


class Cell_N( Cell ):
    vertex_positions : RealTensor[ "num_vertex", "dim_axis" ]
    vertex_cuts      : IntTensor [ "num_vertex", "dim_axis", dict( size = 32 ) ]
    vertex_nbrs      : IntTensor [ "num_vertex", "dim_axis", dict( size = 32 ) ]
    cut_ids          : IntTensor [ "num_cut", dict( size = 32 ) ]

    num_vertex       : Axis[ "nb_vertices" ]
    num_cut          : Axis[ "nb_cuts" ]
    dim_axis         : Axis[ "nb_dims" ]
    nb_vertices      : ShapeVar
    nb_cuts          : ShapeVar
    nb_dims          : CtShapeVar

    default_nb_dims = 3
    _GEOMETRY = ( "vertex_positions", "vertex_cuts", "vertex_nbrs", "cut_ids" )

    # ---- ce que le noyau demande ---------------------------------------------------------------

    def scratch_words( self, cap, fp_size ):
        """`LocalN::words_for( cap )` : `cap` sommets ET `cap` coupes, avec les temporaires de la
        coupe, de la compaction et de la triangulation"""
        d = self.dim
        return ( 4 * d + 5 ) * words_of( fp_size // 8, cap ) + ( 5 * d + 10 ) * words_of( 4, cap )

    def init_capacity( self ):
        return 2 ** self.dim

    def _cut_capacities( self ):
        """Ce qu'une coupe peut produire au plus : une coupe de plus, et pour les sommets, en 3D,
        Euler sur un polytope simple ( `V = 2 F - 4` ), donc `2 ( nc + 1 ) - 4`. Au-delà il n'y a
        pas de borne linéaire en `F` ; chaque sommet neuf est sur une arête traversante, et un
        polytope simple a `nv d / 2` arêtes."""
        d = self.dim
        cap_c = self._cap_c() + 1
        cap_v = 2 * cap_c - 4 if d == 3 else self._cap_v() * ( 1 + d // 2 )
        return max( cap_v, self.init_capacity() ), cap_c

    # ---- lire un item, et ce qui s'en dérive -------------------------------------------------------

    def _item( self, b ):
        d = self.dim
        if self.vertex_positions.raw is None:              # jamais écrite : pas un sommet
            return Item( np.zeros( ( 0, d ) ), np.zeros( 0, int ), vc = np.zeros( ( 0, d ), int ), vn = np.zeros( ( 0, d ), int ) )
        nv, nc = self._count( self.nb_vertices, b ), self._count( self.nb_cuts, b )
        return Item( self._rows( self.vertex_positions, b, nv ).astype( float ),
                     self._rows( self.cut_ids, b, nc ).astype( int ),
                     vc = self._rows( self.vertex_cuts, b, nv ).astype( int ),
                     vn = self._rows( self.vertex_nbrs, b, nv ).astype( int ) )

    def _vertex_cut_indices_of( self, it ):
        return it.vc

    def _edges_of( self, it ):
        """une arête par paire `( a, vn[ a, r ] )`, chacune vue une fois"""
        nv, d = it.vn.shape
        a = np.repeat( np.arange( nv ), d )
        b = it.vn.reshape( -1 )
        keep = a < b
        return np.stack( [ a[ keep ], b[ keep ] ], axis = 1 )

    def _edge_cuts_of( self, it ):
        """l'arête en face de la coupe `r` du sommet `a` est portée par ses `d - 1` autres coupes"""
        nv, d = it.vn.shape
        res = []
        for a in range( nv ):
            for r in range( d ):
                if a < int( it.vn[ a, r ] ):
                    res.append( [ int( it.vc[ a, q ] ) for q in range( d ) if q != r ] )
        return np.asarray( res, int ).reshape( -1, d - 1 )

    def _faces_of( self, it ):
        """En 3D seulement : pour chaque coupe, la marche de voisin en voisin sur sa face -- de `v`
        on va vers `vn[ v, r ]` pour les deux `r` dont la coupe `vc[ v, r ]` n'est PAS la face."""
        nv, d = it.vc.shape
        if d != 3:
            return []
        res = []
        for k in range( len( it.cid ) ):
            on = np.nonzero( ( it.vc == k ).any( axis = 1 ) )[ 0 ]
            if len( on ) < 3:
                continue
            start = int( on[ 0 ] )
            cycle, prev, cur = [ start ], -1, start
            while True:
                nxt = [ int( it.vn[ cur, r ] ) for r in range( d ) if it.vc[ cur, r ] != k ]
                nxt = [ w for w in nxt if w != prev ]
                if not nxt or nxt[ 0 ] == start or len( cycle ) > nv:
                    break
                cycle.append( nxt[ 0 ] )
                prev, cur = cur, nxt[ 0 ]
            if len( cycle ) >= 3:
                res.append( cycle )
        return res

    def _planes_of( self, it ):
        """Le vecteur nul de l'espace engendré par les sommets de chaque face -- par SVD, qui ne
        dépend ni de l'ordre ni de sommets confondus -- orienté vers l'extérieur par le sommet LE
        PLUS LOIN du plan ( pas le premier venu, qui peut être confondu avec la face à 1e-16 près )."""
        nv, d = it.vp.shape
        dirs, offs = np.zeros( ( len( it.cid ), d ) ), np.zeros( len( it.cid ) )
        for k in range( len( it.cid ) ):
            on = np.nonzero( ( it.vc == k ).any( axis = 1 ) )[ 0 ]
            if len( on ) < d:
                continue
            pts = it.vp[ on ]
            _, sv, vt = np.linalg.svd( pts - pts.mean( axis = 0 ), full_matrices = True )
            if len( sv ) >= d and sv[ d - 2 ] <= 1e-9 * max( sv[ 0 ], 1e-300 ):
                continue                                    # pas un hyperplan : face dégénérée
            n = vt[ -1 ]
            off = float( n @ pts[ 0 ] )
            others = np.setdiff1d( np.arange( nv ), on )
            if len( others ):
                sd = it.vp[ others ] @ n - off
                if sd[ np.argmax( np.abs( sd ) ) ] > 0:
                    n, off = -n, -off
            dirs[ k ], offs[ k ] = n, off
        return dirs, offs

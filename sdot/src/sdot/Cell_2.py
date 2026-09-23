"""Un POLYGONE : les sommets en ordre cyclique sont toute la géométrie.

L'invariant : LA COUPE `i` PORTE L'ARÊTE `[ v_i, v_i+1 ]`, donc le sommet `i` est le coin des
coupes `i-1` et `i`, et `nb_cuts == nb_vertices`. Rien d'autre n'est stocké -- le plan d'une coupe
se relit sur son arête ( `_planes_of` ). Côté noyau : `cell/Local2.h`, et sur CPU dans un diagramme
le noyau à registres `cell/Moteur2Reg.h`.
"""

import numpy as np
from loom.tensor import Axis, CtShapeVar, IntTensor, RealTensor, ShapeVar

from .Cell import Cell, Item
from .CellScratch import words_of


class Cell_2( Cell ):
    vertex_positions : RealTensor[ "num_vertex", "dim_axis" ]
    cut_ids          : IntTensor [ "num_cut", dict( size = 32 ) ]

    num_vertex       : Axis[ "nb_vertices" ]
    num_cut          : Axis[ "nb_cuts" ]
    dim_axis         : Axis[ "nb_dims" ]
    nb_vertices      : ShapeVar
    nb_cuts          : ShapeVar
    nb_dims          : CtShapeVar

    default_nb_dims = 2
    _GEOMETRY = ( "vertex_positions", "cut_ids" )

    # ---- ce que le noyau demande ---------------------------------------------------------------

    @staticmethod
    def scratch_words( cap, fp_size ):
        """`Local2::words_for( cap )` : les sommets, les identifiants, les plans, les temporaires"""
        return 8 * words_of( fp_size // 8, cap ) + words_of( 4, cap )

    def init_capacity( self ):
        return 4

    def _cut_capacities( self ):
        """un demi-plan retire un morceau du polygone et ajoute une arête : au plus UN sommet de
        plus, et autant de coupes que de sommets"""
        cap = self._cap_v() + 1
        return cap, cap

    # ---- lire un item, et ce qui s'en dérive -------------------------------------------------------

    def _item( self, b ):
        if self.vertex_positions.raw is None:              # jamais écrite : pas un sommet
            return Item( np.zeros( ( 0, 2 ) ), np.zeros( 0, int ) )
        nv, nc = self._count( self.nb_vertices, b ), self._count( self.nb_cuts, b )
        return Item( self._rows( self.vertex_positions, b, nv ).astype( float ),
                     self._rows( self.cut_ids, b, nc ).astype( int ) )

    def _vertex_cut_indices_of( self, it ):
        """le sommet `i` est le coin des coupes `i-1` et `i`"""
        nv = it.nb_vertices
        i = np.arange( nv )
        return np.stack( [ ( i - 1 ) % nv, i ], axis = 1 ) if nv else np.zeros( ( 0, 2 ), int )

    def _edges_of( self, it ):
        nv = it.nb_vertices
        i = np.arange( nv )
        return np.stack( [ i, ( i + 1 ) % nv ], axis = 1 ) if nv else np.zeros( ( 0, 2 ), int )

    def _edge_cuts_of( self, it ):
        """l'arête `i` est portée par la coupe `i`"""
        return np.arange( it.nb_vertices )[ :, None ]

    def _faces_of( self, it ):
        """une seule face : le polygone lui-même"""
        return [ list( range( it.nb_vertices ) ) ] if it.nb_vertices >= 3 else []

    def _planes_of( self, it ):
        """la normale sortante unitaire de l'arête `[ v_i, v_i+1 ]` -- `( dy, -dx )` pour un polygone
        direct -- et son offset lu sur `v_i`"""
        nv = it.nb_vertices
        dirs, offs = np.zeros( ( len( it.cid ), 2 ) ), np.zeros( len( it.cid ) )
        if nv == 0:
            return dirs, offs
        e = it.vp[ ( np.arange( nv ) + 1 ) % nv ] - it.vp
        n = np.stack( [ e[ :, 1 ], - e[ :, 0 ] ], axis = 1 )
        lng = np.linalg.norm( n, axis = 1 )
        ok = lng > 0
        dirs[ ok ] = n[ ok ] / lng[ ok, None ]
        offs = ( dirs * it.vp ).sum( axis = 1 )
        return dirs, offs

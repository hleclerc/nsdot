"""Un SEGMENT -- ou une demi-droite, ou la droite entière.

Deux sommets, `vertex_positions[ 0 ] <= vertex_positions[ 1 ]`, une coupe par bout ( `cut_ids[ 0 ]`
porte le bout gauche, `cut_ids[ 1 ]` le droit ). Un bout marqué `INFINITE` est posé à une distance
inventée. Côté noyau : `cell/Local1.h`.
"""

import numpy as np
from loom.tensor import Axis, CtShapeVar, IntTensor, RealTensor, ShapeVar

from .Cell import Cell, Item
from .CellScratch import words_of


class Cell_1( Cell ):
    vertex_positions : RealTensor[ "num_vertex", "dim_axis" ]
    cut_ids          : IntTensor [ "num_cut", dict( size = 32 ) ]

    num_vertex       : Axis[ "nb_vertices" ]
    num_cut          : Axis[ "nb_cuts" ]
    dim_axis         : Axis[ "nb_dims" ]
    nb_vertices      : ShapeVar
    nb_cuts          : ShapeVar
    nb_dims          : CtShapeVar

    default_nb_dims = 1
    _GEOMETRY = ( "vertex_positions", "cut_ids" )

    # ---- ce que le noyau demande ---------------------------------------------------------------

    @staticmethod
    def scratch_words( cap, fp_size ):
        """`Local1::words_for( cap )`"""
        return words_of( fp_size // 8, cap ) + words_of( 4, cap )

    def init_capacity( self ):
        return 2

    def _cut_capacities( self ):
        """une coupe déplace un bout : jamais plus de deux sommets"""
        return 2, 2

    # ---- lire un item, et ce qui s'en dérive -------------------------------------------------------

    def _item( self, b ):
        if self.vertex_positions.raw is None:              # jamais écrite : pas un sommet
            return Item( np.zeros( ( 0, 1 ) ), np.zeros( 0, int ) )
        nv, nc = self._count( self.nb_vertices, b ), self._count( self.nb_cuts, b )
        return Item( self._rows( self.vertex_positions, b, nv ).astype( float ),
                     self._rows( self.cut_ids, b, nc ).astype( int ) )

    def _vertex_cut_indices_of( self, it ):
        """le bout `i` est le coin de la coupe `i`"""
        return np.arange( it.nb_vertices )[ :, None ]

    def _edges_of( self, it ):
        return np.array( [ [ 0, 1 ] ] ) if it.nb_vertices == 2 else np.zeros( ( 0, 2 ), int )

    def _edge_cuts_of( self, it ):
        """l'arête -- le segment -- n'est portée par aucune coupe"""
        return np.zeros( ( 1 if it.nb_vertices == 2 else 0, 0 ), int )

    def _faces_of( self, it ):
        return []

    def _planes_of( self, it ):
        dirs = np.zeros( ( len( it.cid ), 1 ) )
        offs = np.zeros( len( it.cid ) )
        if it.nb_vertices == 2:
            dirs[ 0, 0 ], dirs[ 1, 0 ] = -1.0, 1.0
            offs = dirs[ :, 0 ] * it.vp[ :, 0 ]
        return dirs, offs

"""Un polytope convexe, et ce que tous les régimes de dimension ont en commun.

`Cell( nb_dims, ... )` construit la classe qui convient à la dimension -- `Cell_1` ( un segment ),
`Cell_2` ( un polygone ), `Cell_N` ( un polytope simple en dimension >= 3 ) -- chacune dans son
fichier, avec exactement ses tenseurs. Ce fichier ne contient que le CONTRAT et le tronc commun :

  * ce qui est stocké est le format PRATIQUE ( `vertex_positions [ nv, d ]` dans le flottant de
    l'appelant, `cut_ids [ nc ]` ), transféré vers la forme du noyau à chaque appel
    ( `cell/Local*.h`, dans le flottant du noyau -- `kernel_dtype`, `float32` par défaut ) ;
  * les opérations sont des kernels sur un scratch dimensionné par l'hôte ( `CellScratch` ) ;
  * les tenseurs « pratiques » qui ne sont pas stockés -- plans, arêtes, faces, bornage -- sont
    DÉRIVÉS côté hôte, chaque régime disant comment les lire sur ses propres tenseurs.

`cut_ids` porte l'identité des coupes : l'indice du germe d'en face pour une bissectrice, un
entier négatif sinon ( `cell/Ids.h` ). Une paroi `INFINITE` qui porte encore un sommet dit que la
cellule n'est pas bornée.
"""

import os

import numpy as np
from loom.compilation.FfiCode import FfiCodeParallel
from loom.drivers.driver import driver
from loom.tensor import Axis, RealTensor, ShapeVar, Tensor
from loom.util import Aggregate

from .CellScratch import CellScratch, fp_size, merge_call
from . import cell_viz

# les identifiants de coupe qui ne désignent pas un germe -- voir `cell/Ids.h`, qui fait foi
INFINITE = -2 ** 31          # une paroi du simplexe de remplacement d'une cellule non bornée
PIECE    = -2 ** 31 + 1      # un plan de découpe ajouté par une distribution
BOUNDARY = -1                # « pas un germe », sans plus de précision ( = `domain_id( 0 )` )

# LE FLOTTANT DU NOYAU. La géométrie se coupe en `float32` par défaut -- c'est ce qui tient huit
# sommets dans un registre ( `cell/Moteur2Reg.h` ) -- et tout ce qu'on en tire ( une mesure, un
# gradient ) se calcule dans le flottant de l'appelant. `SDOT_KTYPE=FP64` change le défaut ;
# `kernel_dtype = ...` le change pour une cellule ( ou un diagramme ).
DEFAULT_KERNEL_DTYPE = os.environ.get( "SDOT_KTYPE", "FP32" )


def set_kernel_dtype( dtype ):
    """Le flottant du noyau pour les cellules ( et diagrammes ) construits DÉSORMAIS sans
    `kernel_dtype` explicite : `"FP32"` ( le défaut ) ou `"FP64"`. Rend l'ancien réglage."""
    global DEFAULT_KERNEL_DTYPE
    previous = DEFAULT_KERNEL_DTYPE
    DEFAULT_KERNEL_DTYPE = dtype
    return previous


def kernel_dtype():
    return DEFAULT_KERNEL_DTYPE


def cell_class_for( nb_dims ):
    from .Cell_1 import Cell_1
    from .Cell_2 import Cell_2
    from .Cell_N import Cell_N
    return { 1: Cell_1, 2: Cell_2 }.get( int( nb_dims ), Cell_N )


class Item:
    """La géométrie d'UN item d'une cellule ( batchée ou non ), en numpy : `vp [ nv, d ]`,
    `cid [ nc ]`, et ce que le régime y ajoute ( `vc` / `vn` pour `Cell_N` )."""
    def __init__( self, vp, cid, **extra ):
        self.vp = vp
        self.cid = cid
        self.__dict__.update( extra )

    @property
    def nb_vertices( self ):
        return len( self.vp )


class Cell( Aggregate ):
    # ---- ce qu'un régime doit fournir -------------------------------------------------------------
    #
    #   default_nb_dims      la dimension quand la classe est construite sans en donner
    #   _GEOMETRY            les tenseurs qu'une coupe réécrit
    #   scratch_words( cap, fp_size )   la même formule que `Local*::words_for`
    #   init_capacity()      la place d'un hypercube
    #   _cut_capacities()    `( sommets, coupes )` : ce qu'une coupe peut produire au plus
    #   _item( b )           la géométrie de l'item `b`, lue sur ses tenseurs
    #   _vertex_cut_indices_of( it ), _edges_of( it ), _edge_cuts_of( it ), _faces_of( it ),
    #   _planes_of( it )     les tenseurs dérivés, lus sur un `Item`

    def __new__( cls, nb_dims = None, *args, **kwargs ):
        if cls is Cell:
            if nb_dims is None:
                raise TypeError( "Cell( nb_dims, ... ) : la dimension décide de la classe" )
            cls = cell_class_for( nb_dims )
        return super().__new__( cls )

    def __init__( self, nb_dims = None, init_as_unbounded = True, batch_axes = None, kernel_dtype = None ):
        nb_dims = int( nb_dims if nb_dims is not None else self.default_nb_dims )
        self._kernel_dtype = kernel_dtype or DEFAULT_KERNEL_DTYPE
        self.__base_init__( nb_dims = nb_dims, batch_axes = batch_axes )
        if init_as_unbounded:
            self.init_as_unbounded()

    @property
    def kernel_dtype( self ):
        return self._kernel_dtype

    @property
    def dim( self ):
        return int( self.nb_dims.value )

    @classmethod
    def make_hypercube( cls, nb_dims, origin = None, axes = None, cut_id = BOUNDARY, batch_axes = None, **kwargs ):
        res = cls( nb_dims, init_as_unbounded = False, batch_axes = batch_axes, **kwargs )
        res.init_as_hypercube( origin, axes, cut_id )
        return res

    @classmethod
    def make_unbounded( cls, nb_dims, batch_axes = None, **kwargs ):
        return cls( nb_dims, batch_axes = batch_axes, **kwargs )

    def _empty_like_me( self ):
        """Une cellule de même régime, même batch, même noyau, sans géométrie."""
        return type( self )( self.dim, init_as_unbounded = False, batch_axes = self.batch_axes or None,
                             kernel_dtype = self._kernel_dtype )

    # ---- les capacités, et le scratch d'un appel -----------------------------------------------

    def _cap_v( self ):
        """La capacité en sommets dont on part : le compte quand l'hôte le connaît, la capacité
        allouée sinon ( un compte écrit par un kernel est une valeur device sous un tracé )."""
        n = self.nb_vertices.static_count()
        if n is not None:
            return max( int( n ), 1 )
        return int( self.nb_vertices.allocated_capacity() or self.init_capacity() )

    def _cap_c( self ):
        n = self.nb_cuts.static_count()
        if n is not None:
            return max( int( n ), 1 )
        return int( self.nb_cuts.allocated_capacity() or self.init_capacity() )

    def _call_scratch( self, name, cap ):
        """le scratch d'un appel sur CETTE cellule : une ligne par item, `cap` sommets"""
        return CellScratch.for_call( name, self.scratch_words( cap, fp_size( self._kernel_dtype ) ),
                                     self._kernel_dtype, batch_axes = self.batch_axes or None )

    # ---- les kernels ------------------------------------------------------------------------------

    def init_as_unbounded( self, batch_axes = None ):
        """« Tout l'espace », représenté par un SIMPLEXE dont les parois sont marquées `INFINITE`.

        Ces plans-là ne sont pas de vraies coupes : ce sont des bouche-trous, et leurs offsets sont
        inventés. C'est `cut` qui les repousse au fur et à mesure, jusqu'à ce qu'ils ne changent
        plus rien à la coupe en cours ( `Local2::grow_for` ) ; la cellule redevient bornée le jour
        où plus aucun sommet n'en porte.
        """
        if batch_axes is not None:
            self.apply_batch_axes( batch_axes )
        cap = self.dim + 1
        scratch, sc_kwargs = self._call_scratch( "scratch", cap )
        driver.call(
            FfiCodeParallel( name = "init_as_unbounded",
                fwd_code = "cell( batch_index ).init_as_unbounded( scratch( batch_index ) );" ),
            **merge_call( dict(
                output_capacities = { "cell.nb_vertices": cap, "cell.nb_cuts": cap },
                output_attributes = [ "cell" ] ), sc_kwargs ),
            cell = self,
            scratch = scratch,
        )

    def init_as_hypercube( self, origin = None, axes = None, cut_id = BOUNDARY, batch_axes = None ):
        """le parallélotope `origin + sum_j t_j axes[ j ]`, `t` dans `[ 0, 1 ]^d` -- le cube unité
        par défaut. Toutes les coupes portent `cut_id`."""
        if batch_axes is not None:
            self.apply_batch_axes( batch_axes )

        d = self.dim
        origin = RealTensor[ self.dim_axis ]( np.zeros( d ) if origin is None else origin )
        axes = RealTensor[ self.dim_axis, Axis( ShapeVar( d ), name = "num_axis" ) ]( np.eye( d ) if axes is None else axes )

        cap = self.init_capacity()
        scratch, sc_kwargs = self._call_scratch( "scratch", cap )
        driver.call(
            FfiCodeParallel( name = "init_as_hypercube",
                fwd_code = "cell( batch_index ).init_as_hypercube( scratch( batch_index ), origin, axes, cut_id );" ),
            **merge_call( dict(
                output_capacities = { "cell.nb_vertices": cap, "cell.nb_cuts": cap },
                output_attributes = [ "cell" ] ), sc_kwargs ),
            cut_id = cut_id,
            origin = origin,
            axes = axes,
            cell = self,
            scratch = scratch,
        )

    def cut( self, direction, offset, cut_id = BOUNDARY ):
        """Intersecte la cellule avec le demi-espace `direction . x <= offset`, EN PLACE.

        `direction` n'a pas à être normalisée : `offset` est le produit scalaire auquel elle est
        comparée telle quelle. Les entrées et les sorties d'un `driver.call` étant disjointes, le
        kernel écrit dans une cellule NEUVE et la mise à jour en place n'est qu'un rebinding. La
        place que la coupe demande est BORNÉE d'avance ( `_cut_capacities` ) : pas de second tour.
        """
        direction = RealTensor[ self.dim_axis ]( direction )
        offset = RealTensor[ () ]( offset )

        cap_v, cap_c = self._cut_capacities()
        res = self._empty_like_me()
        scratch, sc_kwargs = self._call_scratch( "scratch", max( cap_v, cap_c ) )
        driver.call(
            FfiCodeParallel( name = "cut",
                fwd_code = "cell( batch_index ).cut( res( batch_index ), scratch( batch_index ), direction, offset, cut_id );" ),
            **merge_call( dict(
                output_capacities = { "res.nb_vertices": cap_v, "res.nb_cuts": cap_c },
                output_attributes = [ "res" ] ), sc_kwargs ),
            cut_id = cut_id,
            direction = direction,
            offset = offset,
            cell = self,
            res = res,
            scratch = scratch,
        )
        self._adopt_geometry( res )
        return self

    def _adopt_geometry( self, other ):
        """Reprend sur `self` ce que le kernel vient d'écrire dans `other` : les VALEURS ( le
        stockage des tenseurs, les comptes ), pas les objets `Attribute`, dont l'identité doit
        survivre à la coupe."""
        for name in self._GEOMETRY:
            getattr( self, name ).set( getattr( other, name ) )
        # un compte écrit par un kernel est une valeur DEVICE : `set_count`, pas `set`
        for name in ( "nb_vertices", "nb_cuts" ):
            getattr( self, name ).set_count( getattr( other, name ).raw )

    @property
    def measure( self ) -> Tensor:
        """La mesure de la cellule : longueur, aire, volume -- dans le flottant de l'appelant, quel
        que soit celui du noyau. `TF::max` pour une cellule non bornée. Dérivable par rapport à
        `vertex_positions`."""
        res = RealTensor[ tuple( self.batch_axes ) ]()
        scratch, sc_kwargs = self._call_scratch( "scratch", max( self._cap_v(), self._cap_c() ) )
        driver.call(
            FfiCodeParallel( name = "measure",
                fwd_code = "cell( batch_index ).measure( res( batch_index ), scratch( batch_index ) );",
                bwd_code = "cell( batch_index ).measure_bwd( res( batch_index ), grad_for_res( batch_index ), "
                           "grad_for_cell( batch_index ).vertex_positions, scratch( batch_index ) );" ),
            **merge_call( dict( output_attributes = [ "res" ] ), sc_kwargs ),
            cell = self,
            res = res,
            scratch = scratch,
        )
        return res

    # ---- lire un item ------------------------------------------------------------------------------

    @property
    def nb_items( self ):
        return int( np.prod( [ int( ax.max ) for ax in self.batch_axes ] ) ) if self.batch_axes else 1

    def _count( self, shape_var, b ):
        """le compte de l'item `b` -- un compte écrit par un kernel en a un par item, un compte
        connu de l'hôte est le même pour tous"""
        v = np.atleast_1d( np.asarray( shape_var.value ) ).reshape( -1 ).astype( int )
        return int( v[ b ] if v.size == self.nb_items else v[ 0 ] )

    def _rows( self, tensor, b, count ):
        """les `count` premières lignes de l'item `b` du tenseur `tensor` ( `[ items..., cap, ... ]` ).
        Les axes de batch sont aplatis, et RIEN d'autre : sur un GPU le batch est rembourré à
        l'alignement du device (`Device.batch_alignment`), l'item `b` est à la ligne `b` d'un
        tampon qui en a plus que `nb_items`."""
        raw = np.asarray( tensor.raw )
        return raw.reshape( ( -1, ) + raw.shape[ len( self.batch_axes ) : ] )[ b ][ : count ]

    def _items( self ):
        return [ self._item( b ) for b in range( self.nb_items ) ]

    def _per_item( self, of_item ):
        """`of_item( item )` pour chaque item ; la valeur elle-même si la cellule n'est pas batchée."""
        if not self.batch_axes:
            return of_item( self._item( 0 ) )
        return [ of_item( it ) for it in self._items() ]

    def vertices( self, item = 0 ):
        """`vertex_positions` de l'item `item`, en numpy `[ nb_vertices, d ]` -- sans le padding
        d'un batch."""
        return self._item( item ).vp

    # ---- les tenseurs « pratiques », dérivés ------------------------------------------------------

    def _infinite_cuts_of( self, it ):
        """le masque des coupes `INFINITE` qui portent encore un sommet"""
        alive = np.zeros( len( it.cid ), bool )
        vc = self._vertex_cut_indices_of( it )
        if len( vc ):
            alive[ np.unique( vc.reshape( -1 ) ) ] = True
        return ( it.cid == INFINITE ) & alive

    @property
    def is_bounded( self ):
        """Vrai si aucune paroi `INFINITE` ne porte plus de sommet."""
        return self._per_item( lambda it: not self._infinite_cuts_of( it ).any() )

    @property
    def vertex_cut_indices( self ):
        """`[ nb_vertices, d ]` : les `d` coupes ( indices dans `cut_ids` ) dont chaque sommet est le coin."""
        return self._per_item( self._vertex_cut_indices_of )

    @property
    def edges( self ):
        """`[ nb_edges, 2 ]` : les arêtes, en indices de sommets."""
        return self._per_item( self._edges_of )

    @property
    def faces( self ):
        """Une liste de cycles d'indices de sommets, une face par coupe qui en porte ( 2D et 3D )."""
        return self._per_item( self._faces_of )

    @property
    def cut_planes( self ):
        """`( directions[ nb_cuts, d ], offsets[ nb_cuts ] )`, relus sur la géométrie -- la normale
        sortante UNITAIRE de chaque face et son offset. Une coupe sans sommet a une direction nulle."""
        return self._per_item( self._planes_of )

    @property
    def cut_directions( self ):
        return self._per_item( lambda it: self._planes_of( it )[ 0 ] )

    @property
    def cut_offsets( self ):
        return self._per_item( lambda it: self._planes_of( it )[ 1 ] )

    # ---- l'affichage ------------------------------------------------------------------------------

    def add_to_viz( self, viz, color = None, opacity = 1.0, faces = True, edges = True, points = False ):
        """Se dessine dans un `Visualizer` ( voir `sdot.viz.Visualizer`, et `cell_viz` pour ce qu'une
        cellule non bornée laisse tomber ). Chaque item prend sa couleur à son RANG DANS LE BATCH,
        sur un bloc réservé d'avance : la couleur d'une cellule d'un diagramme dit QUEL germe."""
        return cell_viz.add_to_viz( self, viz, color, opacity, faces, edges, points )

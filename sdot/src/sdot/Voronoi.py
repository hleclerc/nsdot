import numpy as np

from loom.compilation.FfiCode import FfiCodeParallel
from loom.drivers.driver import driver
from loom.tensor import Affine, Axis, CtShapeVar, IntTensor, RealTensor, ShapeVar, Tensor, new_batch_axis
from loom.util import Aggregate

from .Cell import BOUNDARY, Cell


class Voronoi( Aggregate ):
    """Diagramme de Voronoï euclidien, RECONSTRUIT à chaque demande.

    L'objet ne porte pas de diagramme : il porte ses GERMES (`positions`) et le domaine convexe
    qui les borne (`bnd_directions . x <= bnd_offsets`). Une cellule n'existe qu'entre l'instant
    où le kernel la construit et l'instant où il en lit la réponse -- `measures` écrit un volume
    directement dans sa case, puis passe au germe suivant en réutilisant les mêmes tampons.

    Ce que coûte une requête en mémoire est donc ce que coûte UNE cellule, multiplié par le
    nombre de work-items, et jamais par le nombre de germes -- sauf `cells`, qui est la requête
    « garde-les toutes » et n'existe que pour l'affichage : une image demande que tout existe en
    même temps, et c'est la seule chose qui le demande. Deux cellules par work-item, et non
    une : `Cell.cut` écrit son résultat dans une cellule SÉPARÉE (les entrées et les sorties d'un
    appel sont disjointes, cf. `Cell.py::cut`), donc une suite de coupes fait la NAVETTE entre
    deux tampons. Le nombre de work-items est lui-même choisi sur ce budget mémoire
    (`driver.device.nb_threads`), et c'est LUI qui borne le parallélisme -- chaque work-item
    balaie ensuite sa part des germes, à pas fixe.

    Le voisinage n'est pas encore accéléré : chaque cellule est coupée par les `n - 1` bissectrices,
    donc le coût est en `n²`. C'est la structure de recherche (grille / BSP + rayon de sécurité)
    qui manque, pas le reste : elle ne changera que l'énumération des candidats de
    `Voronoi.cxx::make_cell`.
    """

    positions      : RealTensor[ "num_point", "dim" ]

    # le domaine : une liste de demi-espaces, donc n'importe quel convexe polyédrique. Absent
    # (`Unbound`, `NoneTensor` côté C++), les cellules qui partent à l'infini le restent -- et se
    # mesurent comme telles (`Cell::measure` rend `TF::max`).
    bnd_directions : RealTensor[ "num_boundary", "dim" ]
    bnd_offsets    : RealTensor[ "num_boundary" ]

    num_point      : Axis[ "nb_points" ]
    num_boundary   : Axis[ "nb_boundaries" ]
    dim            : Axis[ "nb_dims" ]

    nb_points      : ShapeVar
    nb_boundaries  : ShapeVar
    nb_dims        : CtShapeVar

    def __init__( self, positions, boundaries = None, box = None, max_nb_cuts = None, **kwargs ):
        """`positions` : `[ n, d ]`. Le domaine, au choix :

        - `box = ( mi, ma )` -- le pavé `mi <= x <= ma`, développé ici en `2d` demi-espaces ;
        - `boundaries = ( directions, offsets )` -- les demi-espaces `direction . x <= offset` ;
        - rien -- les cellules du bord restent infinies.

        `max_nb_cuts` dimensionne les tampons d'une cellule (voir `_capacities`). Ce n'est qu'une
        supposition : trop petite, le kernel l'enregistre et la plateforme relance avec le double.
        """
        if box is not None:
            if boundaries is not None:
                raise ValueError( "give either `box` or `boundaries`, not both" )
            boundaries = box_half_spaces( *box )

        # pas de domaine -> on ne NOMME pas les deux tenseurs : les laisser `Unbound` (jamais
        # alloués, `NoneTensor` côté C++) n'est pas la même chose que leur passer `None`, qui est
        # une valeur, et une valeur de rang 0.
        if boundaries is not None:
            kwargs[ "bnd_directions" ], kwargs[ "bnd_offsets" ] = boundaries

        self.__base_init__( positions = positions, **kwargs )

        self._max_nb_cuts = max_nb_cuts


    @property
    def measures( self ) -> Tensor:
        """La mesure de chaque cellule : `[ n ]`, indexé comme `positions`.

        Un seul appel, un seul balayage : chaque work-item construit une cellule dans ses tampons
        à lui, en écrit le volume, et recommence avec le germe suivant. Rien du diagramme n'est
        conservé entre deux germes -- c'est tout l'intérêt.
        """
        d = int( self.nb_dims.value )
        if d < 2:
            raise NotImplementedError( f"Voronoi needs nb_dims >= 2 for now ( nb_dims = { d } )" )

        cap_v, cap_e, cap_c = self._capacities()

        # le budget qui décide du parallélisme : ce qu'UN work-item immobilise. Deux cellules
        # (la navette), plus les scratchs du régime d > 2 -- la table de compaction de la coupe
        # (`corr`) et les apex de la triangulation (`facet_apex`), tous deux dimensionnés sur les
        # DEUX cellules (voir plus bas). Le device en déduit combien de work-items il peut se
        # permettre, plafonné par le nombre de germes (inutile d'en réserver plus qu'il n'y a de
        # travail).
        per_thread = 2 * self._bytes_per_cell( cap_v, cap_e, cap_c )
        if d > 2:
            per_thread += 8 * ( 2 * cap_v + 2 * cap_c + 1 ) + 8 * d * 2 * cap_c
        nt = driver.device.nb_threads( nb_local_bytes_per_thread = per_thread,
                                       batch_axes = [ self.num_point ] )

        # l'axe des work-items est un axe de BATCH : c'est lui qui donne à l'appel son espace
        # d'items, donc un work-item par emplacement de travail. `thread_index` / `nb_threads`
        # (les noms réservés du scaffold) sont alors exactement le rang de ce work-item et leur
        # nombre, et la boucle striée sur les germes se lit directement dessus.
        num_thread = new_batch_axis( nt )

        ws_0 = Cell( d, init_as_unbounded = False, batch_axes = [ num_thread ] )
        ws_1 = Cell( d, init_as_unbounded = False, batch_axes = [ num_thread ] )

        # Les deux scratchs sont dimensionnés en EXPRESSION DES COMPTES DES CELLULES, pas sur des
        # entiers figés : une capacité n'est qu'une supposition, et quand elle ne suffit pas c'est
        # `driver.call` qui la double et relance -- si le scratch ne suivait pas, la coupe suivante
        # écrirait à côté (`corr` est indexé par les anciens sommets PUIS les anciennes coupes,
        # `facet_apex` par le numéro de coupe). La SOMME des deux cellules, et non leur max : les
        # deux ont leur propre compte, la croissance peut n'en toucher qu'un, et une somme majore
        # les deux -- au prix d'un facteur deux sur un scratch déjà petit devant les cellules.
        cuts = Affine.of( ws_0.nb_cuts ) + Affine.of( ws_1.nb_cuts )
        verts = Affine.of( ws_0.nb_vertices ) + Affine.of( ws_1.nb_vertices )

        # Axes NOMMÉS, construits DIRECTEMENT et pas via `Axis[ sv ]( name = ... )` :
        # `Parametrized.__call__` plierait `name` dans les template_kwargs.
        num_corr = Axis( verts + cuts + Affine.constant( 1 ), name = "num_corr" )
        num_level = Axis( ShapeVar( d ), name = "num_level" )
        num_cut_slot = Axis( cuts, name = "num_cut_slot" )

        res = RealTensor[ self.num_point ]()

        # en d <= 2 ces deux-là ne sont PAS déclarés en sortie : ils ne sont donc jamais alloués et
        # arrivent en `NoneTensor`, ce qu'attendent les `if constexpr ( ct_dim > 2 )` du C++.
        scratch = [ "corr", "facet_apex" ] if d > 2 else []

        driver.call(
            FfiCodeParallel( name = "voronoi_measures",
                fwd_code = "voronoi.measures( res, ws_0( batch_index ), ws_1( batch_index ), "
                           "corr( batch_index ), facet_apex( batch_index ), thread_index, nb_threads );" ),
            output_capacities = self._cell_capacities( "ws_0" ) | self._cell_capacities( "ws_1" ),
            output_exceptions = ws_0._face_lattice_exceptions( "ws_0" ) + ws_1._face_lattice_exceptions( "ws_1" ),
            output_attributes = [ "res", "ws_0", "ws_1" ] + scratch,
            # des tampons de travail, pas des résidus : leur contenu ne survit pas à l'appel (les
            # work-items se les repassent d'un germe à l'autre).
            scratch_attributes = [ "ws_0", "ws_1" ] + scratch,
            voronoi = self,
            res = res,
            ws_0 = ws_0,
            ws_1 = ws_1,
            # des INDICES : `IntTensor`, sinon leurs lectures reviendraient en flottants.
            corr = IntTensor[ num_thread, num_corr ](),
            facet_apex = IntTensor[ num_thread, num_level, num_cut_slot ](),
        )

        return res


    @property
    def cells( self ) -> Cell:
        """TOUTES les cellules, en UN appel : une `Cell` batchée sur les germes.

        C'est la requête qui ne réduit pas une cellule à un nombre, donc la seule dont la mémoire
        soit fonction du nombre de germes -- c'est ce qu'est un AFFICHAGE : pour dessiner le
        diagramme il faut que toutes les cellules existent en même temps. On abandonne donc ici, et
        seulement ici, le budget par work-item de `measures` : un work-item par germe, et les deux
        cellules de travail (la navette du clip) coûtent le double de la sortie plutôt qu'un
        forfait. En échange, le tout tient en un `driver.call` -- là où `cell( i )` en fait UN PAR
        COUPE, soit `n²` allers-retours pour dessiner un diagramme.

        La `Cell` rendue se dessine telle quelle : `Cell.add_to_viz` boucle déjà sur les items d'un
        batch, et sait quoi faire d'une cellule non bornée (voir les coupes INFINITE là-bas).
        """
        d = int( self.nb_dims.value )
        if d < 2:
            raise NotImplementedError( f"Voronoi needs nb_dims >= 2 for now ( nb_dims = { d } )" )

        # l'axe des items est celui des CELLULES : un work-item par germe, `batch_index` est donc
        # le germe. Un axe de batch FRAIS et non `self.num_point` : les deux ont la même étendue,
        # mais réutiliser l'axe de `positions` ferait porter à une entrée le nom d'un axe de batch
        # de l'appel, ce qui n'a rien à voir avec ce qu'on veut dire (`positions` est lu EN ENTIER
        # par chaque item, il est indexé par le germe, pas découpé par l'item).
        n = int( self.nb_points.value )
        num_cell = new_batch_axis( n )

        # quel germe ce work-item construit. `batch_index` n'est pas un entier côté C++ mais le
        # multi-indice qui SERT à indexer les tenseurs batchés (un `Tuple<AxisIndex<...>>`) ; le
        # numéro du germe, dont `make_cell` a besoin comme d'un entier, arrive donc par un tenseur
        # d'entrée batché, lu `seeds( batch_index )`. C'est aussi ce qui permettra un jour de ne
        # dessiner qu'un SOUS-ENSEMBLE des cellules sans rien changer au kernel.
        seeds = IntTensor[ num_cell ]( np.arange( n ) )

        cells = Cell( d, init_as_unbounded = False, batch_axes = [ num_cell ] )
        ws_0  = Cell( d, init_as_unbounded = False, batch_axes = [ num_cell ] )
        ws_1  = Cell( d, init_as_unbounded = False, batch_axes = [ num_cell ] )

        # comme dans `measures` : le scratch de compaction suit les comptes des cellules de travail
        # par une expression affine, pour que le doublement d'une capacité l'emmène avec lui.
        num_corr = Axis( Affine.of( ws_0.nb_vertices ) + Affine.of( ws_1.nb_vertices )
                       + Affine.of( ws_0.nb_cuts )     + Affine.of( ws_1.nb_cuts )
                       + Affine.constant( 1 ), name = "num_corr" )

        scratch = [ "corr" ] if d > 2 else []

        driver.call(
            FfiCodeParallel( name = "voronoi_cells",
                fwd_code = "voronoi.build_cell( SI( seeds( batch_index ) ), cells( batch_index ), "
                           "ws_0( batch_index ), ws_1( batch_index ), corr( batch_index ) );" ),
            output_capacities = ( self._cell_capacities( "cells" )
                                | self._cell_capacities( "ws_0" )
                                | self._cell_capacities( "ws_1" ) ),
            output_exceptions = ( cells._face_lattice_exceptions( "cells" )
                                + ws_0._face_lattice_exceptions( "ws_0" )
                                + ws_1._face_lattice_exceptions( "ws_1" ) ),
            output_attributes = [ "cells", "ws_0", "ws_1" ] + scratch,
            # `cells` est LA sortie ; les deux autres et `corr` ne sont que la navette du clip.
            scratch_attributes = [ "ws_0", "ws_1" ] + scratch,
            voronoi = self,
            seeds = seeds,
            cells = cells,
            ws_0 = ws_0,
            ws_1 = ws_1,
            corr = IntTensor[ num_cell, num_corr ](),
        )

        return cells


    def cell( self, i ) -> Cell:
        """La cellule du germe `i`, construite CÔTÉ PYTHON -- un `driver.call` par coupe.

        Le chemin lent, et volontairement : c'est la même géométrie obtenue par une orchestration
        entièrement différente de celle du kernel, donc l'ORACLE des tests, et de quoi inspecter
        une cellule sans écrire de kernel pour ça. Ce n'est PAS le chemin d'affichage : dessiner
        `n` cellules par ici coûte `n²` allers-retours avec le device, voir `cells`.
        """
        d = int( self.nb_dims.value )
        pos = np.asarray( self.positions ).reshape( -1, d )

        res = Cell.make_unbounded( d )
        if self.bnd_directions.is_defined:
            bds = np.asarray( self.bnd_directions ).reshape( -1, d )
            bos = np.asarray( self.bnd_offsets ).reshape( -1 )
            for b in range( len( bds ) ):
                res.cut( bds[ b ], float( bos[ b ] ), BOUNDARY )

        p0 = pos[ i ]
        for j in range( len( pos ) ):
            if j == i:
                continue
            direction = pos[ j ] - p0
            res.cut( direction, float( direction @ ( p0 + pos[ j ] ) / 2 ), j )
        return res


    def add_to_viz( self, viz, **kwargs ):
        """Se dessine dans un `Visualizer` : toutes les cellules, en un appel (voir `cells`)."""
        return self.cells.add_to_viz( viz, **kwargs )


    # -- ce qu'UNE cellule immobilise ---------------------------------------------------------

    def _capacities( self ):
        """`( sommets, arêtes, coupes )` : la taille des tampons d'une cellule.

        Une SUPPOSITION, pas un contrat -- si elle ne suffit pas, le kernel l'enregistre, sort sans
        rien écrire de faux, et la plateforme relance avec le double (voir `driver.call`). On part
        du nombre de coupes parce que c'est la seule des trois qu'on sache estimer : une cellule de
        Voronoï en 3D a une quinzaine de faces, et le reste s'en déduit par les relations d'un
        polytope SIMPLE (ce que `Cell.cut` maintient) -- `V = 2F - 4`, `E = 3F - 6` en 3D, une borne
        beaucoup plus lâche au-delà.
        """
        d = int( self.nb_dims.value )
        cap_c = self._max_nb_cuts or 32
        if d == 2:
            return cap_c, 0, cap_c          # l'invariant du chemin 2D : une coupe par sommet
        if d == 3:
            return 2 * cap_c, 3 * cap_c, cap_c
        return 4 * ( d - 2 ) * cap_c, 8 * ( d - 2 ) * cap_c, cap_c

    def _cell_capacities( self, name ):
        cap_v, cap_e, cap_c = self._capacities()
        res = { f"{ name }.nb_vertices": cap_v, f"{ name }.nb_cuts": cap_c }
        if int( self.nb_dims.value ) > 2:
            res[ f"{ name }.nb_edges" ] = cap_e
        return res

    def _bytes_per_cell( self, cap_v, cap_e, cap_c ):
        d = int( self.nb_dims.value )
        res = cap_v * d + cap_c * d + 2 * cap_c + 1                 # V-rep, H-rep, le drapeau
        if d > 2:
            res += cap_v * d + cap_e * ( d + 1 )                    # le treillis de faces
        return 8 * res


def box_half_spaces( mi, ma ):
    """Le pavé `mi <= x <= ma` en `2d` demi-espaces `direction . x <= offset`."""
    mi = np.asarray( mi, dtype = float ).reshape( -1 )
    ma = np.asarray( ma, dtype = float ).reshape( -1 )
    if mi.size != ma.size:
        raise ValueError( "`box = ( mi, ma )` wants two corners of the same dimension" )
    d = mi.size
    return np.concatenate( [ np.eye( d ), -np.eye( d ) ] ), np.concatenate( [ ma, -mi ] )

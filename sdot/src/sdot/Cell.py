import numpy as np
from loom.compilation.FfiCode import FfiCodeParallel
from loom.drivers.driver import driver
from loom.tensor import Axis, CtShapeVar, IntTensor, RealTensor, ShapeVar, Tensor
from loom.util import Aggregate

INFINITE = -2
BOUNDARY = -1


class Cell( Aggregate ):
    # tensors used for every number of dim
    vertex_positions : RealTensor[ "num_vertex", "dim" ]
    is_fully_bounded : IntTensor

    cut_directions   : RealTensor[ "num_cut", "dim" ]
    cut_offsets      : RealTensor[ "num_cut" ]
    cut_ids          : IntTensor[ "num_cut" ]

    # tensors used if d > 2 ONLY -- the FACE LATTICE. Below 3D the vertices alone describe the
    # cell ( a segment in 1D, a cyclically ordered polygon in 2D ), so these two -- and the
    # `nb_edges` count that sizes one of them -- are left `Unbound`: never allocated, never
    # written, `NoneTensor` on the C++ side ( see `_face_lattice_exceptions` ).
    vertex_indices   : IntTensor[ "num_vertex", "dim" ]
    edge_indices     : IntTensor[ "num_edge", "ein" ]

    # axes and shape vars
    num_vertex       : Axis[ "nb_vertices" ]
    num_edge         : Axis[ "nb_edges" ]
    num_axis         : Axis[ "nb_dims" ]
    num_cut          : Axis[ "nb_cuts" ]
    ein              : Axis[ "nb_dims + 1" ]
    dim              : Axis[ "nb_dims" ]

    nb_vertices      : ShapeVar
    nb_edges         : ShapeVar
    nb_cuts          : ShapeVar

    nb_dims          : CtShapeVar


    def __init__( self, nb_dims, init_as_unbounded = True, batch_axes = None ):
        # `batch_axes = [ new_batch_axis( n ), ... ]` batches this cell: every declared tensor gains
        # those leading axes. `__base_init__` does the work (and records them on `self.batch_axes`);
        # each axis already carries its size (a prescribed `ShapeVar`), so there is nothing else to
        # thread here.
        self.__base_init__( nb_dims = nb_dims, batch_axes = batch_axes )

        if init_as_unbounded:
            self.init_as_unbounded()

    @classmethod
    def make_hypercube( cls, nb_dims, origin = None, axes = None, cut_id = BOUNDARY, batch_axes = None ):
        res = cls( nb_dims, init_as_unbounded = False, batch_axes = batch_axes )
        res.init_as_hypercube( origin, axes, cut_id )
        return res

    @classmethod
    def make_unbounded( cls, nb_dims, batch_axes = None ):
        return cls( nb_dims, batch_axes = batch_axes )


    def init_as_unbounded( self, batch_axes = None ):
        """« Toute la dimension », représentée par un SIMPLEXE dont les plans sont marqués INFINITE.

        Ces plans-là ne sont pas de vraies coupes : ce sont des bouche-trous, et leurs offsets sont
        inventés (le simplexe unité, à l'origine). C'est `cut` qui les repousse au fur et à mesure,
        jusqu'à ce qu'ils ne changent plus rien à la coupe en cours ; la cellule redevient bornée le
        jour où il n'en reste aucun.
        """
        if batch_axes is not None:
            self.apply_batch_axes( batch_axes )

        driver.call(
            FfiCodeParallel( name = "init_as_unbounded", fwd_code = "cell( batch_index ).init_as_unbounded();" ),
            output_capacities = self._init_capacities_for_hypercube(),
            output_exceptions = self._face_lattice_exceptions( "cell" ),
            output_attributes = [ "cell" ],
            cell = self,
        )

    def init_as_hypercube( self, origin = None, axes = None, cut_id = BOUNDARY, batch_axes = None ):
        # batching an already-built cell: prepend the axes to its tensors before we allocate them
        # (`origin` / `axes` stay unbatched -- shared across items). Harmless if the cell was already
        # batched at construction and none is passed here.
        if batch_axes is not None:
            self.apply_batch_axes( batch_axes )

        origin = RealTensor[ self.dim ]( origin )
        axes = RealTensor[ self.num_axis, self.dim ]( axes )

        driver.call(
            FfiCodeParallel( name = "init_as_hypercube",
                fwd_code = "cell( batch_index ).init_as_hypercube( origin, axes, cut_id );",
                bwd_code = "cell( batch_index ).init_as_hypercube_bwd( origin, axes, grad_for_cell( batch_index ), grad_for_origin( batch_index ), grad_for_axes( batch_index ) );"
            ),
            output_capacities = self._init_capacities_for_hypercube(),
            output_exceptions = self._face_lattice_exceptions( "cell" ),
            output_attributes = [ "cell" ],
            cut_id = cut_id,
            origin = origin,
            axes = axes,
            cell = self,
        )

    def cut( self, direction, offset, cut_id = BOUNDARY ):
        """Intersecte la cellule avec le demi-espace `direction . x <= offset`, EN PLACE.

        `direction` n'a pas à être normalisée : `offset` est le produit scalaire auquel elle est
        comparée telle quelle, donc `( 2n, 2o )` désigne le même demi-espace que `( n, o )`.

        DEUX chemins, comme `measure`, et pour la même raison : ils ne réécrivent pas la même
        description de la cellule (voir `_cut_2d` / `_cut_nd`, et les deux surcharges de
        `Cell.h::cut`). En 2D les sommets suffisent, et les coupes les suivent par l'invariant que
        les ordres de `init_as_hypercube` / `init_as_aligned_simplex` établissent : LA COUPE i PORTE
        L'ARÊTE [ v_i, v_i+1 ], donc `nb_cuts == nb_vertices` -- un seul passage cyclique, aucun
        tableau intermédiaire, donc aucun scratch et aucun plafond de threads. Au-delà, c'est le
        TREILLIS DE FACES qui porte la cellule et que le clip réécrit ; la compaction demande alors
        un petit scratch par thread, donc un plafond (cf. `_nb_corr_items`).

        Sur une cellule NON BORNÉE, le clip est précédé d'une poussée : le simplexe factice qui tient
        lieu de « tout le plan » porte des offsets inventés, et il est repoussé jusqu'à ce que cette
        coupe-ci classe ses sommets comme elle le ferait à l'infini (`Cell.h::growth_for_cut`). Une
        cellule bornée n'a rien de factice : la poussée y vaut exactement zéro et ne coûte rien.
        """
        d = int( self.nb_dims.value )
        if d < 2:
            raise NotImplementedError( f"Cell.cut needs nb_dims >= 2 for now ( nb_dims = { d } )" )

        direction = RealTensor[ self.dim ]( direction )
        offset = RealTensor[ () ]( offset )

        # les entrées et les sorties d'un `driver.call` sont DISJOINTES (comme dans XLA) : le kernel
        # ne peut pas réécrire les tenseurs qu'il lit. On alloue donc une cellule NEUVE, et la mise
        # à jour en place n'est qu'un rebinding côté Python une fois l'appel revenu.
        res = self._empty_like_me()

        # et l'ENTRÉE de l'appel ne doit pas être `self` non plus. Le backward n'est construit que
        # plus tard, quand Jax le réclame, et il relit alors les objets qu'on lui a passés -- or
        # `_adopt_geometry` aura entre-temps rebranché `self` sur les buffers de sortie, plus grands
        # d'un cran. Le gradient de `cell.vertex_positions` serait alloué à cette nouvelle capacité,
        # et Jax refuse une cotangente qui n'a pas la taille de sa primale. D'où l'instantané : les
        # mêmes buffers, mais des `Attribute` que la mise à jour ne touchera pas.
        src = self._empty_like_me()
        src._adopt_geometry( self )

        if d == 2:
            self._cut_2d( src, res, direction, offset, cut_id )
        else:
            self._cut_nd( src, res, direction, offset, cut_id )

        self._adopt_geometry( res )
        return self

    def _cut_2d( self, src, res, direction, offset, cut_id ):
        """d == 2 : le clip cyclique, SANS scratch -- donc sans plafond de threads."""
        driver.call(
            FfiCodeParallel( name = "cut",
                fwd_code = "cell( batch_index ).cut( res( batch_index ), direction, offset, cut_id );",
                # le backward ACCUMULE (le clip est un scatter), donc il faut mettre à zéro ce dans
                # quoi il accumule -- une pré-passe unique, avant le corps par item. Une sortie
                # PAR ITEM n'est pas semée par la plateforme (`cpp_seed_member` ne sème qu'une
                # sortie PARTAGÉE d'un appel batché), et sans batch rien ne l'est du tout.
                bwd_setup_code = "cell.cut_bwd_setup( queue, grad_for_cell, grad_for_direction, grad_for_offset );",
                bwd_code = "cell( batch_index ).cut_bwd( direction, offset, grad_for_res( batch_index ), "
                           "grad_for_cell( batch_index ), grad_for_direction, grad_for_offset );" ),
            output_capacities = self._cut_capacities(),
            output_exceptions = self._face_lattice_exceptions( "res" ),
            output_attributes = [ "res" ],
            cut_id = cut_id,
            direction = direction,
            offset = offset,
            cell = src,
            res = res,
        )

    def _cut_nd( self, src, res, direction, offset, cut_id ):
        """d > 2 : le clip réécrit le TREILLIS DE FACES, et compacte -- d'où un scratch par thread.

        `corr` porte les deux tables de renumérotation (anciens sommets, puis anciennes coupes),
        une LIGNE PAR WORK-ITEM : `corr( thread_index )`. C'est le même principe que le scratch de
        `_measure_nd`, en beaucoup plus petit -- une entrée par sommet et par coupe, pas une par
        simplexe. Le `thread_cap` en découle : la place que prennent ces lignes borne le nombre de
        threads simultanés, et il est lu À L'EXÉCUTION sur le tenseur (`corr.shape( 0 )`), jamais
        gravé dans le source, pour qu'un même kernel compilé serve toutes les machines.
        """
        nb_corr_items = self._nb_corr_items()
        nt = driver.device.nb_threads(
            nb_local_bytes_per_thread = 8 * nb_corr_items,
            batch_axes = self.batch_axes,
        )
        num_thread = Axis( ShapeVar( nt ), name = "num_thread" )
        num_corr = Axis( ShapeVar( nb_corr_items ), name = "num_corr" )

        driver.call(
            FfiCodeParallel( name = "cut",
                fwd_code = "cell( batch_index ).cut( res( batch_index ), direction, offset, cut_id, corr( thread_index ) );",
                bwd_setup_code = "cell.cut_bwd_setup( queue, grad_for_cell, grad_for_direction, grad_for_offset );",
                bwd_code = "cell( batch_index ).cut_bwd( direction, offset, grad_for_res( batch_index ), "
                           "grad_for_cell( batch_index ), grad_for_direction, grad_for_offset, corr( thread_index ) );",
                thread_cap = "corr.shape( 0 )" ),
            output_capacities = self._cut_capacities(),
            output_attributes = [ "res", "corr" ],
            # `corr` est un tampon de travail, pas un résidu : son contenu ne survit pas à l'appel
            # (les threads se le repassent d'un item à l'autre), donc le backward s'en réalloue un
            # neuf au lieu de relire celui du forward.
            scratch_attributes = [ "corr" ],
            cut_id = cut_id,
            direction = direction,
            offset = offset,
            cell = src,
            res = res,
            # un tenseur d'INDICES : `IntTensor`, sinon ses lectures reviendraient en flottants et
            # la comparaison `corr( v ) < 0` (le drapeau « sommet supprimé ») partirait en vrille.
            corr = IntTensor[ num_thread, num_corr ](),
        )

    @property
    def measure( self ) -> Tensor:
        """La mesure de la cellule : sa longueur en 1D, son aire en 2D, son volume au-delà.

        DEUX chemins, choisis sur `nb_dims`, et le premier n'est pas un cas particulier du second :
        ils ne lisent pas la même description de la cellule (voir les deux surcharges de
        `Cell.h::measure`). En d <= 2 il n'y a rien à énumérer, donc rien à allouer pour le faire --
        d'où un chemin court qui ne construit ni `item_map` ni son axe de threads.
        """
        if int( self.nb_dims.value ) <= 2:
            return self._measure_up_to_2d()
        return self._measure_nd()

    def _measure_up_to_2d( self ):
        """d <= 2 : `vertex_positions` EST la géométrie, donc AUCUN scratch.

        Un segment en 1D, un polygone en ordre cyclique en 2D : la formule (écart des extrémités,
        lacets) se lit directement sur les sommets. Ni `item_map` ni `nb_map_items` ne sont créés
        ici -- les créer les ferait allouer et remplir pour rien.
        """
        # `res` est une sortie que la méthode fabrique elle-même (pas un membre de l'aggregate), donc
        # elle doit prendre les axes de batch explicitement -- `RealTensor[ *batch_axes ]`, vide pour
        # une `Cell` simple. Le corps indexe alors `res` par `batch_index` comme `cell` : un no-op
        # sans batch (multi-index vide), une écriture par item sinon.
        res = RealTensor[ tuple( self.batch_axes ) ]()

        driver.call(
            FfiCodeParallel( name = "measure",
                fwd_code = "cell( batch_index ).measure( res( batch_index ) );",
                bwd_code = "cell( batch_index ).measure_bwd( res( batch_index ), grad_for_res( batch_index ), "
                           "grad_for_cell( batch_index ).vertex_positions );",
            ),
            output_attributes = [ "res" ],
            input_exceptions = self._measure_input_exceptions(),
            cell = self,
            res = res,
        )

        return res

    def _measure_nd( self ):
        """d > 2 : la cellule est DÉCOUPÉE EN SIMPLEXES, et on somme leurs volumes.

        Le découpage est l'éventail classique -- un sommet de la cellule, coné sur chaque facette
        qui ne le contient pas, chacune découpée pareil une dimension plus bas. Il faut donc, pour
        chaque face rencontrée, UN de ses sommets, et c'est là que va le scratch : `facet_apex`
        garde une ligne par profondeur de récursion et une case par coupe, remplie d'un seul
        passage sur les sommets de la face (voir `Cell.h`). Soit `nb_dims * nb_cuts` mots par
        thread, là où indexer les faces par leur ENSEMBLE de coupes en coûte `nb_cuts^(d-1)`.
        """
        d = int( self.nb_dims.value )
        cap_c = self.nb_cuts.allocated_capacity() or 8

        nt = driver.device.nb_threads(
            nb_local_bytes_per_thread = 8 * d * cap_c,
            batch_axes = self.batch_axes,
        )

        # Axes NOMMÉS, construits DIRECTEMENT et pas via `Axis[ sv ]( name = ... )` :
        # `Parametrized.__call__` plierait `name` dans les template_kwargs, il n'atteindrait jamais
        # le `name` d'`AbstractAxis.__init__`, et l'axe resterait anonyme (`a0`).
        num_thread = Axis( ShapeVar( nt ), name = "num_thread" )
        num_level = Axis( ShapeVar( d ), name = "num_level" )
        num_cut_slot = Axis( ShapeVar( cap_c ), name = "num_cut_slot" )

        # `res` est une sortie que la méthode fabrique elle-même (pas un membre de l'aggregate),
        # donc elle prend les axes de batch explicitement -- vide pour une `Cell` simple. Le
        # scratch, lui, est PAR THREAD et non par item : il n'en prend aucun.
        res = RealTensor[ tuple( self.batch_axes ) ]()

        driver.call(
            FfiCodeParallel( name = "measure",
                fwd_code = "cell( batch_index ).measure( res( batch_index ), facet_apex( thread_index ) );",
                bwd_code = "cell( batch_index ).measure_bwd( res( batch_index ), facet_apex( thread_index ), "
                           "grad_for_res( batch_index ), grad_for_cell( batch_index ).vertex_positions );",
                thread_cap = "facet_apex.shape( 0 )" ),
            output_attributes = [ "res", "facet_apex" ],
            # tampon de travail, pas un résidu : le backward refait le même parcours et s'en
            # réalloue un neuf plutôt que de relire celui du forward.
            scratch_attributes = [ "facet_apex" ],
            input_exceptions = self._measure_input_exceptions(),
            cell = self,
            res = res,
            # des INDICES : `IntTensor`, sinon la case vide (`-1`) reviendrait en flottant.
            facet_apex = IntTensor[ num_thread, num_level, num_cut_slot ](),
        )

        return res

    def add_to_viz( self, viz, color = None, opacity = 1.0, faces = True, edges = True,
                    points = False ):
        """Se dessine dans un `Visualizer` (voir `sdot.viz.Visualizer`).

        C'est la cellule qui choisit ses primitives, pas le visualiseur : selon la dimension et
        selon qu'elle est bornée, elle n'a pas la même chose à montrer.

        - 2D : `vertex_positions` est déjà le polygone en ordre CYCLIQUE (`vertex_ordering_2D`
          côté C++), d'où une face et le tour d'arêtes, sans topologie à reconstruire.
        - 3D : les faces se relisent sur `edge_indices` -- une arête y porte, après ses deux
          sommets, les coupes qui la contiennent ; les arêtes qui partagent une coupe forment le
          cycle de la face correspondante.
        - au-delà, ou non bornée : les faces partent en H-représentation
          (`cut_directions`/`cut_offsets`), la seule description qui reste exploitable. La page en
          montre une COUPE 3D (couper des demi-espaces redonne des demi-espaces) et, en dimension
          > 3, on ajoute le fil de fer PROJETÉ, qui lui ne dépend d'aucun réglage de coupe.

        = Ce qui est FACTICE dans une cellule non bornée

        Une cellule non bornée n'est pas décrite par ses sommets : `init_as_unbounded` tient lieu de
        « tout l'espace » par un SIMPLEXE dont les plans sont marqués `INFINITE` et dont les offsets
        sont inventés, que les coupes repoussent au fur et à mesure. Les dessiner tels quels
        montrerait des parois qui n'existent pas, à une distance qui ne veut rien dire. Trois règles
        en découlent, et elles se lisent toutes sur `cut_ids` :

        - un plan `INFINITE` n'est pas envoyé -- ce qui reste est le vrai polytope, non borné, que
          la page referme elle-même sur la boîte de la scène ;
        - une arête POSÉE sur des plans factices (tous ses plans le sont) n'est pas une arête de la
          cellule : elle n'est pas tracée ;
        - une arête qui n'en TOUCHE un que par un bout est une vraie arête, mais qui part à
          l'infini et qu'on a coupée quelque part : elle est tracée en POINTILLÉS.

        Le cadrage suit la même logique : seuls les sommets réels le fixent, un sommet factice
        n'étant qu'un point de troncature.

        Une cellule batchée pousse un item par élément du batch, chacun dessiné sur SON PROPRE
        compte -- un diagramme de Voronoï, le cas courant d'un batch, n'a pas deux cellules de la
        même taille.
        """
        nb_dims = int( self.nb_dims.value )

        # le nombre d'items se lit sur `is_fully_bounded` -- un scalaire par item, donc jamais de
        # longueur nulle, contrairement aux tableaux de géométrie.
        bounded  = np.asarray( self.is_fully_bounded ).reshape( -1 )
        nb_items = len( bounded )

        nvs = _counts_per_item( self.nb_vertices, nb_items )
        ncs = _counts_per_item( self.nb_cuts, nb_items )
        nes = _counts_per_item( self.nb_edges, nb_items ) if nb_dims > 2 else None

        # les tableaux sont denses au PLUS GRAND des items ; chaque item est ensuite tranché sur son
        # propre compte. Sans sommet nulle part il n'y a rien à montrer -- une cellule vide est un
        # état parfaitement légitime (une coupe a tout exclu), pas une erreur.
        max_v, max_c = int( nvs.max() ), int( ncs.max() )
        if max_v == 0:
            return viz

        vps = np.asarray( self.vertex_positions ).reshape( nb_items, max_v, nb_dims )
        cds = np.asarray( self.cut_directions   ).reshape( nb_items, max_c, nb_dims )
        cos = np.asarray( self.cut_offsets      ).reshape( nb_items, max_c )
        cis = np.asarray( self.cut_ids          ).reshape( nb_items, max_c )
        vis = eis = None
        if nb_dims > 2:
            vis = np.asarray( self.vertex_indices ).reshape( nb_items, max_v, nb_dims )
            eis = np.asarray( self.edge_indices   ).reshape( nb_items, int( nes.max() ), nb_dims + 1 )

        # PREMIÈRE PASSE : ce qui, dans chaque item, tient au simplexe factice. Il faut la vue
        # d'ENSEMBLE avant de dessiner quoi que ce soit -- la longueur qu'on donne à un rayon
        # tronqué se prend sur l'étendue RÉELLE de la scène, jamais sur celle du remplaçant, qui
        # est arbitraire et qui écraserait tout le reste de l'image.
        parts = {}
        for b in range( nb_items ):
            nv, nc = int( nvs[ b ] ), int( ncs[ b ] )
            if nv == 0:
                continue
            parts[ b ] = _infinite_parts(
                nb_dims, cis[ b ][ : nc ] == INFINITE, nv,
                vis[ b ][ : nv ] if vis is not None else None,
                eis[ b ][ : int( nes[ b ] ) ] if eis is not None else None )
        stub = _ray_length( vps, nvs, parts )

        for b in range( nb_items ):
            nv, nc = int( nvs[ b ] ), int( ncs[ b ] )
            if nv == 0:
                continue

            col = color if color is not None else viz.take_color()   # UNE couleur par cellule
            edge_col = viz.darker( col )    # arêtes = la teinte de la face, assombrie

            infinite = cis[ b ][ : nc ] == INFINITE
            fake, ev, hide, dash = parts[ b ]

            # un SEUL tableau, gardé dans une variable et repassé tel quel à chaque `add_*` : le
            # visualiseur reconnaît le vivier de sommets à l'IDENTITÉ de l'objet reçu, donc une
            # tranche refaite à chaque appel lui ferait stocker les mêmes sommets trois fois.
            vp = _truncated_rays( vps[ b ][ : nv ], fake, ev, stub )
            real = vp[ ~fake ] if fake.any() else vp

            # dès qu'il y a du factice, ce vivier-là ne CADRE plus la scène : le bout d'un rayon
            # est à une distance qu'on a choisie, pas mesurée, et la laisser peser sur la boîte
            # écraserait le diagramme au centre de l'image. Ce sont les VRAIS sommets qui cadrent,
            # et eux seuls -- le rayon, lui, sort du cadre, ce qui est exactement ce qu'il dit.
            frames = not fake.any()
            if not frames:
                viz.note_bounds( real if len( real ) else vp )

            if nb_dims > 3 or not bool( bounded[ b ] ):
                keep = ~infinite
                if keep.any():
                    # `faces = False` -> opacité nulle plutôt que pas de faces du tout : elles
                    # restent envoyées, donc écrites dans le z-buffer, donc les arêtes cachées le
                    # restent (c'est la passe de profondeur pure qui s'en charge côté page).
                    # `edges` : au-delà de la 3D ce polytope est une COUPE, et ses arêtes sont
                    # celles de la coupe -- personne d'autre ne les a. En 2D/3D c'est la cellule
                    # elle-même, dont on trace les arêtes plus bas en sachant ce qui est tronqué :
                    # laisser l'énumération les redonner ferait un trait PLEIN jusqu'au bord de la
                    # boîte là où on veut un pointillé qui s'arrête.
                    viz.add_polytope( cds[ b ][ : nc ][ keep ], cos[ b ][ : nc ][ keep ],
                                      nb_dims = nb_dims, color = col,
                                      opacity = opacity if faces else 0,
                                      edges = nb_dims > 3 )
            elif faces and nb_dims >= 2:
                polys = ( [ list( range( nv ) ) ] if nb_dims == 2
                          else _faces_from_edges( eis[ b ][ : int( nes[ b ] ) ], nc ) )
                viz.add_faces( vp, polys, color = col, opacity = opacity, frames = frames )

            if edges and len( ev ):
                # en dimension > 3 le fil de fer est une PROJECTION, pas la cellule : il s'efface
                # derrière la coupe, que la page, elle, montre en vraie grandeur.
                op = 0.55 if nb_dims > 3 else 1.0
                solid = ~hide & ~dash
                if solid.any():
                    viz.add_edges( vp, ev[ solid ], color = edge_col, opacity = op, frames = frames )
                if dash.any():
                    viz.add_edges( vp, ev[ dash ], color = edge_col, opacity = op,
                                   dashed = True, frames = False )

            if points:
                viz.add_points( real if len( real ) else vp, color = edge_col )
        return viz

    # -- ce qui dépend du RÉGIME DE DIMENSION -------------------------------------------------
    # Une cellule de dimension <= 2 se décrit entièrement par ses sommets : un segment en 1D, un
    # polygone en ordre CYCLIQUE en 2D. Une cellule de dimension > 2 ne le peut pas -- il lui faut
    # son treillis de faces (`vertex_indices` / `edge_indices`) et, pour la mesure, le scratch qui
    # énumère les simplexes (`item_map`). Les trois méthodes ci-dessous sont les seuls endroits où
    # cette frontière est tracée ; tout le reste du fichier s'y réfère.

    def _empty_like_me( self ):
        """Une `Cell` de même dimension et même batch, sans géométrie -- ce que `cut` alloue pour
        recevoir sa sortie, et ce sur quoi il prend l'instantané de son entrée."""
        return Cell( self.nb_dims, init_as_unbounded = False, batch_axes = self.batch_axes or None )

    def _cut_capacities( self ):
        """De la place pour UN sommet de plus : couper un polygone convexe par un demi-espace lui
        en laisse au plus autant qu'il en avait, plus un (il perd un morceau et gagne une arête).

        On part de la capacité DÉJÀ ALLOUÉE, pas du compte : le compte est une valeur device dès
        qu'un kernel l'a écrit (illisible sous une trace), la capacité est un fait hôte. Et une
        capacité reste une SUPPOSITION : si elle ne suffit pas, le kernel l'enregistre et sort sans
        rien écrire de faux, la plateforme réserve plus et relance (voir `driver.call`).
        """
        cap_v = self.nb_vertices.allocated_capacity() or 8
        cap_c = self.nb_cuts.allocated_capacity() or 8

        # d <= 2 : un demi-plan retire un morceau du polygone et ajoute une arête -- au plus UN
        # sommet de plus, et autant de coupes que de sommets (l'invariant du chemin 2D).
        if int( self.nb_dims.value ) <= 2:
            return { "res.nb_vertices": cap_v + 1, "res.nb_cuts": cap_v + 1 }

        # d > 2 : plus d'invariant qui lie les trois comptes. Une coupe ajoute AU PLUS une coupe
        # (et en retire souvent), mais elle peut créer plusieurs sommets et plusieurs arêtes -- une
        # facette entière, en fait. La marge de moitié n'est qu'un point de départ raisonnable.
        cap_e = self.nb_edges.allocated_capacity() or 8
        return {
            "res.nb_vertices": cap_v + cap_v // 2 + 4,
            "res.nb_edges":    cap_e + cap_e // 2 + 4,
            "res.nb_cuts":     cap_c + 1,
        }

    def _nb_corr_items( self ):
        """La taille d'une ligne de `corr` : une entrée par ancien sommet, puis une par ancienne
        coupe (plus la nouvelle). Prise sur les CAPACITÉS et non sur les comptes -- un compte
        écrit par un kernel est une valeur device, illisible sous une trace, alors qu'une capacité
        est un fait hôte, et c'en est un majorant."""
        return ( self.nb_vertices.allocated_capacity() or 8 ) + ( self.nb_cuts.allocated_capacity() or 8 ) + 1

    # ce qu'une coupe réécrit -- la V-représentation, la H-représentation, le drapeau de bornage,
    # et au-delà de la 2D le treillis de faces (`vertex_indices` / `edge_indices`, et le compte
    # `nb_edges` qui en dimensionne un). En 2D ces trois-là n'existent pas.
    _CUT_TENSORS = ( "vertex_positions", "is_fully_bounded", "cut_directions", "cut_offsets", "cut_ids",
                     "vertex_indices", "edge_indices" )

    def _cut_counts( self ):
        if int( self.nb_dims.value ) <= 2:
            return ( "nb_vertices", "nb_cuts" )
        return ( "nb_vertices", "nb_cuts", "nb_edges" )

    def _adopt_geometry( self, other ):
        """Reprend sur `self` la géométrie que le kernel vient d'écrire dans `other`.

        C'est le « rebinding côté Python entre deux appels » dont parle `driver.call` : ce sont les
        VALEURS qui passent -- le stockage des tenseurs, le compte des ShapeVars -- pas les objets
        `Attribute`, dont l'identité doit survivre à la coupe (un `AxisId` ou un compte peut être
        partagé avec un autre agrégat).
        """
        for name in self._CUT_TENSORS:
            source = getattr( other, name )
            if source.is_defined:
                getattr( self, name ).set( source )

        # un compte écrit par un kernel est une valeur DEVICE : `set_count`, et surtout pas `set`,
        # qui le prescrirait sur l'hôte (et refuserait un tracer).
        for name in self._cut_counts():
            getattr( self, name ).set_count( getattr( other, name ).raw )

    def _face_lattice_exceptions( self, name ):
        """Le treillis de faces, quand il n'existe pas : les sorties qu'un kernel N'ÉCRIT PAS en
        d <= 2, sous le nom `name` que porte la cellule dans l'appel.

        Un carve-out sur `output_attributes = [ name ]` : sans lui, ces attributs seraient alloués
        et remplis pour rien. Exclus, ils n'ont ni buffer ni compte, et arrivent côté C++ en
        `NoneTensor` -- exactement ce qu'attendent les `if constexpr ( ct_dim > 2 )` de `Cell.cxx`,
        qui ne les touchent pas en dessous de la 3D.

        `nb_edges` en fait partie : c'est le compte de `edge_indices`, et il n'a personne d'autre
        à dimensionner.
        """
        if int( self.nb_dims.value ) > 2:
            return []
        return [ f"{ name }.vertex_indices", f"{ name }.edge_indices", f"{ name }.nb_edges" ]

    def _measure_input_exceptions( self ):
        """Ce que `measure` NE LIT PAS -- forcé `Unbound` plutôt que passé pour rien.

        La H-représentation (`cut_directions` / `cut_offsets` / `cut_ids`, et le compte `nb_cuts`
        qui la dimensionne) n'entre dans aucun des deux chemins : ni la formule des lacets, ni le
        découpage en simplexes ne s'en servent. Elle ne traverse donc pas la FFI et ne devient
        jamais une primale différentiable de cet appel -- ce qui compte, puisque la mesure ne
        dépend vraiment que des sommets. (Combien il y a de coupes, le découpage a bien besoin de
        le savoir : il le lit sur la LARGEUR du scratch, pas sur le compte.)

        `edge_indices` non plus : dans l'éventail une arête est une face comme une autre, atteinte
        par les listes de coupes des sommets. `vertex_indices`, en revanche, EST ce que le
        découpage parcourt -- il n'est donc exclu qu'en d <= 2, où il n'existe pas (voir
        `_face_lattice_exceptions`).
        """
        res = [ "cell.cut_directions", "cell.cut_offsets", "cell.cut_ids", "cell.nb_cuts" ]
        if int( self.nb_dims.value ) > 2:
            res += [ "cell.edge_indices", "cell.nb_edges" ]
        return res

    def _init_capacities_for_hypercube( self ):
        # a hypercube has 2^d vertices, d*2^(d-1) edges and 2d cuts. We allocate ~2x that (a floor
        # of 8, the value hard-coded for 2D) as headroom for the cuts a cell takes over its life,
        # not just its initial shape. A flat 8 was too small in 3D (12 edges) -- an under-provision
        # is now caught (recorded + clamped by ShapeVarView, writes bounded) rather than corrupting,
        # but we still size it right so the call goes through in one shot.
        d = int( self.nb_dims.value )
        res = {
            "cell.nb_vertices": max( 8, 2 * 2 ** d ),
            "cell.nb_cuts":     max( 8, 2 * ( 2 * d ) ),
        }
        # d > 2 only: below that there are no `edge_indices` to size (see
        # `_face_lattice_exceptions`), so `nb_edges` has nothing to be a capacity FOR.
        if d > 2:
            res[ "cell.nb_edges" ] = max( 8, 2 * d * 2 ** ( d - 1 ) )
        return res


def _counts_per_item( shape_var, nb_items ):
    """Le compte de chaque item du batch.

    Un compte qu'un KERNEL écrit en a un par item (rien ne dit que deux items s'accordent, cf.
    `CallArg_ShapeVar`) ; un compte connu de l'hôte reste la valeur unique qu'il est, et on la
    diffuse alors sur les items.
    """
    v = np.atleast_1d( np.asarray( shape_var.value ) ).reshape( -1 ).astype( int )
    return v if v.size == nb_items else np.broadcast_to( v, ( nb_items, ) )


def _infinite_parts( nb_dims, infinite, nb_vertices, vertex_cuts, edge_rows ):
    """Ce qui, dans une cellule, tient au SIMPLEXE FACTICE plutôt qu'à la cellule elle-même.

    Rend `( fake, ev, hide, dash )` : quels sommets sont factices, la liste des arêtes en indices
    de sommets, et lesquelles ne pas tracer / tracer en pointillés (voir `Cell.add_to_viz`).

    `infinite` est le masque des plans marqués `INFINITE`, indexé comme les coupes. Tout le reste
    en découle par le treillis -- et le treillis n'a pas la même forme selon la dimension :

    - d == 2 il est IMPLICITE, porté par l'invariant du chemin 2D (la coupe i porte l'arête
      `[ v_i, v_i+1 ]`, donc `v_i` est le coin des coupes `i-1` et `i`, et `nb_cuts == nb_vertices`) ;
    - d > 2 il est STOCKÉ, dans `vertex_indices` / `edge_indices` ;
    - d == 1 il n'y en a pas : un segment n'a qu'une arête, et rien à classer.
    """
    if nb_dims >= 3:
        vc = np.asarray( vertex_cuts )
        ev = np.asarray( edge_rows )[ :, :2 ].astype( int )
        ec = np.asarray( edge_rows )[ :, 2: ].astype( int )
    elif nb_dims == 2:
        i = np.arange( nb_vertices )
        vc = np.stack( [ ( i - 1 ) % nb_vertices, i ], axis = 1 )
        ev = np.stack( [ i, ( i + 1 ) % nb_vertices ], axis = 1 )
        ec = i[ :, None ]
    else:
        ev = np.array( [ [ 0, 1 ] ] ) if nb_vertices >= 2 else np.zeros( ( 0, 2 ), int )
        return ( np.zeros( nb_vertices, bool ), ev,
                 np.zeros( len( ev ), bool ), np.zeros( len( ev ), bool ) )

    # un sommet est factice dès qu'UN de ses plans l'est : il n'est pas à l'intersection des plans
    # de la cellule, mais à celle d'une paroi inventée -- donc quelque part sur un rayon, à une
    # distance qui ne veut rien dire.
    fake = infinite[ vc ].any( axis = 1 ) if len( infinite ) else np.zeros( nb_vertices, bool )

    # une arête POSÉE sur des plans factices (tous les siens le sont) n'est pas une arête de la
    # cellule. Une arête qui n'en TOUCHE un que par un bout en est une, mais tronquée : pointillés.
    hide = infinite[ ec ].all( axis = 1 ) if len( infinite ) else np.zeros( len( ev ), bool )
    dash = ( fake[ ev[ :, 0 ] ] | fake[ ev[ :, 1 ] ] ) & ~hide
    return fake, ev, hide, dash


def _ray_length( vps, nvs, parts ):
    """La longueur à donner à un rayon tronqué : une fraction de l'étendue RÉELLE de la scène.

    Un sommet factice est posé là où `Cell::cut` a dû repousser le simplexe de remplacement pour
    que le classement soit celui de l'infini -- c'est-à-dire assez loin pour ne plus rien changer,
    donc sans commune mesure avec la cellule. Le dessiner là écraserait toute l'image sur un point
    (et le cadrage de la scène avec). On garde donc du rayon sa DIRECTION, qui veut dire quelque
    chose, et on lui donne une longueur qui n'en veut aucune -- ce que le pointillé dit déjà.

    Prise sur TOUS les items à la fois : un diagramme de Voronoï doit avoir le même moignon partout,
    et une cellule prise isolément n'a pas d'échelle à elle (deux sommets réels, parfois aucun).
    """
    real = [ vps[ b ][ : int( nvs[ b ] ) ][ ~parts[ b ][ 0 ] ] for b in parts ]
    real = [ r for r in real if len( r ) ]
    if not real:
        # aucune cellule n'a de vrai sommet ( un unique germe, sans domaine ) : plus rien ne donne
        # d'échelle, on garde alors le simplexe tel quel plutôt que d'en inventer une.
        return None
    pts = np.concatenate( real, axis = 0 )
    # les percentiles et non le min/max : un sommet de Voronoï est un centre de cercle circonscrit,
    # et trois germes presque alignés en donnent un très loin. Il est RÉEL, il a donc sa place dans
    # la boîte de la scène -- mais il ne dit rien de l'échelle à laquelle on regarde, et le prendre
    # pour telle allongerait tous les moignons du diagramme.
    lo, hi = np.percentile( pts, 5, axis = 0 ), np.percentile( pts, 95, axis = 0 )
    diag = float( np.linalg.norm( hi - lo ) )
    return 0.2 * diag if diag > 0 else None


def _truncated_rays( vp, fake, ev, stub ):
    """`vp` avec ses sommets FACTICES ramenés à `stub` du sommet réel dont ils partent.

    Le sommet réel se lit sur les arêtes : une arête qui joint un vrai sommet à un faux EST le
    rayon, et c'est de son extrémité réelle qu'il part. Un sommet factice qui n'en a pas (une arête
    dont les deux bouts sont faux) est ramené par rapport au centre des sommets réels de la cellule
    -- à défaut de rayon identifiable, il reste au moins dans le cadre.
    """
    if stub is None or not fake.any():
        return vp

    src = np.full( len( vp ), -1 )
    for a, b in ev:
        if fake[ a ] and not fake[ b ]: src[ a ] = b
        if fake[ b ] and not fake[ a ]: src[ b ] = a

    real = vp[ ~fake ]
    centre = real.mean( axis = 0 ) if len( real ) else vp.mean( axis = 0 )

    res = np.array( vp, dtype = float )
    for f in np.nonzero( fake )[ 0 ]:
        origin = vp[ src[ f ] ] if src[ f ] >= 0 else centre
        ray = vp[ f ] - origin
        n = float( np.linalg.norm( ray ) )
        if n > 1e-12:
            res[ f ] = origin + ray / n * stub
    return res


def _faces_from_edges( edge_indices, nb_cuts ):
    """Les faces d'une cellule 3D, une par coupe, en ordre cyclique.

    Une ligne de `edge_indices` est `[ v0, v1, coupes... ]` : en 3D une arête porte exactement
    deux coupes, donc les arêtes citant une même coupe `c` sont exactement les côtés de la face
    portée par `c` -- il ne reste qu'à les chaîner de proche en proche pour retrouver le cycle
    (ce que `add_faces` attend pour trianguler en éventail).
    """
    res = []
    for c in range( nb_cuts ):
        adj = {}
        for row in edge_indices:
            if not any( int( v ) == c for v in row[ 2 : ] ):
                continue
            a, b = int( row[ 0 ] ), int( row[ 1 ] )
            adj.setdefault( a, [] ).append( b )
            adj.setdefault( b, [] ).append( a )
        if len( adj ) < 3:
            continue
        start = next( iter( adj ) )
        cycle, prev, cur = [ start ], None, start
        while len( cycle ) <= len( adj ):
            nxt = [ v for v in adj[ cur ] if v != prev ]
            if not nxt or nxt[ 0 ] == start:
                break
            cycle.append( nxt[ 0 ] )
            prev, cur = cur, nxt[ 0 ]
        if len( cycle ) >= 3:
            res.append( cycle )
    return res

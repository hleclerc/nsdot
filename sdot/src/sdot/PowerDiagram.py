"""Diagramme de puissance ( Laguerre ) -- LA VUE, et ce que tous les stockages ont en commun.

La cellule du germe `i` est là où sa DISTANCE DE PUISSANCE gagne :

    |x - d_i|² - w_i  <=  |x - d_j|² - w_j   pour tout autre j

Développée, l'inégalité perd son `|x|²` des deux côtés et devient un demi-espace : un diagramme de
puissance coûte exactement ce que coûte un Voronoï, un plan par rival et la même coupe. Seules les
DIFFÉRENCES de poids atteignent les plans : « tous égaux » et « pas de poids du tout » sont le même
objet, et le cas euclidien s'appelle `Voronoi` ( voir `Voronoi.py` ).

`PowerDiagram( positions, weights, ... )` ne porte pas de diagramme : il porte ses GERMES et le
domaine convexe qui les borne, et reconstruit ce qu'on lui demande, cellule par cellule, dans le
scratch d'un work-item ( `diagram/Ops.h` ). Ce fichier est le CONTRAT -- ce qu'un utilisateur lit et
écrit : `positions`, `weights`, `measures`, `cells`, `cell( i )` -- et le tronc commun : le domaine,
la distribution, le scratch, les trois kernels. COMMENT les germes sont rangés est l'affaire d'une
spécialisation, choisie à la construction :

  * `PowerDiagram_Plain` -- les germes tels qu'ils sont venus, chaque cellule coupée par les `n - 1`
    bissectrices. Le plancher, et ce qui reste quand les positions sont un traceur ;
  * `PowerDiagram_Bsp`   -- les germes dans l'ordre d'un arbre BSP ( `AaBsp` ), une feuille se
    lisant d'un seul tenant, et chaque cellule coupée par les seuls germes que l'arbre n'a pas
    su écarter. Le défaut dès que les positions sont concrètes.

Le voisinage est ACCÉLÉRABLE, pas le résultat : un accélérateur ne peut que taire des coupes qui
n'auraient rien enlevé, donc les cellules sont les MÊMES, aux erreurs d'arrondi près.
"""

import numpy as np

from loom.compilation.FfiCode import FfiCodeParallel
from loom.drivers.driver import driver
from loom.tensor import Axis, CtShapeVar, IntTensor, RealTensor, ShapeVar, Tensor, new_batch_axis
from loom.util import Aggregate

from .Cell import BOUNDARY, Cell
from .CellScratch import CellScratch, fp_size, merge_call


def diagram_class_for( positions, weights, accelerator ):
    """La spécialisation qui range ces germes : l'arbre BSP sauf quand on n'en veut pas
    ( `accelerator = "plain"` ) ou qu'on ne peut pas en bâtir un ici -- des germes TRACÉS, positions
    ou poids : l'arbre se bâtit côté hôte, et sous un `jit` même une constante sort tracée d'un
    kernel. Qui veut l'arbre sous une trace le bâtit dehors et le passe ( `accelerator = tree` )."""
    from .PowerDiagram_Bsp import PowerDiagram_Bsp
    from .PowerDiagram_Plain import PowerDiagram_Plain
    if accelerator == "plain":
        return PowerDiagram_Plain
    if accelerator is None and any( driver.is_traced( getattr( x, "raw", x ) ) for x in ( positions, weights ) if x is not None ):
        return PowerDiagram_Plain
    return PowerDiagram_Bsp


class PowerDiagram( Aggregate ):
    # ---- ce qu'une spécialisation fournit -----------------------------------------------------------
    #
    #   _init_seeds( positions, weights, accelerator )   range les germes dans ses tenseurs
    #   positions / weights            propriétés, en lecture et en écriture, dans l'ORDRE de l'utilisateur
    #   _ranks_of_items()              pour `cells` : le rang ( ordre du stockage ) du germe `i`
    #
    # et côté C++ ( `diagram/Ops.h` ) : `point( k )`, `weight( k )`, `user_id( k )`, `fournisseur( k0 )`.

    # LA CELLULE DE DÉPART, quand on en connaît une meilleure que « tout l'espace » : le pavé
    # `box_min <= x <= box_max`, posé d'un trait par `Cell.init_as_hypercube`. Absent ( `Unbound` ),
    # chaque cellule naît comme un SIMPLEXE DE REMPLACEMENT non borné dont toute coupe doit d'abord
    # repousser les plans infinis. Ce n'est pas un second domaine : c'est le même, exprimé sous la
    # forme qu'on sait poser directement ; ce qu'un pavé ne dit pas reste dans `bnd_*`.
    box_min        : RealTensor[ "dim" ]
    box_max        : RealTensor[ "dim" ]

    # le domaine : une liste de demi-espaces, donc n'importe quel convexe polyédrique. Absent, les
    # cellules qui partent à l'infini le restent -- et se mesurent comme telles ( `TF::max` ).
    bnd_directions : RealTensor[ "num_boundary", "dim" ]
    bnd_offsets    : RealTensor[ "num_boundary" ]

    num_point      : Axis[ "nb_points" ]
    num_boundary   : Axis[ "nb_boundaries" ]
    dim            : Axis[ "nb_dims" ]

    nb_points      : ShapeVar
    nb_boundaries  : ShapeVar
    nb_dims        : CtShapeVar

    def __new__( cls, positions = None, weights = None, *args, accelerator = None, **kwargs ):
        if cls is PowerDiagram:
            cls = diagram_class_for( positions, weights, accelerator )
        return super().__new__( cls )

    def __init__( self, positions, weights = None, boundaries = None, accelerator = None,
                  distribution = None, kernel_dtype = None, scratch_capacity = None ):
        """`positions` : `[ n, d ]`. `weights` : `[ n ]`, ou rien ( le cas euclidien ). Le domaine :

        - `boundaries = ( directions, offsets )` -- les demi-espaces `direction . x <= offset`.
          Un pavé s'écrit `box_half_spaces( mi, ma )`, qui est là pour ça ;
        - rien -- les cellules du bord restent infinies.

        `accelerator` : `None` ( un arbre BSP, bâti ici ), un `AaBsp` déjà bâti sur ces positions
        ( ce qu'il faut pour dériver par rapport à des positions tracées ), ou `"plain"`. Sans effet
        sur le RÉSULTAT -- seulement sur ce qu'il coûte.

        `distribution` : CONTRE QUOI intégrer ( `Image`, `SumOfGaussians`, ... ). Absente,
        `measures` rend le volume des cellules ; présente, l'intégrale de sa densité dessus,
        NORMALISÉE ici une fois pour toutes. Si elle a un SUPPORT borné, il s'ajoute au domaine.

        `kernel_dtype` : le flottant dans lequel la géométrie se coupe ( `FP32` par défaut,
        `SDOT_KTYPE` pour changer le défaut ). `scratch_capacity` : pour combien de sommets par
        cellule le scratch d'un work-item est taillé au départ -- une supposition, que loom double
        sur débordement.
        """
        # le SUPPORT de la distribution borne le domaine, gratuitement et sans rien changer au
        # résultat : ce qui dépasse n'apporte aucune masse. Le domaine de l'appelant est INTERSECTÉ
        # avec, pas remplacé.
        if distribution is not None:
            support = distribution.bounding_half_spaces()
            if support is not None:
                if boundaries is None:
                    boundaries = support
                else:
                    boundaries = ( np.concatenate( [ np.asarray( boundaries[ 0 ], dtype = float ), support[ 0 ] ] ),
                                   np.concatenate( [ np.asarray( boundaries[ 1 ], dtype = float ), support[ 1 ] ] ) )

        # pas de domaine -> on ne NOMME pas les deux tenseurs : les laisser `Unbound` ( jamais
        # alloués, `NoneTensor` côté C++ ) n'est pas la même chose que leur passer `None`.
        kwargs = {}
        if boundaries is not None:
            # D'OÙ PARTIR, lu sur les demi-espaces eux-mêmes : ceux qu'un pavé exprime déjà sortent
            # de la liste -- ils sont `2d` et reviendraient sur chaque cellule ( 25 %, mesuré ).
            start_box = axis_aligned_box( *boundaries )
            if start_box is not None:
                mi, ma, kept = start_box
                kwargs[ "box_min" ], kwargs[ "box_max" ] = mi, ma
                boundaries = ( np.asarray( boundaries[ 0 ], dtype = float )[ kept ],
                               np.asarray( boundaries[ 1 ], dtype = float )[ kept ] )
            if len( boundaries[ 1 ] ):
                kwargs[ "bnd_directions" ], kwargs[ "bnd_offsets" ] = boundaries

        pos = positions if hasattr( positions, "shape" ) else np.asarray( positions, dtype = float )
        if len( pos.shape ) != 2:
            raise ValueError( f"`positions` has to be [ n, d ] ( got { tuple( pos.shape ) } )" )
        n, d = int( pos.shape[ 0 ] ), int( pos.shape[ 1 ] )

        self._kernel_dtype = kernel_dtype
        self._scratch_capacity = int( scratch_capacity or { 2: 64, 3: 128 }.get( d, 256 ) )
        self.__base_init__( nb_dims = d, nb_points = n, **self._init_seeds( pos, weights, accelerator ), **kwargs )

        # PAS un champ : la distribution est un argument d'APPEL, normalisée DÈS ICI plutôt qu'à
        # chaque `measures` -- « le diagramme intègre CETTE mesure-là » est une propriété de l'objet
        self.distribution = None
        if distribution is not None:
            dd = int( distribution.nb_dims.value )
            if dd != d:
                raise ValueError( f"the distribution lives in { dd }D, this diagram in { d }D" )
            self.distribution = distribution.normalized_version()

    @property
    def dim_count( self ):
        return int( self.nb_dims.value )

    @property
    def kernel_dtype( self ):
        return self._domain_cell().kernel_dtype

    # ---- le domaine, et la distribution ------------------------------------------------------------

    def _domain_cell( self ):
        """Le domaine comme POLYTOPE, calculé UNE FOIS -- ce dont chaque cellule part. Construit avec
        le MÊME code que l'oracle `cell( i )`, ce qui garde les deux descriptions du domaine
        littéralement identiques. C'est le type de cette cellule-là qui décide, côté C++, de la forme
        locale dans laquelle chaque cellule est construite."""
        if getattr( self, "_dom_cell", None ) is None:
            self._dom_cell = self._start_cell()
            if self.bnd_directions.is_defined:
                # les VALEURS du backend, pas du numpy : sous un `jit` les demi-espaces peuvent être
                # tracés. `stop_gradient` : le domaine est une constante du problème ( ses coupes
                # portent `BOUNDARY`, « pas un germe », et n'ont nulle part où envoyer une dérivée ).
                dirs = driver.stop_gradient( self.bnd_directions.raw )
                offs = driver.stop_gradient( self.bnd_offsets.raw )
                for b in range( int( self.bnd_directions.shape[ 0 ] ) ):
                    self._dom_cell.cut( dirs[ b ], offs[ b ], BOUNDARY )
        return self._dom_cell

    def _start_cell( self ):
        """le pavé de départ, ou tout l'espace"""
        d = self.dim_count
        kw = dict( kernel_dtype = self._kernel_dtype )
        if self.box_min.is_defined:
            mi = np.asarray( self.box_min ).reshape( -1 )
            ma = np.asarray( self.box_max ).reshape( -1 )
            return Cell.make_hypercube( d, mi, np.diag( ma - mi ), **kw )
        return Cell.make_unbounded( d, **kw )

    def _dist_for( self ):
        """Comment un appel nomme sa distribution : `( expression C++, expression de sa cotangente,
        kwargs de l'appel )`. Sans distribution, `unit_density()` -- une valeur que le C++ fabrique
        lui-même -- et une cotangente `0` que `UnitDensity` ignore."""
        if self.distribution is None:
            return "power_diagram.unit_density()", "0", {}
        return "distribution", "grad_for_distribution", { "distribution": self.distribution }

    # ---- le scratch --------------------------------------------------------------------------------

    def _scratch_words( self, cap, nb_cells, with_grad ):
        """Ce qu'un work-item immobilise, en mots : `nb_cells` cellules locales de `cap` sommets, et
        pour l'adjoint une cotangente par sommet -- LA MÊME FORMULE que `diagram::words_for`."""
        dom = self._domain_cell()
        words = nb_cells * dom.scratch_words( cap, fp_size( dom.kernel_dtype ) )
        if with_grad:
            words += -( -self.dim_count * cap * 8 // 32 ) * 8
        return words

    def _nb_work_cells( self ):
        """une distribution qui DÉCOUPE la cellule demande une seconde cellule locale"""
        return 2 if self.distribution is not None and getattr( self.distribution, "cuts_pieces", False ) else 1

    # ---- ce qu'on lit -----------------------------------------------------------------------------

    @property
    def measures( self ) -> Tensor:
        """La mesure de chaque cellule : `[ n ]`, indexé comme `positions`.

        Un seul appel, un seul balayage : chaque work-item construit une cellule dans son scratch,
        en écrit le volume, et recommence avec le germe suivant. Rien du diagramme n'est conservé.
        Avec une `distribution`, c'est l'INTÉGRALE de sa densité sur la cellule ( même balayage ).

        DÉRIVABLE par rapport aux germes, `positions` comme `weights`, et par rapport aux VALEURS de
        la distribution ( `diagram::measures_bwd` refait le même balayage ). Le DOMAINE est une
        constante : une coupe qui en vient porte un identifiant négatif, donc sa part ne va nulle part.
        """
        dom = self._domain_cell()
        # le budget qui décide du parallélisme : ce qu'UN work-item immobilise -- son scratch, taillé
        # pour le backward dès le forward ( il refait le balayage sur un scratch de même forme )
        nb_words = self._scratch_words( self._scratch_capacity, self._nb_work_cells(), True )
        nt = driver.device.nb_threads( nb_local_bytes_per_thread = 4 * nb_words, batch_axes = [ self.num_point ] )

        # l'axe des work-items est un axe de BATCH porté par le scratch : `thread_index` /
        # `nb_threads` sont le rang de ce work-item et leur nombre, la boucle striée se lit dessus
        num_thread = new_batch_axis( nt, prefix = "thread" )
        scratch, sc_kwargs = CellScratch.for_call( "scratch", nb_words, dom.kernel_dtype, batch_axes = [ num_thread ] )

        res = RealTensor[ self.num_point ]()
        dist_expr, grad_dist_expr, dist_kwargs = self._dist_for()

        driver.call(
            FfiCodeParallel( name = "power_diagram_measures",
                fwd_code = "power_diagram.measures( res, dom_cell, scratch( batch_index ), "
                           f"{ dist_expr }, thread_index, nb_threads );",
                # les gradients sur les germes sont PARTAGÉS par tous les items : chaque work-item y
                # accumule ( `atomic_add` côté C++ ), et la plateforme les met à zéro avant le corps
                bwd_code = "power_diagram.measures_bwd( res, dom_cell, grad_for_res, "
                           f"{ self._grad_seeds_expr() }, "
                           f"scratch( batch_index ), { dist_expr }, { grad_dist_expr }, "
                           "thread_index, nb_threads );" ),
            **merge_call( dict( output_attributes = [ "res" ] ), sc_kwargs ),
            power_diagram = self,
            dom_cell = dom,
            res = res,
            scratch = scratch,
            **dist_kwargs,
        )
        return res

    @property
    def moments( self ):
        """`( masses, first, second )` : pour chaque cellule, `int rho`, `int x rho` ( `[ n, d ]` ) et
        `int |x|^2 rho` -- de quoi écrire un COÛT DE TRANSPORT, `sum_i int_{cell_i} |x - p_i|^2 rho
        = second - 2 p . first + |p|^2 mass`, et les barycentres `first / mass`. Même balayage que
        `measures`, sur une distribution constante par morceaux ( `Image`, ou rien ) seulement.
        PAS dérivable : un coût de transport se dérive par le théorème de l'enveloppe, aux poids
        ajustés -- `2 mass_i ( p_i - b_i )` -- ce que `OtPlan` fait tout seul."""
        dom = self._domain_cell()
        nb_words = self._scratch_words( self._scratch_capacity, self._nb_work_cells(), False )
        nt = driver.device.nb_threads( nb_local_bytes_per_thread = 4 * nb_words, batch_axes = [ self.num_point ] )
        num_thread = new_batch_axis( nt, prefix = "thread" )
        scratch, sc_kwargs = CellScratch.for_call( "scratch", nb_words, dom.kernel_dtype, batch_axes = [ num_thread ] )

        mass = RealTensor[ self.num_point ]()
        first = RealTensor[ self.num_point, self.dim ]()
        second = RealTensor[ self.num_point ]()
        dist_expr, _, dist_kwargs = self._dist_for()

        driver.call(
            FfiCodeParallel( name = "power_diagram_moments",
                fwd_code = "power_diagram.moments( mass, first, second, dom_cell, scratch( batch_index ), "
                           f"{ dist_expr }, thread_index, nb_threads );" ),
            **merge_call( dict( output_attributes = [ "mass", "first", "second" ] ), sc_kwargs ),
            power_diagram = self,
            dom_cell = dom,
            mass = mass, first = first, second = second,
            scratch = scratch,
            **dist_kwargs,
        )
        return mass, first, second

    def hessian_rows( self ):
        """`( nb_nbrs, ids, vals )` : pour chaque cellule `i`, ses voisins `j` ( `ids[ i, :nb_nbrs[ i ] ]`,
        indexés comme `positions` ; négatifs pour le domaine, à ignorer ) et `vals[ i, r ] =
        int_{facette ij} rho / ( 2 | p_i - p_j | )` -- de quoi assembler la JACOBIENNE des mesures par
        rapport aux poids, `d m_i / d w_j = - vals`, `d m_i / d w_i = + sum_j vals`, qui est aussi la
        hessienne de la fonctionnelle duale d'un transport ( `OtPlan`, `objective = "newton"` ).
        Tableaux hôtes. Une distribution constante par morceaux seulement. Un appel batché sur les
        cellules, comme `cells` ( le nombre de voisins par cellule a une capacité que loom double )."""
        n, d = int( self.nb_points.value ), self.dim_count
        num_cell = new_batch_axis( n, prefix = "cell" )
        ranks = IntTensor[ num_cell ]( self._ranks_of_items() )

        dom = self._domain_cell()
        cap = self._scratch_capacity
        nbrs = Neighbors( batch_axes = [ num_cell ] )

        nb_words = self._scratch_words( cap, self._nb_work_cells(), False )
        nt = driver.device.nb_threads( nb_local_bytes_per_thread = 4 * nb_words, batch_axes = [ num_cell ] )
        scratch, sc_kwargs = CellScratch.for_call( "scratch", nb_words, dom.kernel_dtype, nb_threads = nt )
        dist_expr, _, dist_kwargs = self._dist_for()

        driver.call(
            FfiCodeParallel( name = "power_diagram_hessian_rows",
                fwd_code = "power_diagram.hessian_row( SI( ranks( batch_index ) ), dom_cell, nbrs( batch_index ), "
                           f"scratch, thread_index, { dist_expr } );",
                thread_cap = "scratch.words.shape( 0 )" ),
            **merge_call( dict(
                output_capacities = { "nbrs.nb_nbrs": 16 },
                output_attributes = [ "nbrs" ] ), sc_kwargs ),
            power_diagram = self,
            dom_cell = dom,
            ranks = ranks,
            scratch = scratch,
            nbrs = nbrs,
            **dist_kwargs,
        )
        counts = np.asarray( nbrs.nb_nbrs.value ).reshape( -1 ).astype( int )
        ids = np.asarray( nbrs.ids ).reshape( n, -1 )
        vals = np.asarray( nbrs.vals ).reshape( n, -1 )
        return counts, ids, vals

    @property
    def cells( self ) -> Cell:
        """TOUTES les cellules, en UN appel : une `Cell` batchée sur les germes, dans l'ordre de
        `positions`, ses `cut_ids` désignant les germes dans ce même ordre.

        La requête qui ne réduit pas une cellule à un nombre, donc la seule dont la mémoire soit
        fonction du nombre de germes -- ce qu'est un AFFICHAGE. Le scratch, lui, reste PAR
        WORK-ITEM ( `thread_cap` ). La `Cell` rendue se dessine telle quelle.
        """
        n, d = int( self.nb_points.value ), self.dim_count
        num_cell = new_batch_axis( n, prefix = "cell" )
        ranks = IntTensor[ num_cell ]( self._ranks_of_items() )

        dom = self._domain_cell()
        cap = self._scratch_capacity
        cells = Cell( d, init_as_unbounded = False, batch_axes = [ num_cell ], kernel_dtype = dom.kernel_dtype )

        nb_words = self._scratch_words( cap, 1, False )
        nt = driver.device.nb_threads( nb_local_bytes_per_thread = 4 * nb_words, batch_axes = [ num_cell ] )
        scratch, sc_kwargs = CellScratch.for_call( "scratch", nb_words, dom.kernel_dtype, nb_threads = nt )

        driver.call(
            FfiCodeParallel( name = "power_diagram_cells",
                fwd_code = "power_diagram.build_cell( SI( ranks( batch_index ) ), dom_cell, cells( batch_index ), "
                           "scratch, thread_index );",
                thread_cap = "scratch.words.shape( 0 )" ),
            **merge_call( dict(
                output_capacities = { "cells.nb_vertices": cap, "cells.nb_cuts": cap },
                output_attributes = [ "cells" ] ), sc_kwargs ),
            power_diagram = self,
            dom_cell = dom,
            ranks = ranks,
            scratch = scratch,
            cells = cells,
        )
        return cells

    def cell( self, i ) -> Cell:
        """La cellule du germe `i`, construite CÔTÉ PYTHON -- un `driver.call` par coupe.

        Le chemin lent, et volontairement : la même géométrie obtenue par une orchestration
        entièrement différente de celle du kernel, donc l'ORACLE des tests. Ce n'est PAS le chemin
        d'affichage ( `n²` allers-retours ), voir `cells`.
        """
        d = self.dim_count
        pos = np.asarray( self.positions ).reshape( -1, d )
        w = np.asarray( self.weights ).reshape( -1 ) if self.weights.is_defined else None

        res = self._start_cell()
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
            offset = float( direction @ ( p0 + pos[ j ] ) / 2 )
            if w is not None:
                offset += float( w[ i ] - w[ j ] ) / 2
            res.cut( direction, offset, j )
        return res

    def add_to_viz( self, viz, **kwargs ):
        """Se dessine dans un `Visualizer` : toutes les cellules, en un appel ( voir `cells` )."""
        return self.cells.add_to_viz( viz, **kwargs )


class Neighbors( Aggregate ):
    """les voisins d'UNE cellule et le poids de chaque facette ( voir `PowerDiagram.hessian_rows` ) --
    batché sur les cellules, `nb_nbrs` par cellule"""
    ids     : IntTensor [ "num_nbr", dict( size = 32 ) ]
    vals    : RealTensor[ "num_nbr" ]
    num_nbr : Axis[ "nb_nbrs" ]
    nb_nbrs : ShapeVar


# ---- le domaine, lu sur des demi-espaces ---------------------------------------------------------

def axis_aligned_box( directions, offsets ):
    """`( mi, ma, gardés )` : le pavé que ces demi-espaces bornent, et lesquels d'entre eux il ne
    remplace PAS. `None` s'ils ne bornent pas de pavé.

    On ne cherche pas à reconnaître un pavé « écrit comme il faut » : on cherche, axe par axe, la
    borne la plus serrée que les demi-espaces ALIGNÉS SUR CET AXE donnent. Un domaine qui n'est pas
    un pavé mais qui en contient un ( un octogone ) fournit donc quand même un point de départ, et
    ce qui dépasse est retiré par les plans restants. `None` dès qu'un axe n'est pas borné des deux
    côtés : la cellule de départ doit être un polytope BORNÉ.
    """
    try:
        dirs = np.asarray( directions, dtype = float )
        offs = np.asarray( offsets, dtype = float ).reshape( -1 )
    except ( TypeError, ValueError ):
        return None                              # géométrie non lisible ici ( un tracer ) : tant pis
    if dirs.ndim != 2 or len( dirs ) != len( offs ):
        return None

    d = dirs.shape[ 1 ]
    mi, ma = np.full( d, -np.inf ), np.full( d, np.inf )
    for k in range( len( dirs ) ):
        nz = np.flatnonzero( dirs[ k ] )
        if nz.size != 1:
            continue                             # pas aligné sur un axe : il sera coupé, c'est tout
        a = int( nz[ 0 ] )
        c = dirs[ k, a ]
        if c > 0:
            ma[ a ] = min( ma[ a ], offs[ k ] / c )
        else:
            mi[ a ] = max( mi[ a ], offs[ k ] / c )

    if not np.isfinite( mi ).all() or not np.isfinite( ma ).all():
        return None
    if not ( mi < ma ).all():                    # un pavé vide ne se pose pas : le chemin général videra
        return None

    # QUELS plans le pavé exprime déjà : ceux, alignés, qui atteignent la borne retenue sur leur axe
    kept = np.ones( len( dirs ), dtype = bool )
    for k in range( len( dirs ) ):
        nz = np.flatnonzero( dirs[ k ] )
        if nz.size != 1:
            continue
        a = int( nz[ 0 ] )
        c = dirs[ k, a ]
        bound = ma[ a ] if c > 0 else mi[ a ]
        kept[ k ] = offs[ k ] / c != bound
    return mi, ma, kept


def box_half_spaces( mi, ma ):
    """Le pavé `mi <= x <= ma` en `2d` demi-espaces `direction . x <= offset`."""
    mi = np.asarray( mi, dtype = float ).reshape( -1 )
    ma = np.asarray( ma, dtype = float ).reshape( -1 )
    if mi.size != ma.size:
        raise ValueError( "`box_half_spaces( mi, ma )` wants two corners of the same dimension" )
    d = mi.size
    return np.concatenate( [ np.eye( d ), -np.eye( d ) ] ), np.concatenate( [ ma, -mi ] )

import numpy as np

from loom.compilation.FfiCode import FfiCodeParallel
from loom.drivers.driver import driver
from loom.tensor import Affine, Axis, CtShapeVar, IntTensor, RealTensor, ShapeVar, Tensor, new_batch_axis
from loom.util import Aggregate

from .Cell import BOUNDARY, Cell


class PowerDiagram( Aggregate ):
    """Diagramme de puissance (Laguerre), RECONSTRUIT à chaque demande.

    La cellule du germe `i` est là où sa DISTANCE DE PUISSANCE gagne :

        |x - d_i|² - w_i  <=  |x - d_j|² - w_j   pour tout autre j

    Développée, l'inégalité perd son `|x|²` des deux côtés et devient un demi-espace -- c'est
    toute la raison d'être de cette convention : un diagramme de puissance coûte exactement ce que
    coûte un Voronoï, un plan par rival et le même clip. Le plan est la médiatrice euclidienne
    DÉCALÉE le long de sa normale par l'écart des poids (voir `PowerDiagram.cxx::make_cell`).

    Seules les DIFFÉRENCES de poids atteignent les plans : ajouter la même constante à tous les
    poids ne change rien au diagramme, donc « tous égaux » et « pas de poids du tout » sont le même
    objet. C'est pourquoi `weights` est FACULTATIF -- absent, il ne coûte pas un tampon de zéros à
    lire mais un `NoneTensor` dont le compilateur supprime le terme, et le cas euclidien s'appelle
    `Voronoi` (voir `Voronoi.py`).

    L'objet ne porte pas de diagramme : il porte ses GERMES (`positions`, `weights`) et le domaine
    convexe qui les borne (`bnd_directions . x <= bnd_offsets`). Une cellule n'existe qu'entre l'instant
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

    Le voisinage est ACCÉLÉRABLE. Par défaut chaque cellule est coupée par les `n - 1`
    bissectrices, donc le coût est en `n²` ; passer un `accelerator` (voir `AaBsp`) remplace cette
    énumération par une descente d'arbre qui ne va voir que les régions capables d'entamer la
    cellule -- et RIEN D'AUTRE ne change, ni la géométrie, ni les mesures, ni les dérivées. C'est
    la propriété qu'il faut retenir : un accélérateur ne peut que taire des coupes qui n'auraient
    rien enlevé, donc les cellules obtenues sont les MÊMES, aux erreurs d'arrondi près (voir
    `SpatialAccelerator.py` pour le contrat, et `cell_may_be_cut` pour le test qui le garantit).

    CONTRE QUOI on mesure est un second point d'extension, indépendant du premier : par défaut la
    mesure de Lebesgue (le volume), mais `distribution = Image( ... )` fait rendre à `measures`
    l'intégrale d'une densité sur chaque cellule. Là encore le diagramme ne sait rien de la
    distribution : elle DÉCOUPE la cellule en morceaux sur lesquels sa densité est simple, et le
    diagramme intègre sur un morceau -- de sorte qu'une autre grandeur que la mesure (un
    barycentre, un moment second) s'écrira comme une seconde fonctionnelle sur le MÊME découpage.
    « Pas de distribution » est un cas ordinaire de ce code-là, `UnitDensity`, et non une absence
    de code. Voir `distributions/Distribution.py`.
    """

    positions      : RealTensor[ "num_point", "dim" ]

    # NOS positions, dans l'ordre où l'accélérateur balaie les germes -- `positions[ seed_at( k ) ]`.
    # Une copie, et c'est tout l'objet : sans elle, essayer les germes d'une feuille allait chercher
    # `positions( i1 )` à des endroits épars d'un tableau qui pèse 16 Mo à 1e6 germes, alors que les
    # germes d'une feuille sont justement voisins DANS CET ORDRE-LÀ.
    #
    # Elle est ici et pas dans l'accélérateur, et c'est la leçon d'un bug : un accélérateur qui
    # fournirait les positions rendrait celles du DIAGRAMME non déterminantes, donc ses dérivées
    # fausses dès que l'accélérateur a été bâti sur un autre nuage (ce que
    # `the_accelerator_changes_nothing_to_the_derivatives` vérifie). L'accélérateur ne répond que de
    # l'ÉNUMÉRATION -- combinatoire, constante du trace, de gradient nul.
    #
    # Le rassemblement est fait par le backend (voir `_sorted_positions`), sous un `stop_gradient` :
    # la dérivée par rapport aux positions est celle que `measures_bwd` calcule analytiquement et
    # dépose dans `grad_positions`, donc la faire passer une seconde fois par cette copie la
    # compterait deux fois. Absente (pas d'accélérateur), elle est `Unbound` et le kernel lit
    # `positions( i1 )` comme avant.
    sorted_positions : RealTensor[ "num_point", "dim" ]

    # FACULTATIF, exactement comme le domaine ci-dessous : absent (`Unbound`, `NoneTensor` côté
    # C++), le terme de poids disparaît du plan À LA COMPILATION et le diagramme est l'euclidien.
    # Ce n'est pas une optimisation gratuite mais la bonne façon de dire ce qu'on veut dire : un
    # poids constant ne se voit nulle part dans le diagramme, donc « pas de poids » est un ÉTAT,
    # pas un tableau de zéros.
    weights        : RealTensor[ "num_point" ]

    # LA CELLULE DE DÉPART, quand on en connaît une meilleure que « tout l'espace » : le pavé
    # `box_min <= x <= box_max`, posé d'un trait par `Cell::init_as_hypercube`. Absent (`Unbound`,
    # `NoneTensor`), chaque cellule naît comme un SIMPLEXE DE REMPLACEMENT non borné dont toute
    # coupe doit d'abord repousser les plans infinis -- et il en faut `2d` pour la borner.
    #
    # Ce n'est PAS un second domaine : c'est le même, exprimé sous la forme qu'on sait poser
    # directement. Ce qu'un pavé ne dit pas reste dans `bnd_directions` / `bnd_offsets` et se coupe
    # comme avant, de sorte qu'un domaine quelconque, ou un support de distribution qui n'est pas un
    # pavé, marche exactement comme aujourd'hui. Voir `Distribution.bounding_box`, qui est l'autre
    # source de ce pavé.
    box_min        : RealTensor[ "dim" ]
    box_max        : RealTensor[ "dim" ]

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

    def __init__( self, positions, weights = None, boundaries = None, max_nb_cuts = None,
                  accelerator = None, distribution = None, **kwargs ):
        """`positions` : `[ n, d ]`. `weights` : `[ n ]`, ou rien (le cas euclidien). Le domaine :

        - `boundaries = ( directions, offsets )` -- les demi-espaces `direction . x <= offset`.
          Un pavé s'écrit `box_half_spaces( mi, ma )`, qui est là pour ça ;
        - rien -- les cellules du bord restent infinies.

        `max_nb_cuts` dimensionne les tampons d'une cellule (voir `_capacities`). Ce n'est qu'une
        supposition : trop petite, le kernel l'enregistre et la plateforme relance avec le double.

        `accelerator` : de quoi n'essayer que les germes qui peuvent servir (`AaBsp.of( pd )`).
        Facultatif, et sans effet sur le RÉSULTAT -- seulement sur ce qu'il coûte.

        `distribution` : CONTRE QUOI intégrer (`Image`, `SumOfGaussians`, ...). Absente, `measures`
        rend le volume des cellules ; présente, l'intégrale de sa densité dessus. Elle est
        NORMALISÉE ici, une fois pour toutes (`normalized_version`), de sorte que la somme des
        mesures soit sa masse cible et pas ce que ses valeurs brutes valaient. Si elle a un SUPPORT
        borné, il s'ajoute au domaine -- gratuit et sans effet sur le résultat, voir
        `Distribution.bounding_half_spaces`. Voir `distributions/Distribution.py` pour le contrat.
        """

        # le SUPPORT de la distribution borne le domaine, gratuitement et sans rien changer au
        # résultat : ce qui dépasse n'apporte aucune masse. Voir `Distribution.bounding_half_spaces`
        # pour ce que ça fait gagner ; le domaine de l'appelant, s'il y en a un, est INTERSECTÉ avec
        # (les demi-espaces s'ajoutent), pas remplacé.
        if distribution is not None:
            support = distribution.bounding_half_spaces()
            if support is not None:
                if boundaries is None:
                    boundaries = support
                else:
                    boundaries = ( np.concatenate( [ np.asarray( boundaries[ 0 ], dtype = float ), support[ 0 ] ] ),
                                   np.concatenate( [ np.asarray( boundaries[ 1 ], dtype = float ), support[ 1 ] ] ) )

        # pas de domaine -> on ne NOMME pas les deux tenseurs : les laisser `Unbound` (jamais
        # alloués, `NoneTensor` côté C++) n'est pas la même chose que leur passer `None`, qui est
        # une valeur, et une valeur de rang 0.
        if boundaries is not None:
            # D'OÙ PARTIR, lu sur les demi-espaces eux-mêmes : les demi-espaces restent la seule
            # façon d'ÉCRIRE le domaine, et le pavé n'en est qu'une lecture. Ceux qu'il exprime
            # déjà sortent de la liste -- ils sont `2d` et reviendraient sur chaque cellule.
            #
            # Les DEUX chemins de construction doivent alors partir du même endroit, sans quoi le
            # `cell( i )` coupe par coupe se retrouve sans domaine : voir `cell`, qui part du même
            # pavé. C'est ce qui manquait à une première version, et ça se voyait à des mesures
            # infinies.
            start_box = axis_aligned_box( *boundaries )
            if start_box is not None:
                mi, ma, kept = start_box
                kwargs[ "box_min" ], kwargs[ "box_max" ] = mi, ma
                boundaries = ( np.asarray( boundaries[ 0 ], dtype = float )[ kept ],
                               np.asarray( boundaries[ 1 ], dtype = float )[ kept ] )

            if len( boundaries[ 1 ] ):
                kwargs[ "bnd_directions" ], kwargs[ "bnd_offsets" ] = boundaries

        # même règle pour les poids : `None` n'est pas « des zéros », c'est « pas de poids ». On ne
        # nomme donc pas le membre plutôt que de lui passer une valeur.
        if weights is not None:
            kwargs[ "weights" ] = weights

        self.__base_init__( positions = positions, **kwargs )

        self._max_nb_cuts = max_nb_cuts

        # PAS un champ de l'agrégat : l'accélérateur est un argument d'APPEL, pas un morceau du
        # diagramme. Le mettre en membre obligerait « pas d'accélérateur » à être un agrégat vide,
        # que le C++ engendré ne sait pas écrire (une structure sans membre n'a pas de paramètre
        # de template) -- alors que côté kernel l'absence a déjà un nom, `EverySeed`, qui est un
        # accélérateur comme un autre et se fabrique sur place (`PowerDiagram::every_seed`).
        self.accelerator = accelerator

        # PAS un champ non plus, et pour la même raison que l'accélérateur : c'est un argument
        # d'APPEL. Normalisée DÈS ICI plutôt qu'à chaque `measures` -- « le diagramme intègre CETTE
        # mesure-là » est une propriété de l'objet, et la normalisation est un calcul (la masse
        # totale) qu'on ne veut pas refaire à chaque requête.
        self.distribution = None
        if distribution is not None:
            nd = int( self.nb_dims.value )
            dd = int( distribution.nb_dims.value )
            if dd != nd:
                raise ValueError( f"the distribution lives in { dd }D, this diagram in { nd }D" )
            self.distribution = distribution.normalized_version()


    def _dist_for( self ):
        """Comment cet appel nomme sa distribution : `( expression C++, expression de sa cotangente,
        kwargs de l'appel )`.

        Sans distribution, l'expression est `power_diagram.unit_density()` -- une valeur que le C++
        fabrique lui-même -- et la cotangente un `0` littéral, que `UnitDensity` ignore. Rien n'est
        donc passé depuis Python dans ce cas : pas d'agrégat vide à engendrer, et une seule
        signature C++ par méthode. Exactement le montage de `_acc_for`.
        """
        if self.distribution is None:
            return "power_diagram.unit_density()", "0", {}
        return "distribution", "grad_for_distribution", { "distribution": self.distribution }


    def _extra_cuts_per_piece( self ):
        """Combien de coupes de plus qu'une cellule un morceau peut porter -- 0 sans distribution
        (le morceau EST la cellule, donc pas de cellules de découpe du tout)."""
        if self.distribution is None:
            return 0
        return int( self.distribution.extra_cuts_per_piece( int( self.nb_dims.value ) ) )


    def _domain_cell( self ):
        """Le domaine comme POLYTOPE, calculé UNE FOIS -- ce dont chaque cellule part.

        Avant, `make_cell` refaisait le domaine pour CHACUN des germes : la cellule de départ, puis
        une coupe par plan restant. Sur un domaine qui n'est pas un pavé -- un octogone, le support
        d'une distribution, un domaine tronqué -- ces plans étaient donc reclippés autant de fois
        qu'il y a de germes, et on a mesuré que quatre plans redondants coûtaient déjà 25 %. Ici,
        chaque germe le RECOPIE (`Cell::copy_into`), ce qui passe le coût par germe de `k` coupes à
        une copie.

        Ça vaut même pour un pavé : `init_as_hypercube` inverse le repère, donc résout un système
        `d x d` par cellule, là où la copie ne fait que lire.

        Construit avec le MÊME code que l'oracle `cell( i )` -- un `driver.call` par plan -- ce qui
        ne coûte rien puisque c'est une fois par diagramme, et garde les deux descriptions du
        domaine littéralement identiques. Mis en cache sur l'instance : le domaine ne change pas.
        """
        if getattr( self, "_dom_cell", None ) is None:
            d = int( self.nb_dims.value )
            if self.box_min.is_defined:
                mi = np.asarray( self.box_min ).reshape( -1 )
                ma = np.asarray( self.box_max ).reshape( -1 )
                cell = Cell.make_hypercube( d, mi, np.diag( ma - mi ) )
            else:
                cell = Cell.make_unbounded( d )
            if self.bnd_directions.is_defined:
                # les VALEURS du backend, pas du numpy : sous un `jit` les demi-espaces peuvent être
                # tracés, et les rapatrier côté hôte échouerait. Seul leur NOMBRE est lu ici, et il
                # est connu (c'est une entrée prescrite).
                #
                # `stop_gradient` dit ce que `the_domain_carries_no_gradient` vérifie : le domaine
                # est une constante du problème. Une coupe qui en vient porte `BOUNDARY`, c'est-à-dire
                # « pas un germe », donc sa part n'a de toute façon nulle part où aller (voir
                # `measures`) -- le dire ici évite que jax cherche à la faire remonter par la
                # construction de cette cellule.
                dirs = driver.stop_gradient( self.bnd_directions.raw )
                offs = driver.stop_gradient( self.bnd_offsets.raw )
                for b in range( int( self.bnd_directions.shape[ 0 ] ) ):
                    cell.cut( dirs[ b ], offs[ b ], BOUNDARY )
            self._dom_cell = cell
        return self._dom_cell


    def _sorted_positions( self ):
        """NOS positions rangées dans l'ordre de l'accélérateur -- voir le champ `sorted_positions`.

        `stop_gradient` : la valeur est celle du diagramme, mais la dérivée qui la concerne est déjà
        produite analytiquement par `measures_bwd`. Sans lui, jax ajouterait la contribution du
        rassemblement à celle du kernel, et le gradient serait compté deux fois.
        """
        if self.accelerator is None:
            return
        order = self.accelerator.seed_indices.raw
        if order is None:
            return
        self.sorted_positions = driver.stop_gradient( self.positions.raw[ order ] )


    def _acc_for( self, batch_axis ):
        """Comment cet appel nomme son accélérateur : `( expression C++, expression du scratch,
        kwargs de l'appel, noms à déclarer en scratch )`.

        Sans accélérateur, l'expression est `power_diagram.every_seed()` -- une valeur que le C++
        fabrique lui-même à partir d'un compte qu'il a déjà -- et le scratch est un `0` littéral,
        que `EverySeed` ignore. Rien n'est donc passé depuis Python dans ce cas : pas d'agrégat vide à
        engendrer, pas de tampon à allouer, et une seule signature C++ par méthode.
        """
        if self.accelerator is None:
            return "power_diagram.every_seed()", "0", {}, []

        # l'accélérateur INDEXE nos germes : construit sur un autre nuage, ses indices désignent
        # autre chose, et la réponse serait fausse sans rien qui le signale. Le compte est ce qu'on
        # peut vérifier ici pour rien, et il attrape le mésusage courant (`AaBsp` gardé d'un pas
        # précédent). Voir `AaBsp.of` pour la façon de ne pas se poser la question.
        n = int( self.nb_points.value )
        nb = self.accelerator.nb_seeds()
        if nb is not None and nb != n:
            raise ValueError( f"the accelerator was built on { nb } seeds, this diagram has { n }" )

        ws = self.accelerator.thread_scratch( batch_axis )
        if ws is None:
            return "accelerator", "0", { "accelerator": self.accelerator }, []
        return ( "accelerator", "acc_ws( batch_index )",
                 { "accelerator": self.accelerator, "acc_ws": ws }, [ "acc_ws" ] )


    @property
    def measures( self ) -> Tensor:
        """La mesure de chaque cellule : `[ n ]`, indexé comme `positions`.

        Un seul appel, un seul balayage : chaque work-item construit une cellule dans ses tampons
        à lui, en écrit le volume, et recommence avec le germe suivant. Rien du diagramme n'est
        conservé entre deux germes -- c'est tout l'intérêt.

        Avec une `distribution`, ce n'est plus le volume mais l'INTÉGRALE de sa densité sur la
        cellule -- même balayage, même mémoire : la distribution découpe la cellule en morceaux où
        sa densité est simple, et on intègre morceau par morceau sans jamais en garder un
        (`PowerDiagram.cxx::integrate_into`, et `distributions/Distribution.py` pour le contrat).
        La distribution ayant été normalisée à la construction, la somme des mesures est sa masse
        cible dès que les cellules la recouvrent.

        DÉRIVABLE par rapport aux germes, `positions` comme `weights` (voir
        `PowerDiagram.cxx::measures_bwd`), et par rapport aux VALEURS de la distribution -- la
        masse en étant linéaire, la dérivée par rapport à la valeur d'un morceau est le volume de
        ce morceau. Le backward refait le même balayage : les cellules n'ayant pas été gardées, il
        les REconstruit plutôt que de les relire (les morceaux aussi), ce qui lui coûte un forward
        de plus et rien en mémoire. Le DOMAINE, lui, est traité comme une constante : une coupe
        venue de `bnd_directions` porte `cut_id == BOUNDARY`, qui dit « pas un germe » et non
        LEQUEL, donc sa part n'a nulle part où aller. La GÉOMÉTRIE d'une distribution
        (`origin` / `frame` / `knots` d'une image) est constante pour la même raison : les plans
        d'un morceau qui en viennent portent `BOUNDARY` eux aussi.
        """
        d = int( self.nb_dims.value )
        if d < 2:
            raise NotImplementedError( f"PowerDiagram needs nb_dims >= 2 for now ( nb_dims = { d } )" )

        cap_v, cap_e, cap_c = self._capacities()

        # les cellules de DÉCOUPE, s'il y a une distribution : un morceau porte les coupes de la
        # cellule plus celles que le découpage ajoute (`extra_cuts_per_piece`), donc elles sont
        # plus grandes. Deux, parce qu'une suite de coupes fait la navette entre deux tampons --
        # et deux DE PLUS plutôt que réutiliser celles du clip : la cellule, elle, doit survivre à
        # tous ses morceaux (voir `PieceWorkspace.h`).
        extra = self._extra_cuts_per_piece()
        pce_v, pce_e, pce_c = self._capacities( extra )

        # le budget qui décide du parallélisme : ce qu'UN work-item immobilise. Les cellules de
        # travail (la navette du clip, plus celle du découpage), les cotangentes par sommet du
        # backward (`grad_vp`), plus les scratchs du régime d > 2 -- la table de compaction de la
        # coupe (`corr`) et les apex de la triangulation (`facet_apex`). Tous les trois sont
        # dimensionnés sur la SOMME des cellules de travail, comme les axes plus bas. Le device en
        # déduit combien de work-items il peut se permettre, plafonné par le nombre de germes
        # (inutile d'en réserver plus qu'il n'y a de travail).
        tot_v, tot_c = 2 * cap_v, 2 * cap_c
        per_thread = 2 * self._bytes_per_cell( cap_v, cap_e, cap_c )
        if extra:
            tot_v += 2 * pce_v
            tot_c += 2 * pce_c
            per_thread += 2 * self._bytes_per_cell( pce_v, pce_e, pce_c )
        per_thread += 8 * tot_v * d                     # `grad_vp`, le tampon du backward
        if d > 2:
            per_thread += 8 * ( tot_v + tot_c + 1 ) + 8 * d * tot_c
        if self.accelerator is not None:
            per_thread += self.accelerator.bytes_per_thread()
        nt = driver.device.nb_threads( nb_local_bytes_per_thread = per_thread,
                                       batch_axes = [ self.num_point ] )

        # l'axe des work-items est un axe de BATCH : c'est lui qui donne à l'appel son espace
        # d'items, donc un work-item par emplacement de travail. `thread_index` / `nb_threads`
        # (les noms réservés du scaffold) sont alors exactement le rang de ce work-item et leur
        # nombre, et la boucle striée sur les germes se lit directement dessus.
        num_thread = new_batch_axis( nt, prefix = "thread" )

        def work_cell():
            return Cell( d, init_as_unbounded = False, batch_axes = [ num_thread ] )

        ws_0 = work_cell()
        ws_1 = work_cell()

        # `ws_2` / `ws_3` n'existent QUE s'il y a quelque chose à découper : sans distribution le
        # morceau est la cellule, donc pas un tampon de plus, pas une coupe de plus, et le C++ reçoit
        # une paire bidon (`UnitDensity` n'y touche jamais).
        piece_cells = [ "ws_2", "ws_3" ] if extra else []
        piece_kwargs = { n: work_cell() for n in piece_cells }
        work_cells = [ ws_0, ws_1 ] + list( piece_kwargs.values() )

        # Les scratchs sont dimensionnés en EXPRESSION DES COMPTES DES CELLULES, pas sur des
        # entiers figés : une capacité n'est qu'une supposition, et quand elle ne suffit pas c'est
        # `driver.call` qui la double et relance -- si le scratch ne suivait pas, la coupe suivante
        # écrirait à côté (`corr` est indexé par les anciens sommets PUIS les anciennes coupes,
        # `facet_apex` par le numéro de coupe). La SOMME des cellules, et non leur max : chacune a
        # son propre compte, la croissance peut n'en toucher qu'une, et une somme les majore
        # toutes -- au prix d'un facteur sur un scratch déjà petit devant les cellules.
        cuts = _sum_of( Affine.of( c.nb_cuts ) for c in work_cells )
        verts = _sum_of( Affine.of( c.nb_vertices ) for c in work_cells )

        # Axes NOMMÉS, construits DIRECTEMENT et pas via `Axis[ sv ]( name = ... )` :
        # `Parametrized.__call__` plierait `name` dans les template_kwargs.
        num_corr = Axis( verts + cuts + Affine.constant( 1 ), name = "num_corr" )
        num_level = Axis( ShapeVar( d ), name = "num_level" )
        num_cut_slot = Axis( cuts, name = "num_cut_slot" )
        # le seul tampon que le BACKWARD ajoute : une cotangente par sommet de la cellule courante,
        # ce que `Cell::measure_bwd` écrit et ce que la remontée vers les plans relit. Dimensionné
        # comme les autres sur une expression affine des comptes, pour suivre un doublement de
        # capacité -- et sur la somme, donc il couvre aussi un MORCEAU, qui a plus de sommets que la
        # cellule dont il sort. Il est déclaré dès le forward parce qu'un scratch appartient à
        # L'APPEL, pas à l'une de ses deux directions -- le forward l'alloue et n'y touche pas.
        num_grad_vertex = Axis( verts, name = "num_grad_vertex" )

        res = RealTensor[ self.num_point ]()

        # en d <= 2 ces deux-là ne sont PAS déclarés en sortie : ils ne sont donc jamais alloués et
        # arrivent en `NoneTensor`, ce qu'attendent les `if constexpr ( ct_dim > 2 )` du C++.
        scratch = [ "corr", "facet_apex" ] if d > 2 else []

        self._sorted_positions()
        acc_expr, acc_ws_expr, acc_kwargs, acc_scratch = self._acc_for( num_thread )
        dist_expr, grad_dist_expr, dist_kwargs = self._dist_for()

        # le scratch de découpe, monté CÔTÉ C++ par agrégation (`PieceWorkspace.h`) : une seule
        # expression à passer, donc une seule signature, qu'il y ait une distribution ou non.
        a, b = ( piece_cells if extra else [ "ws_0", "ws_1" ] )
        piece_ws_expr = f"PieceWorkspace{{ { a }( batch_index ), { b }( batch_index ), corr( batch_index ) }}"

        capacities = {}
        exceptions = []
        for name, cell in zip( [ "ws_0", "ws_1" ] + piece_cells, work_cells ):
            capacities |= self._cell_capacities( name, extra if name in piece_cells else 0 )
            exceptions += cell._face_lattice_exceptions( name )

        driver.call(
            FfiCodeParallel( name = "power_diagram_measures",
                fwd_code = "power_diagram.measures( res, dom_cell, ws_0( batch_index ), ws_1( batch_index ), "
                           "corr( batch_index ), facet_apex( batch_index ), "
                           f"{ acc_expr }, { acc_ws_expr }, { dist_expr }, { piece_ws_expr }, "
                           "thread_index, nb_threads );",
                # `grad_for_power_diagram.positions` / `.weights` sont PARTAGÉS par tous les items
                # (ils ne portent pas l'axe de batch) : chaque work-item y accumule pour ses germes,
                # d'où les `atomic_add` côté C++ -- et la plateforme les met à zéro avant le corps,
                # ce qu'elle fait justement pour une sortie flottante partagée d'un appel batché
                # (`CallArg_Tensor.cpp_seed_member`), donc pas de `bwd_setup_code` ici. Idem pour
                # `grad_for_distribution.values`, qui reçoit les mêmes `atomic_add`.
                bwd_code = "power_diagram.measures_bwd( res, dom_cell, grad_for_res, "
                           "grad_for_power_diagram.positions, grad_for_power_diagram.weights, "
                           "ws_0( batch_index ), ws_1( batch_index ), corr( batch_index ), "
                           "facet_apex( batch_index ), grad_vp( batch_index ), "
                           f"{ acc_expr }, { acc_ws_expr }, { dist_expr }, { grad_dist_expr }, "
                           f"{ piece_ws_expr }, thread_index, nb_threads );" ),
            output_capacities = capacities,
            output_exceptions = exceptions,
            output_attributes = [ "res", "ws_0", "ws_1", "grad_vp" ] + piece_cells + scratch + acc_scratch,
            # des tampons de travail, pas des résidus : leur contenu ne survit pas à l'appel (les
            # work-items se les repassent d'un germe à l'autre).
            scratch_attributes = [ "ws_0", "ws_1", "grad_vp" ] + piece_cells + scratch + acc_scratch,
            power_diagram = self,
            dom_cell = self._domain_cell(),
            res = res,
            **acc_kwargs,
            **dist_kwargs,
            ws_0 = ws_0,
            ws_1 = ws_1,
            **piece_kwargs,
            # des INDICES : `IntTensor`, sinon leurs lectures reviendraient en flottants.
            corr = IntTensor[ num_thread, num_corr ](),
            facet_apex = IntTensor[ num_thread, num_level, num_cut_slot ](),
            grad_vp = RealTensor[ num_thread, num_grad_vertex, self.dim ](),
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
            raise NotImplementedError( f"PowerDiagram needs nb_dims >= 2 for now ( nb_dims = { d } )" )

        # l'axe des items est celui des CELLULES : un work-item par germe, `batch_index` est donc
        # le germe. Un axe de batch FRAIS et non `self.num_point` : les deux ont la même étendue,
        # mais réutiliser l'axe de `positions` ferait porter à une entrée le nom d'un axe de batch
        # de l'appel, ce qui n'a rien à voir avec ce qu'on veut dire (`positions` est lu EN ENTIER
        # par chaque item, il est indexé par le germe, pas découpé par l'item).
        n = int( self.nb_points.value )
        num_cell = new_batch_axis( n, prefix = "cell" )

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

        acc_expr, acc_ws_expr, acc_kwargs, acc_scratch = self._acc_for( num_cell )

        driver.call(
            FfiCodeParallel( name = "power_diagram_cells",
                fwd_code = "power_diagram.build_cell( SI( seeds( batch_index ) ), dom_cell, cells( batch_index ), "
                           "ws_0( batch_index ), ws_1( batch_index ), corr( batch_index ), "
                           f"{ acc_expr }, { acc_ws_expr } );" ),
            output_capacities = ( self._cell_capacities( "cells" )
                                | self._cell_capacities( "ws_0" )
                                | self._cell_capacities( "ws_1" ) ),
            output_exceptions = ( cells._face_lattice_exceptions( "cells" )
                                + ws_0._face_lattice_exceptions( "ws_0" )
                                + ws_1._face_lattice_exceptions( "ws_1" ) ),
            output_attributes = [ "cells", "ws_0", "ws_1" ] + scratch + acc_scratch,
            # `sorted_positions` est la copie de NOS positions dans l'ordre de l'accélérateur, et
            # elle s'indexe par le RANG du germe. Ici l'item EST le germe : il n'y a pas de rang à
            # offrir, donc on force le membre à rester non lié (un `measures` précédent sur le même
            # objet l'aurait laissé lié). `build_cell` le vérifie à la compilation.
            input_exceptions = [ "power_diagram.sorted_positions" ],
            # `cells` est LA sortie ; les deux autres et `corr` ne sont que la navette du clip.
            scratch_attributes = [ "ws_0", "ws_1" ] + scratch + acc_scratch,
            power_diagram = self,
            dom_cell = self._domain_cell(),
            seeds = seeds,
            **acc_kwargs,
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

        # le MÊME point de départ que le kernel (voir `__init__`) : l'oracle est une autre
        # orchestration de la même géométrie, pas un autre domaine.
        if self.box_min.is_defined:
            mi = np.asarray( self.box_min ).reshape( -1 )
            ma = np.asarray( self.box_max ).reshape( -1 )
            res = Cell.make_hypercube( d, mi, np.diag( ma - mi ) )
        else:
            res = Cell.make_unbounded( d )
        if self.bnd_directions.is_defined:
            bds = np.asarray( self.bnd_directions ).reshape( -1, d )
            bos = np.asarray( self.bnd_offsets ).reshape( -1 )
            for b in range( len( bds ) ):
                res.cut( bds[ b ], float( bos[ b ] ), BOUNDARY )

        w = None
        if self.weights.is_defined:
            w = np.asarray( self.weights ).reshape( -1 )

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
        """Se dessine dans un `Visualizer` : toutes les cellules, en un appel (voir `cells`)."""
        return self.cells.add_to_viz( viz, **kwargs )


    # -- ce qu'UNE cellule immobilise ---------------------------------------------------------

    def _capacities( self, extra_cuts = 0 ):
        """`( sommets, arêtes, coupes )` : la taille des tampons d'une cellule.

        `extra_cuts` dit combien de coupes de plus qu'une cellule le tampon doit encaisser -- ce que
        demande une cellule de DÉCOUPE, un morceau portant les coupes de la cellule plus celles que
        la distribution ajoute (voir `Distribution.extra_cuts_per_piece`).

        Une SUPPOSITION, pas un contrat -- si elle ne suffit pas, le kernel l'enregistre, sort sans
        rien écrire de faux, et la plateforme relance avec le double (voir `driver.call`). On part
        du nombre de coupes parce que c'est la seule des trois qu'on sache estimer : une cellule de
        Voronoï en 3D a une quinzaine de faces, et le reste s'en déduit par les relations d'un
        polytope SIMPLE (ce que `Cell.cut` maintient) -- `V = 2F - 4`, `E = 3F - 6` en 3D, une borne
        beaucoup plus lâche au-delà.
        """
        d = int( self.nb_dims.value )
        cap_c = ( self._max_nb_cuts or 32 ) + extra_cuts
        if d == 2:
            return cap_c, 0, cap_c          # l'invariant du chemin 2D : une coupe par sommet
        if d == 3:
            return 2 * cap_c, 3 * cap_c, cap_c
        return 4 * ( d - 2 ) * cap_c, 8 * ( d - 2 ) * cap_c, cap_c

    def _cell_capacities( self, name, extra_cuts = 0 ):
        cap_v, cap_e, cap_c = self._capacities( extra_cuts )
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


def _sum_of( affines ):
    """La somme d'une suite d'`Affine` (`Affine` n'a pas d'élément neutre à donner à `sum`)."""
    res = None
    for a in affines:
        res = a if res is None else res + a
    return res


def axis_aligned_box( directions, offsets ):
    """`( mi, ma, gardes )` : le pavé que ces demi-espaces bornent, et lesquels d'entre eux il ne
    remplace PAS. `None` s'ils ne bornent pas de pavé.

    On ne cherche pas à reconnaître un pavé « écrit comme il faut » : on cherche, axe par axe, la
    borne la plus serrée que les demi-espaces ALIGNÉS SUR CET AXE donnent. Un domaine qui n'est pas
    un pavé mais qui en contient un (un octogone, une boîte plus deux plans obliques) fournit donc
    quand même un point de départ, et ce qui dépasse est retiré par les plans restants -- qui sont
    de toute façon tous coupés.

    `None` dès qu'un axe n'est pas borné des deux côtés : la cellule de départ doit être un polytope
    BORNÉ, sans quoi il n'y a rien à poser d'un trait.
    """
    try:
        dirs = np.asarray( directions, dtype = float )
        offs = np.asarray( offsets, dtype = float ).reshape( -1 )
    except ( TypeError, ValueError ):
        return None                              # géométrie non lisible ici (un tracer) : tant pis
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
    # un pavé vide ou dégénéré ne se pose pas : `init_as_hypercube` en ferait un polytope retourné.
    # Le chemin général, lui, videra la cellule proprement.
    if not ( mi < ma ).all():
        return None

    # QUELS plans le pavé exprime déjà : ceux, alignés, qui atteignent la borne retenue sur leur
    # axe. Ils sont RETIRÉS de la liste, et ce n'est pas une élégance -- les repasser coûte 25 %
    # (mesuré) : ils sont `2d` et ils reviennent sur CHAQUE cellule, alors qu'une cellule ne porte
    # que six coupes utiles. Un plan alignés plus lâche, ou un plan oblique, reste dans la liste.
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

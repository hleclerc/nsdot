import math

import numpy as np

# `loom.tensor` D'ABORD : `AaBsp` est le premier module que `sdot/__init__.py` importe, et
# `loom.drivers.driver` importé avant lui coupe le cycle `driver <-> tensor` du mauvais côté.
from loom.tensor import Axis, CtShapeVar, IntTensor, RealTensor, ShapeVar, new_batch_axis
from loom.compilation.FfiCode import FfiCodeParallel
from loom.drivers.driver import driver
from loom.util import Aggregate

from .SpatialAccelerator import SpatialAccelerator


class AaBsp( SpatialAccelerator ):
    """Un BSP ALIGNÉ SUR LES AXES : un arbre binaire de boîtes, chaque feuille tenant une
    poignée de germes.

    L'arbre est bâti par coupes médianes sur l'axe le plus long, jusqu'à ce qu'une feuille
    n'ait plus que `max_seeds_per_leaf` germes : plus petit, l'arbre coûte plus en descentes qu'il
    ne fait gagner en coupes évitées ; plus grand, on paie des bissectrices dont on savait déjà
    qu'elles ne serviraient à rien.

    L'équilibre est DIX, et il est mesuré (Xeon W-2145 + RTX 2080 Ti, 1e6 germes en 2D) : le
    plateau va de 6 à 12, et 30 -- l'ancien défaut, mesuré sur un autre processeur -- y coûte 15 %.
    C'est un réglage qui suit la MACHINE et pas le problème : il arbitre entre le coût d'une coupe
    et celui d'une éviction de boîte, et les deux ne bougent pas ensemble d'un processeur à
    l'autre. À rouvrir dès que l'un des deux change (voir le banc `pd accelerated`, qui le balaie :
    `./run bench "test_PowerDiagram::pd accelerated" --leaf-size=6,10,16,30`).

    = Ce que chaque nœud porte, et pourquoi

    La BOÎTE (`node_box`, `lo` puis `hi`) contient tous les germes du sous-arbre, et un MAJORANT
    AFFINE de leurs poids : `w( y ) <= node_wa . y + node_wb` pour tout germe `y` du sous-arbre.
    Les deux ensemble suffisent à répondre « rien là-dedans ne peut couper cette cellule », et le
    majorant affine est ce qui rend la réponse fine : la borne classique est un majorant CONSTANT
    (le poids max du nœud), qui traite toute la boîte comme si le germe le plus lourd était
    partout. Un poids qui varie régulièrement dans l'espace -- ce qui est exactement le régime du
    transport optimal semi-discret, où les poids sont un potentiel -- est alors très mal borné.

    L'autre raison, moins évidente et décisive : le degré 1 ne coûte RIEN de plus à tester. Le
    minimum de `|p - y|² - wa . y` sur une boîte est SÉPARABLE par axe, son minimum libre est en
    `y = p + wa / 2`, et un `clamp` par axe donne la réponse exacte. La borne constante fait le
    même travail avec `wa = 0`. Un majorant de degré 2 casserait cette séparabilité.

    Le majorant est choisi À LA CONSTRUCTION, nœud par nœud, entre l'affine ajusté aux moindres
    carrés et le constant : l'affine n'est retenu que s'il resserre franchement l'ÉTALEMENT des
    résidus, qui est précisément ce qui fait le mou de la borne (voir `_weight_majorant`). Sans
    poids du tout, `node_wa` / `node_wb` ne sont pas nommés : ils restent `Unbound`, arrivent en
    `NoneTensor`, et le terme disparaît du kernel à la COMPILATION.

    = La marche

    Une descente en profondeur, l'enfant le plus proche d'abord (voir `AaBsp.cxx`). Le premier
    nœud atteint est donc la feuille du germe lui-même : la cellule se réduit tout de suite sur
    ses voisins immédiats, et tout ce qui suit est élagué contre une cellule déjà petite. C'est
    ce qui rend la pile SUFFISANTE là où il faudrait sinon une file de priorité : à chaque niveau
    on dépile un nœud et on en empile deux, donc la pile ne dépasse jamais la PROFONDEUR de
    l'arbre -- une capacité connue à la construction (`max_depth`), et pas une capacité à
    deviner puis à doubler.

    = Où il est construit

    En KERNEL, un appel par NIVEAU de l'arbre (`_build_in_kernel` + `bsp_build_level.h`), et un
    work-item par nœud du niveau. Ce qui le permet est que la FORME de l'arbre ne dépend pas des
    données : la coupe est MÉDIANE, donc la profondeur vaut `ceil( log2( n / leaf_size ) ) + 1` et
    les nœuds sont ceux d'un arbre binaire PARFAIT de cette profondeur, l'un et l'autre fonction de
    `n` seul (voir `max_depth_for` / `max_nb_nodes_for`, et le test
    `the_tree_shape_does_not_depend_on_the_data`, qui le vérifie jusqu'à des nuages entièrement
    dégénérés). Il n'y a donc AUCUNE capacité à deviner -- ni pour la pile de la descente, ni pour
    les tableaux de nœuds -- et le NOMBRE D'APPELS lui-même est connu avant de regarder un point.

    La boucle sur les niveaux reste côté hôte, ainsi que l'arithmétique d'indices entre deux
    niveaux : des tableaux de la taille d'un niveau, jamais du nuage. C'est ce qui empêche encore
    cette construction de passer sous un `jit` -- mais un `AaBsp` est une CONSTANTE du trace (voir
    plus bas), donc ce n'est pas ce qu'on lui demande.

    Le NUAGE, lui, ne redescend jamais : `positions` / `weights` sont passés au kernel tels qu'ils
    arrivent, sans `np.asarray`. Des germes qui vivent sur le GPU y restent -- ce qui compte pour
    celui qui reconstruit l'arbre à chaque pas de Newton, où un aller-retour hôte coûterait deux
    fois le nuage par pas et ne servirait à rien. Ce qui redescend est de la taille d'un NIVEAU
    (les boîtes, les `mid`), plus la permutation finale : mesuré, 0.11 s sur les 4.5 s d'un arbre
    à 1e6 germes, le reste étant les kernels eux-mêmes -- et pour l'essentiel les tout premiers
    niveaux, où deux ou quatre work-items balaient tout le nuage.

    = La DÉRIVATION, et pourquoi il n'y en a pas

    L'arbre est un objet COMBINATOIRE, et le gradient juste à travers lui est exactement ZÉRO --
    l'élagage ne change pas l'ensemble des coupes survivantes, seulement lesquelles on essaie. Ses
    sorties flottantes (boîtes, majorants) ne sont donc pas dérivables, et la construction hôte
    l'obtient gratuitement en les rendant constantes du trace.
    """

    # les germes, RÉORDONNÉS : les indices des germes groupés par feuille, chaque feuille
    # occupant la tranche `[ node_begin, node_end )`. C'est ce regroupement qui fait que lire
    # une feuille est une lecture contiguë et pas une collecte d'indices épars.
    seed_indices : IntTensor[ "num_bsp_seed" ]


    # l'arbre, numeroté EN TAS : le nœud `k` a ses enfants en `2k+1` / `2k+2`, la racine est 0, et
    # le niveau `L` occupe `[ 2^L - 1, 2^(L+1) - 1 )`. `node_left < 0` DIT feuille, et n'arrive
    # qu'au DERNIER niveau -- un nœud qui n'a plus rien à couper passe sa tranche entière à son fils
    # gauche et rien au droit (voir `_build_in_kernel`), de sorte qu'un fils VIDE (`begin == end`)
    # est la
    # seule autre chose à distinguer, ce que `for_each_candidate` fait en deux lectures d'entier.
    node_left    : IntTensor[ "num_bsp_node" ]
    node_right   : IntTensor[ "num_bsp_node" ]
    node_begin   : IntTensor[ "num_bsp_node" ]
    node_end     : IntTensor[ "num_bsp_node" ]

    # la boîte englobante du sous-arbre -- `lo` PUIS `hi`, DANS LE MÊME TABLEAU, et c'est le point :
    # la marche est du pointer-chasing, donc ce qui coûte n'est pas le nombre d'octets lus mais le
    # nombre de LIGNES DE CACHE touchées. Deux tableaux séparés, ce sont deux lignes à deux endroits
    # de la mémoire pour une seule boîte ; entrelacés, la boîte d'un nœud tient dans une lecture
    # contiguë (32 octets en 2D FP64).
    #
    # Ça compte parce que l'arbre ne tient dans aucun cache : ~16 Mo à 1e6 germes, contre 1 Mo de L2
    # par cœur et 11 Mo de L3 pour tous. Mesuré : les défauts L2 PAR CELLULE passent de 15 à un
    # thread à 163 à huit, à localité par cœur pourtant identique -- le L3 de Skylake-SP est un
    # cache de VICTIMES non inclusif, donc à huit cœurs chacun n'a plus qu'un huitième du
    # rattrapage. Diviser le nombre de lignes touchées est la seule prise là-dessus.
    node_box     : RealTensor[ "num_bsp_node", "num_lohi", "dim" ]
    node_wa      : RealTensor[ "num_bsp_node", "dim" ]
    node_wb      : RealTensor[ "num_bsp_node" ]

    num_bsp_seed : Axis[ "nb_bsp_seeds" ]
    num_bsp_node : Axis[ "nb_bsp_nodes" ]
    num_lohi     : Axis[ "nb_lohi" ]
    dim          : Axis[ "nb_dims" ]

    nb_bsp_seeds : ShapeVar
    nb_bsp_nodes : ShapeVar
    nb_lohi      : CtShapeVar
    nb_dims      : CtShapeVar


    def __init__( self, positions, weights = None, max_seeds_per_leaf = 10 ):
        """`positions` : `[ n, d ]`. `weights` : `[ n ]`, ou rien (le cas euclidien).

        `max_seeds_per_leaf` est le grain de l'arbre -- voir la docstring de la classe.

        Le nuage n'est PAS converti en numpy : il part au kernel tel qu'il arrive (`Tensor.set`
        lit sa forme sans toucher ses données), donc des germes qui vivent sur le GPU y restent.
        Seule une FORME est lue ici, et une forme n'est pas une donnée.
        """
        pos = positions if hasattr( positions, "shape" ) else np.asarray( positions, dtype = float )
        # sous un `jit`, `positions` est un tracer : sa forme se lit, mais la boucle par niveau
        # relit les `mid` côté hôte et n'a rien à lire sur un tracer. Le dire ICI plutôt que de
        # laisser remonter l'erreur du backend quinze lignes plus loin : ce n'est pas un accident,
        # c'est la limite assumée de la construction côté hôte (voir la docstring de la classe).
        if driver.is_traced( pos ):
            raise TypeError( "`AaBsp` is built on the HOST, from concrete positions: it cannot be "
                             "built from a traced array (inside a `jit`). Build it outside, and "
                             "pass it in -- the tree is a constant of the trace, which is also what "
                             "makes it invisible to the gradients." )
        if len( pos.shape ) != 2:
            raise ValueError( f"`positions` has to be [ n, d ] ( got { tuple( pos.shape ) } )" )
        w = None if weights is None else ( weights if hasattr( weights, "shape" ) else np.asarray( weights, dtype = float ) )
        if w is not None and int( np.prod( w.shape ) ) != int( pos.shape[ 0 ] ):
            raise ValueError( "`weights` has to hold one weight per position" )
        if int( pos.shape[ 0 ] ) == 0:
            raise ValueError( "an accelerator over no seed at all has nothing to accelerate" )

        tree = _build_in_kernel( pos, w, int( max_seeds_per_leaf ) )

        # la profondeur, qui est EXACTEMENT `max_depth_for( n, leaf )` : l'arbre a désormais la
        # forme fixe que ce majorant décrivait (voir `_build_in_kernel`). C'est elle qui dimensionne
        # la pile
        # de la descente, et une pile trop courte serait une marche qui saute des germes.
        self.max_depth = tree[ "max_depth" ]
        self.max_seeds_per_leaf = int( max_seeds_per_leaf )
        self.nb_leaves = tree[ "nb_leaves" ]

        kwargs = dict(
            seed_indices = tree[ "seed_indices" ],
            node_left    = tree[ "node_left"    ],
            node_right   = tree[ "node_right"   ],
            node_begin   = tree[ "node_begin"   ],
            node_end     = tree[ "node_end"     ],
            node_box     = tree[ "node_box"     ],
        )
        # pas de poids -> on ne NOMME pas les deux tenseurs du majorant : les laisser `Unbound`
        # (jamais alloués, `NoneTensor` côté C++) supprime le terme du kernel, là où des zéros
        # seraient un tableau à lire. Même règle que `PowerDiagram.weights`, et pour la même
        # raison : « pas de poids » est un ÉTAT, pas une valeur.
        if w is not None:
            kwargs[ "node_wa" ] = tree[ "node_wa" ]
            kwargs[ "node_wb" ] = tree[ "node_wb" ]

        self.__base_init__( nb_dims = int( pos.shape[ 1 ] ), nb_lohi = 2, **kwargs )


    @staticmethod
    def max_depth_for( nb_seeds, max_seeds_per_leaf = 10 ):
        """Un MAJORANT de la profondeur, SANS voir les points -- et il est ATTEINT dès que les
        germes sont distincts : la coupe est médiane, donc l'arbre est équilibré et sa forme ne
        dépend que de `n`. Un nuage dégénéré (des germes confondus) ferme des feuilles plus tôt,
        donc il ne fait que rétrécir l'arbre, jamais l'inverse. C'est ce qui dit qu'une
        construction en kernel n'a aucune capacité à deviner.
        """
        n, leaf = int( nb_seeds ), max( int( max_seeds_per_leaf ), 1 )
        return 1 if n <= leaf else math.ceil( math.log2( n / leaf ) ) + 1

    @staticmethod
    def max_nb_nodes_for( nb_seeds, max_seeds_per_leaf = 10 ):
        """Un majorant du nombre de nœuds, `n` seul -- voir `max_depth_for`. Un arbre binaire dont
        toutes les feuilles sont au même niveau en a `2 * feuilles - 1`, et les feuilles sont au
        plus `2 ** ( profondeur - 1 )`."""
        return 2 * 2 ** ( AaBsp.max_depth_for( nb_seeds, max_seeds_per_leaf ) - 1 ) - 1

    @classmethod
    def of( cls, power_diagram, max_seeds_per_leaf = 10 ):
        """L'accélérateur des germes de `power_diagram` -- ses positions ET ses poids.

        Le raccourci qu'on veut presque toujours : un BSP construit sur d'autres poids que ceux
        du diagramme resterait CORRECT (le majorant ne servirait qu'à élaguer moins bien) mais
        n'aurait aucune raison d'être bon.
        """
        # les TAMPONS du diagramme, pas leur copie hôte : `Tensor.raw` est le tableau du backend, et
        # `__init__` le passe au kernel sans y toucher. Un diagramme dont les germes sont sur le GPU
        # y bâtit donc son arbre sans que le nuage ne redescende -- et il redescendait deux fois,
        # une par `np.asarray` et une par le ré-upload.
        pos = power_diagram.positions.raw
        w = power_diagram.weights.raw if power_diagram.weights.is_defined else None
        return cls( pos, w, max_seeds_per_leaf = max_seeds_per_leaf )


    # -- ce que l'appelant a besoin de savoir ---------------------------------------------------

    def nb_seeds( self ):
        return int( self.nb_bsp_seeds.value )

    def thread_scratch( self, num_thread ):
        """La pile de la descente : un entier par niveau de l'arbre, par work-item.

        `max_depth + 2` et pas `max_depth` : on dépile un nœud pour en empiler deux, donc la pile
        gagne un cran par niveau descendu, et il faut la place de la racine plus celle du dernier
        frère empilé. C'est une borne EXACTE -- rien à doubler ici. Et elle ne demande même pas
        d'avoir vu les points : `max_depth` vaut toujours `max_depth_for( n, leaf_size )`.
        """
        num_slot = Axis( ShapeVar( self.max_depth + 2 ), name = "num_bsp_stack" )
        return IntTensor[ num_thread, num_slot ]()

    def bytes_per_thread( self ):
        return 8 * ( self.max_depth + 2 )


def _weight_majorant( pos, w ):
    """`( a, b )` tels que `w_i <= a . pos_i + b` pour tout germe du nœud, le plus serré qu'on
    sache faire vite.

    Deux candidats : le CONSTANT (`a = 0`, `b = max w`), et l'AFFINE ajusté aux moindres carrés
    puis relevé jusqu'à majorer. On garde l'affine seulement s'il resserre franchement
    l'ÉTALEMENT des résidus -- c'est-à-dire `max( w - a.y ) - min( w - a.y )`, qui est exactement
    le mou de la borne : un majorant vaut ce que vaut l'écart entre lui et le poids réel du germe
    qui l'atteint. Les deux ne sont pas comparables dans l'absolu (l'affine crédite moins le côté
    « poids faible » de la boîte, mais plus le côté opposé), et l'étalement est la façon honnête
    de trancher sans dépendre d'où on regarde la boîte.

    Comparé à QUOI, en revanche, demande une précaution : un ajustement à `d + 1` paramètres sur
    `m` points resserre l'étalement même quand il n'y a rien à ajuster, d'autant plus fort que
    `m` est petit -- et une feuille est petite par construction. Le seuil est donc le
    resserrement que le HASARD donne déjà, `sqrt( 1 - d / ( m - 1 ) )` (mesuré : 0.93 pour
    `d = 2, m = 13`, 0.76 pour `d = 3, m = 8`), et l'affine doit faire nettement mieux que lui.
    Sans cette correction, un nœud de poids purement aléatoires retenait l'affine une fois sur
    trois -- toujours VALIDE (le relevé s'en charge), mais un vecteur de plus à lire par nœud
    pour une borne qui ne vaut pas mieux.
    """
    d = pos.shape[ 1 ]
    if w is None:
        return np.zeros( d ), 0.0

    m = len( w )
    spread = float( w.max() - w.min() )
    a = np.zeros( d )
    if m >= 2 * ( d + 1 ) and spread > 0:
        # centré : les moindres carrés sur `pos` brut seraient mal conditionnés dès que le nœud
        # est loin de l'origine. La constante ne change pas l'étalement, elle est reprise par `b`.
        q = pos - pos.mean( axis = 0 )
        fit = np.linalg.lstsq( q, w - w.mean(), rcond = None )[ 0 ]
        r = w - pos @ fit
        by_chance = np.sqrt( max( 1.0 - d / ( m - 1 ), 0.0 ) )
        if float( r.max() - r.min() ) < 0.85 * by_chance * spread:
            a = fit

    b = float( ( w - pos @ a ).max() )

    # une MARGE d'arrondi sur la constante, et sur elle seule. La boîte, elle, n'en a pas besoin :
    # `float32( min( y ) ) == min( float32( y ) )` (un arrondi est monotone), donc `node_box`
    # reste exact une fois converti. `b`, au contraire, est le seul terme que l'hôte
    # et le kernel calculent DIFFÉREMMENT -- ici `w - a . y` en double, là-bas en `TF` -- et un `b`
    # arrondi vers le bas cesserait de majorer. Grossir `b` ne peut qu'élaguer moins, jamais mentir.
    scale = abs( b ) + float( w.max() - w.min() ) + float( np.abs( pos @ a ).max() )
    return a, b + 1e-6 * scale


# -- la construction, NIVEAU PAR NIVEAU ---------------------------------------------------------


class _BspCloud( Aggregate ):
    """Le nuage EN COURS DE TRI : les germes rangés dans l'ordre où l'arbre les regroupe, plus
    l'indice d'origine de chacun.

    Les positions (et les poids) sont tenues PERMUTÉES à côté des indices, et pas relues à travers
    eux : un nœud lit alors ses points d'un seul tenant, là où une indirection par `order` en ferait
    une collecte éparse. Ça compte partout, et surtout aux premiers niveaux, où très peu de
    work-items balaient tout le nuage.

    Il en faut DEUX par niveau, l'un lu et l'autre écrit : les entrées et les sorties d'un appel sont
    disjointes (voir `driver.call`), et le tri d'un niveau est une permutation, donc chaque case de
    la sortie est écrite par le work-item du nœud qui la contient -- une et une seule fois, sans
    atomique ni barrière, parce que les tranches d'un niveau PARTITIONNENT `[ 0, n )`.
    """

    positions : RealTensor[ "num_point", "dim" ]
    weights   : RealTensor[ "num_point" ]
    order     : IntTensor[ "num_point" ]

    num_point : Axis[ "nb_points" ]
    dim       : Axis[ "nb_dims" ]

    nb_points : ShapeVar
    nb_dims   : CtShapeVar


class _BspLevel( Aggregate ):
    """Ce qu'UN niveau de l'arbre porte, PAR NŒUD -- batché sur les nœuds du niveau, donc un
    work-item par nœud.

    `begin` / `end` sont l'ENTRÉE (la tranche du nœud, décidée par le niveau d'au-dessus) ; tout le
    reste est la sortie. `mid` dit où couper : le fils gauche reçoit `[ begin, mid )`, le droit
    `[ mid, end )`, et `mid == end` est un nœud qui n'avait plus rien à couper et propage tout à
    gauche (voir `bsp_build_level.h`).
    """

    begin : IntTensor
    end   : IntTensor
    mid   : IntTensor

    box   : RealTensor[ "num_lohi", "dim" ]
    wa    : RealTensor[ "dim" ]
    wb    : RealTensor

    num_lohi : Axis[ "nb_lohi" ]
    dim      : Axis[ "nb_dims" ]
    nb_lohi  : CtShapeVar
    nb_dims  : CtShapeVar


def _preorder_of_heap( depth ):
    """`p[ i ]` = ou le noeud de rang-tas `i` atterrit en PREORDRE (DFS).

    = Pourquoi changer la numerotation

    En TAS, les fils du noeud `i` sont en `2i+1` / `2i+2` : un chemin racine -> feuille saute vers
    des adresses qui DIVERGENT exponentiellement, et le pire est en bas de l'arbre, la ou les
    niveaux sont les plus gros -- a 1e6 germes, le niveau 17 fait 65 536 noeuds, donc les fils d'un
    noeud profond sont a deux megaoctets de lui. Or c'est un chemin racine -> feuille que la marche
    parcourt POUR CHAQUE CELLULE.

    En PREORDRE, le fils gauche est en `i+1` et le droit en `i + 2^(h-1)`, ou `h` est la hauteur du
    sous-arbre : les sauts RETRECISSENT en descendant, et les derniers niveaux -- les plus nombreux
    et les plus visites -- tiennent a quelques noeuds les uns des autres. La propriete genante est
    exactement inversee.

    = Ce que ca ne change pas

    Ni la forme de l'arbre, ni les tranches, ni la marche : c'est une PERMUTATION des memes noeuds.
    Ce qui change est l'adresse a laquelle chacun est ecrit -- et c'est mesure comme etant ce qui
    compte : a instructions egales (17 500 par cellule contre 26 950 pour pysdot, donc MOINS), on
    generait 199 defauts de cache par cellule a huit coeurs la ou pysdot -- un quadtree en ordre Z,
    donc a sous-arbres contigus -- en genere 5.9.
    """
    nb_nodes = 2 ** depth - 1
    pre = np.zeros( nb_nodes, dtype = np.int64 )
    for level in range( depth - 1 ):
        h = depth - level                       # hauteur du sous-arbre d'un noeud de ce niveau
        idx = np.arange( 2 ** level - 1, 2 ** ( level + 1 ) - 1 )
        pre[ 2 * idx + 1 ] = pre[ idx ] + 1
        pre[ 2 * idx + 2 ] = pre[ idx ] + 2 ** ( h - 1 )
    return pre


def _build_in_kernel( pos, w, leaf_size ):
    """Le même arbre que `_build`, construit par `bsp_build_level.h` au lieu de numpy.

    = Pourquoi un appel PAR NIVEAU

    Un niveau lit les tranches que le précédent a produites, et il n'y a pas de barrière GLOBALE
    dans un kernel SYCL -- seulement au sein d'un work-group. La barrière est donc la fin du
    lancement, et l'hôte enchaîne `depth` appels. Ce n'est pas un pis-aller : `depth` vaut
    `max_depth_for( n, leaf_size )`, une fonction de `n` SEUL (coupe médiane), donc le nombre
    d'appels est connu d'avance et ne dépend d'aucune donnée -- une quinzaine à 1e6 germes, contre
    les ~130 000 tours de boucle Python que la version hôte fait par nœud.

    = Ce qui reste côté hôte, et ce que ça coûte

    L'arithmétique d'indices entre deux niveaux (`[ begin, mid )` / `[ mid, end )`) et le
    recollement des niveaux en un seul tableau de nœuds. Des tableaux de la taille d'un NIVEAU,
    jamais du nuage. C'est aussi ce qui empêche encore cette construction de passer sous un `jit`
    -- mais un `AaBsp` est de toute façon une CONSTANTE du trace (voir la docstring de la classe),
    donc ce n'est pas ce qu'on lui demande.

    = Le nom de l'axe de batch

    Un axe frais par niveau donnerait `depth` sources C++ différentes, donc `depth` compilations
    (voir `loom.tensor.batch`). Les tenseurs d'un niveau sont donc RECOPIÉS en numpy et le niveau
    relâché avant le suivant : le nom revient à la réserve, les `depth` appels partagent une seule
    source, et seul le premier compile.
    """
    n, d = pos.shape

    depth = AaBsp.max_depth_for( n, leaf_size )
    num_param = Axis( ShapeVar( 1 ), name = "num_bsp_param" )

    src = _BspCloud( nb_dims = d, positions = pos, order = np.arange( n, dtype = np.int64 ),
                     **( {} if w is None else { "weights": w } ) )

    beg = np.zeros( 1, dtype = np.int64 )
    end = np.full( 1, n, dtype = np.int64 )

    begs, ends, boxes, was, wbs = [], [], [], [], []

    for level in range( depth ):
        # le nuage de sortie PARTAGE l'axe des points (donc son compte) avec l'entrée : c'est la
        # même permutation, réarrangée.
        dst = _BspCloud( nb_dims = d, num_point = src.num_point )

        num_node = new_batch_axis( 2 ** level, prefix = "bspnode" )
        lvl = _BspLevel( nb_dims = d, nb_lohi = 2, batch_axes = [ num_node ], begin = beg, end = end )

        perm = IntTensor[ src.num_point ]()
        leaf = IntTensor[ num_param ]()
        leaf.set( np.array( [ leaf_size ], dtype = np.int64 ) )

        # sans poids, ni le nuage ni le majorant n'ont de tenseur : laissés HORS des sorties, ils
        # restent `Unbound`, arrivent en `NoneTensor`, et les deux blocs correspondants du kernel
        # disparaissent à la compilation (même règle que `PowerDiagram.weights`).
        no_weights = [] if w is not None else [ "dst.weights", "lvl.wa", "lvl.wb" ]

        driver.call(
            FfiCodeParallel( name = "bsp_build_level",
                includes = [ "sdot/bsp_build_level.h" ],
                fwd_code = "bsp_build_level( src, dst, perm, "
                           "lvl.begin( batch_index ), lvl.end( batch_index ), "
                           "lvl.box( batch_index ), "
                           "lvl.wa( batch_index ), lvl.wb( batch_index ), lvl.mid( batch_index ), "
                           "SI( leaf_size( 0 ) ) );" ),
            output_attributes = [ "dst", "lvl", "perm" ],
            # `begin` / `end` sont l'ENTRÉE du niveau : sous une sortie nommée, il faut les en
            # retirer explicitement pour qu'elles restent lues et non allouées.
            output_exceptions = [ "lvl.begin", "lvl.end" ] + no_weights,
            scratch_attributes = [ "perm" ],
            # toutes les tailles sont prescrites en amont (elles ne dépendent que de `n` et du
            # niveau) : aucun compte n'est décidé par le kernel, donc rien ne peut déborder et le
            # test d'exécution -- une synchro device -> hôte par appel -- n'a rien à surveiller.
            has_dynamic_capacity = False,
            src = src, dst = dst, lvl = lvl, perm = perm, leaf_size = leaf,
        )

        mid = np.asarray( lvl.mid ).reshape( -1 )
        begs.append( beg )
        ends.append( end )
        boxes.append( np.asarray( lvl.box ).reshape( -1, 2, d ).copy() )
        if w is not None:
            was.append( np.asarray( lvl.wa ).reshape( -1, d ).copy() )
            wbs.append( np.asarray( lvl.wb ).reshape( -1 ).copy() )

        if level + 1 < depth:
            nb = np.empty( 2 * len( beg ), dtype = np.int64 )
            ne = np.empty( 2 * len( beg ), dtype = np.int64 )
            nb[ 0::2 ], nb[ 1::2 ] = beg, mid
            ne[ 0::2 ], ne[ 1::2 ] = mid, end
            beg, end = nb, ne

        order = np.asarray( dst.order ).reshape( -1 ).copy()
        src = dst

        # RENDRE le nom de l'axe avant d'en emprunter un autre : le niveau suivant le prend au
        # DÉBUT de son tour, donc tant que celui-ci est vivant il en faut un neuf -- et un nom neuf,
        # c'est une source C++ de plus, donc une compilation de plus (voir `loom.tensor.batch`).
        del lvl, num_node, perm, dst, leaf

    nb_nodes = 2 ** depth - 1

    # les niveaux se concatènent dans l'ordre, ce qui donne la numérotation EN TAS : le nœud `k` du
    # niveau `L` est le global `2^L - 1 + k`. C'est la forme dans laquelle ils SORTENT du kernel.
    node_begin = np.concatenate( begs )
    node_end   = np.concatenate( ends )
    node_box   = np.concatenate( boxes )
    node_wa    = np.concatenate( was ) if w is not None else np.zeros( ( nb_nodes, d ) )
    node_wb    = np.concatenate( wbs ) if w is not None else np.zeros( nb_nodes )

    is_leaf = np.zeros( nb_nodes, dtype = bool )
    is_leaf[ 2 ** ( depth - 1 ) - 1: ] = True              # le dernier niveau : QUE des feuilles

    # ... puis on les RANGE en préordre, ce qui est une pure permutation : même arbre, mêmes
    # tranches, même marche, seules les ADRESSES changent (voir `_preorder_of_heap`).
    pre = _preorder_of_heap( depth )
    def in_preorder( a ):
        r = np.empty_like( a )
        r[ pre ] = a
        return r

    heap = np.arange( nb_nodes, dtype = np.int64 )
    node_left = np.where( is_leaf, -1, pre[ np.where( is_leaf, 0, 2 * heap + 1 ) ] )
    node_right = np.where( is_leaf, -1, pre[ np.where( is_leaf, 0, 2 * heap + 2 ) ] )

    node_end_pre = in_preorder( node_end )
    node_begin_pre = in_preorder( node_begin )

    return dict(
        seed_indices = order,
        node_left    = in_preorder( node_left ),
        node_right   = in_preorder( node_right ),
        node_begin   = node_begin_pre,
        node_end     = node_end_pre,
        node_box     = in_preorder( node_box ),
        node_wa      = in_preorder( node_wa ),
        node_wb      = in_preorder( node_wb ),
        max_depth    = depth,
        nb_leaves    = int( ( in_preorder( is_leaf ) & ( node_end_pre > node_begin_pre ) ).sum() ),
    )

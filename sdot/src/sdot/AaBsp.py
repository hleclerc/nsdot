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
    n'ait plus que `max_seeds_per_leaf` germes -- une trentaine est l'équilibre mesuré : plus
    petit, l'arbre coûte plus en descentes qu'il ne fait gagner en coupes évitées ; plus grand,
    on paie des bissectrices dont on savait déjà qu'elles ne serviraient à rien.

    = Ce que chaque nœud porte, et pourquoi

    La BOÎTE (`node_lo` / `node_hi`) contient tous les germes du sous-arbre, et un MAJORANT
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
    plus bas), donc ce n'est pas ce qu'on lui demande. `_build` garde la même construction en
    numpy, et un test vérifie que les deux rendent le même arbre.

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
    # gauche et rien au droit (voir `_build`), de sorte qu'un fils VIDE (`begin == end`) est la
    # seule autre chose à distinguer, ce que `for_each_candidate` fait en deux lectures d'entier.
    node_left    : IntTensor[ "num_bsp_node" ]
    node_right   : IntTensor[ "num_bsp_node" ]
    node_begin   : IntTensor[ "num_bsp_node" ]
    node_end     : IntTensor[ "num_bsp_node" ]

    # la boîte englobante du sous-arbre, et le majorant affine de ses poids (voir la docstring).
    node_lo      : RealTensor[ "num_bsp_node", "dim" ]
    node_hi      : RealTensor[ "num_bsp_node", "dim" ]
    node_wa      : RealTensor[ "num_bsp_node", "dim" ]
    node_wb      : RealTensor[ "num_bsp_node" ]

    num_bsp_seed : Axis[ "nb_bsp_seeds" ]
    num_bsp_node : Axis[ "nb_bsp_nodes" ]
    dim          : Axis[ "nb_dims" ]

    nb_bsp_seeds : ShapeVar
    nb_bsp_nodes : ShapeVar
    nb_dims      : CtShapeVar


    def __init__( self, positions, weights = None, max_seeds_per_leaf = 30, in_kernel = True ):
        """`positions` : `[ n, d ]`. `weights` : `[ n ]`, ou rien (le cas euclidien).

        `max_seeds_per_leaf` est le grain de l'arbre -- voir la docstring de la classe.

        `in_kernel` dit QUI construit l'arbre : le kernel (`_build_in_kernel`, un appel par niveau)
        ou numpy (`_build`, une boucle Python par nœud). Les deux rendent le MÊME arbre -- même
        forme, mêmes tranches, mêmes boîtes -- et c'est ce qu'un test vérifie ; `False` n'est là que
        pour lui, et pour pouvoir bâtir un arbre sans rien compiler.
        """
        try:
            pos = np.asarray( positions, dtype = float )
        except Exception as e:
            # sous un `jit`, `positions` est un tracer et il n'y a rien à mesurer dessus. Le dire
            # ici plutôt que de laisser remonter l'erreur du backend : ce n'est pas un accident,
            # c'est la limite assumée de la construction côté hôte (voir la docstring de la classe).
            raise TypeError( "`AaBsp` is built on the HOST, from concrete positions: it cannot be "
                             "built from a traced array (inside a `jit`). Build it outside, and "
                             "pass it in -- the tree is a constant of the trace, which is also what "
                             "makes it invisible to the gradients." ) from e
        if pos.ndim != 2:
            raise ValueError( f"`positions` has to be [ n, d ] ( got { pos.shape } )" )
        w = None if weights is None else np.asarray( weights, dtype = float ).reshape( -1 )
        if w is not None and w.size != len( pos ):
            raise ValueError( "`weights` has to hold one weight per position" )
        if len( pos ) == 0:
            raise ValueError( "an accelerator over no seed at all has nothing to accelerate" )

        build = _build_in_kernel if in_kernel else _build
        tree = build( pos, w, int( max_seeds_per_leaf ) )

        # la profondeur, qui est EXACTEMENT `max_depth_for( n, leaf )` : l'arbre a désormais la
        # forme fixe que ce majorant décrivait (voir `_build`). C'est elle qui dimensionne la pile
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
            node_lo      = tree[ "node_lo"      ],
            node_hi      = tree[ "node_hi"      ],
        )
        # pas de poids -> on ne NOMME pas les deux tenseurs du majorant : les laisser `Unbound`
        # (jamais alloués, `NoneTensor` côté C++) supprime le terme du kernel, là où des zéros
        # seraient un tableau à lire. Même règle que `PowerDiagram.weights`, et pour la même
        # raison : « pas de poids » est un ÉTAT, pas une valeur.
        if w is not None:
            kwargs[ "node_wa" ] = tree[ "node_wa" ]
            kwargs[ "node_wb" ] = tree[ "node_wb" ]

        self.__base_init__( nb_dims = pos.shape[ 1 ], **kwargs )


    @staticmethod
    def max_depth_for( nb_seeds, max_seeds_per_leaf = 30 ):
        """Un MAJORANT de la profondeur, SANS voir les points -- et il est ATTEINT dès que les
        germes sont distincts : la coupe est médiane, donc l'arbre est équilibré et sa forme ne
        dépend que de `n`. Un nuage dégénéré (des germes confondus) ferme des feuilles plus tôt,
        donc il ne fait que rétrécir l'arbre, jamais l'inverse. C'est ce qui dit qu'une
        construction en kernel n'a aucune capacité à deviner.
        """
        n, leaf = int( nb_seeds ), max( int( max_seeds_per_leaf ), 1 )
        return 1 if n <= leaf else math.ceil( math.log2( n / leaf ) ) + 1

    @staticmethod
    def max_nb_nodes_for( nb_seeds, max_seeds_per_leaf = 30 ):
        """Un majorant du nombre de nœuds, `n` seul -- voir `max_depth_for`. Un arbre binaire dont
        toutes les feuilles sont au même niveau en a `2 * feuilles - 1`, et les feuilles sont au
        plus `2 ** ( profondeur - 1 )`."""
        return 2 * 2 ** ( AaBsp.max_depth_for( nb_seeds, max_seeds_per_leaf ) - 1 ) - 1

    @classmethod
    def of( cls, power_diagram, max_seeds_per_leaf = 30 ):
        """L'accélérateur des germes de `power_diagram` -- ses positions ET ses poids.

        Le raccourci qu'on veut presque toujours : un BSP construit sur d'autres poids que ceux
        du diagramme resterait CORRECT (le majorant ne servirait qu'à élaguer moins bien) mais
        n'aurait aucune raison d'être bon.
        """
        d = int( power_diagram.nb_dims.value )
        pos = np.asarray( power_diagram.positions ).reshape( -1, d )
        w = np.asarray( power_diagram.weights ).reshape( -1 ) if power_diagram.weights.is_defined else None
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
    # `float32( min( y ) ) == min( float32( y ) )` (un arrondi est monotone), donc `node_lo` /
    # `node_hi` restent exacts une fois convertis. `b`, au contraire, est le seul terme que l'hôte
    # et le kernel calculent DIFFÉREMMENT -- ici `w - a . y` en double, là-bas en `TF` -- et un `b`
    # arrondi vers le bas cesserait de majorer. Grossir `b` ne peut qu'élaguer moins, jamais mentir.
    scale = abs( b ) + float( w.max() - w.min() ) + float( np.abs( pos @ a ).max() )
    return a, b + 1e-6 * scale


def _build( pos, w, leaf_size ):
    """L'arbre, a plat, dans un tableau de la taille d'un arbre binaire PARFAIT de profondeur
    `max_depth_for( n, leaf_size )` : le noeud `k` a ses enfants en `2k+1` / `2k+2`, la racine est 0,
    et le niveau `L` occupe `[ 2^L - 1, 2^(L+1) - 1 )`.

    = Pourquoi une forme FIXE plutot que les noeuds reellement produits

    Parce que la forme n'a jamais depend des donnees (coupe MEDIANE, voir `max_depth_for`), et que
    la seule chose qui l'empechait de s'ecrire ainsi etait la facon dont l'HOTE la construisait, en
    ajoutant les noeuds au fur et a mesure. Numerotee en tas, la place de chaque noeud est connue
    AVANT de le calculer : un niveau entier se remplit sans se demander ou, ce qui est exactement ce
    qu'il faut pour qu'un kernel ecrive le niveau `L` en parallele sur ses noeuds (voir
    `bsp_build_level.h`). Rien ici ne se reserve, rien ne se compte.

    = Un noeud « fini » ne s'arrete pas, il se PROPAGE

    Un noeud dont la tranche tient deja dans une feuille (ou qu'aucune coupe ne separerait, tous ses
    germes etant confondus) ne peut pas simplement cesser : sa tranche doit continuer d'occuper un
    emplacement aux niveaux suivants, sinon ses germes ne seraient plus couverts par aucun noeud et
    le niveau suivant ne saurait plus les ecrire. Il passe donc TOUT a son fils gauche et rien a son
    fils droit, jusqu'au dernier niveau -- ou tout le monde est feuille (`node_left < 0`).

    Ca ne change ni les feuilles ni leur taille : un noeud de 31 germes avec `leaf_size = 30` se
    coupait deja en 15 et 16. Ca ajoute seulement, pour une feuille qui se ferme tot, une chaine de
    noeuds a un seul fils que la marche traverse -- un `pop` de plus par niveau saute, et un fils
    droit VIDE que `for_each_candidate` reconnait a `begin == end` et saute sans rien tester.

    = Ce qui reste du profil d'avant

    Un noeud n'est pas une liste d'indices mais une TRANCHE `[ begin, end )` d'une permutation
    rearrangee en place, les positions (et les poids) tenues permutees en parallele. Un noeud lit
    donc ses points par une VUE, et `seed_indices` EST la permutation a la fin -- rien a recoller
    feuille par feuille, et un kernel lit une feuille d'un seul tenant.
    """
    n, d = pos.shape

    depth = AaBsp.max_depth_for( n, leaf_size )
    nb_nodes = 2 ** depth - 1

    order = np.arange( n, dtype = np.int64 )
    # une COPIE, toujours : on permute ce tableau en place, et `pos` appartient a l'appelant (et
    # peut tres bien etre en lecture seule). `ascontiguousarray` ne copierait pas si l'entree est
    # deja contigue -- et permuter les positions du diagramme sous ses pieds ne se voit pas tout
    # de suite, ca se voit aux mesures.
    P = np.array( pos, dtype = float )
    W = None if w is None else np.array( w, dtype = float )

    node_left  = np.full( nb_nodes, -1, dtype = np.int64 )
    node_right = np.full( nb_nodes, -1, dtype = np.int64 )
    node_begin = np.zeros( nb_nodes, dtype = np.int64 )
    node_end   = np.zeros( nb_nodes, dtype = np.int64 )
    node_lo    = np.zeros( ( nb_nodes, d ) )
    node_hi    = np.zeros( ( nb_nodes, d ) )
    node_wa    = np.zeros( ( nb_nodes, d ) )
    node_wb    = np.zeros( nb_nodes )

    node_end[ 0 ] = n

    for level in range( depth ):
        first, last = 2 ** level - 1, 2 ** ( level + 1 ) - 1
        is_last = level == depth - 1

        for k in range( first, last ):
            b, e = int( node_begin[ k ] ), int( node_end[ k ] )

            split, ax = False, 0
            if e > b:
                p = P[ b:e ]
                lo = p.min( axis = 0 )
                hi = p.max( axis = 0 )
                node_lo[ k ], node_hi[ k ] = lo, hi
                if W is not None:
                    node_wa[ k ], node_wb[ k ] = _weight_majorant( p, W[ b:e ] )
                ax = int( np.argmax( hi - lo ) )
                # `hi[ ax ] <= lo[ ax ]` : tous les germes au meme endroit. Aucune coupe ne les
                # separerait, et insister ferait une descente sans fin -- le noeud se propage,
                # quitte a finir en une feuille plus grosse que `leaf_size`.
                split = e - b > leaf_size and hi[ ax ] > lo[ ax ]

            if is_last:
                continue

            node_left[ k ], node_right[ k ] = 2 * k + 1, 2 * k + 2
            if split:
                # la MEDIANE, pas le milieu de la boite : c'est ce qui borne la profondeur par
                # `log2( n / leaf_size )` quelle que soit la distribution -- un nuage tres
                # inhomogene ferait degenerer une coupe geometrique en une chaine.
                h = ( e - b ) // 2
                part = np.argpartition( P[ b:e, ax ], h )

                # la permutation, appliquee EN PLACE a la tranche : les deux enfants sont alors ses
                # deux moities, et il n'y a plus rien a transporter en descendant.
                order[ b:e ] = order[ b:e ][ part ]
                P[ b:e ] = P[ b:e ][ part ]
                if W is not None:
                    W[ b:e ] = W[ b:e ][ part ]
                mid = b + h
            else:
                mid = e

            node_begin[ 2 * k + 1 ], node_end[ 2 * k + 1 ] = b, mid
            node_begin[ 2 * k + 2 ], node_end[ 2 * k + 2 ] = mid, e

    return dict(
        seed_indices = order,
        node_left    = node_left,
        node_right   = node_right,
        node_begin   = node_begin,
        node_end     = node_end,
        node_lo      = node_lo,
        node_hi      = node_hi,
        node_wa      = node_wa,
        node_wb      = node_wb,
        max_depth    = depth,
        nb_leaves    = int( ( ( node_left < 0 ) & ( node_end > node_begin ) ).sum() ),
    )


# -- la construction EN KERNEL ---------------------------------------------------------------------


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
    gauche (voir `_build`).
    """

    begin : IntTensor
    end   : IntTensor
    mid   : IntTensor

    lo    : RealTensor[ "dim" ]
    hi    : RealTensor[ "dim" ]
    wa    : RealTensor[ "dim" ]
    wb    : RealTensor

    dim     : Axis[ "nb_dims" ]
    nb_dims : CtShapeVar


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

    begs, ends, los, his, was, wbs = [], [], [], [], [], []

    for level in range( depth ):
        # le nuage de sortie PARTAGE l'axe des points (donc son compte) avec l'entrée : c'est la
        # même permutation, réarrangée.
        dst = _BspCloud( nb_dims = d, num_point = src.num_point )

        num_node = new_batch_axis( 2 ** level, prefix = "bspnode" )
        lvl = _BspLevel( nb_dims = d, batch_axes = [ num_node ], begin = beg, end = end )

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
                           "lvl.lo( batch_index ), lvl.hi( batch_index ), "
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
        los.append( np.asarray( lvl.lo ).reshape( -1, d ).copy() )
        his.append( np.asarray( lvl.hi ).reshape( -1, d ).copy() )
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

    # la numérotation EN TAS : le nœud global `g` a ses enfants en `2g+1` / `2g+2`, et le nœud `k`
    # du niveau `L` est le global `2^L - 1 + k` -- ce qui fait tomber les enfants exactement sur les
    # nœuds `2k` et `2k+1` du niveau suivant. Rien à renuméroter, donc rien à recoller : les
    # niveaux se concatènent dans l'ordre.
    node_left = np.arange( nb_nodes, dtype = np.int64 ) * 2 + 1
    node_left[ 2 ** ( depth - 1 ) - 1: ] = -1               # le dernier niveau : QUE des feuilles
    node_right = np.where( node_left < 0, -1, node_left + 1 )

    node_begin = np.concatenate( begs )
    node_end   = np.concatenate( ends )

    return dict(
        seed_indices = order,
        node_left    = node_left,
        node_right   = node_right,
        node_begin   = node_begin,
        node_end     = node_end,
        node_lo      = np.concatenate( los ),
        node_hi      = np.concatenate( his ),
        node_wa      = np.concatenate( was ) if w is not None else np.zeros( ( nb_nodes, d ) ),
        node_wb      = np.concatenate( wbs ) if w is not None else np.zeros( nb_nodes ),
        max_depth    = depth,
        nb_leaves    = int( ( ( node_left < 0 ) & ( node_end > node_begin ) ).sum() ),
    )

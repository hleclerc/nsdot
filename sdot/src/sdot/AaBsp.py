import math

import numpy as np

from loom.tensor import Axis, CtShapeVar, IntTensor, RealTensor, ShapeVar

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

    = Où il est construit, et où il POURRAIT l'être

    Aujourd'hui côté HÔTE, en numpy, donc il lui faut des positions concrètes : il ne se construit
    pas sous un `jit`. C'est un état de ce code, pas une propriété de la structure -- rien ici ne
    résiste à une construction dans un kernel, et `_build` est la seule chose qu'il faudrait
    remplacer (la marche C++ ne lit que les tableaux ci-dessus).

    En particulier la FORME de l'arbre ne dépend pas des données : la coupe est MÉDIANE, donc la
    profondeur vaut `ceil( log2( n / leaf_size ) ) + 1` et le nombre de nœuds est majoré par
    `2 ** ceil( log2( n / leaf_size ) ) * 2 - 1`, l'un et l'autre fonction de `n` seul (voir
    `max_depth_for` / `max_nb_nodes_for`, et le test `the_tree_shape_does_not_depend_on_the_data`,
    qui le vérifie jusqu'à des nuages entièrement dégénérés). Il n'y a donc AUCUNE capacité à
    deviner : ni pour la pile de la descente, ni pour les tableaux de nœuds.

    Ce qui resterait à traiter est la DÉRIVATION, et c'est une décision plutôt qu'une difficulté :
    l'arbre est un objet COMBINATOIRE, et le gradient juste à travers lui est exactement ZÉRO --
    l'élagage ne change pas l'ensemble des coupes survivantes, seulement lesquelles on essaie. Ses
    sorties flottantes (boîtes, majorants) doivent donc être déclarées non dérivables. La
    construction côté hôte l'obtient gratuitement, en les rendant constantes du trace ; une
    construction en kernel devrait le dire.
    """

    # les germes, RÉORDONNÉS : les indices des germes groupés par feuille, chaque feuille
    # occupant la tranche `[ node_begin, node_end )`. C'est ce regroupement qui fait que lire
    # une feuille est une lecture contiguë et pas une collecte d'indices épars.
    seed_indices : IntTensor[ "num_bsp_seed" ]

    # l'arbre. `node_left < 0` DIT feuille -- un nœud interne a toujours ses deux enfants (les
    # coupes médianes ne produisent pas de fils vide), donc il n'y a rien d'autre à distinguer.
    # Les nœuds sont numérotés en PRÉORDRE, donc la racine est le nœud 0.
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


    def __init__( self, positions, weights = None, max_seeds_per_leaf = 30 ):
        """`positions` : `[ n, d ]`. `weights` : `[ n ]`, ou rien (le cas euclidien).

        `max_seeds_per_leaf` est le grain de l'arbre -- voir la docstring de la classe.
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

        tree = _build( pos, w, int( max_seeds_per_leaf ) )

        # la profondeur MESURÉE, pas une borne : c'est elle qui dimensionne la pile de la
        # descente, et une pile trop courte serait une marche qui saute des germes.
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
    """L'arbre, à plat : un nœud par entrée des tableaux, numérotés en largeur (racine = 0).

    Deux décisions font tout le coût de la construction, et elles viennent d'un profil, pas d'une
    intuition -- la première version, récursive et par nœud, passait 9 us PAR NOEUD, dont 43 % en
    surcharge d'appel numpy sur des tableaux de trente lignes (`min` / `max`), 26 % dans le corps
    Python de la récursion, 13 % dans `argpartition`.

      * Un nœud n'est pas une liste d'indices mais une TRANCHE `[ begin, end )` d'une permutation
        réarrangée en place, les positions (et les poids) tenues permutées en parallèle. Un nœud lit
        donc ses points par une VUE, et `seed_indices` EST la permutation à la fin -- rien à
        recoller feuille par feuille.
      * On avance PAR NIVEAU, pas par nœud. À un niveau donné les tranches sont disjointes et
        ordonnées, donc les boîtes de tout le niveau se lisent en DEUX appels `reduceat` au lieu de
        deux par nœud. C'était le premier poste du profil, et il disparaît.

    Ce qui reste par nœud est la partition médiane elle-même (`argpartition` + l'application de la
    permutation), qui ne se vectorise pas à travers des tranches de tailles différentes.

    La numérotation est en LARGEUR et non plus en préordre. Rien ne dépend de l'ordre -- le kernel
    descend par `node_left` / `node_right`, et la racine est 0 dans les deux cas.
    """
    n, d = pos.shape

    order = np.arange( n, dtype = np.int64 )
    # une COPIE, toujours : on permute ce tableau en place, et `pos` appartient à l'appelant (et
    # peut très bien être en lecture seule). `ascontiguousarray` ne copierait pas si l'entrée est
    # déjà contiguë -- et permuter les positions du diagramme sous ses pieds ne se voit pas tout
    # de suite, ça se voit aux mesures.
    P = np.array( pos, dtype = float )
    W = None if w is None else np.array( w, dtype = float )

    node_left, node_right, node_begin, node_end = [], [], [], []
    node_lo, node_hi, node_wa, node_wb = [], [], [], []

    # le niveau courant : une tranche et l'identifiant déjà réservé pour elle
    beg = np.zeros( 1, dtype = np.int64 )
    end = np.full( 1, n, dtype = np.int64 )
    nid = np.zeros( 1, dtype = np.int64 )
    node_left.append( -1 ); node_right.append( -1 )
    node_begin.append( 0 ); node_end.append( n )
    node_lo.append( None ); node_hi.append( None )
    node_wa.append( None ); node_wb.append( 0.0 )

    max_depth = 0
    while len( beg ):
        max_depth += 1

        # ---- les boîtes de TOUT le niveau, en deux appels.
        # `reduceat` réduit de `bounds[i]` à `bounds[i+1]`, donc on intercale les trous : les
        # entrées PAIRES sont nos tranches, les impaires ce qui les sépare, et on jette celles-là.
        # Le dernier indice doit rester < n (la dernière réduction va jusqu'au bout du tableau).
        bounds = np.empty( 2 * len( beg ), dtype = np.int64 )
        bounds[ 0::2 ] = beg
        bounds[ 1::2 ] = end
        if bounds[ -1 ] >= n:
            bounds = bounds[ :-1 ]
        lo = np.minimum.reduceat( P, bounds, axis = 0 )[ 0::2 ]
        hi = np.maximum.reduceat( P, bounds, axis = 0 )[ 0::2 ]

        ax = np.argmax( hi - lo, axis = 1 )
        rows = np.arange( len( beg ) )
        # `hi[ ax ] <= lo[ ax ]` : tous les germes au même endroit. Aucune coupe ne les
        # sépareraient, et insister ferait une récursion infinie -- on en fait une feuille, quitte
        # à ce qu'elle soit grosse.
        leaf = ( end - beg <= leaf_size ) | ( hi[ rows, ax ] <= lo[ rows, ax ] )

        for i in range( len( beg ) ):
            k = int( nid[ i ] )
            node_lo[ k ], node_hi[ k ] = lo[ i ], hi[ i ]
            node_wa[ k ], node_wb[ k ] = ( ( np.zeros( d ), 0.0 ) if W is None else
                                           _weight_majorant( P[ beg[ i ]:end[ i ] ],
                                                             W[ beg[ i ]:end[ i ] ] ) )

        # ---- les enfants du niveau suivant
        nb, ne, nn = [], [], []
        for i in np.flatnonzero( ~leaf ):
            b, e, a = int( beg[ i ] ), int( end[ i ] ), int( ax[ i ] )

            # la MÉDIANE, pas le milieu de la boîte : c'est ce qui borne la profondeur par
            # `log2( n / leaf_size )` quelle que soit la distribution -- un nuage très inhomogène
            # ferait dégénérer une coupe géométrique en une chaîne.
            h = ( e - b ) // 2
            p = P[ b:e ]
            part = np.argpartition( p[ :, a ], h )

            # la permutation, appliquée EN PLACE à la tranche : les deux enfants sont alors ses
            # deux moitiés, et il n'y a plus rien à transporter en descendant.
            order[ b:e ] = order[ b:e ][ part ]
            P[ b:e ] = p[ part ]
            if W is not None:
                W[ b:e ] = W[ b:e ][ part ]

            for cb, ce in ( ( b, b + h ), ( b + h, e ) ):
                node_left.append( -1 ); node_right.append( -1 )
                node_begin.append( cb ); node_end.append( ce )
                node_lo.append( None ); node_hi.append( None )
                node_wa.append( None ); node_wb.append( 0.0 )
                nb.append( cb ); ne.append( ce ); nn.append( len( node_left ) - 1 )

            k = int( nid[ i ] )
            node_left[ k ], node_right[ k ] = nn[ -2 ], nn[ -1 ]

        beg = np.array( nb, dtype = np.int64 )
        end = np.array( ne, dtype = np.int64 )
        nid = np.array( nn, dtype = np.int64 )

    return dict(
        seed_indices = order,
        node_left    = np.array( node_left, dtype = np.int64 ),
        node_right   = np.array( node_right, dtype = np.int64 ),
        node_begin   = np.array( node_begin, dtype = np.int64 ),
        node_end     = np.array( node_end, dtype = np.int64 ),
        node_lo      = np.array( node_lo, dtype = float ),
        node_hi      = np.array( node_hi, dtype = float ),
        node_wa      = np.array( node_wa, dtype = float ),
        node_wb      = np.array( node_wb, dtype = float ),
        max_depth    = max_depth,
        nb_leaves    = sum( 1 for l in node_left if l < 0 ),
    )

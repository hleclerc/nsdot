"""Visualiseur de géométries en dimension quelconque : on lui DONNE des primitives, il les
STOCKE, et le choix de la sortie vient après (`write_html`, `write_vtk`).

Pendant de `otrec.viz.points_html` (un nuage de points 2D sur un <canvas>), étendu à ce dont une
CELLULE a besoin : des arêtes avec traitement des parties CACHÉES, des faces éventuellement
PLEINES, en 2D, en 3D, ou au-delà (au-delà de 3, on regarde une COUPE).

= Ce qu'on lui donne

Des primitives en coordonnées MONDE, chacune renvoyant à un jeu de positions :

    v = Visualizer()
    v.add_points( positions )                       # [n, d]
    v.add_edges ( positions, edges )                # [n, d] + [m, 2] (ou None : ligne brisée)
    v.add_faces ( positions, faces )                # [n, d] + polygones (listes d'indices)
    v.add_polytope( cut_directions, cut_offsets )   # H-représentation { x : dir . x <= off }
    v.write_html( "cut.html" )                      # ou v.write_vtk( "cut.vtu" )

La DIMENSION n'est pas déclarée : elle se lit sur la taille des vecteurs positions (dernier axe),
et tous les ajouts d'une même scène doivent s'accorder dessus.

= Le temps (ou un paramètre)

`new_frame( valeur )` ouvre une IMAGE : tout ce qui suit y est rangé. Une image est un état
COMPLET et indépendant -- ni les sommets ni la connectivité n'ont à se correspondre d'une image à
l'autre, des cellules peuvent apparaître, disparaître, changer de forme. C'est ce qu'il faut pour
rejouer une descente ; c'est aussi pourquoi ce n'est PAS la même mécanique que les coupes en
dimension > 3, où l'on regarde un même objet sous un autre angle.

    for k, x in enumerate( iterations ):
        if k: v.new_frame( x )
        ...

L'axe est nommé (`frame_axis`, « temps » par défaut) et se déroule tout seul s'il a un sens à être
parcouru (`playable`) : la page a alors une barre et un bouton lecture/pause, et ParaView reçoit
une série temporelle.

Un objet qui sait se dessiner expose `add_to_viz( viz )` (voir `Cell.add_to_viz`) et s'ajoute avec
`v.add( obj )` -- c'est l'objet qui décide de ses primitives, le visualiseur n'en connaît aucun.

= Comment c'est stocké

Rien n'est mis en forme à l'ajout, tout est gardé sous la forme la plus PROCHE de ce qu'on a reçu,
pour que chaque sortie en fasse ce qu'elle veut (VTK voudra les polygones tels quels, le HTML les
triangulera) :

- UN vivier de sommets pour toute la scène. Deux ajouts qui reçoivent le MÊME tableau de positions
  (les faces et les arêtes d'une cellule) le partagent -- l'identité du tableau suffit à le dire.
- une primitive n'est que des INDICES dans ce vivier, plus un indice de couleur dans une petite
  table. C'est ce qui tient le poids d'une sortie : un sommet de cube sert à 6 triangles et 3
  arêtes, il n'est stocké qu'une fois, et sa couleur ne coûte rien par sommet.
- les polygones gardent leur longueur libre (découpés par un tableau d'offsets), les faces ne sont
  jamais triangulées au stockage.

= Ce que la page HTML sait faire

Le rendu passe en **WebGL2 écrit à la main** plutôt qu'en canvas 2D : une cellule 3D remplie coûte
cher en triangles/arêtes, et c'est le GPU qui doit encaisser ça. Aucune dépendance : une lib type
three.js imposerait un CDN, donc une connexion réseau, ce qui casserait l'autonomie du fichier
(données en base64 dans le HTML, aucun fichier annexe, aucun serveur -- s'ouvre en `file://`).

- faces pleines (opacité réglable) éclairées en double face ;
- ARÊTES CACHÉES : une passe de profondeur PURE (les faces écrites dans le z-buffer sans toucher
  la couleur) précède le tracé des arêtes, ce qui donne l'élimination des parties cachées MÊME
  quand les faces ne sont pas affichées. Trois régimes, touche [e] : cachées supprimées, cachées
  en fantôme (le classique fil de fer lisible), ou tout visible ;
- 2D : caméra orthographique verrouillée à plat, pan/zoom comme `points_html` ;
- 3D : rotation libre (l'orientation est un quaternion : pas de plan de référence, donc le même
  geste produit le même mouvement quelle que soit l'orientation de départ) + pan + zoom,
  projection perspective ou orthographique ([o]). Zoom et rotation VISENT la scène : un rayon
  est lancé dans les primitives déjà présentes côté CPU (`pickWorld`), de sorte que le zoom
  garde immobile le point sous le curseur et que la rotation pivote autour de ce que l'écran
  montre en son centre, à la BONNE profondeur -- pas à celle, arbitraire, de la cible ;
- 4D et plus : on choisit les 3 dimensions montrées (sélecteurs X/Y/Z) et on fixe les AUTRES avec
  un curseur. Les polytopes (H-représentation) sont alors vraiment COUPÉS par l'hyperplan choisi
  -- la coupe reste un polytope, ses sommets sont ré-énumérés dans le navigateur à chaque
  mouvement de curseur --, tandis que les maillages déjà triangulés sont, eux, simplement PROJETÉS
  (couper un maillage n'a pas de sens : l'intersection d'une arête et d'un hyperplan est un point).

Un polytope non borné est affiché coupé par une boîte englobante, sans quoi il n'aurait aucun
sommet à montrer.

= La sortie ParaView

`write_vtk` : XML binaire compressé, une image par `.vtu` plus un `.pvd` qui les rassemble. Les
détails (et les trois choix qu'elle tranche) sont dans `vtk_writer`. Elle a besoin des sommets
d'un polytope, que `polytope` énumère en Python -- en dimension quelconque.
"""
import base64
import colorsys
import json
import re
from pathlib import Path

import numpy as np


#: couleurs par défaut, attribuées dans l'ordre des ajouts (mi-tons : lisibles sur fond clair
#: COMME sur fond sombre, la page bascule de l'un à l'autre à la volée)
# L'ÉCHELLE DE COULEURS : une roue des teintes, PARCOURUE AU NOMBRE D'OR.
#
# Continue et cyclique, donc sans nombre de couleurs à annoncer -- ce qui compte n'est pas
# combien il y en a mais le PAS entre deux indices. Avancer d'un cran ne doit pas bouger la teinte
# d'un cheveu (deux cellules voisines seraient jumelles) : le pas est donc un tour d'or, la façon
# connue de tenir n'importe quel nombre d'indices bien séparés -- les 10 premiers comme les 100,
# sans savoir à l'avance lequel des deux on aura.
GOLDEN_STRIDE = 0.6180339887498949      # φ - 1


def scale_color( index ):
    """La couleur numéro `index` de l'échelle. Une FONCTION de l'indice, et rien d'autre."""
    index = int( index )
    # deux cycles courts sur la saturation et la valeur, en plus de la teinte : le pas d'or garde
    # les indices VOISINS loin les uns des autres, mais deux indices distants finissent par se
    # rapprocher en teinte (13 crans, 21 crans...) et ce sont eux que ces cycles séparent.
    h = ( index * GOLDEN_STRIDE ) % 1.0
    s = 0.55 + 0.13 * ( index % 3 )
    v = 0.95 - 0.13 * ( index % 2 )
    return colorsys.hsv_to_rgb( h, s, v )


def _np( x, dtype = np.float32 ):
    """Un `Tensor` (loom), un array, une liste... -> un `np.ndarray` du dtype demandé.

    `np.asarray` suffit pour les trois : un `Tensor` expose `__array__`, qui rend déjà sa vue
    DENSE (le padding de capacité retiré) -- pas de `.raw[ :n ]` à écrire ici.
    """
    return np.asarray( x, dtype = dtype )


def _rgb( color ):
    """`"#rgb"` / `"#rrggbb"` / `(r, g, b)` dans [0,1] -> trois flottants dans [0,1]."""
    if isinstance( color, str ):
        s = color.lstrip( "#" )
        if len( s ) == 3:
            s = "".join( 2 * c for c in s )
        return tuple( int( s[ i : i + 2 ], 16 ) / 255 for i in ( 0, 2, 4 ) )
    return tuple( float( c ) for c in color )[ :3 ]


def _darker( color, factor = 0.72 ):
    return tuple( c * factor for c in _rgb( color ) )


def _cat( blocks, dtype ):
    """Concatène des blocs (un par ajout) en un seul tableau à plat du dtype voulu."""
    if not blocks:
        return np.zeros( 0, dtype = dtype )
    return np.concatenate( [ np.asarray( b ).reshape( -1 ) for b in blocks ] ).astype( dtype )


def _b64( arr ):
    return base64.b64encode( np.ascontiguousarray( arr ).tobytes() ).decode( "ascii" )


class Visualizer:
    """Collecte des primitives géométriques ; les sorties (`write_html`, ...) viennent après.

    `title` nomme la scène (l'onglet, côté HTML).
    """

    def __init__( self, title = "sdot", background = None, frame_axis = "temps",
                  playable = True, fps = 5.0 ):
        self.title      = title
        self.background = background      # None -> clair/sombre selon la préférence système
        self.nb_dims    = None            # déduit du premier ajout (taille des vecteurs positions)

        # axe des IMAGES (voir `new_frame`) : son nom (le temps, ou un paramètre), et si on peut
        # le dérouler tout seul -- une lecture n'a de sens que sur un axe qu'on parcourt, ce qui
        # est le cas du temps et pas forcément d'un paramètre.
        self.frame_axis = frame_axis
        self.playable   = playable
        self.fps        = fps

        self._pools    = []               # viviers de sommets, chacun [n, d]
        self._bb_pts   = []               # ceux qui CADRENT la scène (voir `_pool( frames = )`)
        self._keep     = []               # les tableaux D'ORIGINE : `id()` doit rester valide
        self._base     = {}               # id( tableau reçu ) -> offset de son vivier
        self._size     = {}               # offset -> nombre de sommets du vivier
        self._nb_verts = 0

        self._colors   = []               # table des couleurs distinctes, en (r, g, b, a)
        self._col_ids  = {}

        self._poly_v, self._poly_len, self._poly_c = [], [], []   # polygones (longueur libre)
        self._edge_v, self._edge_c                 = [], []
        self._pnt_v,  self._pnt_c,  self._pnt_r    = [], [], []
        self._hpolys                               = []           # polytopes en H-représentation

        # le prochain indice libre de l'échelle. Remis à zéro à chaque IMAGE (voir `new_frame`) :
        # une couleur doit dire QUOI, pas COMBIEN a été dessiné avant.
        self._next_color = 0

        # Une IMAGE ne retient que l'endroit où elle commence dans les listes ci-dessus : tout ce
        # qu'on ajoute va dans la DERNIÈRE, donc chaque image y occupe une plage contiguë. Rien
        # n'est dupliqué, et une image n'a aucune structure propre à porter.
        self._nb_edges  = 0
        self._nb_points = 0
        self._frames    = [ self._frame_mark( 0.0 ) ]

    # ---- couleurs -----------------------------------------------------------------------------

    def color_at( self, index ):
        """La couleur numéro `index` de l'échelle -- voir `scale_color`.

        C'est l'entrée à préférer dès qu'un objet a un NUMÉRO à lui : la cellule du dirac `i` se
        colore `color_at( i )`, et sa couleur ne dépend alors que de `i` -- ni de l'ordre des
        appels, ni de ce qui a été dessiné avant, ni de l'image. Un dirac garde sa couleur d'un
        pas de descente au suivant, et un voisin qui perd sa cellule ne décale plus personne.
        """
        return scale_color( index )

    def next_color( self ):
        """La prochaine couleur libre (celle qu'un `add_*` sans `color` prendrait), sans avancer."""
        return self.color_at( self._next_color )

    def reserve_colors( self, nb = 1 ):
        """Réserve `nb` indices CONSÉCUTIFS sur l'échelle et rend le premier.

        Un bloc plutôt qu'un par un : ce qui doit numéroter les couleurs d'un objet composite est
        le rang de ses parties DANS l'objet (l'item `b` d'une cellule batchée prend `base + b`),
        pas le nombre de parties effectivement dessinées -- sinon une cellule vide, qu'on saute,
        décalerait toutes les suivantes.

        La numérotation repart de zéro à chaque image (`new_frame`), de sorte que le même objet
        dessiné au même rang dans chaque image y garde la même couleur.
        """
        res = self._next_color
        self._next_color += int( nb )
        return res

    def take_color( self ):
        """Réserve UN indice et rend sa couleur -- pour un objet qui n'a pas de numéro à lui.

        À prendre quand un même objet donne PLUSIEURS primitives (une cellule : ses faces, ses
        arêtes, ses sommets) et qu'on les veut de la même couleur : sans ça, chaque `add_*` sans
        `color` en consommerait une nouvelle et la cellule serait bariolée.
        """
        return self.color_at( self.reserve_colors() )

    @staticmethod
    def darker( color, factor = 0.72 ):
        """La même teinte, assombrie -- de quoi tirer une couleur d'arête d'une couleur de face."""
        return _darker( color, factor )

    def _color_id( self, color, opacity ):
        """Indice de `(color, opacity)` dans la table -- une primitive ne porte que cet indice."""
        if color is None:
            color = self.take_color()
        rgba = ( *_rgb( color ), float( opacity ) )
        if rgba not in self._col_ids:
            self._col_ids[ rgba ] = len( self._colors )
            self._colors.append( rgba )
        return self._col_ids[ rgba ]

    # ---- vivier de sommets --------------------------------------------------------------------

    def _pool( self, positions, frames = True ):
        """Enregistre `positions` (une fois) et rend l'offset de son vivier.

        Le partage se lit sur l'IDENTITÉ du tableau reçu : deux ajouts qui reçoivent le même objet
        (les faces et les arêtes d'une cellule) renvoient au même vivier, et les sommets ne sont
        stockés qu'une fois. C'est aussi ici que la dimension de la scène se décide -- la taille
        des vecteurs positions, rien d'autre.

        `frames = False` dessine sans CADRER : ces sommets-là n'entrent pas dans la boîte de la
        scène (`bounds`). Pour ce qui est tracé à une distance qui ne veut rien dire -- le moignon
        d'une arête qui part à l'infini (`Cell.add_to_viz`) -- et qui, compté, écraserait sur un
        point ce qu'on est venu regarder. C'est le PREMIER ajout d'un vivier qui en décide, les
        suivants retrouvant le même offset sans repasser par ici.
        """
        key = id( positions )
        if key in self._base:
            return self._base[ key ]

        pos = _np( positions )
        if pos.ndim == 1:
            pos = pos.reshape( 1, -1 )
        pos = pos.reshape( -1, pos.shape[ -1 ] )
        self._note_dims( pos.shape[ 1 ] )

        self._base[ key ] = base = self._nb_verts
        self._size[ base ] = len( pos )
        self._pools.append( pos )
        if frames:
            self._bb_pts.append( pos )
        self._keep.append( positions )
        self._nb_verts += len( pos )
        return base

    def _note_dims( self, nb_dims ):
        if nb_dims < 2:
            raise ValueError( f"Visualizer: dimension { nb_dims } (au moins 2 attendues)" )
        if self.nb_dims is None:
            self.nb_dims = int( nb_dims )
        elif int( nb_dims ) != self.nb_dims:
            raise ValueError( f"Visualizer: dimension { nb_dims } incompatible avec la scène "
                              f"(déjà en { self.nb_dims }D)" )

    # ---- images (temps, ou paramètre) ---------------------------------------------------------

    def _frame_mark( self, value ):
        return { "value": float( value ),
                 "poly": len( self._poly_len ), "edge": self._nb_edges,
                 "pnt": self._nb_points, "hpoly": len( self._hpolys ) }

    def new_frame( self, value = None ):
        """Ouvre une nouvelle IMAGE : tout ce qu'on ajoutera ensuite y sera rangé.

        Une image est un ÉTAT COMPLET, indépendant des autres : ni les sommets ni la connectivité
        n'ont à se correspondre d'une image à l'autre (des cellules apparaissent, disparaissent,
        changent de forme -- c'est ce qu'il faut pour rejouer une descente). Elles ne partagent
        que le vivier de sommets, et seulement là où l'appelant repasse le MÊME tableau.

        `value` est l'abscisse sur l'axe (`frame_axis`) : un instant, ou la valeur du paramètre.
        Par défaut, le rang de l'image.

        La numérotation des couleurs repart de zéro ici. C'est ce qui fait qu'une animation ne
        clignote pas : ce qu'on redessine au même rang dans chaque image y reprend sa couleur, au
        lieu d'en prendre une nouvelle parce que l'image précédente en a consommé.
        """
        self._next_color = 0
        self._frames.append( self._frame_mark(
            len( self._frames ) if value is None else value ) )
        return self

    @property
    def nb_frames( self ):
        return len( self._frames )

    def _frame_ranges( self, key ):
        """Les bornes des images sur une des listes de primitives : `nb_frames + 1` entiers."""
        total = { "poly": len( self._poly_len ), "edge": self._nb_edges,
                  "pnt": self._nb_points, "hpoly": len( self._hpolys ) }[ key ]
        return np.array( [ f[ key ] for f in self._frames ] + [ total ], dtype = np.int32 )

    # ---- primitives ---------------------------------------------------------------------------

    def add( self, obj, **kwargs ):
        """Ajoute un objet qui sait se dessiner (il expose `add_to_viz( viz, ... )`).

        Le visualiseur ne connaît aucun type métier : c'est l'objet qui traduit sa géométrie en
        primitives (`Cell.add_to_viz` -> `add_faces` / `add_edges` / `add_polytope`).
        """
        obj.add_to_viz( self, **kwargs )
        return self

    def note_bounds( self, positions ):
        """Étend la boîte de la scène à `positions`, SANS rien dessiner.

        Pour ce qu'on affiche autrement que par des sommets : un polytope donné en demi-espaces
        n'a pas d'étendue lisible, alors que l'objet qui l'envoie, lui, la connaît souvent (une
        cellule garde ses sommets) -- et c'est cette boîte qui cadre la caméra, donne leur plage
        aux curseurs de coupe et fixe le rognage des polytopes non bornés.
        """
        self._pool( positions )
        return self

    def add_points( self, positions, radius = 0.0, color = None, opacity = 1.0, frames = True ):
        """`positions` : `[n, d]`. `radius` : le rayon MONDE des points (un scalaire, ou un rayon
        par point). `0` (défaut) = le point n'a pas de taille propre (un dirac) : c'est le curseur
        de la page qui fixe son rayon d'affichage -- même convention que `points_html`.
        """
        base = self._pool( positions, frames )
        n = self._pool_len( base )
        if n == 0:
            return self
        rad = np.full( n, float( radius ), dtype = np.float32 ) \
            if np.isscalar( radius ) else _np( radius ).reshape( -1 )
        if len( rad ) != n:
            raise ValueError( f"radius: { len( rad ) } rayons pour { n } points" )
        ci = self._color_id( color, opacity )
        self._pnt_v.append( np.arange( base, base + n, dtype = np.int32 ) )
        self._pnt_r.append( rad )
        self._pnt_c.append( np.full( n, ci, dtype = np.uint16 ) )
        self._nb_points += n
        return self

    def add_edges( self, positions, edges = None, color = None, opacity = 1.0, closed = False,
                   dashed = False, nb_dashes = 7, frames = True ):
        """`positions` : `[n, d]`. `edges` : `[m, 2]` d'indices, ou `None` pour relier les points
        consécutifs (`closed` referme alors la boucle -- le cas d'un polygone donné en ordre
        cyclique, ce que rend une cellule 2D).

        `dashed` trace des POINTILLÉS. Le tiret n'est pas un style porté jusqu'à la sortie : chaque
        segment est simplement DÉCOUPÉ ici, en `nb_dashes` morceaux séparés par autant de trous, et
        ce qui part ensuite est une suite d'arêtes ordinaires. Deux raisons : le rendu (WebGL écrit
        à la main) n'a pas de motif de ligne, et le VTK non plus -- une géométrie découpée est la
        seule chose que les DEUX sorties savent afficher pareil. Le prix est que la longueur d'un
        tiret est en unités MONDE, pas en pixels : c'est un nombre fixe de tirets par arête, donc
        la densité ne change pas avec le zoom.
        """
        base = self._pool( positions, frames )
        n = self._pool_len( base )
        if edges is None:
            if n < 2:
                return self
            idx = np.stack( [ np.arange( n - 1 ), np.arange( 1, n ) ], axis = 1 )
            if closed:
                idx = np.concatenate( [ idx, [ [ n - 1, 0 ] ] ], axis = 0 )
        else:
            idx = np.asarray( edges, dtype = np.int64 ).reshape( -1, 2 )
        if len( idx ) == 0:
            return self
        if dashed:
            return self._add_dashes( positions, idx, color, opacity, nb_dashes, frames )
        ci = self._color_id( color, opacity )
        self._edge_v.append( idx.astype( np.int32 ) + base )
        self._edge_c.append( np.full( len( idx ), ci, dtype = np.uint16 ) )
        self._nb_edges += len( idx )
        return self

    def _add_dashes( self, positions, idx, color, opacity, nb_dashes, frames = True ):
        """Les `idx` segments de `positions`, en pointillés : un vivier de sommets À PART, tenant
        les bouts des tirets, et des arêtes ordinaires dessus (voir `add_edges( dashed = True )`).
        """
        p = _np( positions )
        a, b = p[ idx[ :, 0 ] ], p[ idx[ :, 1 ] ]

        # `2 k - 1` intervalles -> `k` tirets séparés par `k - 1` trous, et une arête qui COMMENCE
        # et FINIT par un tiret : ses deux extrémités restent donc visibles là où elles comptent
        # (le sommet réel d'un côté, la direction de fuite de l'autre).
        nb_dashes = max( 1, int( nb_dashes ) )
        k = 2 * nb_dashes - 1
        t = np.arange( k + 1, dtype = np.float32 ) / k
        pts = a[ :, None, : ] + ( b - a )[ :, None, : ] * t[ None, :, None ]

        starts = ( np.arange( len( idx ) )[ :, None ] * ( k + 1 )
                 + np.arange( 0, k, 2 )[ None, : ] )
        seg = np.stack( [ starts, starts + 1 ], axis = -1 ).reshape( -1, 2 )
        return self.add_edges( pts.reshape( -1, p.shape[ 1 ] ), seg, color = color,
                               opacity = opacity, frames = frames )

    def add_faces( self, positions, faces, color = None, opacity = 1.0, frames = True ):
        """`positions` : `[n, d]`. `faces` : une suite de POLYGONES (listes d'indices, longueurs
        libres). Ils sont gardés TELS QUELS -- c'est la sortie qui décide si elle les triangule.
        """
        base = self._pool( positions, frames )
        ci = self._color_id( color, opacity )
        nb = 0
        for f in faces:
            f = [ int( i ) + base for i in f ]
            if len( f ) < 3:
                continue
            self._poly_v += f
            self._poly_len.append( len( f ) )
            nb += 1
        self._poly_c.append( np.full( nb, ci, dtype = np.uint16 ) )
        return self

    def add_polytope( self, cut_directions, cut_offsets, color = None, opacity = 1.0,
                      edge_color = None, nb_dims = None, edges = True ):
        """Ajoute un polytope par sa H-représentation : `{ x : dir_i . x <= off_i }`.

        C'est la forme à donner quand la V-représentation ne suffit PAS : en dimension > 3 (la
        page en montre une COUPE 3D, et couper des demi-espaces redonne des demi-espaces, donc
        un polytope, ré-énuméré dans le navigateur à chaque déplacement de curseur), ou pour un
        polytope NON BORNÉ (aucun sommet à énumérer côté Python -- la page le coupe par une boîte
        englobante et montre ce qui reste).

        `edges = False` n'en garde que les FACES. Pour l'appelant qui trace lui-même les arêtes et
        en sait plus que l'énumération : celle d'un polytope non borné s'arrête sur la boîte de la
        scène, elle rendrait donc pleine et jusqu'au bord une arête qui part à l'infini -- là où
        `Cell.add_to_viz`, lui, sait qu'elle est tronquée et la met en pointillés.
        """
        dirs = _np( cut_directions )
        dirs = dirs.reshape( -1, dirs.shape[ -1 ] )
        offs = _np( cut_offsets ).reshape( -1 )
        if len( dirs ) != len( offs ):
            raise ValueError( f"polytope: { len( dirs ) } directions pour { len( offs ) } offsets" )
        self._note_dims( nb_dims if nb_dims is not None else dirs.shape[ 1 ] )

        if color is None:
            color = self.take_color()
        self._hpolys.append( {
            "dirs": _b64( dirs.astype( np.float32 ) ),
            "offs": _b64( offs.astype( np.float32 ) ),
            "nb"  : int( len( dirs ) ),
            "fcol": [ *_rgb( color ), float( opacity ) ],
            "ecol": [ *_rgb( edge_color if edge_color is not None else _darker( color ) ), 1.0 ],
            "edg" : bool( edges ),
        } )
        return self

    # ---- lecture de la scène ------------------------------------------------------------------

    def _pool_len( self, base ):
        """Le nombre de sommets du vivier commençant à `base`."""
        return self._size[ base ]

    @property
    def positions( self ):
        """Le vivier de sommets de toute la scène, `[nb_verts, d]` -- les indices des primitives
        y renvoient."""
        if not self._pools:
            return np.zeros( ( 0, self.nb_dims or 2 ), dtype = np.float32 )
        return np.concatenate( self._pools, axis = 0 )

    @property
    def polygons( self ):
        """Les faces, comme une liste de listes d'indices (longueurs libres, non triangulées)."""
        res, o = [], 0
        for n in self._poly_len:
            res.append( self._poly_v[ o : o + n ] )
            o += n
        return res

    @property
    def edges( self ):
        """Les arêtes, `[m, 2]` d'indices."""
        return _cat( self._edge_v, np.int32 ).reshape( -1, 2 )

    @property
    def points( self ):
        """Les points, `[k]` d'indices."""
        return _cat( self._pnt_v, np.int32 )

    @property
    def point_radii( self ):
        """Le rayon monde de chaque point (0 = pas de taille propre, cf. `add_points`)."""
        return _cat( self._pnt_r, np.float32 )

    @property
    def colors( self ):
        """La table des couleurs distinctes, en `(r, g, b, a)` -- les primitives y renvoient."""
        return list( self._colors )

    @property
    def polygon_colors( self ):
        """L'indice de couleur de chaque polygone, dans `colors`."""
        return _cat( self._poly_c, np.uint16 )

    @property
    def edge_colors( self ):
        """L'indice de couleur de chaque arête, dans `colors`."""
        return _cat( self._edge_c, np.uint16 )

    @property
    def point_colors( self ):
        """L'indice de couleur de chaque point, dans `colors`."""
        return _cat( self._pnt_c, np.uint16 )

    def frame( self, i ):
        """Les primitives de l'image `i`, déjà découpées -- de quoi écrire une sortie sans rien
        savoir du découpage interne.

        Rend un dict : `value`, `polygons`, `polygon_colors`, `edges`, `edge_colors`, `points`,
        `point_colors`, `point_radii`, `polytopes`. Les indices renvoient au vivier `positions`,
        commun à TOUTES les images.
        """
        def rng( key ):
            r = self._frame_ranges( key )
            return int( r[ i ] ), int( r[ i + 1 ] )

        p0, p1 = rng( "poly" )
        e0, e1 = rng( "edge" )
        v0, v1 = rng( "pnt" )
        h0, h1 = rng( "hpoly" )
        return {
            "value"         : self._frames[ i ][ "value" ],
            "polygons"      : self.polygons[ p0 : p1 ],
            "polygon_colors": self.polygon_colors[ p0 : p1 ],
            "edges"         : self.edges[ e0 : e1 ],
            "edge_colors"   : self.edge_colors[ e0 : e1 ],
            "points"        : self.points[ v0 : v1 ],
            "point_colors"  : self.point_colors[ v0 : v1 ],
            "point_radii"   : self.point_radii[ v0 : v1 ],
            "polytopes"     : self.polytopes[ h0 : h1 ],
        }

    @property
    def polytopes( self ):
        """Les polytopes en H-représentation, `[ ( dirs [c, d], offs [c], rgba, edges ), ... ]`.

        Ils n'ont PAS de sommets : une sortie qui en veut (VTK) doit les énumérer elle-même,
        comme le fait la page HTML à chaque coupe.
        """
        res = []
        for h in self._hpolys:
            dirs = np.frombuffer( base64.b64decode( h[ "dirs" ] ), np.float32 ).reshape( -1, self.nb_dims )
            offs = np.frombuffer( base64.b64decode( h[ "offs" ] ), np.float32 )
            res.append( ( dirs, offs, tuple( h[ "fcol" ] ), h[ "edg" ] ) )
        return res

    def bounds( self ):
        """Boîte englobante `[ [lo, hi] ] * d`, dimension par dimension.

        Elle cadre la caméra ET donne leur plage aux curseurs de coupe. Un polytope n'a pas de
        sommets à mesurer : on l'encadre par la distance de ses plans à l'origine, ce qui est
        grossier mais du bon ordre de grandeur (et sans objet dès qu'un vivier est là aussi).
        """
        d = self.nb_dims
        # ce qui CADRE, pas tout ce qui est stocké (voir `_pool( frames = )`). Une scène dont tout
        # aurait renoncé à cadrer retombe sur ses sommets : mieux vaut une boîte trop grande que
        # pas de boîte du tout.
        pts = self._bb_pts or self._pools
        if pts:
            allp = np.concatenate( pts, axis = 0 )
            lo, hi = allp.min( axis = 0 ), allp.max( axis = 0 )
        else:
            r = 1.0
            for h in self._hpolys:
                dirs = np.frombuffer( base64.b64decode( h[ "dirs" ] ), dtype = np.float32 ).reshape( -1, d )
                offs = np.frombuffer( base64.b64decode( h[ "offs" ] ), dtype = np.float32 )
                nrm  = np.linalg.norm( dirs, axis = 1 )
                ok   = nrm > 1e-12
                if ok.any():
                    r = max( r, float( np.abs( offs[ ok ] / nrm[ ok ] ).max() ) )
            lo, hi = np.full( d, -r, np.float32 ), np.full( d, r, np.float32 )

        span = np.maximum( hi - lo, 1e-9 )
        pad  = 0.05 * span.max()
        return [ [ float( a - pad ), float( b + pad ) ] for a, b in zip( lo, hi ) ]

    # ---- sorties ------------------------------------------------------------------------------

    def write_vtk( self, filename, axes = ( 0, 1, 2 ) ):
        """Écrit la scène pour ParaView (XML binaire compressé). Renvoie le chemin écrit.

        Une seule image -> un `.vtu`. Plusieurs -> un `.vtu` par image plus un `.pvd` qui les
        rassemble avec leur abscisse sur l'axe (c'est le `.pvd` qu'on ouvre, et c'est lui qui est
        rendu). `axes` choisit les 3 dimensions écrites en GÉOMÉTRIE ; au-delà de 3, les autres
        coordonnées partent en données de points, pour que ParaView fasse ses coupes lui-même.
        """
        from .vtk_writer import write_vtk
        return write_vtk( self, filename, axes = axes )

    def write_html( self, filename ):
        """Écrit une page HTML autonome (voir l'en-tête du module). Renvoie le chemin écrit."""
        path = Path( filename )
        if self.nb_dims is None:
            raise ValueError( "Visualizer: rien à afficher (aucune primitive ajoutée)" )

        d       = self.nb_dims
        bounds  = self.bounds()
        span    = max( hi - lo for lo, hi in bounds )
        r0      = span / 150                                    # rayon de départ des points nus

        poly_off = np.concatenate( [ [ 0 ], np.cumsum( self._poly_len ) ] ).astype( np.int32 )

        subs = {
            "__TITLE__"   : self.title,
            "__D__"       : str( d ),
            "__BOUNDS__"  : json.dumps( bounds ),
            "__POS__"     : _b64( self.positions.astype( np.float32 ) ),
            "__COLORS__"  : json.dumps( [ list( c ) for c in self._colors ] ),
            "__POLY_V__"  : _b64( np.asarray( self._poly_v, dtype = np.int32 ) ),
            "__POLY_OFF__": _b64( poly_off ),
            "__POLY_C__"  : _b64( self.polygon_colors ),
            "__EDG_V__"   : _b64( self.edges.reshape( -1 ) ),
            "__EDG_C__"   : _b64( self.edge_colors ),
            "__PNT_V__"   : _b64( self.points ),
            "__PNT_C__"   : _b64( self.point_colors ),
            "__PNT_R__"   : _b64( self.point_radii ),
            "__HPOLY__"   : json.dumps( self._hpolys ),
            "__FR_VAL__"  : _b64( np.array( [ f[ "value" ] for f in self._frames ], np.float32 ) ),
            "__FR_POLY__" : _b64( self._frame_ranges( "poly" ) ),
            "__FR_EDG__"  : _b64( self._frame_ranges( "edge" ) ),
            "__FR_PNT__"  : _b64( self._frame_ranges( "pnt" ) ),
            "__FR_HP__"   : _b64( self._frame_ranges( "hpoly" ) ),
            "__AXIS__"    : json.dumps( self.frame_axis ),
            "__PLAYABLE__": "true" if self.playable else "false",
            "__FPS__"     : repr( float( self.fps ) ),
            "__R0__"      : repr( float( r0 ) ),
            "__RMIN__"    : repr( float( r0 / 30 ) ),
            "__RMAX__"    : repr( float( r0 * 60 ) ),
            "__BG__"      : json.dumps( None if self.background is None
                                        else list( _rgb( self.background ) ) ),
        }
        html = re.sub( r"__[A-Z0-9_]+__", lambda m: subs.get( m.group( 0 ), m.group( 0 ) ), _HTML )

        path.parent.mkdir( parents = True, exist_ok = True )
        path.write_text( html )
        print( f"OUTPUT: file://{ path.absolute() }" )
        return path


_HTML = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>__TITLE__</title>
<style>
  html, body { margin: 0; padding: 0; overflow: hidden; background: #ffffff; color: #111; }
  canvas { display: block; }
  #controls {
    position: fixed; top: 10px; left: 10px; z-index: 1;
    background: rgba(255,255,255,0.88); padding: 8px 12px; border-radius: 6px;
    font-family: sans-serif; font-size: 13px; box-shadow: 0 1px 4px rgba(0,0,0,0.3);
    width: 244px; box-sizing: border-box;
  }
  #controls *, #controls { box-sizing: border-box; }
  #controls label { display: block; margin: 2px 0; }
  #controls input[type=range] { vertical-align: middle; width: 100%; }
  #controls select { font-size: 12px; }
  #controls .hint { color: #666; margin-top: 6px; }
  #controls .sec { margin-top: 8px; padding-top: 8px; border-top: 1px solid #ddd; }
  #controls .val { color: #666; font-variant-numeric: tabular-nums; }
  #timeControls .row { display: flex; align-items: center; gap: 6px; }
  #timeControls input[type=range] { flex: 1 1 auto; width: auto; min-width: 0; }
  #play {
    cursor: pointer; border: none; border-radius: 4px; background: #444; color: #fff;
    width: 26px; height: 26px; font-size: 12px; line-height: 1; flex: none;
  }
  #tval { flex: none; text-align: right; font-variant-numeric: tabular-nums; }
  body.dark #play { background: #8ab4f8; color: #172033; }
  #slices .row { display: flex; align-items: center; gap: 6px; }
  #slices .row span { flex: none; width: 62px; }
  #axes select { width: 58px; }
  #help {
    display: none; position: fixed; top: 10px; right: 10px; z-index: 2;
    background: rgba(255,255,255,0.95); padding: 10px 14px; border-radius: 6px;
    font-family: sans-serif; font-size: 12px; box-shadow: 0 1px 4px rgba(0,0,0,0.3);
  }
  #help table { border-collapse: collapse; }
  #help td { padding: 1px 0; }
  #help td:first-child { padding-right: 10px; color: #333; white-space: nowrap; }
  #help td:last-child { color: #666; }
  body.dark { background: #1a1a1a; color: #ddd; }
  body.dark #controls { background: rgba(30,30,30,0.92); color: #ddd; }
  body.dark #controls .hint, body.dark #controls .val { color: #aaa; }
  body.dark #controls .sec { border-top-color: #555; }
  body.dark #controls input[type=range] { accent-color: #8ab4f8; }
  body.dark #help { background: rgba(30,30,30,0.95); color: #ccc; }
  body.dark #help td:first-child { color: #ddd; }
  body.dark #help td:last-child { color: #aaa; }
</style>
</head>
<body data-theme="light">
<div id="controls">
  <div id="counts"></div>
  <div class="sec" id="timeControls" style="display:none">
    <div class="row">
      <button id="play">&#9654;</button>
      <input id="t" type="range" min="0" max="0" step="1" value="0">
      <span id="tval">1 / 1</span>
    </div>
    <div class="val" id="tname"></div>
  </div>
  <div class="sec">
    <label><input type="checkbox" id="cbFaces" checked> faces</label>
    <label>opacité <span class="val" id="opVal"></span>
      <input id="op" type="range" min="0.05" max="1" step="0.01" value="1"></label>
    <label>arêtes : <select id="edgeMode">
      <option value="hide">cachées supprimées</option>
      <option value="ghost" selected>cachées en fantôme</option>
      <option value="all">toutes visibles</option>
      <option value="none">aucune</option>
    </select></label>
  </div>
  <div class="sec" id="ptBox">
    <label><input type="checkbox" id="cbPoints" checked> points</label>
    <label>rayon <span class="val" id="rVal"></span>
      <input id="r" type="range" min="0" max="1" step="0.001" value="0.5"></label>
  </div>
  <div class="sec" id="axes" style="display:none">
    vues sur <select id="axX"></select><select id="axY"></select><select id="axZ"></select>
  </div>
  <div id="slices"></div>
  <div class="hint"><b>?</b> : aide · <b>d</b> : <span id="modeLabel">clair</span></div>
</div>
<div id="help">
  <b>Raccourcis clavier</b>
  <table>
    <tr><td>&larr; / &rarr;</td><td id="hTime">image -1 / +1 (barre de temps active)</td></tr>
    <tr><td>Maj + &larr;/&rarr;</td><td>image, pas plus large</td></tr>
    <tr><td>Début / Fin</td><td>première / dernière image</td></tr>
    <tr><td>Espace</td><td>lecture / pause</td></tr>
    <tr><td>glisser</td><td id="hDrag">orbite</td></tr>
    <tr><td>Maj + glisser</td><td>déplacer (pan)</td></tr>
    <tr><td>molette</td><td>déplacer · Ctrl/pincement : zoom vers le curseur</td></tr>
    <tr><td>&larr; &rarr; &uarr; &darr;</td><td id="hArrows">tourner (Maj/Ctrl : plus vite)</td></tr>
    <tr><td>+ / -</td><td>zoom avant / arrière</td></tr>
    <tr><td>[ / ]</td><td>rayon des points - / +</td></tr>
    <tr><td>f</td><td>faces</td></tr>
    <tr><td>e</td><td>régime des arêtes cachées</td></tr>
    <tr><td>p</td><td>points</td></tr>
    <tr><td>o</td><td>projection ortho / perspective</td></tr>
    <tr><td>0 · double-clic</td><td>réinitialiser la vue</td></tr>
    <tr><td>d</td><td>clair / sombre</td></tr>
    <tr><td>?</td><td>afficher / masquer cette aide</td></tr>
  </table>
</div>
<canvas id="c"></canvas>
<script>
// ============================================================================================
// données -- tout est encodé en base64 dans la page (fichier autonome, ouvrable en file://)
// ============================================================================================
function b64bytes(s) {
  const raw = atob(s), b = new Uint8Array(raw.length);
  for (let i = 0; i < raw.length; i++) b[i] = raw.charCodeAt(i);
  return b;
}
function decF32(s) { const b = b64bytes(s); return new Float32Array(b.buffer, 0, b.length / 4); }
function decI32(s) { const b = b64bytes(s); return new Int32Array(b.buffer, 0, b.length / 4); }
function decU16(s) { const b = b64bytes(s); return new Uint16Array(b.buffer, 0, b.length / 2); }

const D      = __D__;                  // dimension du MONDE (taille des vecteurs positions)
const BOUNDS = __BOUNDS__;             // [lo, hi] par dimension

// UN seul vivier de sommets pour toute la scène (POS, de pas D) ; une primitive n'y renvoie que
// des INDICES, et sa couleur n'est qu'un indice dans COLORS. C'est ce qui tient le poids du
// fichier : un sommet de cube sert à 6 triangles et 3 arêtes, il n'est stocké qu'une fois, et
// une couleur ne coûte que 2 octets par primitive au lieu de 4 par sommet.
const POS    = decF32("__POS__");
const COLORS = __COLORS__;             // table des couleurs distinctes, en [r, g, b, a]

// polygones de longueur LIBRE : POLY_OFF découpe POLY_V (POLY_OFF[p] .. POLY_OFF[p+1]). Ils ne
// sont triangulés qu'ici, à l'affichage -- le fichier, lui, garde les faces telles quelles.
const POLY_V = decI32("__POLY_V__"), POLY_OFF = decI32("__POLY_OFF__"), POLY_C = decU16("__POLY_C__");
const EDG_V  = decI32("__EDG_V__"),  EDG_C    = decU16("__EDG_C__");
const PNT_V  = decI32("__PNT_V__"),  PNT_C    = decU16("__PNT_C__"), PNT_R = decF32("__PNT_R__");

const HPOLY  = __HPOLY__.map(h => ({                                     // polytopes en H-rep
  nb: h.nb, dirs: decF32(h.dirs), offs: decF32(h.offs), fcol: h.fcol, ecol: h.ecol, edg: h.edg,
}));
// IMAGES : chacune n'est qu'une PLAGE dans les listes ci-dessus (elles ne partagent rien
// d'autre que le vivier de sommets) -- FR_* donne les bornes, FR_VAL l'abscisse sur l'axe.
const FR_VAL = decF32("__FR_VAL__");
const FR_POLY = decI32("__FR_POLY__"), FR_EDG = decI32("__FR_EDG__");
const FR_PNT = decI32("__FR_PNT__"), FR_HP = decI32("__FR_HP__");
const NB_FRAMES = FR_VAL.length, AXIS = __AXIS__, PLAYABLE = __PLAYABLE__, FPS = __FPS__;

const RMIN = __RMIN__, RMAX = __RMAX__, R0 = __R0__;
const BG = __BG__;

const NB_V = POS.length / D, NB_POLY = POLY_OFF.length - 1, NB_EDG = EDG_V.length / 2;

// ============================================================================================
// projection monde -> 3D : quelles dimensions on regarde, et où on coupe les autres
// ============================================================================================
// AX = les dimensions montrées en x, y, z (z = -1 en 2D : tout est à plat). SLICE fixe la valeur
// des dimensions NON montrées -- c'est le réglage de la coupe, seuls les polytopes la subissent
// vraiment (voir sliceHalfspaces) ; un maillage déjà triangulé est simplement projeté.
let AX = [0, 1, D >= 3 ? 2 : -1];
const SLICE = new Float32Array(D);
for (let k = 0; k < D; k++) SLICE[k] = 0.5 * (BOUNDS[k][0] + BOUNDS[k][1]);

function px(src, i) { return AX[0] >= 0 ? src[i * D + AX[0]] : 0; }
function py(src, i) { return AX[1] >= 0 ? src[i * D + AX[1]] : 0; }
function pz(src, i) { return AX[2] >= 0 ? src[i * D + AX[2]] : 0; }

// ============================================================================================
// polytopes : coupe puis énumération des sommets (dans le navigateur, à chaque changement)
// ============================================================================================
// { dir . x <= off } en dimension D, restreint aux dimensions montrées et aux coupes fixées :
// dir_vu . y <= off - somme( dir_k * SLICE[k] ) sur les dimensions cachées. Une coupe de
// demi-espaces reste un jeu de demi-espaces -- d'où un polytope, qu'on sait ré-énumérer.
function sliceHalfspaces(h) {
  const A = [], b = [];
  for (let i = 0; i < h.nb; i++) {
    const a0 = AX[0] >= 0 ? h.dirs[i * D + AX[0]] : 0;
    const a1 = AX[1] >= 0 ? h.dirs[i * D + AX[1]] : 0;
    const a2 = AX[2] >= 0 ? h.dirs[i * D + AX[2]] : 0;
    let rhs = h.offs[i];
    for (let k = 0; k < D; k++)
      if (k !== AX[0] && k !== AX[1] && k !== AX[2]) rhs -= h.dirs[i * D + k] * SLICE[k];
    const n = Math.hypot(a0, a1, a2);
    // plan devenu constant : soit la contrainte est vide (la coupe ne rencontre pas le polytope),
    // soit elle est toujours vraie et ne dit plus rien.
    if (n < 1e-12) { if (rhs < -1e-9) return null; continue; }
    A.push([a0 / n, a1 / n, a2 / n]); b.push(rhs / n);   // plans NORMALISÉS : un seul eps partout
  }
  return { A, b };
}

// boîte englobante ajoutée à tout polytope : sans elle, un polytope non borné n'aurait aucun
// sommet à énumérer, donc rien à montrer. C'est EXACTEMENT la boîte de la scène, celle que cadre
// la caméra -- et elle est déjà marginée côté Python, donc elle ne rogne rien de borné.
function clipPlanes() {
  const c = [], r = [];
  for (let a = 0; a < 3; a++) {
    const k = AX[a];
    const lo = k >= 0 ? BOUNDS[k][0] : -1, hi = k >= 0 ? BOUNDS[k][1] : 1;
    const mid = 0.5 * (lo + hi), half = Math.max(1e-6, 0.5 * (hi - lo));
    const e = [0, 0, 0]; e[a] = 1;
    c.push(e.slice()); r.push(mid + half);
    c.push(e.map(v => -v)); r.push(-(mid - half));
  }
  return { A: c, b: r };
}

function feasible(A, b, x, eps) {
  for (let m = 0; m < A.length; m++)
    if (A[m][0] * x[0] + A[m][1] * x[1] + A[m][2] * x[2] > b[m] + eps) return false;
  return true;
}

// V-représentation d'un polytope 3D donné en demi-espaces : tout sommet est l'intersection de 3
// plans (et est admissible pour tous les autres) ; une face est l'ensemble des sommets portés par
// un même plan, remis en ordre cyclique autour de leur centre.
function polyhedron3D(A, b) {
  const n = A.length, eps = 1e-6 * Math.max(1, Math.max(...b.map(Math.abs)));
  const V = [], key = new Map();
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) for (let k = j + 1; k < n; k++) {
    const a = A[i], c = A[j], e = A[k];
    const det = a[0] * (c[1] * e[2] - c[2] * e[1])
              - a[1] * (c[0] * e[2] - c[2] * e[0])
              + a[2] * (c[0] * e[1] - c[1] * e[0]);
    if (Math.abs(det) < 1e-9) continue;                       // plans (quasi) liés : pas de sommet
    const d0 = b[i], d1 = b[j], d2 = b[k];
    const x = ( d0 * (c[1] * e[2] - c[2] * e[1]) - a[1] * (d1 * e[2] - c[2] * d2)
              + a[2] * (d1 * e[1] - c[1] * d2) ) / det;
    const y = ( a[0] * (d1 * e[2] - c[2] * d2) - d0 * (c[0] * e[2] - c[2] * e[0])
              + a[2] * (c[0] * d2 - d1 * e[0]) ) / det;
    const z = ( a[0] * (c[1] * d2 - d1 * e[1]) - a[1] * (c[0] * d2 - d1 * e[0])
              + d0 * (c[0] * e[1] - c[1] * e[0]) ) / det;
    const p = [x, y, z];
    if (!feasible(A, b, p, eps)) continue;
    const kk = p.map(v => Math.round(v / (eps * 10))).join(",");   // sommets confondus fusionnés
    if (key.has(kk)) continue;
    key.set(kk, V.length); V.push(p);
  }
  if (V.length < 3) return null;

  // sur quels plans chaque sommet se tient : c'est ce qui permet ensuite de reconnaître une
  // arête portée par la BOÎTE DE ROGNAGE, qui n'est pas une arête du polytope (voir buildScene).
  const vpl = V.map(() => []);
  const faces = [], fpl = [];
  for (let p = 0; p < n; p++) {
    const on = [];
    for (let v = 0; v < V.length; v++) {
      const q = V[v];
      if (Math.abs(A[p][0] * q[0] + A[p][1] * q[1] + A[p][2] * q[2] - b[p]) < eps * 10) {
        on.push(v); vpl[v].push(p);
      }
    }
    if (on.length < 3) continue;
    const nz = A[p];
    let u = Math.abs(nz[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
    u = [u[0] - nz[0] * (u[0] * nz[0] + u[1] * nz[1] + u[2] * nz[2]),
         u[1] - nz[1] * (u[0] * nz[0] + u[1] * nz[1] + u[2] * nz[2]),
         u[2] - nz[2] * (u[0] * nz[0] + u[1] * nz[1] + u[2] * nz[2])];
    const ul = Math.hypot(u[0], u[1], u[2]); u = [u[0] / ul, u[1] / ul, u[2] / ul];
    const w = [nz[1] * u[2] - nz[2] * u[1], nz[2] * u[0] - nz[0] * u[2], nz[0] * u[1] - nz[1] * u[0]];
    const ctr = [0, 0, 0];
    for (const v of on) { ctr[0] += V[v][0]; ctr[1] += V[v][1]; ctr[2] += V[v][2]; }
    ctr[0] /= on.length; ctr[1] /= on.length; ctr[2] /= on.length;
    // angle autour du centre dans le repère (u, w) du plan : ordre cyclique, orienté vers
    // l'extérieur (w = normale x u, et la normale sort du polytope).
    const ang = q => {
      const dx = q[0] - ctr[0], dy = q[1] - ctr[1], dz = q[2] - ctr[2];
      return Math.atan2(dx * w[0] + dy * w[1] + dz * w[2], dx * u[0] + dy * u[1] + dz * u[2]);
    };
    on.sort((p1, p2) => ang(V[p1]) - ang(V[p2]));
    faces.push(on); fpl.push(p);
  }
  return { V, faces, fpl, vpl };
}

// même chose à plat : un sommet est l'intersection de 2 droites, et il n'y a qu'une face.
function polygon2D(A, b) {
  const n = A.length, eps = 1e-6 * Math.max(1, Math.max(...b.map(Math.abs)));
  const V = [], key = new Map();
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
    const det = A[i][0] * A[j][1] - A[i][1] * A[j][0];
    if (Math.abs(det) < 1e-9) continue;
    const x = (b[i] * A[j][1] - A[i][1] * b[j]) / det;
    const y = (A[i][0] * b[j] - b[i] * A[j][0]) / det;
    const p = [x, y, 0];
    if (!feasible(A, b, p, eps)) continue;
    const kk = p.map(v => Math.round(v / (eps * 10))).join(",");
    if (key.has(kk)) continue;
    key.set(kk, V.length); V.push(p);
  }
  if (V.length < 3) return null;
  const vpl = V.map(() => []);
  for (let p = 0; p < n; p++) for (let v = 0; v < V.length; v++)
    if (Math.abs(A[p][0] * V[v][0] + A[p][1] * V[v][1] - b[p]) < eps * 10) vpl[v].push(p);
  const ctr = V.reduce((a, q) => [a[0] + q[0] / V.length, a[1] + q[1] / V.length, 0], [0, 0, 0]);
  const idx = V.map((_, i) => i);
  idx.sort((p1, p2) => Math.atan2(V[p1][1] - ctr[1], V[p1][0] - ctr[0])
                     - Math.atan2(V[p2][1] - ctr[1], V[p2][0] - ctr[0]));
  return { V, faces: [idx], fpl: [-1], vpl };
}

// ============================================================================================
// construction de la scène rendue (positions 3D projetées + polytopes coupés)
// ============================================================================================
let TRI = { pos: null, nrm: null, col: null, n: 0 };
let LIN = { a: null, b: null, col: null, n: 0 };
let PNT = { pos: null, col: null, rad: null, n: 0 };

// Le vivier est projeté UNE fois par choix de dimensions vues, pas à chaque image : c'est le
// seul tableau qui porte toute la scène, alors que le reste d'une reconstruction ne coûte que
// l'image affichée.
let PR = new Float32Array(0);
function projectPool() {
  PR = new Float32Array(NB_V * 3);
  for (let i = 0; i < NB_V; i++) {
    PR[3 * i] = px(POS, i); PR[3 * i + 1] = py(POS, i); PR[3 * i + 2] = pz(POS, i);
  }
}

let frameIdx = 0;

function buildScene() {
  // 1. bornes de l'image affichée dans chaque liste de primitives
  const p0 = FR_POLY[frameIdx], p1 = FR_POLY[frameIdx + 1];
  const e0 = FR_EDG[frameIdx],  e1 = FR_EDG[frameIdx + 1];
  const v0 = FR_PNT[frameIdx],  v1 = FR_PNT[frameIdx + 1];
  const h0 = FR_HP[frameIdx],   h1 = FR_HP[frameIdx + 1];

  // 2. polytopes : coupe -> énumération -> triangles (éventail) + arêtes (dédoublonnées)
  const hp = [], hpc = [], hl = [], hlc = [];
  for (let hi = h0; hi < h1; hi++) {
    const h = HPOLY[hi];
    const s = sliceHalfspaces(h);
    if (!s) continue;                                   // coupe vide : rien à montrer
    const box = clipPlanes();
    const A = s.A.concat(box.A), b = Array.from(s.b).concat(box.b);
    const poly = AX[2] >= 0 ? polyhedron3D(A, b) : polygon2D(A, b);
    if (!poly) continue;
    // La boîte est un moyen de MONTRER un polytope ouvert, pas une partie de lui : ses plans
    // remplissent la face par où le polytope sort du champ (sans quoi on verrait dedans), mais
    // ils ne donnent aucune arête -- une arête posée sur la boîte, c'est la boîte qu'on dessine,
    // et elle réapparaîtrait seule dès qu'on décoche les faces.
    const nReal = s.A.length;
    const onBox = (i0, i1) => poly.vpl[i0].some(q => q >= nReal && poly.vpl[i1].indexOf(q) >= 0);
    const seen = new Set();
    for (const f of poly.faces) {
      for (let k = 1; k + 1 < f.length; k++) {
        for (const v of [f[0], f[k], f[k + 1]])
          hp.push(poly.V[v][0], poly.V[v][1], poly.V[v][2]);
        hpc.push(h.fcol);
      }
      if (!h.edg) continue;                             // l'appelant trace ses arêtes lui-même
      for (let k = 0; k < f.length; k++) {
        const i0 = f[k], i1 = f[(k + 1) % f.length];
        const kk = Math.min(i0, i1) + "," + Math.max(i0, i1);
        if (seen.has(kk)) continue;                     // une arête est portée par 2 faces
        seen.add(kk);
        if (onBox(i0, i1)) continue;
        hl.push(poly.V[i0][0], poly.V[i0][1], poly.V[i0][2],
                poly.V[i1][0], poly.V[i1][1], poly.V[i1][2]);
        hlc.push(h.ecol);
      }
    }
  }

  // 3. triangles. La normale est calculée APRÈS projection (elle n'a de sens que dans l'espace
  //    vu) -> ombrage plat, donc un sommet par coin de triangle : c'est un tampon d'affichage,
  //    pas un format de stockage, la dépense reste dans la mémoire du GPU.
  let nTri = hp.length / 9;
  for (let p = p0; p < p1; p++) nTri += Math.max(0, POLY_OFF[p + 1] - POLY_OFF[p] - 2);
  TRI = { pos: new Float32Array(nTri * 9), nrm: new Float32Array(nTri * 9),
          col: new Float32Array(nTri * 12), n: nTri };
  let t = 0;
  function putTri(a, b, c, col) {
    const o = t * 9;
    TRI.pos[o    ] = a[0]; TRI.pos[o + 1] = a[1]; TRI.pos[o + 2] = a[2];
    TRI.pos[o + 3] = b[0]; TRI.pos[o + 4] = b[1]; TRI.pos[o + 5] = b[2];
    TRI.pos[o + 6] = c[0]; TRI.pos[o + 7] = c[1]; TRI.pos[o + 8] = c[2];
    let nx = (b[1] - a[1]) * (c[2] - a[2]) - (b[2] - a[2]) * (c[1] - a[1]);
    let ny = (b[2] - a[2]) * (c[0] - a[0]) - (b[0] - a[0]) * (c[2] - a[2]);
    let nz = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
    const nl = Math.hypot(nx, ny, nz);
    if (nl < 1e-20) { nx = 0; ny = 0; nz = 1; } else { nx /= nl; ny /= nl; nz /= nl; }
    for (let k = 0; k < 3; k++) {
      TRI.nrm[o + k * 3] = nx; TRI.nrm[o + k * 3 + 1] = ny; TRI.nrm[o + k * 3 + 2] = nz;
      for (let q = 0; q < 4; q++) TRI.col[t * 12 + k * 4 + q] = col[q];
    }
    t++;
  }
  const at = i => [PR[3 * i], PR[3 * i + 1], PR[3 * i + 2]];
  for (let p = p0; p < p1; p++) {
    const o0 = POLY_OFF[p], o1 = POLY_OFF[p + 1], col = COLORS[POLY_C[p]];
    for (let k = o0 + 1; k + 1 < o1; k++)                 // éventail depuis le premier sommet
      putTri(at(POLY_V[o0]), at(POLY_V[k]), at(POLY_V[k + 1]), col);
  }
  for (let i = 0; i + 8 < hp.length; i += 9)
    putTri(hp.slice(i, i + 3), hp.slice(i + 3, i + 6), hp.slice(i + 6, i + 9), hpc[i / 9]);

  // 4. arêtes : une INSTANCE par arête (deux extrémités), élargie en quadrilatère dans le
  //    nuanceur -- gl.lineWidth est plafonné à 1 pixel sur la plupart des pilotes.
  const nLin = (e1 - e0) + hl.length / 6;
  LIN = { a: new Float32Array(nLin * 3), b: new Float32Array(nLin * 3),
          col: new Float32Array(nLin * 4), n: nLin };
  let e = 0;
  function putLin(a, b, col) {
    LIN.a[e * 3] = a[0]; LIN.a[e * 3 + 1] = a[1]; LIN.a[e * 3 + 2] = a[2];
    LIN.b[e * 3] = b[0]; LIN.b[e * 3 + 1] = b[1]; LIN.b[e * 3 + 2] = b[2];
    for (let q = 0; q < 4; q++) LIN.col[e * 4 + q] = col[q];
    e++;
  }
  for (let i = e0; i < e1; i++)
    putLin(at(EDG_V[2 * i]), at(EDG_V[2 * i + 1]), COLORS[EDG_C[i]]);
  for (let i = 0; i + 5 < hl.length; i += 6)
    putLin(hl.slice(i, i + 3), hl.slice(i + 3, i + 6), hlc[i / 6]);

  // 5. points
  const nPnt = v1 - v0;
  PNT = { pos: new Float32Array(nPnt * 3), col: new Float32Array(nPnt * 4),
          rad: PNT_R.subarray(v0, v1), n: nPnt };
  for (let i = 0; i < nPnt; i++) {
    const v = PNT_V[v0 + i], col = COLORS[PNT_C[v0 + i]];
    PNT.pos[i * 3] = PR[3 * v]; PNT.pos[i * 3 + 1] = PR[3 * v + 1]; PNT.pos[i * 3 + 2] = PR[3 * v + 2];
    for (let q = 0; q < 4; q++) PNT.col[i * 4 + q] = col[q];
  }

  upload();
  pivotDirty = true;                                    // la scène a changé : plus rien de visé
  document.getElementById("counts").textContent =
    PNT.n + " points · " + LIN.n + " arêtes · " + TRI.n + " triangles";
}

// ============================================================================================
// algèbre linéaire (juste ce dont la caméra a besoin)
// ============================================================================================
function mul(a, e) {                                   // a * e, matrices 4x4 en colonne-major
  const o = new Float32Array(16);
  for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) {
    let s = 0;
    for (let k = 0; k < 4; k++) s += a[k * 4 + r] * e[c * 4 + k];
    o[c * 4 + r] = s;
  }
  return o;
}
function perspective(fov, asp, n, f) {
  const t = 1 / Math.tan(fov / 2), o = new Float32Array(16);
  o[0] = t / asp; o[5] = t; o[10] = (f + n) / (n - f); o[11] = -1; o[14] = 2 * f * n / (n - f);
  return o;
}
function ortho(hw, hh, n, f) {
  const o = new Float32Array(16);
  o[0] = 1 / hw; o[5] = 1 / hh; o[10] = 2 / (n - f); o[14] = (f + n) / (n - f); o[15] = 1;
  return o;
}

// ============================================================================================
// caméra
// ============================================================================================
const FLAT = D <= 2;                                   // 2D : à plat, pas d'orbite
const FOV = 45 * Math.PI / 180;

// L'orientation de la caméra est un QUATERNION, pas un couple (azimut, élévation). Il n'y a ici
// aucun plan de référence -- pas de sol, pas de « haut » du monde : deux angles d'Euler en
// imposeraient un, et le même geste ne produirait alors pas le même mouvement selon l'orientation
// de départ (écrasement près des pôles, butée d'élévation, roulis impossible). Avec un quaternion,
// un glisser applique toujours la MÊME rotation, exprimée dans le repère de l'ÉCRAN.
const cam = { target: [0, 0, 0], dist: 3, rot: [0, 0, 0, 1], pivotZ: 3, ortho: FLAT };

function qMul(a, b) {                                  // produit de Hamilton : `b` d'abord, puis `a`
  return [a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
          a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
          a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
          a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2]];
}
function qAxis(ax, ang) {                              // rotation d'angle `ang` autour de `ax`
  const h = 0.5 * ang, s = Math.sin(h);
  return [ax[0] * s, ax[1] * s, ax[2] * s, Math.cos(h)];
}
function qNorm(q) {                                    // les arrondis dérivent : renormaliser
  const l = Math.hypot(q[0], q[1], q[2], q[3]) || 1;
  return [q[0] / l, q[1] / l, q[2] / l, q[3] / l];
}
function qApply(q, v) {                                // v tourné par q
  const t = [2 * (q[1] * v[2] - q[2] * v[1]),
             2 * (q[2] * v[0] - q[0] * v[2]),
             2 * (q[0] * v[1] - q[1] * v[0])];
  return [v[0] + q[3] * t[0] + q[1] * t[2] - q[2] * t[1],
          v[1] + q[3] * t[1] + q[2] * t[0] - q[0] * t[2],
          v[2] + q[3] * t[2] + q[0] * t[1] - q[1] * t[0]];
}

function sceneSphere() {                               // centre + rayon des dimensions VUES
  const c = [0, 0, 0]; let r = 0;
  for (let a = 0; a < 3; a++) {
    const k = AX[a];
    const lo = k >= 0 ? BOUNDS[k][0] : 0, hi = k >= 0 ? BOUNDS[k][1] : 0;
    c[a] = 0.5 * (lo + hi); r += 0.25 * (hi - lo) * (hi - lo);
  }
  return { c, r: Math.max(Math.sqrt(r), 1e-6) };
}
function resetView() {
  const s = sceneSphere();
  cam.target = s.c;
  // le champ de vision est VERTICAL : sur une fenêtre plus haute que large, s'ajuster dessus
  // laisserait la scène déborder sur les côtés -- d'où le recul en 1/rapport.
  const asp = canvas.width / Math.max(canvas.height, 1);
  cam.dist = s.r / Math.sin(FOV / 2) * 1.05 / Math.min(1, asp);
  // vue de trois quarts par défaut -- une orientation de départ comme une autre, sans statut
  // particulier : rien dans la suite ne la reprend comme référence.
  cam.rot = FLAT ? [0, 0, 0, 1]
                 : qNorm(qMul(qAxis([0, 1, 0], 0.6), qAxis([1, 0, 0], -0.35)));
  cam.pivotZ = cam.dist; pivotDirty = true;
  draw();
}
function camEye() {
  const b = qApply(cam.rot, [0, 0, 1]);                // l'axe qui « sort » de l'écran
  return [cam.target[0] + cam.dist * b[0],
          cam.target[1] + cam.dist * b[1],
          cam.target[2] + cam.dist * b[2]];
}
function camBasis() {                                  // droite / haut de l'écran, en monde
  return { right: qApply(cam.rot, [1, 0, 0]),
           up:    qApply(cam.rot, [0, 1, 0]),
           fwd:   qApply(cam.rot, [0, 0, -1]) };
}
function viewMatrix() {                                // monde -> caméra, tirée directement de `rot`
  // pas de `lookAt` : il lui faudrait un « haut » du monde, exactement l'hypothèse qu'on refuse
  // (et il dégénérerait quand la caméra le regarde en face).
  const b = camBasis(), e = camEye(), r = b.right, u = b.up, f = b.fwd;
  return new Float32Array([
    r[0], u[0], -f[0], 0,
    r[1], u[1], -f[1], 0,
    r[2], u[2], -f[2], 0,
    -(r[0] * e[0] + r[1] * e[1] + r[2] * e[2]),
    -(u[0] * e[0] + u[1] * e[1] + u[2] * e[2]),
      f[0] * e[0] + f[1] * e[1] + f[2] * e[2], 1,
  ]);
}
function orthoHalfH() { return cam.dist * Math.tan(FOV / 2); }

// ============================================================================================
// visée : quel point de la scène se trouve sous un pixel
// ============================================================================================
// Le zoom vers le curseur et le pivot de rotation ont besoin d'un point 3D, pas d'un pixel. Le
// lire dans le tampon de profondeur est hors de portée en WebGL2 (`readPixels` ne sait relire
// qu'une couleur), et une passe de rendu supplémentaire pour ça coûterait cher. On lance donc un
// rayon dans les primitives de l'image affichée, qui sont DÉJÀ là côté CPU (`TRI`/`LIN`/`PNT`) :
// coût linéaire, mais payé une fois par GESTE, jamais par image rendue.

function aimPlane(sx, sy) {                            // point visé, dans le plan de la cible
  const b = camBasis(), hh = orthoHalfH();
  const w = Math.max(canvas.clientWidth, 1), h = Math.max(canvas.clientHeight, 1);
  const u = (2 * sx / w - 1) * hh * (w / h), v = (1 - 2 * sy / h) * hh;
  return [cam.target[0] + u * b.right[0] + v * b.up[0],
          cam.target[1] + u * b.right[1] + v * b.up[1],
          cam.target[2] + u * b.right[2] + v * b.up[2]];
}
function screenRay(sx, sy) {
  const b = camBasis(), e = camEye(), p = aimPlane(sx, sy);
  if (cam.ortho)                                       // rayons parallèles : c'est l'origine qui bouge
    return { o: [p[0] - cam.dist * b.fwd[0], p[1] - cam.dist * b.fwd[1],
                 p[2] - cam.dist * b.fwd[2]], d: b.fwd };
  const d = [p[0] - e[0], p[1] - e[1], p[2] - e[2]];
  const l = Math.hypot(d[0], d[1], d[2]) || 1;
  return { o: e, d: [d[0] / l, d[1] / l, d[2] / l] };
}

const AIM_PX = 18;                                     // tolérance de visée, en pixels

function pickWorld(sx, sy) {
  const r = screenRay(sx, sy), o = r.o, d = r.d;
  const h = Math.max(canvas.clientHeight, 1);
  // largeur d'un pixel, en monde, à la distance `t` le long du rayon
  const wpx = cam.ortho ? () => 2 * orthoHalfH() / h
                        : t => 2 * Math.tan(FOV / 2) * t / h;

  // 1. les faces d'abord : intersection EXACTE (Möller-Trumbore). C'est ce que l'oeil voit.
  let best = Infinity;
  const P = TRI.pos;
  for (let i = 0; i < TRI.n; i++) {
    const j = i * 9;
    const ax = P[j], ay = P[j + 1], az = P[j + 2];
    const e1x = P[j + 3] - ax, e1y = P[j + 4] - ay, e1z = P[j + 5] - az;
    const e2x = P[j + 6] - ax, e2y = P[j + 7] - ay, e2z = P[j + 8] - az;
    const hx = d[1] * e2z - d[2] * e2y, hy = d[2] * e2x - d[0] * e2z, hz = d[0] * e2y - d[1] * e2x;
    const det = e1x * hx + e1y * hy + e1z * hz;
    if (det > -1e-12 && det < 1e-12) continue;         // rayon parallèle au plan du triangle
    const f = 1 / det, wx = o[0] - ax, wy = o[1] - ay, wz = o[2] - az;
    const u = f * (wx * hx + wy * hy + wz * hz);
    if (u < 0 || u > 1) continue;
    const qx = wy * e1z - wz * e1y, qy = wz * e1x - wx * e1z, qz = wx * e1y - wy * e1x;
    const v = f * (d[0] * qx + d[1] * qy + d[2] * qz);
    if (v < 0 || u + v > 1) continue;
    const t = f * (e2x * qx + e2y * qy + e2z * qz);
    if (t > 1e-9 && t < best) best = t;
  }

  // 2. sinon arêtes et points : rien de PLEIN à percer (fil de fer, nuage), on accepte donc ce
  //    qui passe assez près du rayon -- tolérance en pixels, pour qu'elle ne dépende pas du zoom.
  if (best === Infinity) {
    const A = LIN.a, B = LIN.b;
    for (let i = 0; i < LIN.n; i++) {
      const k = i * 3;
      const vx = B[k] - A[k], vy = B[k + 1] - A[k + 1], vz = B[k + 2] - A[k + 2];
      const wx = o[0] - A[k], wy = o[1] - A[k + 1], wz = o[2] - A[k + 2];
      const bb = d[0] * vx + d[1] * vy + d[2] * vz, cc = vx * vx + vy * vy + vz * vz;
      const dd = d[0] * wx + d[1] * wy + d[2] * wz, ee = vx * wx + vy * wy + vz * wz;
      const den = cc - bb * bb;                        // (|d| = 1) ; nul <=> rayon et arête parallèles
      let s = Math.abs(den) < 1e-12 ? 0 : (ee - bb * dd) / den;
      s = Math.max(0, Math.min(1, s));
      const px2 = A[k] + s * vx, py2 = A[k + 1] + s * vy, pz2 = A[k + 2] + s * vz;
      const t = (px2 - o[0]) * d[0] + (py2 - o[1]) * d[1] + (pz2 - o[2]) * d[2];
      if (t <= 1e-9 || t >= best) continue;
      if (Math.hypot(px2 - o[0] - t * d[0], py2 - o[1] - t * d[1], pz2 - o[2] - t * d[2])
          < AIM_PX * wpx(t)) best = t;
    }
    const Q = PNT.pos, rad = PNT.rad;
    for (let i = 0; i < PNT.n; i++) {
      const k = i * 3;
      const t = (Q[k] - o[0]) * d[0] + (Q[k + 1] - o[1]) * d[1] + (Q[k + 2] - o[2]) * d[2];
      if (t <= 1e-9 || t >= best) continue;
      if (Math.hypot(Q[k] - o[0] - t * d[0], Q[k + 1] - o[1] - t * d[1], Q[k + 2] - o[2] - t * d[2])
          < rad[i] + AIM_PX * wpx(t)) best = t;
    }
  }
  return best === Infinity ? null
                           : [o[0] + best * d[0], o[1] + best * d[1], o[2] + best * d[2]];
}

function depthAt(sx, sy) {                             // profondeur du point visé, le long de fwd
  const c = pickWorld(sx, sy);
  if (c === null) return cam.dist;                     // fond : faute de mieux, le plan de la cible
  const e = camEye(), f = camBasis().fwd;
  return Math.max(1e-6, (c[0] - e[0]) * f[0] + (c[1] - e[1]) * f[1] + (c[2] - e[2]) * f[2]);
}
function aimAt(sx, sy, z) {                            // le point du rayon situé à la profondeur z
  const r = screenRay(sx, sy), f = camBasis().fwd;
  const c = r.d[0] * f[0] + r.d[1] * f[1] + r.d[2] * f[2];   // 1 en orthographique
  const e = camEye(), o = r.o;
  const t = (z - ((o[0] - e[0]) * f[0] + (o[1] - e[1]) * f[1] + (o[2] - e[2]) * f[2])) / c;
  return [o[0] + t * r.d[0], o[1] + t * r.d[1], o[2] + t * r.d[2]];
}

// Profondeur du pivot de rotation : celle de ce que la scène montre AU CENTRE DE L'ÉCRAN. Elle
// n'est recalculée que si la vue a bougé AUTREMENT que par une rotation -- une rotation ne doit
// surtout pas redéfinir son propre pivot en cours de geste, il dériverait sous la main.
let pivotDirty = true;
function pivotDepth() {
  if (pivotDirty) {
    pivotDirty = false;
    cam.pivotZ = depthAt(canvas.clientWidth / 2, canvas.clientHeight / 2);
  }
  return cam.pivotZ;
}

// ============================================================================================
// WebGL
// ============================================================================================
const canvas = document.getElementById("c");
const gl = canvas.getContext("webgl2", { antialias: true, alpha: false });
if (!gl) document.body.innerHTML = "<p style='font-family:sans-serif;padding:20px'>WebGL2 "
  + "indisponible dans ce navigateur.</p>";

function prog(vs, fs) {
  function sh(t, src) {
    const s = gl.createShader(t); gl.shaderSource(s, src); gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw gl.getShaderInfoLog(s);
    return s;
  }
  const p = gl.createProgram();
  gl.attachShader(p, sh(gl.VERTEX_SHADER, vs)); gl.attachShader(p, sh(gl.FRAGMENT_SHADER, fs));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw gl.getProgramInfoLog(p);
  return p;
}

const pFace = prog(`#version 300 es
in vec3 aPos; in vec3 aNrm; in vec4 aCol;
uniform mat4 uMVP; uniform mat4 uMV;
out vec4 vCol; out vec3 vN;
void main() { gl_Position = uMVP * vec4(aPos, 1.0); vN = mat3(uMV) * aNrm; vCol = aCol; }`,
`#version 300 es
precision highp float;
in vec4 vCol; in vec3 vN; out vec4 oCol;
uniform float uAlpha;
void main() {
  // éclairage frontal, DOUBLE FACE (abs) : l'intérieur d'une cellule ouverte reste lisible.
  vec3 n = normalize(vN);
  float d = abs(dot(n, normalize(vec3(0.25, 0.45, 1.0))));
  oCol = vec4(vCol.rgb * (0.45 + 0.55 * d), vCol.a * uAlpha);
}`);

const pLine = prog(`#version 300 es
in vec2 aCorner; in vec3 aA; in vec3 aB; in vec4 aCol;
uniform mat4 uMVP; uniform vec2 uHalfVP; uniform float uW;
out vec4 vCol;
void main() {
  // une arête = une INSTANCE élargie en quadrilatère à l'ÉCRAN (gl.lineWidth ne dépasse pas 1px
  // sur la plupart des pilotes).
  vec4 ca = uMVP * vec4(aA, 1.0), cb = uMVP * vec4(aB, 1.0);
  vec2 sa = ca.xy / max(ca.w, 1e-5) * uHalfVP, sb = cb.xy / max(cb.w, 1e-5) * uHalfVP;
  vec2 dir = sb - sa;
  float l = length(dir);
  dir = l > 1e-6 ? dir / l : vec2(1.0, 0.0);
  vec4 c = mix(ca, cb, aCorner.x);
  c.xy += vec2(-dir.y, dir.x) * (uW * 0.5 * aCorner.y) / uHalfVP * max(c.w, 1e-5);
  gl_Position = c; vCol = aCol;
}`,
`#version 300 es
precision highp float;
in vec4 vCol; out vec4 oCol; uniform float uAlpha;
void main() { oCol = vec4(vCol.rgb, vCol.a * uAlpha); }`);

const pPoint = prog(`#version 300 es
in vec3 aPos; in vec4 aCol; in float aRad;
uniform mat4 uMVP; uniform mat4 uMV; uniform float uSizeK; uniform float uOrtho; uniform float uR;
out vec4 vCol;
void main() {
  gl_Position = uMVP * vec4(aPos, 1.0);
  // un rayon nul = un point sans taille propre (un dirac) : c'est le curseur qui le dimensionne.
  float r = aRad > 0.0 ? aRad : uR;
  float zv = -(uMV * vec4(aPos, 1.0)).z;
  gl_PointSize = clamp(uSizeK * r / (uOrtho > 0.5 ? 1.0 : max(zv, 1e-4)), 1.0, 1024.0);
  vCol = aCol;
}`,
`#version 300 es
precision highp float;
in vec4 vCol; out vec4 oCol;
void main() {
  vec2 d = gl_PointCoord * 2.0 - 1.0;
  float q = dot(d, d);
  if (q > 1.0) discard;                                // disque, pas carré
  float l = 0.55 + 0.45 * sqrt(max(0.0, 1.0 - q));     // ombrage de sphère
  oCol = vec4(vCol.rgb * l, vCol.a);
}`);

const buf = {};
["triPos", "triNrm", "triCol", "linA", "linB", "linCol", "linCorner", "pntPos", "pntCol", "pntRad"]
  .forEach(k => buf[k] = gl.createBuffer());

gl.bindBuffer(gl.ARRAY_BUFFER, buf.linCorner);
gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([0, -1, 1, -1, 1, 1, 0, -1, 1, 1, 0, 1]), gl.STATIC_DRAW);

const vaoFace = gl.createVertexArray(), vaoLine = gl.createVertexArray(), vaoPoint = gl.createVertexArray();

function attr(p, name, b, size, divisor) {
  const loc = gl.getAttribLocation(p, name);
  if (loc < 0) return;
  gl.bindBuffer(gl.ARRAY_BUFFER, b);
  gl.enableVertexAttribArray(loc);
  gl.vertexAttribPointer(loc, size, gl.FLOAT, false, 0, 0);
  if (divisor) gl.vertexAttribDivisor(loc, divisor);
}
gl.bindVertexArray(vaoFace);
attr(pFace, "aPos", buf.triPos, 3); attr(pFace, "aNrm", buf.triNrm, 3); attr(pFace, "aCol", buf.triCol, 4);
gl.bindVertexArray(vaoLine);
attr(pLine, "aCorner", buf.linCorner, 2);
attr(pLine, "aA", buf.linA, 3, 1); attr(pLine, "aB", buf.linB, 3, 1); attr(pLine, "aCol", buf.linCol, 4, 1);
gl.bindVertexArray(vaoPoint);
attr(pPoint, "aPos", buf.pntPos, 3); attr(pPoint, "aCol", buf.pntCol, 4); attr(pPoint, "aRad", buf.pntRad, 1);
gl.bindVertexArray(null);

function up(b, data) { gl.bindBuffer(gl.ARRAY_BUFFER, b); gl.bufferData(gl.ARRAY_BUFFER, data, gl.DYNAMIC_DRAW); }
function upload() {
  up(buf.triPos, TRI.pos); up(buf.triNrm, TRI.nrm); up(buf.triCol, TRI.col);
  up(buf.linA, LIN.a); up(buf.linB, LIN.b); up(buf.linCol, LIN.col);
  up(buf.pntPos, PNT.pos); up(buf.pntCol, PNT.col); up(buf.pntRad, PNT.rad);
}

// ============================================================================================
// dessin
// ============================================================================================
const ui = {
  faces: document.getElementById("cbFaces"), points: document.getElementById("cbPoints"),
  edgeMode: document.getElementById("edgeMode"), op: document.getElementById("op"),
  r: document.getElementById("r"),
};
function pointRadius() { return RMIN * Math.pow(RMAX / RMIN, parseFloat(ui.r.value)); }

function draw() {
  const w = canvas.width, h = canvas.height;
  const dark = document.body.classList.contains("dark");
  gl.viewport(0, 0, w, h);
  const bg = BG !== null ? BG : (dark ? [0.10, 0.10, 0.10] : [1, 1, 1]);
  gl.clearColor(bg[0], bg[1], bg[2], 1);
  gl.clearDepth(1);
  gl.enable(gl.DEPTH_TEST);
  gl.depthMask(true);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

  const s = sceneSphere();
  const near = Math.max(cam.dist - 4 * s.r, 1e-3 * s.r), far = cam.dist + 4 * s.r;
  const asp = w / Math.max(h, 1);
  const P = cam.ortho ? ortho(orthoHalfH() * asp, orthoHalfH(), -far, far)
                      : perspective(FOV, asp, near, far);
  const V = viewMatrix();
  const MVP = mul(P, V);
  const alpha = parseFloat(ui.op.value);
  const showFaces = ui.faces.checked, mode = ui.edgeMode.value;

  gl.enable(gl.BLEND);
  gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

  // (1) passe de PROFONDEUR PURE sur les faces : c'est elle qui rend possible l'élimination des
  // arêtes cachées, y compris quand les faces ne sont pas affichées ou sont transparentes.
  const needDepth = TRI.n > 0 && (mode === "hide" || mode === "ghost" || showFaces);
  gl.enable(gl.POLYGON_OFFSET_FILL);
  gl.polygonOffset(1.0, 1.0);
  if (needDepth) {
    gl.useProgram(pFace); gl.bindVertexArray(vaoFace);
    gl.uniformMatrix4fv(gl.getUniformLocation(pFace, "uMVP"), false, MVP);
    gl.uniformMatrix4fv(gl.getUniformLocation(pFace, "uMV"), false, V);
    gl.uniform1f(gl.getUniformLocation(pFace, "uAlpha"), 1);
    gl.colorMask(false, false, false, false);
    gl.depthMask(true);
    gl.drawArrays(gl.TRIANGLES, 0, TRI.n * 3);
    gl.colorMask(true, true, true, true);
  }

  // (2) faces visibles -- la profondeur est déjà écrite, on ne fait que colorier ce qui est devant
  if (showFaces && TRI.n > 0) {
    gl.useProgram(pFace); gl.bindVertexArray(vaoFace);
    gl.uniformMatrix4fv(gl.getUniformLocation(pFace, "uMVP"), false, MVP);
    gl.uniformMatrix4fv(gl.getUniformLocation(pFace, "uMV"), false, V);
    gl.uniform1f(gl.getUniformLocation(pFace, "uAlpha"), alpha);
    gl.depthFunc(gl.LEQUAL);
    gl.depthMask(!needDepth);
    gl.drawArrays(gl.TRIANGLES, 0, TRI.n * 3);
    gl.depthMask(true);
  }

  gl.disable(gl.POLYGON_OFFSET_FILL);

  // (3) arêtes. "ghost" = deux passes : d'abord TOUTES, sans test de profondeur et très pâles
  // (les cachées), puis les seules visibles en pleine couleur par-dessus.
  if (mode !== "none" && LIN.n > 0) {
    gl.useProgram(pLine); gl.bindVertexArray(vaoLine);
    gl.uniformMatrix4fv(gl.getUniformLocation(pLine, "uMVP"), false, MVP);
    gl.uniform2f(gl.getUniformLocation(pLine, "uHalfVP"), w / 2, h / 2);
    gl.uniform1f(gl.getUniformLocation(pLine, "uW"), 1.6 * (window.devicePixelRatio || 1));
    gl.depthMask(false);
    if (mode === "ghost" || mode === "all") {
      gl.disable(gl.DEPTH_TEST);
      gl.uniform1f(gl.getUniformLocation(pLine, "uAlpha"), mode === "all" ? 1 : 0.22);
      gl.drawArraysInstanced(gl.TRIANGLES, 0, 6, LIN.n);
      gl.enable(gl.DEPTH_TEST);
    }
    if (mode !== "all") {
      gl.depthFunc(gl.LEQUAL);
      gl.uniform1f(gl.getUniformLocation(pLine, "uAlpha"), 1);
      gl.drawArraysInstanced(gl.TRIANGLES, 0, 6, LIN.n);
    }
    gl.depthMask(true);
  }

  // (4) points
  if (ui.points.checked && PNT.n > 0) {
    gl.useProgram(pPoint); gl.bindVertexArray(vaoPoint);
    gl.uniformMatrix4fv(gl.getUniformLocation(pPoint, "uMVP"), false, MVP);
    gl.uniformMatrix4fv(gl.getUniformLocation(pPoint, "uMV"), false, V);
    gl.uniform1f(gl.getUniformLocation(pPoint, "uOrtho"), cam.ortho ? 1 : 0);
    gl.uniform1f(gl.getUniformLocation(pPoint, "uSizeK"),
      cam.ortho ? (h / 2) / orthoHalfH() : (h / 2) / Math.tan(FOV / 2));
    gl.uniform1f(gl.getUniformLocation(pPoint, "uR"), pointRadius());
    gl.depthFunc(gl.LEQUAL);
    gl.drawArrays(gl.POINTS, 0, PNT.n);
  }
  gl.bindVertexArray(null);
}

function resize() {
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(window.innerWidth * dpr);
  canvas.height = Math.round(window.innerHeight * dpr);
  canvas.style.width = window.innerWidth + "px";
  canvas.style.height = window.innerHeight + "px";
  draw();
}

// ============================================================================================
// interactions
// ============================================================================================
// La molette part et revient : on ne veut pas re-viser à chaque cran. La profondeur saisie n'est
// reprise que si le curseur a bougé, ou après une pause -- entre-temps le geste garde la sienne.
let wheelAim = { x: 1e9, y: 1e9, z: 0, t: -1e9 };
function wheelDepth(sx, sy, now) {
  if (Math.hypot(sx - wheelAim.x, sy - wheelAim.y) > 8 || now - wheelAim.t > 250)
    wheelAim.z = depthAt(sx, sy);
  wheelAim.x = sx; wheelAim.y = sy; wheelAim.t = now;
  return wheelAim.z;
}
function panBy(dxPix, dyPix, z) {
  const b = camBasis();
  // pixels -> monde à la profondeur `z` de ce qu'on DÉPLACE (celle du point saisi), et non à
  // celle de la cible : en perspective les deux diffèrent, et c'est la première qui fait que le
  // décor suit exactement le curseur. En orthographique l'échelle ne dépend pas de la profondeur.
  const d = cam.ortho ? cam.dist : (z === undefined ? pivotDepth() : z);
  const k = 2 * d * Math.tan(FOV / 2) / Math.max(canvas.clientHeight, 1);
  for (let i = 0; i < 3; i++)
    cam.target[i] += -dxPix * k * b.right[i] + dyPix * k * b.up[i];
  pivotDirty = true;
}
function zoomBy(f, sx, sy, z) {
  // La visée passe AVANT le changement de `dist` : l'oeil en dépend, donc viser après lancerait
  // le rayon depuis une caméra qui a déjà bougé.
  const p = sx === undefined ? null : aimAt(sx, sy, z === undefined ? depthAt(sx, sy) : z);
  const d0 = cam.dist;
  cam.dist = Math.max(1e-6, Math.min(cam.dist * f, 1e9));
  const g = cam.dist / d0;                             // facteur RÉELLEMENT appliqué (bornes)
  // Le point visé reste immobile : oeil et cible se rapprochent de lui du même facteur `g`.
  // C'est exact dans les DEUX projections, parce que la position à l'écran d'un point ne dépend
  // de la cible que par l'écart `p - cible`, lequel est ici multiplié par `g` tout comme l'est
  // la demi-hauteur de la vue. Rien à viser (fond) : `depthAt` rend le plan de la cible, et le
  // zoom retombe alors sur le comportement usuel, centré sur le curseur.
  if (p) for (let i = 0; i < 3; i++) cam.target[i] = p[i] + g * (cam.target[i] - p[i]);
  // tout ce qui restait visible s'est rapproché du même facteur ; le pivot, lui, est à revoir --
  // un zoom vers le curseur fait GLISSER la scène, donc change ce qui occupe le centre.
  wheelAim.z *= g;
  pivotDirty = true;
  return g;
}
function zoomAtCenter(f) {                             // clavier : pas de curseur, on vise le centre
  zoomBy(f, canvas.clientWidth / 2, canvas.clientHeight / 2, pivotDepth());
}
function orbitBy(dx, dy) {
  if (FLAT) { panBy(dx, dy); return; }                 // en 2D il n'y a rien à faire tourner
  // Le pivot est au CENTRE DE L'ÉCRAN, à la profondeur de ce qui s'y trouve -- pas à celle de la
  // cible, qui ne veut plus rien dire après un déplacement latéral. `cam.dist` n'est PAS touché :
  // il porte le niveau de zoom en orthographique, et le repivotage doit rester invisible.
  const z = pivotDepth(), e = camEye(), f = camBasis().fwd;
  const pv = [e[0] + z * f[0], e[1] + z * f[1], e[2] + z * f[2]];

  // La rotation est exprimée dans le repère de l'ÉCRAN, donc POST-multipliée : `dx` tourne autour
  // du haut de l'écran, `dy` autour de sa droite -- quelle que soit l'orientation courante. D'où
  // l'absence de butée : il n'y a pas de pôle, et le roulis apparaît naturellement le long d'un
  // geste courbe (deux rotations d'axes différents ne commutent pas), comme sur une vraie boule.
  const k = 0.008;
  cam.rot = qNorm(qMul(cam.rot,
                       qMul(qAxis([0, 1, 0], -dx * k), qAxis([1, 0, 0], -dy * k))));

  // la cible se replace pour que le PIVOT, lui, ne bouge pas : l'oeil finit à la distance `z` de
  // `pv`, toujours dans son axe -- donc `pv` reste au centre de l'écran, à la même profondeur.
  const g = camBasis().fwd, s = cam.dist - z;
  cam.target = [pv[0] + s * g[0], pv[1] + s * g[1], pv[2] + s * g[2]];
}

// Coordonnées d'un évènement DANS le canevas -- il occupe la fenêtre, mais on ne le suppose pas.
function evPos(e) { const r = canvas.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; }

let drag = null;
canvas.style.cursor = FLAT ? "grab" : "move";
canvas.addEventListener("mousedown", e => {
  const pan = e.shiftKey || e.button === 1 || e.button === 2;
  pivotDirty = true;                                   // nouveau geste : le pivot est à revoir
  // La profondeur du déplacement est celle du point SAISI, figée pour tout le geste : la
  // recalculer à chaque mouvement ferait sauter la vitesse dès que le curseur franchit un bord.
  drag = { x: e.clientX, y: e.clientY, pan, z: pan ? depthAt(...evPos(e)) : 0 };
});
window.addEventListener("mousemove", e => {
  if (!drag) return;
  const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
  drag.x = e.clientX; drag.y = e.clientY;
  if (drag.pan) panBy(dx, dy, drag.z); else orbitBy(dx, dy);
  draw();
});
window.addEventListener("mouseup", () => { drag = null; });
canvas.addEventListener("contextmenu", e => e.preventDefault());
canvas.addEventListener("dblclick", resetView);

// molette : le pincement d'un pavé tactile arrive comme un `wheel` avec ctrlKey (convention
// partagée avec Ctrl+molette d'une souris) -> zoom ; sinon défilement = déplacement.
canvas.addEventListener("wheel", e => {
  e.preventDefault();
  const [sx, sy] = evPos(e);
  if (e.ctrlKey || e.metaKey) zoomBy(Math.exp(e.deltaY * 0.01), sx, sy);
  else panBy(-e.deltaX, -e.deltaY, wheelDepth(sx, sy, e.timeStamp));
  draw();
}, { passive: false });

let touch = null;
canvas.addEventListener("touchstart", e => {
  e.preventDefault();
  pivotDirty = true;                                   // nouveau geste : le pivot est à revoir
  if (e.touches.length === 1) touch = { mode: "rot", x: e.touches[0].clientX, y: e.touches[0].clientY };
  else if (e.touches.length === 2) {
    const [a, b] = e.touches;
    const cx = (a.clientX + b.clientX) / 2, cy = (a.clientY + b.clientY) / 2;
    const r = canvas.getBoundingClientRect();
    // profondeur du point pincé, figée pour le geste -- comme au déplacement à la souris
    touch = { mode: "pinch", d: Math.hypot(b.clientX - a.clientX, b.clientY - a.clientY),
              x: cx, y: cy, z: depthAt(cx - r.left, cy - r.top) };
  }
}, { passive: false });
canvas.addEventListener("touchmove", e => {
  e.preventDefault();
  if (!touch) return;
  if (touch.mode === "rot" && e.touches.length === 1) {
    orbitBy(e.touches[0].clientX - touch.x, e.touches[0].clientY - touch.y);
    touch.x = e.touches[0].clientX; touch.y = e.touches[0].clientY;
  } else if (touch.mode === "pinch" && e.touches.length === 2) {
    const [a, b] = e.touches;
    const d = Math.hypot(b.clientX - a.clientX, b.clientY - a.clientY);
    const mx = (a.clientX + b.clientX) / 2, my = (a.clientY + b.clientY) / 2;
    const r = canvas.getBoundingClientRect();
    // le pincement zoome sur ce qu'il TIENT, puis déplace à la profondeur de ce même point --
    // dont on s'est rapproché du facteur que `zoomBy` a réellement appliqué.
    touch.z *= zoomBy(touch.d / Math.max(d, 1), mx - r.left, my - r.top, touch.z);
    panBy(mx - touch.x, my - touch.y, touch.z);
    touch.d = d; touch.x = mx; touch.y = my;
  }
  draw();
}, { passive: false });
canvas.addEventListener("touchend", e => { if (e.touches.length === 0) touch = null; });

function pick(e, a, b, c) { return (e.ctrlKey || e.metaKey) ? c : (e.shiftKey ? b : a); }
const helpPanel = document.getElementById("help");
window.addEventListener("keydown", e => {
  if (e.key === "?") { e.preventDefault(); helpPanel.style.display = helpPanel.style.display === "block" ? "none" : "block"; return; }
  if (e.key === "Escape") { helpPanel.style.display = "none"; return; }
  // Espace passe AVANT le renvoi ci-dessous : la barre de temps garde le focus (pour que les
  // flèches la pilotent nativement), et la lecture doit rester accessible dans cet état.
  if (e.key === " " && NB_FRAMES > 1) { e.preventDefault(); togglePlay(); return; }
  if (e.target.tagName === "SELECT" || e.target.tagName === "INPUT") return;
  const s = pick(e, 12, 40, 90);
  switch (e.key) {
    case "ArrowLeft":  e.preventDefault(); orbitBy(-s, 0); break;
    case "ArrowRight": e.preventDefault(); orbitBy( s, 0); break;
    case "ArrowUp":    e.preventDefault(); orbitBy(0, -s); break;
    case "ArrowDown":  e.preventDefault(); orbitBy(0,  s); break;
    case "+": case "=": e.preventDefault(); zoomAtCenter(1 / Math.pow(1.2, pick(e, 1, 2, 5))); break;
    case "-": case "_": e.preventDefault(); zoomAtCenter(Math.pow(1.2, pick(e, 1, 2, 5))); break;
    case "0": e.preventDefault(); resetView(); return;
    case "[": case "]":
      e.preventDefault();
      ui.r.value = Math.max(0, Math.min(1, parseFloat(ui.r.value) + (e.key === "]" ? 0.03 : -0.03)));
      syncLabels(); break;
    case "f": case "F": ui.faces.checked = !ui.faces.checked; break;
    case "p": case "P": ui.points.checked = !ui.points.checked; break;
    case "e": case "E": {
      const opts = ["hide", "ghost", "all", "none"];
      ui.edgeMode.value = opts[(opts.indexOf(ui.edgeMode.value) + 1) % opts.length];
      break;
    }
    case "o": case "O": cam.ortho = !cam.ortho; break;
    default: return;
  }
  draw();
});

// ============================================================================================
// barre de temps / de paramètre
// ============================================================================================
const tSlider = document.getElementById("t"), tLabel = document.getElementById("tval");
const tName = document.getElementById("tname"), playBtn = document.getElementById("play");
let playing = false, playTimer = null;

function setFrame(i) {
  frameIdx = Math.max(0, Math.min(NB_FRAMES - 1, i));
  tSlider.value = frameIdx;
  tLabel.textContent = (frameIdx + 1) + " / " + NB_FRAMES;
  tName.textContent = AXIS + " = " + FR_VAL[frameIdx].toPrecision(4);
  buildScene();                                       // seule l'image affichée est reconstruite
  draw();
}
function stopPlaying() {
  playing = false;
  playBtn.innerHTML = "&#9654;";
  clearInterval(playTimer);
}
function togglePlay() {
  if (!PLAYABLE || NB_FRAMES < 2) return;
  if (playing) { stopPlaying(); return; }
  playing = true;
  playBtn.innerHTML = "&#10074;&#10074;";
  if (frameIdx >= NB_FRAMES - 1) setFrame(0);
  playTimer = setInterval(() => {
    if (frameIdx >= NB_FRAMES - 1) { stopPlaying(); return; }
    setFrame(frameIdx + 1);
  }, 1000 / FPS);
}
if (NB_FRAMES > 1) {
  document.getElementById("timeControls").style.display = "block";
  tSlider.max = NB_FRAMES - 1;
  tSlider.addEventListener("input", () => { stopPlaying(); setFrame(parseInt(tSlider.value, 10)); });
  if (PLAYABLE) playBtn.addEventListener("click", togglePlay);
  else playBtn.style.display = "none";
}

// ============================================================================================
// panneau : opacité, rayon, dimensions vues, coupes
// ============================================================================================
function syncLabels() {
  document.getElementById("opVal").textContent = parseFloat(ui.op.value).toFixed(2);
  document.getElementById("rVal").textContent = pointRadius().toPrecision(3);
  draw();
}
ui.op.addEventListener("input", syncLabels);
ui.r.addEventListener("input", syncLabels);
ui.faces.addEventListener("change", draw);
ui.points.addEventListener("change", draw);
ui.edgeMode.addEventListener("change", draw);
if (PNT_V.length === 0) document.getElementById("ptBox").style.display = "none";

// au-delà de 3 dimensions : on choisit les 3 qu'on regarde, et on fixe les autres (la COUPE).
if (D > 3) {
  document.getElementById("axes").style.display = "block";
  const sels = [document.getElementById("axX"), document.getElementById("axY"), document.getElementById("axZ")];
  sels.forEach((sel, a) => {
    for (let k = 0; k < D; k++) sel.add(new Option("x" + k, k));
    sel.value = AX[a];
    sel.addEventListener("change", () => {
      const v = parseInt(sel.value, 10);
      const other = sels.findIndex((s, i) => i !== a && parseInt(s.value, 10) === v);
      if (other >= 0) { sels[other].value = AX[a]; AX[other] = AX[a]; }   // on ÉCHANGE
      AX[a] = v;
      projectPool(); buildSlices(); buildScene(); resetView();
    });
  });
  buildSlices();
}
function buildSlices() {
  const box = document.getElementById("slices");
  box.innerHTML = "";
  const hidden = [];
  for (let k = 0; k < D; k++) if (k !== AX[0] && k !== AX[1] && k !== AX[2]) hidden.push(k);
  if (!hidden.length) return;
  box.className = "sec";
  const t = document.createElement("div");
  t.textContent = "coupe";
  box.appendChild(t);
  for (const k of hidden) {
    const row = document.createElement("div");
    row.className = "row";
    const lab = document.createElement("span");
    const inp = document.createElement("input");
    inp.type = "range"; inp.min = BOUNDS[k][0]; inp.max = BOUNDS[k][1];
    inp.step = (BOUNDS[k][1] - BOUNDS[k][0]) / 400 || 0.01;
    inp.value = SLICE[k];
    const upd = () => { lab.textContent = "x" + k + " = " + parseFloat(inp.value).toPrecision(3); };
    upd();
    inp.addEventListener("input", () => { SLICE[k] = parseFloat(inp.value); upd(); buildScene(); draw(); });
    row.appendChild(lab); row.appendChild(inp);
    box.appendChild(row);
  }
}

if (FLAT) {
  document.getElementById("hDrag").textContent = "déplacer (pan)";
  document.getElementById("hArrows").textContent = "déplacer (Maj/Ctrl : plus vite)";
}

// mode sombre : préférence système au chargement, puis bascule par [d].
(function () {
  const ml = window.matchMedia("(prefers-color-scheme: dark)");
  let cur = ml.matches;
  function setTheme(on) {
    cur = on;
    document.body.classList.toggle("dark", on);
    document.body.dataset.theme = on ? "dark" : "light";
    document.getElementById("modeLabel").textContent = on ? "sombre" : "clair";
    draw();
  }
  setTheme(cur);
  ml.addEventListener("change", e => setTheme(e.matches));
  document.addEventListener("keydown", e => {
    if ((e.key === "d" || e.key === "D") && e.target.tagName !== "SELECT" && e.target.tagName !== "INPUT")
      { e.preventDefault(); setTheme(!cur); }
  });
})();

window.addEventListener("resize", resize);
projectPool();
setFrame(NB_FRAMES - 1);          // l'état final d'abord -- c'est ce qu'on veut voir en premier
syncLabels();
resize();                         // AVANT le cadrage : celui-ci dépend du rapport de la fenêtre
resetView();
// La barre de temps prend le focus : les flèches la pilotent NATIVEMENT (et Début/Fin aussi),
// sans rien enlever à la caméra -- un clic sur la vue rend les flèches à l'orbite.
if (NB_FRAMES > 1) tSlider.focus();
</script>
</body>
</html>
"""

"""Dessiner une cellule dans un `Visualizer` -- et ce qu'une cellule NON BORNÉE laisse tomber.

C'est la cellule qui choisit ses primitives, pas le visualiseur : en 2D le polygone, en 3D les
faces relues sur le treillis, au-delà la H-représentation ( la page en montre une coupe 3D ) avec le
fil de fer projeté. Tout se lit sur les tenseurs dérivés que chaque régime sait fournir
( `Cell._edges_of`, `_faces_of`, `_planes_of`, ... ) : rien ici ne dépend de la dimension.

= Ce qui est FACTICE dans une cellule non bornée

Les parois `INFINITE` n'existent pas, et leurs sommets sont posés là où les coupes ont dû les
repousser, à une distance qui ne veut rien dire. Trois règles, lues sur `cut_ids` :

  * une paroi `INFINITE` n'est pas envoyée -- ce qui reste est le vrai polytope, non borné, que la
    page referme elle-même sur la boîte de la scène ;
  * une arête POSÉE sur des parois factices ( tous ses plans le sont ) n'est pas une arête de la
    cellule : elle n'est pas tracée ;
  * une arête qui n'en touche une que par un bout est une vraie arête, tronquée quelque part :
    elle est tracée en POINTILLÉS, ramenée à une longueur prise sur l'étendue RÉELLE de la scène.

Le cadrage suit la même logique : seuls les sommets réels le fixent.
"""

import numpy as np


def add_to_viz( cell, viz, color, opacity, faces, edges, points ):
    from .Cell import INFINITE

    nb_dims = cell.dim
    items = cell._items()
    first_color = viz.reserve_colors( len( items ) )   # un par item, vide ou non ( voir `Cell.add_to_viz` )

    if max( ( it.nb_vertices for it in items ), default = 0 ) == 0:
        return viz

    # PREMIÈRE PASSE : ce qui, dans chaque item, tient au simplexe factice -- il faut la vue
    # d'ensemble avant de dessiner, pour donner aux rayons tronqués une longueur prise sur la scène
    parts = { b: _infinite_parts( cell, it ) for b, it in enumerate( items ) if it.nb_vertices }
    stub = _ray_length( [ it.vp for it in items ], parts )

    for b, it in enumerate( items ):
        if it.nb_vertices == 0:
            continue
        col = color if color is not None else viz.color_at( first_color + b )
        edge_col = viz.darker( col )

        fake, ev, hide, dash = parts[ b ]
        vp = _truncated_rays( it.vp, fake, ev, stub )
        real = vp[ ~fake ] if fake.any() else vp
        bounded = not fake.any()
        if not bounded:
            # ce vivier-là ne CADRE plus la scène : le bout d'un rayon est à une distance choisie
            viz.note_bounds( real if len( real ) else vp )

        if nb_dims > 3 or not bounded:
            dirs, offs = cell._planes_of( it )
            keep = ( it.cid != INFINITE ) & ( np.abs( dirs ).sum( axis = 1 ) > 0 )
            if keep.any():
                # `faces = False` -> opacité nulle plutôt que pas de faces : elles restent dans le
                # z-buffer, donc les arêtes cachées le restent. `edges` : au-delà de la 3D ce polytope
                # est une COUPE, et ses arêtes sont celles de la coupe ; en 2D / 3D c'est la cellule
                # qui trace les siennes plus bas, en sachant ce qui est tronqué.
                viz.add_polytope( dirs[ keep ], offs[ keep ], nb_dims = nb_dims, color = col,
                                  opacity = opacity if faces else 0, edges = nb_dims > 3 )
        elif faces and nb_dims >= 2:
            viz.add_faces( vp, cell._faces_of( it ), color = col, opacity = opacity, frames = bounded )

        if edges and len( ev ):
            # en dimension > 3 le fil de fer est une PROJECTION : il s'efface derrière la coupe
            op = 0.55 if nb_dims > 3 else 1.0
            solid = ~hide & ~dash
            if solid.any():
                viz.add_edges( vp, ev[ solid ], color = edge_col, opacity = op, frames = bounded )
            if dash.any():
                viz.add_edges( vp, ev[ dash ], color = edge_col, opacity = op, dashed = True, frames = False )

        if points:
            viz.add_points( real if len( real ) else vp, color = edge_col )
    return viz


def _infinite_parts( cell, it ):
    """`( fake, ev, hide, dash )` : quels sommets sont factices, la liste des arêtes, et lesquelles
    ne pas tracer / tracer en pointillés. Un sommet est factice dès qu'UN de ses plans l'est."""
    from .Cell import INFINITE

    infinite = it.cid == INFINITE
    vc = cell._vertex_cut_indices_of( it )
    ev = cell._edges_of( it )
    ec = cell._edge_cuts_of( it )
    fake = infinite[ vc ].any( axis = 1 ) if it.nb_vertices else np.zeros( 0, bool )
    hide = infinite[ ec ].all( axis = 1 ) if len( ev ) and ec.shape[ 1 ] else np.zeros( len( ev ), bool )
    dash = ( fake[ ev[ :, 0 ] ] | fake[ ev[ :, 1 ] ] ) & ~hide if len( ev ) else np.zeros( 0, bool )
    return fake, ev, hide, dash


def _ray_length( vps, parts ):
    """La longueur à donner à un rayon tronqué : une fraction de l'étendue RÉELLE de la scène, prise
    sur tous les items à la fois ( un diagramme doit avoir le même moignon partout ). Les percentiles
    et non le min / max : un sommet de Voronoï de trois germes presque alignés est réel mais ne dit
    rien de l'échelle à laquelle on regarde."""
    real = [ vps[ b ][ ~parts[ b ][ 0 ] ] for b in parts ]
    real = [ r for r in real if len( r ) ]
    if not real:
        return None                      # aucun sommet réel nulle part : on garde le simplexe tel quel
    pts = np.concatenate( real, axis = 0 )
    lo, hi = np.percentile( pts, 5, axis = 0 ), np.percentile( pts, 95, axis = 0 )
    diag = float( np.linalg.norm( hi - lo ) )
    return 0.2 * diag if diag > 0 else None


def _truncated_rays( vp, fake, ev, stub ):
    """`vp` avec ses sommets FACTICES ramenés à `stub` du sommet réel dont ils partent -- lu sur les
    arêtes : celle qui joint un vrai sommet à un faux EST le rayon. Un sommet factice sans rayon
    identifiable est ramené par rapport au centre des sommets réels."""
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

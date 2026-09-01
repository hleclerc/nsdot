"""Écrit une page HTML autonome (SVG inline) montrant une ou plusieurs courbes de convergence
(perte, résidu, ...) au fil des itérations d'un solveur -- LBFGS aujourd'hui, Newton demain,
la même fonction pour les deux : elle ne sait rien de qui l'appelle, seulement des SUITES de
valeurs.

Une seule page en dit long sur un solveur : la PENTE en échelle log dit si la convergence est
linéaire (LBFGS -- une droite) ou quadratique (Newton -- une droite qui se met à plonger).
D'où l'échelle log par défaut sur l'axe des y. SVG plutôt que le canvas+WebGL de `Visualizer` :
une poignée de points par courbe (le nombre d'itérations), pas des millions -- pas besoin de sa
mécanique base64.
"""
import numpy as np


#: mêmes teintes que `Visualizer.scale_color` (pas de dépendance croisée : juste la même
#: constante), pour qu'une courbe et le diagramme qu'elle résume, s'ils apparaissent dans la
#: même expérience, ne se contredisent pas sur ce qu'une couleur veut dire.
_GOLDEN_STRIDE = 0.6180339887498949


def _color( index ):
    import colorsys
    h = ( index * _GOLDEN_STRIDE ) % 1.0
    r, g, b = colorsys.hsv_to_rgb( h, 0.65, 0.75 )
    return f"rgb({ int( r * 255 ) },{ int( g * 255 ) },{ int( b * 255 ) })"


def _nice_log_ticks( lo, hi ):
    """Les puissances de 10 couvrant `[ lo, hi ]` (`lo > 0`), au moins deux."""
    import math
    a, b = math.floor( math.log10( lo ) ), math.ceil( math.log10( hi ) )
    if a == b:
        a, b = a - 1, b + 1
    return list( range( a, b + 1 ) )


def _nice_linear_ticks( lo, hi, count = 6 ):
    if hi <= lo:
        return [ lo ]
    step = ( hi - lo ) / max( count - 1, 1 )
    return [ lo + k * step for k in range( count ) ]


def write_convergence_html( series, out_path, title = "convergence", xlabel = "itération",
                            ylabel = "résidu", log_y = True ):
    """`series` : `{ nom: [ y0, y1, ... ] }` (une abscisse implicite `0, 1, 2, ...`), ou
    `{ nom: [ ( x0, y0 ), ( x1, y1 ), ... ] }` pour une abscisse explicite (des pas non
    consécutifs, par exemple). Renvoie le chemin écrit.

    `log_y` : échelle log sur l'axe des y (le défaut -- une courbe de convergence ne dit rien en
    échelle linéaire, elle s'écrase contre l'axe dès les deux premiers pas). Les valeurs `<= 0`
    n'ont alors pas de point : `0` EST le but, mais ne se place pas sur un axe log -- seule son
    approche se lit, dans la pente.
    """
    curves = {}
    for name, ys in series.items():
        pts = [ tuple( p ) if isinstance( p, ( tuple, list ) ) else ( i, p )
               for i, p in enumerate( ys ) ]
        if log_y:
            pts = [ ( x, y ) for x, y in pts if y > 0 ]
        if pts:
            curves[ name ] = pts

    W, H = 760, 420
    ml, mr, mt, mb = 64, 16, 34, 44           # marges : place pour les graduations et le titre

    all_x = [ x for pts in curves.values() for x, _ in pts ]
    all_y = [ y for pts in curves.values() for _, y in pts ]
    x_lo, x_hi = ( min( all_x ), max( all_x ) ) if all_x else ( 0, 1 )
    y_lo, y_hi = ( min( all_y ), max( all_y ) ) if all_y else ( 1e-12, 1.0 )
    if x_hi <= x_lo:
        x_hi = x_lo + 1
    if log_y:
        y_ticks = _nice_log_ticks( y_lo, y_hi )
        y_lo_l, y_hi_l = y_ticks[ 0 ], y_ticks[ -1 ]
        def to_y( y ):
            return H - mb - ( np.log10( y ) - y_lo_l ) / ( y_hi_l - y_lo_l ) * ( H - mt - mb )
    else:
        if y_hi <= y_lo:
            y_hi = y_lo + 1
        pad = 0.05 * ( y_hi - y_lo )
        y_lo, y_hi = y_lo - pad, y_hi + pad
        y_ticks = _nice_linear_ticks( y_lo, y_hi )
        def to_y( y ):
            return H - mb - ( y - y_lo ) / ( y_hi - y_lo ) * ( H - mt - mb )

    x_ticks = _nice_linear_ticks( x_lo, x_hi )
    def to_x( x ):
        return ml + ( x - x_lo ) / ( x_hi - x_lo ) * ( W - ml - mr )

    svg = [ f'<svg viewBox="0 0 { W } { H }" xmlns="http://www.w3.org/2000/svg" '
           f'font-family="sans-serif" font-size="11">' ]
    svg.append( f'<rect x="0" y="0" width="{ W }" height="{ H }" fill="#ffffff"/>' )
    svg.append( f'<text x="{ W / 2 }" y="18" text-anchor="middle" font-size="14">{ title }</text>' )

    # grille + graduations. Les valeurs formatées le sont SANS espace avant `}` -- tout ce qui
    # suit `:` dans un f-string est pris LITTÉRALEMENT comme spécificateur de format, un espace y
    # devient donc PARTIE du format (`ValueError: Invalid format specifier '.1f '`), contrairement
    # à un simple nom (`{ ml }`) où l'espace est du Python normal.
    for yt in y_ticks:
        y = to_y( 10 ** yt if log_y else yt )
        y_s = f"{ y:.1f}"
        svg.append( f'<line x1="{ ml }" y1="{ y_s }" x2="{ W - mr }" y2="{ y_s }" '
                   f'stroke="#e5e5e5" stroke-width="1"/>' )
        label = f"1e{ yt }" if log_y else f"{ yt:.3g}"
        svg.append( f'<text x="{ ml - 6 }" y="{ y + 3:.1f}" text-anchor="end">{ label }</text>' )
    for xt in x_ticks:
        x_s = f"{ to_x( xt ):.1f}"
        svg.append( f'<line x1="{ x_s }" y1="{ mt }" x2="{ x_s }" y2="{ H - mb }" '
                   f'stroke="#f2f2f2" stroke-width="1"/>' )
        svg.append( f'<text x="{ x_s }" y="{ H - mb + 16 }" text-anchor="middle">{ xt:.3g}</text>' )
    svg.append( f'<text x="{ ( ml + W - mr ) / 2 }" y="{ H - 6 }" text-anchor="middle">{ xlabel }</text>' )
    svg.append( f'<text x="14" y="{ ( mt + H - mb ) / 2 }" text-anchor="middle" '
               f'transform="rotate(-90 14 { ( mt + H - mb ) / 2 })">{ ylabel }</text>' )
    svg.append( f'<rect x="{ ml }" y="{ mt }" width="{ W - ml - mr }" height="{ H - mt - mb }" '
               f'fill="none" stroke="#999"/>' )

    # courbes + légende
    legend_y = mt + 4
    for i, ( name, pts ) in enumerate( curves.items() ):
        color = _color( i )
        path = " ".join( f'{ "M" if k == 0 else "L" }{ to_x( x ):.1f},{ to_y( y ):.1f}'
                        for k, ( x, y ) in enumerate( pts ) )
        svg.append( f'<path d="{ path }" fill="none" stroke="{ color }" stroke-width="2"/>' )
        for x, y in pts:
            svg.append( f'<circle cx="{ to_x( x ):.1f}" cy="{ to_y( y ):.1f}" r="2" fill="{ color }"/>' )
        lx = W - mr - 10
        svg.append( f'<circle cx="{ lx - 90 }" cy="{ legend_y }" r="4" fill="{ color }"/>' )
        svg.append( f'<text x="{ lx - 82 }" y="{ legend_y + 4 }">{ name }</text>' )
        legend_y += 16

    svg.append( "</svg>" )

    html = ( f'<!DOCTYPE html><html><head><meta charset="utf-8"><title>{ title }</title></head>'
           f'<body style="margin:0;display:flex;justify-content:center;padding:24px 0;">'
           + "".join( svg ) + "</body></html>" )

    from pathlib import Path
    path = Path( out_path )
    path.parent.mkdir( parents = True, exist_ok = True )
    path.write_text( html )
    print( f"OUTPUT: file://{ path.absolute() }" )
    return path

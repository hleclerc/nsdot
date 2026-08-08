"""Export d'un nuage de points 2D en page HTML autonome, avec un <canvas> plein écran et un
disque dessiné par point -- alternative à `matplotlib.pyplot.plot(..., '.')`, peu lisible et peu
interactif à grande échelle (pas de zoom, marqueurs de taille fixe en pixels écran).

Fichier UNIQUE et autonome (les points sont encodés en base64 directement dans le HTML, pas de
fichier annexe / pas de serveur -- s'ouvre directement depuis le disque, `file://`).
"""
import base64

import numpy as np

from sdot import Tensor


_HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>__TITLE__</title>
<style>
  html, body { margin: 0; padding: 0; overflow: hidden; background: #ffffff; }
  canvas { display: block; }
  #controls {
    position: fixed; top: 10px; left: 10px; z-index: 1;
    background: rgba(255,255,255,0.88); padding: 8px 12px; border-radius: 6px;
    font-family: sans-serif; font-size: 13px; box-shadow: 0 1px 4px rgba(0,0,0,0.3);
  }
  #controls label { display: block; }
  #controls input[type=range] { vertical-align: middle; width: 200px; }
</style>
</head>
<body>
<div id="controls">
  <label>rayon : <span id="rval">__RADIUS__</span>
    <input id="r" type="range" min="__RMIN__" max="__RMAX__" step="__RSTEP__" value="__RADIUS__">
  </label>
  <div>__N__ points</div>
</div>
<canvas id="c"></canvas>
<script>
const raw = atob("__B64__");
const buf = new ArrayBuffer(raw.length);
const bytes = new Uint8Array(buf);
for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);
const pts = new Float32Array(buf);           // [x0,y0, x1,y1, ...]
const n = pts.length / 2;
const bound = __BOUND__;                     // demi-étendue du monde affiché ([-bound,bound]^2)

const canvas = document.getElementById('c');
const ctx = canvas.getContext('2d');
const rSlider = document.getElementById('r');
const rLabel = document.getElementById('rval');

function draw() {
  const w = canvas.width, h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  const scale = Math.min(w, h) / (2 * bound);
  const cx = w / 2, cy = h / 2;
  const r = Math.max(parseFloat(rSlider.value) * scale, 0.4);
  ctx.fillStyle = '#000000';
  const path = new Path2D();
  for (let i = 0; i < n; i++) {
    const x = cx + pts[2 * i] * scale;
    const y = cy - pts[2 * i + 1] * scale;
    path.moveTo(x + r, y);
    path.arc(x, y, r, 0, 2 * Math.PI);
  }
  ctx.fill(path);
}

function resize() {
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;
  draw();
}

rSlider.addEventListener('input', () => { rLabel.textContent = rSlider.value; draw(); });
window.addEventListener('resize', resize);
resize();
</script>
</body>
</html>
"""


def export_positions_html(
    positions, extent: float, out_path: str, point_radius: float = 0.1,
    radius_range: tuple[float, float] | None = None, max_points: int = 500_000,
    title: str = "reconstruction", seed: int = 0,
):
    """Écrit `out_path`, une page HTML autonome affichant `positions` (Tensor ou array [n,2])
    comme un disque par point sur un <canvas> plein écran, avec un slider pour ajuster le rayon
    des disques (en unités MONDE, pas pixels -- reste cohérent quelle que soit la taille de la
    fenêtre). `extent` fixe la fenêtre affichée ([-extent/2, extent/2]^2, mêmes conventions que
    `Sinogram`).

    `max_points` : sous-échantillonne au-delà de ce compte -- un fichier HTML à 1e7 points
    encoderait ~80 Mo de coordonnées (base64 ~+33%), lourd à charger pour un gain visuel nul
    (les disques se recouvrent de toute façon bien avant cette densité à l'écran).
    """
    p = positions.raw if isinstance( positions, Tensor ) else np.asarray( positions )
    p = np.asarray( p, dtype = np.float32 )
    if len( p ) > max_points:
        idx = np.random.default_rng( seed ).choice( len( p ), max_points, replace = False )
        p = p[ idx ]

    b64 = base64.b64encode( p.tobytes() ).decode( "ascii" )
    bound = extent / 2

    if radius_range is None:
        radius_range = ( point_radius / 20, point_radius * 20 )
    rmin, rmax = radius_range
    rstep = ( rmax - rmin ) / 400

    html = ( _HTML
        .replace( "__TITLE__", title )
        .replace( "__RADIUS__", repr( point_radius ) )
        .replace( "__RMIN__", repr( rmin ) )
        .replace( "__RMAX__", repr( rmax ) )
        .replace( "__RSTEP__", repr( rstep ) )
        .replace( "__N__", str( len( p ) ) )
        .replace( "__B64__", b64 )
        .replace( "__BOUND__", repr( float( bound ) ) )
    )
    with open( out_path, "w" ) as f:
        f.write( html )
    print( f"html sauvé: { out_path } ({ len( p ) } points, { len( html ) / 1e6:.1f} Mo)" )

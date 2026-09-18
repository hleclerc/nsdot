#!/usr/bin/env python3
"""L'animation d'un ou deux chemins de resolution, depuis les trames JSONL de `newton --dump` et
`homotopie --dump` : un HTML autonome, deux panneaux, curseur et lecture.

    python3 scripts/animation.py sortie.html "Newton direct" newton.jsonl "Homotopie" homotopie.jsonl
"""
import sys, json

def lit( f ):
    return [ json.loads( l ) for l in open( f ) if l.strip() ]

GABARIT = r"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>Deux chemins vers les aires egales</title>
<!--
  CE QU'ON REGARDE

  Le meme nuage ( 2000 diracs sur 5 lignes, cible : aires toutes egales a 1/n ), resolu par deux
  chemins. A gauche, NEWTON AMORTI depuis Voronoi ( le pas par les limites ) : une trame par
  iteration acceptee. A droite, L'HOMOTOPIE SUR LA PRESCRIPTION : la cible glisse `nu_s = a_0 +
  s ( nu - a_0 )`, Newton CONVERGE sur chaque cible intermediaire ( tolerance 10 % ) avant que `s`
  ne double ; une trame par iteration du correcteur, l'etiquette dit le palier `s`.

  La COULEUR d'une cellule est log10( aire / nu ) : bleu quand elle est trop petite, gris quand
  elle a la bonne aire, orange quand elle est trop grande ( echelle -2 .. +2 ). En mode
  « voisinage », les cellules dont l'ensemble des voisins a CHANGE depuis la trame precedente sont
  peintes en noir : c'est le compte des epoques combinatoires, ce qui coute des iterations.

  Molette : zoom sur le curseur. Cliquer-glisser : deplacer. Espace : lecture / pause.
-->
<style>
  body { margin: 0; font: 13px/1.4 system-ui, sans-serif; color: #1f2937; background: #fafafa; }
  h1 { font-size: 16px; font-weight: 600; margin: 12px 16px 4px; }
  .note { margin: 0 16px 10px; color: #6b7280; }
  .panneaux { display: flex; gap: 16px; padding: 0 16px 16px; flex-wrap: wrap; }
  .panneau { background: #fff; border: 1px solid #e5e7eb; border-radius: 6px; padding: 10px; }
  .panneau h2 { font-size: 14px; font-weight: 600; margin: 0 0 6px; }
  canvas { display: block; border: 1px solid #e5e7eb; cursor: grab; }
  .ctl { display: flex; gap: 8px; align-items: center; margin: 8px 0 4px; }
  .ctl input[type=range] { flex: 1; }
  .stats { font-family: ui-monospace, monospace; font-size: 12px; white-space: pre; color: #374151; margin-top: 4px; }
  .stats b { color: #1f2937; }
  button { font: inherit; padding: 2px 10px; }
  .legende { display: flex; gap: 12px; align-items: center; margin: 0 16px 8px; color: #6b7280; }
  .barre { width: 160px; height: 10px; background: linear-gradient( to right, #2a78d6, #d9dde3, #eb6834 ); border: 1px solid #e5e7eb; }
  .noir { width: 14px; height: 10px; background: #1f2937; display: inline-block; }
</style>
</head>
<body>
<h1>Deux chemins vers les aires égales — lignes, n = 2000</h1>
<p class="note">Couleur : log₁₀(aire / ν), de −2 (bleu, trop petite) à +2 (orange, trop grande). Mode « voisinage » : en noir, les cellules dont les voisins ont changé depuis la trame précédente. Molette : zoom ; glisser : déplacer ; espace : lecture.</p>
<div class="legende"><span>−2</span><div class="barre"></div><span>+2</span><span style="margin-left:16px"><span class="noir"></span> voisinage changé</span>
  <label style="margin-left:16px">mode <select id="mode"><option value="aire">aire</option><option value="voisinage">voisinage</option></select></label>
  <label style="margin-left:16px"><input type="checkbox" id="lie" checked> panneaux liés (même vue)</label>
</div>
<div class="panneaux" id="panneaux"></div>
<script>
const DONNEES = __DONNEES__;
const TAILLE = 620;
const vue = { cx: 0.5, cy: 0.5, zoom: 1 };
const panneaux = [];

function couleur( r ) {                       // r = aire / nu ; bleu -> gris -> orange
  let v = Math.log10( Math.max( r, 1e-9 ) ) / 2; v = Math.max( -1, Math.min( 1, v ) );
  const a = [ 42, 120, 214 ], m = [ 217, 221, 227 ], b = [ 235, 104, 52 ];
  const c = v < 0 ? [ 0, 1, 2 ].map( i => m[ i ] + ( a[ i ] - m[ i ] ) * -v ) : [ 0, 1, 2 ].map( i => m[ i ] + ( b[ i ] - m[ i ] ) * v );
  return `rgb(${c[0]|0},${c[1]|0},${c[2]|0})`;
}

function dessine( p ) {
  const tr = p.trames[ p.k ], ctx = p.ctx, mode = document.getElementById( "mode" ).value;
  const z = TAILLE * vue.zoom, ox = TAILLE / 2 - vue.cx * z, oy = TAILLE / 2 + vue.cy * z;
  ctx.fillStyle = "#fff"; ctx.fillRect( 0, 0, TAILLE, TAILLE );
  ctx.lineWidth = 0.5;
  for ( let i = 0; i < tr.cellules.length; ++i ) {
    const c = tr.cellules[ i ];
    if ( c.length < 6 ) continue;
    ctx.beginPath();
    for ( let j = 0; j < c.length; j += 2 ) { const x = ox + c[ j ] * z, y = oy - c[ j + 1 ] * z; j ? ctx.lineTo( x, y ) : ctx.moveTo( x, y ); }
    ctx.closePath();
    ctx.fillStyle = mode == "aire" ? couleur( tr.aires[ i ] ) : ( tr.change[ i ] ? "#1f2937" : "#eef0f3" );
    ctx.fill();
    ctx.strokeStyle = mode == "aire" && tr.change[ i ] ? "#1f2937" : "#c9cdd3";
    ctx.lineWidth = mode == "aire" && tr.change[ i ] ? 1.2 : 0.4;
    ctx.stroke();
  }
  const s = tr.etiquette == "newton" ? "" : `palier s = ${tr.s}   `;
  p.stats.innerHTML = `trame <b>${p.k}</b> / ${p.trames.length - 1}   ${tr.etiquette}   ${s}it ${tr.it}   pas t = ${tr.t}\n` +
    `max|a-ν|/ν = <b>${tr.pire}</b>   |r|₂ = ${tr.r2}   voisinages changés : <b>${tr.changes}</b> / ${tr.cellules.length}`;
  p.curseur.value = p.k;
}
function redessine() { for ( const p of panneaux ) dessine( p ); }

for ( const d of DONNEES ) {
  const div = document.createElement( "div" ); div.className = "panneau";
  div.innerHTML = `<h2>${d.nom} — ${d.trames.length} trames</h2><canvas width="${TAILLE}" height="${TAILLE}"></canvas>
    <div class="ctl"><button class="lect">▶</button><input type="range" min="0" max="${d.trames.length - 1}" value="0"><span class="pos"></span></div><div class="stats"></div>`;
  document.getElementById( "panneaux" ).appendChild( div );
  const p = { trames: d.trames, k: 0, ctx: div.querySelector( "canvas" ).getContext( "2d" ), curseur: div.querySelector( "input" ),
              stats: div.querySelector( ".stats" ), lect: div.querySelector( ".lect" ), canvas: div.querySelector( "canvas" ), timer: null };
  p.curseur.oninput = () => { p.k = +p.curseur.value; dessine( p ); };
  p.lect.onclick = () => {
    if ( p.timer ) { clearInterval( p.timer ); p.timer = null; p.lect.textContent = "▶"; return; }
    p.lect.textContent = "❚❚";
    p.timer = setInterval( () => { p.k = ( p.k + 1 ) % p.trames.length; dessine( p ); }, 700 );
  };
  // zoom et deplacement
  p.canvas.onwheel = e => {
    e.preventDefault();
    const r = p.canvas.getBoundingClientRect(), mx = e.clientX - r.left, my = e.clientY - r.top;
    const z = TAILLE * vue.zoom, wx = vue.cx + ( mx - TAILLE / 2 ) / z, wy = vue.cy - ( my - TAILLE / 2 ) / z;
    const f = e.deltaY < 0 ? 1.25 : 0.8;
    vue.zoom = Math.max( 1, Math.min( 400, vue.zoom * f ));
    const z2 = TAILLE * vue.zoom;
    vue.cx = wx - ( mx - TAILLE / 2 ) / z2; vue.cy = wy + ( my - TAILLE / 2 ) / z2;
    redessine();
  };
  let glisse = null;
  p.canvas.onmousedown = e => { glisse = [ e.clientX, e.clientY ]; };
  window.addEventListener( "mouseup", () => { glisse = null; } );
  p.canvas.onmousemove = e => {
    if ( ! glisse ) return;
    const z = TAILLE * vue.zoom;
    vue.cx -= ( e.clientX - glisse[ 0 ] ) / z; vue.cy += ( e.clientY - glisse[ 1 ] ) / z;
    glisse = [ e.clientX, e.clientY ]; redessine();
  };
  panneaux.push( p );
  dessine( p );
}
document.getElementById( "mode" ).onchange = redessine;
// `?k=5&mode=voisinage&zoom=3&cx=0.4&cy=0.6` : une vue initiale, pour les captures
{
  const q = new URLSearchParams( location.search );
  if ( q.get( "mode" ) ) document.getElementById( "mode" ).value = q.get( "mode" );
  if ( q.get( "zoom" ) ) { vue.zoom = +q.get( "zoom" ); vue.cx = +( q.get( "cx" ) || 0.5 ); vue.cy = +( q.get( "cy" ) || 0.5 ); }
  if ( q.get( "k" ) ) for ( const p of panneaux ) p.k = Math.min( +q.get( "k" ), p.trames.length - 1 );
  redessine();
}
window.addEventListener( "keydown", e => { if ( e.code == "Space" ) { e.preventDefault(); for ( const p of panneaux ) p.lect.onclick(); } } );
</script>
</body>
</html>
"""

def main( out, *args ):
    donnees = []
    for k in range( 0, len( args ), 2 ):
        donnees.append( { "nom": args[ k ], "trames": lit( args[ k + 1 ] ) } )
    html = GABARIT.replace( "__DONNEES__", json.dumps( donnees, separators = ( ",", ":" ) ) )
    open( out, "w" ).write( html )
    print( "ecrit", out, sum( len( d[ "trames" ] ) for d in donnees ), "trames" )

if __name__ == "__main__":
    main( *sys.argv[ 1: ] )

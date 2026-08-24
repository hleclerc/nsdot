"""Minimal self-contained HTML viewer for a 2D point cloud, or a SEQUENCE of
frames (e.g. one snapshot per optimizer step) with a play/pause time slider.

A much simpler cousin of `otrec/src/otrec/viz/points_html.py`: fixed-radius
dots, pan/zoom, no per-point radii/markers/vertex-offsets/dark-mode/touch.
"""
import base64

import numpy as np


_HTML = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>__TITLE__</title>
<style>
  html, body { margin: 0; padding: 0; overflow: hidden; background: #fff; }
  canvas { display: block; }
  #bar {
    position: fixed; top: 8px; left: 8px; z-index: 1; display: __BAR_DISPLAY__;
    align-items: center; gap: 8px; font: 12px sans-serif;
    background: rgba(255,255,255,0.9); padding: 6px 10px; border-radius: 6px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.3);
  }
  #bar input[type=range] { width: 260px; }
  #play { cursor: pointer; width: 22px; }
</style></head>
<body>
<div id="bar">
  <button id="play">&#9654;</button>
  <input id="t" type="range" min="0" max="__T_MAX__" step="1" value="__T_MAX__">
  <span id="tlabel"></span>
</div>
<canvas id="c"></canvas>
<script>
function decodeF32(b64) {
  const raw = atob(b64), bytes = new Uint8Array(raw.length);
  for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);
  return new Float32Array(bytes.buffer);
}
const pts = decodeF32("__POINTS_B64__");     // all frames concatenated: [x0,y0, x1,y1, ...]
const COUNTS = __COUNTS__;                   // points per frame
const OFFSETS = COUNTS.reduce((o, c) => (o.push(o.length ? o[o.length - 1] + COUNTS[o.length - 1] : 0), o), []);
const nFrames = COUNTS.length;
const bound = __BOUND__;                     // half-extent of the world shown ([-bound,bound]^2)
const radiusPx = __RADIUS_PX__;
const fps = __FPS__;

const canvas = document.getElementById('c');
const ctx = canvas.getContext('2d');
const tSlider = document.getElementById('t');
const tLabel = document.getElementById('tlabel');
const playBtn = document.getElementById('play');

let zoom = 1, panX = 0, panY = 0, frameIdx = nFrames - 1;

function draw() {
  const w = canvas.width, h = canvas.height;
  ctx.fillStyle = '#fff';
  ctx.fillRect(0, 0, w, h);
  const scale = Math.min(w, h) / (2 * bound) * zoom;
  const ox = w / 2 + panX, oy = h / 2 + panY;
  ctx.fillStyle = '#000';
  const path = new Path2D();
  const off = OFFSETS[frameIdx], cnt = COUNTS[frameIdx];
  for (let i = 0; i < cnt; i++) {
    const x = ox + pts[2 * (off + i)] * scale;
    const y = oy - pts[2 * (off + i) + 1] * scale;
    path.moveTo(x + radiusPx, y);
    path.arc(x, y, radiusPx, 0, 2 * Math.PI);
  }
  ctx.fill(path);
}

function resize() {
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;
  draw();
}
window.addEventListener('resize', resize);

canvas.addEventListener('wheel', (e) => {
  e.preventDefault();
  const factor = Math.exp(-e.deltaY * 0.01);
  zoom = Math.min(Math.max(zoom * factor, 0.05), 200);
  draw();
}, { passive: false });

let dragging = false, lastX = 0, lastY = 0;
canvas.addEventListener('mousedown', (e) => { dragging = true; lastX = e.clientX; lastY = e.clientY; });
window.addEventListener('mousemove', (e) => {
  if (!dragging) return;
  panX += e.clientX - lastX; panY += e.clientY - lastY;
  lastX = e.clientX; lastY = e.clientY;
  draw();
});
window.addEventListener('mouseup', () => { dragging = false; });
canvas.addEventListener('dblclick', () => { zoom = 1; panX = 0; panY = 0; draw(); });

let playing = false, playTimer = null;
function setFrame(i) {
  frameIdx = Math.max(0, Math.min(nFrames - 1, i));
  tSlider.value = frameIdx;
  tLabel.textContent = `${frameIdx + 1} / ${nFrames}`;
  draw();
}
function stopPlaying() {
  playing = false; playBtn.innerHTML = '&#9654;'; clearInterval(playTimer);
}
function togglePlay() {
  if (playing) { stopPlaying(); return; }
  playing = true; playBtn.innerHTML = '&#10074;&#10074;';
  if (frameIdx >= nFrames - 1) setFrame(0);
  playTimer = setInterval(() => {
    if (frameIdx >= nFrames - 1) { stopPlaying(); return; }
    setFrame(frameIdx + 1);
  }, 1000 / fps);
}
if (nFrames > 1) {
  tSlider.addEventListener('input', () => { stopPlaying(); setFrame(parseInt(tSlider.value, 10)); });
  playBtn.addEventListener('click', togglePlay);
}

setFrame(frameIdx);
resize();
</script>
</body></html>
"""


def export_points_html(frames, extent, out_path, title="reconstruction",
                       fps=5.0, max_points=200_000, radius_px=2.0, seed=0):
    """Write `out_path`, a self-contained HTML page showing `frames` as dots
    on a full-window <canvas> (pan: drag, zoom: wheel). `frames` is either a
    single point cloud (`[n, 2]`) or a sequence of them (one per optimizer
    step, e.g. `Tracker.frames`) -- with more than one frame the page adds a
    time slider + play/pause. `extent` sets the world window shown
    (`[-extent/2, extent/2]^2`, same convention as `Sinogram`).

    `max_points`: subsamples beyond this count, per frame, with the SAME
    indices across frames when every frame has the same point count -- so an
    animated point keeps its identity instead of the frames looking like
    independent noise.
    """
    if isinstance(frames, np.ndarray) and frames.ndim == 2:
        frames = [frames]
    frames = [np.asarray(f, dtype=np.float32) for f in frames]
    counts = [len(f) for f in frames]

    rng = np.random.default_rng(seed)
    if len(set(counts)) == 1 and counts[0] > max_points:
        idx = rng.choice(counts[0], max_points, replace=False)
        frames = [f[idx] for f in frames]
    else:
        frames = [f[rng.choice(len(f), max_points, replace=False)] if len(f) > max_points else f
                  for f in frames]
    counts = [len(f) for f in frames]

    points_b64 = base64.b64encode(np.concatenate(frames, axis=0).tobytes()).decode("ascii")

    html = (_HTML
        .replace("__TITLE__", title)
        .replace("__BAR_DISPLAY__", "flex" if len(frames) > 1 else "none")
        .replace("__T_MAX__", str(len(frames) - 1))
        .replace("__POINTS_B64__", points_b64)
        .replace("__COUNTS__", "[" + ",".join(str(c) for c in counts) + "]")
        .replace("__BOUND__", repr(float(extent) / 2))
        .replace("__RADIUS_PX__", repr(float(radius_px)))
        .replace("__FPS__", repr(float(fps)))
    )
    with open(out_path, "w") as f:
        f.write(html)
    print(f"OUTPUT: {out_path}")

"""The three pages the generator writes: the map, the cube, the replay.

Each is one self-contained file that works when double-clicked. Every colour and
space comes from the tokens in one place, every bulk toggle carries its inverse,
and no layout scrolls sideways.
"""

from __future__ import annotations

import json

from inmotion.bus.demand.config import ScenarioConfig

#: The design tokens, in one place: surfaces, ink, and one hue per quantity.
#: Values are the only thing the stylesheet and the JavaScript both need, so
#: they are emitted as custom properties and nothing else hard-codes a colour.
TOKENS = {
    "surface": "#0e1418",
    "surface-raised": "#161f26",
    "surface-sunken": "#0a0f13",
    "line": "#22303a",
    "ink": "#e8eef2",
    "ink-muted": "#93a4b0",
    "ink-faint": "#5d707d",
    "boarded": "#38bdf8",
    "waiting": "#f4b942",
    "stranded": "#f2555a",
    "focus": "#7dd3fc",
}

#: The grid everything is spaced on. An 8-point scale, used through the
#: stylesheet, so no margin is invented at the point of use.
SPACING = {"s1": "4px", "s2": "8px", "s3": "12px", "s4": "16px", "s5": "24px", "s6": "32px"}


def _css_tokens() -> str:
    """The token set as custom properties: the stylesheet's only colour source."""
    lines = [f"      --{name}: {value};" for name, value in TOKENS.items()]
    lines.extend(f"      --{name}: {value};" for name, value in SPACING.items())
    return "\n".join(lines)


def _escape(text: str) -> str:
    """Text going into HTML rather than into the JSON payload."""
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


LEAFLET_CSS = "https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
LEAFLET_JS = "https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"
LEAFLET_CSS_SRI = "sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY="
LEAFLET_JS_SRI = "sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo="

THREE_JS = "https://unpkg.com/three@0.147.0/build/three.min.js"
THREE_JS_SRI = "sha256-80RGv4dbX7Dc2TgZ/+HZ4YLUZjTuhV9dkExsSsfNvJU="
ORBIT_JS = "https://unpkg.com/three@0.147.0/examples/js/controls/OrbitControls.js"
ORBIT_JS_SRI = "sha256-tMblP5hThTWxH9ZlVifUf7WHezrZclaL/wpwJsS21cQ="

COMMON_CSS = """
  :root {
__TOKENS__
  }
  * { box-sizing: border-box; }
  html, body { height: 100%; }
  body {
    margin: 0;
    overflow-x: clip;
    background: var(--surface);
    color: var(--ink);
    font: 15px/1.55 ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
    -webkit-font-smoothing: antialiased;
  }
  .num { font-variant-numeric: tabular-nums; font-feature-settings: "tnum" 1; }
  .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
  .app { display: grid; height: 100vh; max-width: 100%; }
  .panel {
    display: flex; flex-direction: column; gap: var(--s5); padding: var(--s5);
    background: var(--surface-raised); border-right: 1px solid var(--line);
    max-width: 100%; overflow-wrap: anywhere; overflow-y: auto; height: 100vh;
  }
  h1 { font-size: 20px; line-height: 1.25; margin: 0 0 var(--s1); letter-spacing: -0.01em; }
  h2 {
    font-size: 11px; text-transform: uppercase; letter-spacing: 0.09em;
    color: var(--ink-muted); margin: 0 0 var(--s3); font-weight: 600;
  }
  p { margin: 0 0 var(--s3); color: var(--ink-muted); }
  .eyebrow { font-size: 12px; color: var(--ink-faint); margin: 0; }
  .stats { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: var(--s3); }
  .stat {
    background: var(--surface-sunken); border: 1px solid var(--line);
    border-radius: 8px; padding: var(--s3);
  }
  .stat dt {
    font-size: 11px; text-transform: uppercase; letter-spacing: 0.07em; color: var(--ink-faint);
  }
  .stat dd { margin: var(--s1) 0 0; font-size: 20px; font-weight: 600; }
  .row { display: flex; flex-wrap: wrap; gap: var(--s2); align-items: center; }
  label.layer {
    display: flex; align-items: center; gap: var(--s2);
    background: var(--surface-sunken); border: 1px solid var(--line);
    border-radius: 999px; padding: 6px 12px; font-size: 13px; cursor: pointer;
  }
  label.layer:focus-within { outline: 2px solid var(--focus); outline-offset: 2px; }
  input[type="checkbox"] { accent-color: var(--boarded); }
  select, input[type="search"], input[type="number"] {
    background: var(--surface-sunken); color: var(--ink); border: 1px solid var(--line);
    border-radius: 8px; padding: 6px 10px; font: inherit; font-size: 13px; max-width: 100%;
  }
  input[type="range"] { width: 100%; accent-color: var(--boarded); }
  button {
    background: transparent; color: var(--ink); border: 1px solid var(--line);
    border-radius: 999px; padding: 6px 12px; font: inherit; font-size: 13px; cursor: pointer;
  }
  button:hover { border-color: var(--ink-faint); }
  button:focus-visible, select:focus-visible, input:focus-visible {
    outline: 2px solid var(--focus); outline-offset: 2px;
  }
  .scroll {
    max-height: 168px; overflow-y: auto; border: 1px solid var(--line);
    border-radius: 8px; padding: var(--s2);
  }
  .legend { display: flex; flex-direction: column; gap: var(--s2); }
  .legend-item {
    display: flex; align-items: center; gap: var(--s2); font-size: 13px; color: var(--ink-muted);
  }
  .swatch { width: 12px; height: 12px; border-radius: 3px; flex: 0 0 auto; }
  .swatch.line { height: 3px; border-radius: 2px; width: 22px; }
  .equation {
    background: var(--surface-sunken); border: 1px solid var(--line); border-radius: 8px;
    padding: var(--s3); font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 13px; overflow-x: auto;
  }
  .equation div { white-space: nowrap; margin-bottom: var(--s1); }
  .terms { list-style: none; padding: 0; margin: var(--s3) 0 0; }
  .terms li {
    display: flex; gap: var(--s2); font-size: 12.5px; color: var(--ink-muted); margin-bottom: 6px;
  }
  .terms b {
    color: var(--ink); font-weight: 600; min-width: 88px; font-family: ui-monospace, monospace;
  }
  .callout {
    background: var(--surface-sunken); border: 1px solid var(--waiting);
    border-radius: 8px; padding: var(--s3); font-size: 12.5px; color: var(--ink);
    margin-top: var(--s3);
  }
  .detail { min-height: 92px; font-size: 13px; }
  .detail h3 { margin: 0 0 var(--s1); font-size: 14px; }
  .detail dl {
    display: grid; grid-template-columns: auto 1fr; gap: 2px var(--s3); margin: var(--s2) 0 0;
  }
  .detail dt { color: var(--ink-faint); }
  .detail dd { margin: 0; text-align: right; }
  .sparkline {
    display: flex; align-items: flex-end; gap: 2px; height: 46px; margin-top: var(--s2);
  }
  .sparkline button {
    flex: 1 1 auto; padding: 0; border: 0; border-radius: 2px 2px 0 0; min-width: 4px;
    background: var(--boarded); opacity: 0.45; cursor: pointer;
  }
  .sparkline button.in-window { opacity: 1; }
  .sparkline button:hover { outline: 2px solid var(--focus); }
  footer {
    font-size: 12px; color: var(--ink-faint);
    border-top: 1px solid var(--line); padding-top: var(--s3);
  }
  footer a { color: var(--ink-muted); }
  @media (max-width: 1024px) {
    .app { grid-template-columns: 1fr; height: auto; }
    .panel {
      border-right: 0; border-bottom: 1px solid var(--line);
      height: auto; overflow-y: visible;
    }
  }
  @media (max-width: 560px) {
    .stats { grid-template-columns: 1fr; }
    .panel { padding: var(--s4); gap: var(--s4); }
  }
"""

FILTER_MARKUP = """
    <section>
      <h2>Basemap</h2>
      <div class="row">
        <select id="basemap" aria-label="Basemap"></select>
      </div>
      <p class="eyebrow" id="basemap-note"></p>
    </section>

    <section>
      <h2>Filters</h2>
      <div class="layers">
        <div class="row" id="group-row"></div>
        <div class="row" id="operator-row"></div>
        <div class="row">
          <input type="search" id="line-search" placeholder="filter lines"
                 aria-label="Filter lines">
          <button id="toggle-all" type="button">Deselect all</button>
          <button id="reset-view" type="button">Reset view</button>
        </div>
        <div class="scroll" id="line-row"></div>
        <label class="eyebrow" for="min-passengers">
          At least <span id="min-label">1</span> passenger(s)
        </label>
        <input type="range" id="min-passengers" min="1" max="30" value="1">
        <input type="search" id="stop-search" placeholder="find a stop"
               aria-label="Find a stop">
      </div>
      <p class="eyebrow" id="visible" style="margin-top: var(--s2)"></p>
    </section>
"""

TIME_MARKUP = """
    <section>
      <h2>Time</h2>
      <div class="row">
        <button id="play" type="button">Play the day</button>
        <span class="eyebrow num" id="window-label"></span>
      </div>
      <div class="row" style="margin-top: var(--s2)">
        <label class="eyebrow" for="hour-from">From</label>
        <input type="range" id="hour-from" min="0" max="24" value="0" style="flex: 1 1 120px">
        <label class="eyebrow" for="hour-to">To</label>
        <input type="range" id="hour-to" min="0" max="24" value="24" style="flex: 1 1 120px">
      </div>
      <div class="sparkline" id="sparkline" role="group" aria-label="Riders per hour"></div>
      <p class="eyebrow">Bar height is riders boarded in that hour; click one to jump to it.</p>
    </section>
"""

FILTER_JS = """
function options(container, values, state, onChange) {
  const row = document.getElementById(container);
  row.textContent = '';
  values.forEach(function (value) {
    const label = document.createElement('label');
    label.className = 'layer';
    const box = document.createElement('input');
    box.type = 'checkbox';
    box.checked = state[value] !== false;
    box.addEventListener('change', function () {
      state[value] = box.checked;
      onChange();
    });
    const text = document.createElement('span');
    text.textContent = value;
    label.appendChild(box);
    label.appendChild(text);
    row.appendChild(label);
  });
}

function lineRows(data, state, onChange) {
  const row = document.getElementById('line-row');
  const needle = document.getElementById('line-search').value.trim().toLowerCase();
  row.textContent = '';
  data.lines
    .filter(function (line) { return !needle || line.toLowerCase().indexOf(needle) >= 0; })
    .forEach(function (line) {
      const label = document.createElement('label');
      label.className = 'layer';
      label.style.margin = '2px';
      const box = document.createElement('input');
      box.type = 'checkbox';
      box.checked = state.lines[line] !== false;
      box.addEventListener('change', function () {
        state.lines[line] = box.checked;
        onChange();
      });
      const text = document.createElement('span');
      text.textContent = line;
      label.appendChild(box);
      label.appendChild(text);
      row.appendChild(label);
    });
}

function lineMatches(data, state, feature) {
  if (state.allLines) { return true; }
  const own = (feature.properties.lines || '').split(' ').filter(Boolean);
  if (!own.length) { return state.allLines; }
  return own.some(function (line) { return state.lines[line] !== false; });
}

function bulkToggle(data, state, flipTo) {
  const on = flipTo === undefined ? !allOn(data, state) : flipTo;
  data.groups.forEach(function (group) { state[group] = on; });
  data.operators.forEach(function (operator) { state[operator] = on; });
  data.lines.forEach(function (line) { state.lines[line] = on; });
  state.allLines = on;
  document.querySelectorAll('#group-row input, #operator-row input, #line-row input')
    .forEach(function (box) { box.checked = on; });
  document.getElementById('toggle-all').textContent = on ? 'Deselect all' : 'Select all';
  return on;
}

function allOn(data, state) {
  return data.groups.every(function (group) { return state[group]; })
    && data.operators.every(function (operator) { return state[operator]; })
    && (state.allLines || data.lines.every(function (line) { return state.lines[line]; }));
}

function hourWindow(data, state) {
  const span = 1 + Math.max(data.hours.max - data.hours.min, 0);
  const from = Math.min(state.from, state.to);
  const to = Math.max(state.from, state.to);
  return { from: from, to: to, span: span };
}

function inWindow(hour, window) {
  return hour >= window.from && hour <= window.to;
}

function windowLabel(state) {
  const clock = function (hour) {
    return (hour < 10 ? '0' : '') + hour + ':00';
  };
  return clock(state.from) + ' – ' + clock(state.to + 1);
}

function hourRiders(feature, window) {
  return (feature.properties.hours || []).reduce(function (total, slot) {
    return slot[0] >= window.from && slot[0] <= window.to ? total + slot[1] : total;
  }, 0);
}
"""

SPARKLINE_JS = """
function sparkline(data, state, onJump) {
  const host = document.getElementById('sparkline');
  if (host.childElementCount) {
    Array.prototype.forEach.call(host.children, function (bar, hour) {
      const inside = hour >= state.from && hour <= state.to;
      bar.className = inside ? 'in-window' : '';
    });
    return;
  }
  host.textContent = '';
  const totals = {};
  data.flows.forEach(function (feature) {
    const hour = feature.properties.hour;
    totals[hour] = (totals[hour] || 0) + feature.properties.passengers;
  });
  const best = Math.max(1, ...Object.values(totals));
  data.hours.labels.forEach(function (label, hour) {
    const bar = document.createElement('button');
    bar.type = 'button';
    bar.title = label + ' — ' + (totals[hour] || 0) + ' riders';
    bar.setAttribute('aria-label', bar.title);
    bar.style.height = Math.max(4, 100 * (totals[hour] || 0) / best) + '%';
    if (hour >= state.from && hour <= state.to) { bar.className = 'in-window'; }
    bar.addEventListener('click', function () { onJump(hour); });
    host.appendChild(bar);
  });
}

"""

CUBE_JS = """
const scene = new THREE.Scene();
scene.background = new THREE.Color(TOKENS.surface);
const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 4000);
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio || 1);
document.getElementById('cube').appendChild(renderer.domElement);
const controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

const span = { x: Math.max(DATA.stops.length - 1, 1) };
span.y = Math.max(DATA.hours.max - DATA.hours.min, 1);
const size = Math.max(span.x, span.y);
span.halfX = size / 2;
span.halfY = Math.max(span.y / 2, 1);

const box = new THREE.Box3(
  new THREE.Vector3(-span.halfX - 1, -0.5, -span.halfX - 1),
  new THREE.Vector3(span.halfX + 1, span.y + 0.5, span.halfX + 1)
);
const helper = new THREE.Box3Helper(box, new THREE.Color('#33424e'));
scene.add(helper);
scene.add(new THREE.AmbientLight(0xffffff, 0.9));
const key = new THREE.DirectionalLight(0xffffff, 0.7);
key.position.set(1, 2, 1);
scene.add(key);
const fill = new THREE.DirectionalLight(0xffffff, 0.35);
fill.position.set(-1, -0.5, -1);
scene.add(fill);

const geometry = new THREE.BoxGeometry(0.6, 0.6, 0.6);
const material = new THREE.MeshLambertMaterial({
  color: 0xffffff, transparent: true, opacity: 0.92
});
const mesh = new THREE.InstancedMesh(geometry, material, Math.max(DATA.points.length, 1));
mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
scene.add(mesh);

function hourColour(hour, span) {
  const ratio = span > 0 ? (hour - DATA.hours.min) / span : 0;
  return new THREE.Color().setHSL(0.55 - 0.42 * Math.min(Math.max(ratio, 0), 1), 0.75, 0.6);
}

const state = {
  mode: 'entries', from: DATA.hours.min, to: DATA.hours.max,
  min: 1, lines: {}, allLines: true
};
const dummy = new THREE.Object3D();
const colour = new THREE.Color();
let shown = [];

function rebuild() {
  const hourOf = function (point) { return state.mode === 'entries' ? point.b : point.a; };
  shown = DATA.points.filter(function (point) {
    if (point.n < state.min) { return false; }
    if (!state.allLines && state.lines[DATA.lines[point.l]] === false) { return false; }
    return hourOf(point) >= state.from && hourOf(point) <= state.to;
  });
  shown.sort(function (a, b) { return a.n - b.n; });
  mesh.count = shown.length;
  shown.forEach(function (point, index) {
    const scale = 0.35 + Math.min(point.n, 12) / 12;
    dummy.position.set(
      point.o - span.halfX,
      hourOf(point) - DATA.hours.min,
      point.d - span.halfX
    );
    dummy.scale.set(scale, scale, scale);
    dummy.updateMatrix();
    mesh.setMatrixAt(index, dummy.matrix);
    colour.copy(hourColour(hourOf(point), span.y));
    mesh.setColorAt(index, colour);
  });
  mesh.instanceMatrix.needsUpdate = true;
  if (mesh.instanceColor) { mesh.instanceColor.needsUpdate = true; }
  document.getElementById('cube-count').textContent =
    shown.length + ' rides shown of ' + DATA.points.length + ' on the lattice';
}

function frame() {
  controls.update();
  renderer.render(scene, camera);
  labels();
  requestAnimationFrame(frame);
}

function resize() {
  const host = document.getElementById('cube');
  const width = host.clientWidth;
  const height = host.clientHeight;
  renderer.setSize(width, height);
  camera.aspect = width / Math.max(height, 1);
  camera.updateProjectionMatrix();
}
window.addEventListener('resize', resize);

const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2(-2, -2);
const tooltip = document.getElementById('tooltip');
renderer.domElement.addEventListener('pointermove', function (event) {
  const rect = renderer.domElement.getBoundingClientRect();
  pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
  raycaster.setFromCamera(pointer, camera);
  const hits = raycaster.intersectObject(mesh, false);
  if (!hits.length || hits[0].instanceId >= shown.length) {
    tooltip.hidden = true;
    return;
  }
  const point = shown[hits[0].instanceId];
  const origin = DATA.stops[point.o];
  const destination = DATA.stops[point.d];
  tooltip.hidden = false;
  tooltip.textContent = '';
  const title = document.createElement('strong');
  title.textContent = origin.name + ' → ' + destination.name;
  const body = document.createElement('div');
  body.className = 'num';
  body.textContent = point.n + ' riders · line ' + DATA.lines[point.l] +
    ' · boarded ' + DATA.hours.labels[point.b] + ' · alighted ' + DATA.hours.labels[point.a];
  tooltip.appendChild(title);
  tooltip.appendChild(body);
  tooltip.style.left = Math.min(event.clientX - rect.left + 12, rect.width - 260) + 'px';
  tooltip.style.top = (event.clientY - rect.top + 12) + 'px';
});

const labelHost = document.getElementById('labels');
const anchors = [];
const projected = new THREE.Vector3();

function buildLabels() {
  const step = Math.max(1, Math.round(span.y / 8));
  for (let hour = DATA.hours.min; hour <= DATA.hours.max; hour += step) {
    anchors.push({
      text: DATA.hours.labels[hour],
      position: new THREE.Vector3(-span.halfX - 3, hour - DATA.hours.min, -span.halfX - 3),
      side: 'right'
    });
  }
  DATA.stops.slice(0, 12).forEach(function (stop, index) {
    if (index % 3) { return; }
    anchors.push({
      text: shortName(stop.name) + ' (from)',
      position: new THREE.Vector3(index - span.halfX, -2.2, span.halfX + 4),
      side: 'above'
    });
    anchors.push({
      text: shortName(stop.name) + ' (to)',
      position: new THREE.Vector3(span.halfX + 4, -2.2, index - span.halfX),
      side: 'left'
    });
  });
  anchors.forEach(function (anchor) {
    const node = document.createElement('span');
    node.className = 'axis-label' + (anchor.text.indexOf(':') >= 0 ? ' num' : '');
    node.textContent = anchor.text;
    labelHost.appendChild(node);
    anchor.node = node;
  });
}

function labels() {
  if (!anchors.length) { return; }
  const host = renderer.domElement;
  anchors.forEach(function (anchor) {
    projected.copy(anchor.position).project(camera);
    anchor.node.style.left = (projected.x * 0.5 + 0.5) * host.clientWidth + 'px';
    anchor.node.style.top = (-projected.y * 0.5 + 0.5) * host.clientHeight + 'px';
    anchor.node.style.transform =
      anchor.side === 'right' ? 'translate(4px, -50%)'
        : anchor.side === 'left' ? 'translate(-100%, -50%)'
        : 'translate(-50%, -100%)';
    anchor.node.style.opacity = projected.z < 1 ? 1 : 0;
  });
}

function shortName(name) {
  return name.length > 16 ? name.slice(0, 15) + '…' : name;
}
"""

CUBE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__ — space-time cube</title>
<script src="__THREE__" integrity="__THREE_SRI__" crossorigin=""></script>
<script src="__ORBIT__" integrity="__ORBIT_SRI__" crossorigin=""></script>
<style>
__CSS__
  .app { grid-template-columns: 360px 1fr; }
  .stage { position: relative; height: 100vh; background: var(--surface-sunken); }
  #cube { position: absolute; inset: 0; }
  #cube canvas { display: block; }
  #labels { position: absolute; inset: 0; pointer-events: none; overflow: hidden; }
  .axis-label {
    position: absolute; font-size: 10.5px; color: var(--ink-muted);
    white-space: nowrap; text-shadow: 0 0 6px var(--surface);
  }
  #tooltip {
    position: absolute; max-width: 250px; background: var(--surface-raised);
    border: 1px solid var(--line); border-radius: 8px; padding: var(--s2) var(--s3);
    font-size: 12.5px; pointer-events: none; z-index: 5;
  }
  .modes { display: flex; gap: var(--s2); flex-wrap: wrap; }
</style>
</head>
<body>
<div class="app">
  <main class="panel">
    <div>
      <p class="eyebrow">inMotion · origin × destination × time</p>
      <h1>__TITLE__</h1>
      <p class="eyebrow mono">seed __SEED__ · __FLEET_POLICY__ · __FINGERPRINT__</p>
    </div>

    <dl class="stats">
      <div class="stat"><dt>Rides placed</dt><dd class="num">__POINTS__</dd></div>
      <div class="stat"><dt>Stops on the lattice</dt><dd class="num">__STOP_COUNT__</dd></div>
      <div class="stat"><dt>Hours spanned</dt><dd class="num">__HOUR_COUNT__</dd></div>
      <div class="stat"><dt>Boardings</dt><dd class="num">__BOARDED__</dd></div>
      <div class="stat"><dt>Alightings</dt><dd class="num">__ALIGHTED__</dd></div>
      <div class="stat"><dt>Stops omitted</dt><dd class="num">__OMITTED__</dd></div>
    </dl>

    <section>
      <h2>What the height means</h2>
      <div class="modes">
        <label class="layer">
          <input type="radio" name="mode" value="entries" checked>
          entries — boarded at the origin
        </label>
        <label class="layer">
          <input type="radio" name="mode" value="exits">
          exits — alighted at the destination
        </label>
      </div>
      <p class="eyebrow" style="margin-top: var(--s3)">
        The same rides read twice: a rider belongs to the hour they stepped on when
        you look at entries, and to the hour they stepped off when you look at
        exits. __SHIFTED__ of __POINTS__ rides change hour between the two.
      </p>
    </section>

    <section>
      <h2>Time</h2>
      <div class="row">
        <button id="play" type="button">Play the day</button>
      </div>
      <div class="row" style="margin-top: var(--s2)">
        <label class="eyebrow" for="hour-from">From</label>
        <input type="range" id="hour-from" min="__HOUR_MIN__" max="__HOUR_MAX__"
               value="__HOUR_MIN__" style="flex: 1 1 110px">
        <label class="eyebrow" for="hour-to">To</label>
        <input type="range" id="hour-to" min="__HOUR_MIN__" max="__HOUR_MAX__"
               value="__HOUR_MAX__" style="flex: 1 1 110px">
      </div>
      <p class="eyebrow num" id="window-label" style="margin-top: var(--s2)"></p>
    </section>

    <section>
      <h2>Routes</h2>
      <div class="row">
        <input type="search" id="line-search" placeholder="filter lines" aria-label="Filter lines">
        <button id="toggle-all" type="button">Deselect all</button>
        <button id="reset-view" type="button">Reset view</button>
      </div>
      <div class="scroll" id="line-row" style="margin-top: var(--s3)"></div>
      <label class="eyebrow" for="min-passengers" style="margin-top: var(--s3)">
        At least <span id="min-label">1</span> rider(s) per ride
      </label>
      <input type="range" id="min-passengers" min="1" max="20" value="1">
      <p class="eyebrow" id="cube-count"></p>
    </section>

    <section>
      <h2>Legend</h2>
      <div class="legend">
        <div class="legend-item">
          <span class="swatch" style="background: var(--boarded)"></span>
          early in the day
        </div>
        <div class="legend-item">
          <span class="swatch" style="background: var(--waiting)"></span>
          late in the day (colour ramps with the hour)
        </div>
        <div class="legend-item">
          <span class="swatch line" style="background: var(--ink-faint)"></span>
          the lattice: origin across, destination back, hours up
        </div>
        <div class="legend-item">
          <span class="swatch" style="background: var(--stranded)"></span>
          cube size — riders on that ride
        </div>
      </div>
      <p class="eyebrow" style="margin-top: var(--s3)">
        Drag to rotate, scroll to zoom, right-drag to pan, hover a cube for the ride
        behind it.
      </p>
    </section>

    <footer>
      Stops: the __STOP_TOTAL__ of the network with coordinates (the busiest
      __STOP_COUNT__ are on the lattice; __OMITTED__ are omitted, so the cube stays
      readable). Coordinates from
      <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors (ODbL).
    </footer>
  </main>

  <div class="stage">
    <div id="cube" role="application" aria-label="Origin, destination and time cube"></div>
    <div id="labels"></div>
    <div id="tooltip" hidden></div>
  </div>
</div>

<script>
const DATA = __DATA__;
const TOKENS = DATA.tokens;
__FILTER_JS__
__CUBE_JS__
DATA.lines.forEach(function (line) { state.lines[line] = true; });

function refresh() {
  rebuild();
  document.getElementById('window-label').textContent = windowLabel(state);
}
document.getElementById('min-passengers').addEventListener('input', function (event) {
  state.min = Number(event.target.value);
  document.getElementById('min-label').textContent = state.min;
  refresh();
});
document.getElementById('hour-from').addEventListener('input', function (event) {
  state.from = Number(event.target.value);
  refresh();
});
document.getElementById('hour-to').addEventListener('input', function (event) {
  state.to = Number(event.target.value);
  refresh();
});
document.querySelectorAll('input[name="mode"]').forEach(function (radio) {
  radio.addEventListener('change', function () { state.mode = radio.value; refresh(); });
});
document.getElementById('line-search').addEventListener('input', function () {
  lineRows(DATA, state, refresh);
});
document.getElementById('toggle-all').addEventListener('click', function () {
  const on = bulkToggle(DATA, state);
  lineRows(DATA, state, refresh);
  document.getElementById('toggle-all').textContent = on ? 'Deselect all' : 'Select all';
});
const corners = [];
for (let x = -1; x <= 1; x += 2) {
  for (let y = 0; y <= 1; y += 1) {
    for (let z = -1; z <= 1; z += 2) {
      corners.push(new THREE.Vector3(x * (span.halfX + 1), y * span.y, z * (span.halfX + 1)));
    }
  }
}
const probe = new THREE.Vector3();

function fitDistance() {
  let worst = 0;
  corners.forEach(function (corner) {
    probe.copy(corner).project(camera);
    worst = Math.max(worst, Math.abs(probe.x), Math.abs(probe.y));
  });
  return worst;
}

function frameCube() {
  controls.target.set(0, span.y / 2, 0);
  const direction = new THREE.Vector3(0.62, 0.5, 0.78).normalize();
  let distance = Math.hypot(span.halfX, span.y, span.halfX) * 1.6;
  for (let pass = 0; pass < 4; pass += 1) {
    camera.position.copy(direction).multiplyScalar(distance).add(controls.target);
    camera.lookAt(controls.target);
    camera.updateMatrixWorld();
    const reach = fitDistance();
    if (reach > 0) { distance *= reach / 0.86; }
  }
  controls.update();
}
document.getElementById('reset-view').addEventListener('click', frameCube);

let playing = null;
document.getElementById('play').addEventListener('click', function (event) {
  if (playing) {
    clearInterval(playing);
    playing = null;
    event.target.textContent = 'Play the day';
    return;
  }
  event.target.textContent = 'Pause';
  playing = setInterval(function () {
    state.from += 1;
    state.to += 1;
    if (state.from > DATA.hours.max) { state.from = DATA.hours.min; state.to = DATA.hours.min; }
    document.getElementById('hour-from').value = state.from;
    document.getElementById('hour-to').value = state.to;
    refresh();
  }, 700);
});

lineRows(DATA, state, refresh);
buildLabels();
resize();
frameCube();
document.getElementById('min-label').textContent = state.min;
refresh();
frame();
</script>
</body>
</html>
"""


def render_cube(
    cube: dict,
    config: ScenarioConfig,
    report,
    *,
    hours: dict,
    shifted: int,
    stop_total: int,
) -> str:
    """The space-time cube page: the OD relation as a tensor you can turn."""
    payload = {
        "stops": cube["stops"],
        "points": cube["points"],
        "lines": cube["lines"],
        "operators": cube["operators"],
        "groups": cube["groups"],
        "hours": hours,
        "tokens": TOKENS,
    }
    return (
        CUBE_TEMPLATE.replace("__CSS__", COMMON_CSS.replace("__TOKENS__", _css_tokens()))
        .replace("__THREE__", THREE_JS)
        .replace("__THREE_SRI__", THREE_JS_SRI)
        .replace("__ORBIT__", ORBIT_JS)
        .replace("__ORBIT_SRI__", ORBIT_JS_SRI)
        .replace("__FILTER_JS__", FILTER_JS)
        .replace("__CUBE_JS__", CUBE_JS)
        .replace("__DATA__", json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
        .replace("__TITLE__", _escape(report.label))
        .replace("__SEED__", str(config.seed))
        .replace("__FLEET_POLICY__", _escape(config.fleet_policy))
        .replace("__FINGERPRINT__", _escape(report.fingerprint))
        .replace("__POINTS__", f"{len(cube['points']):,}")
        .replace("__STOP_COUNT__", f"{len(cube['stops']):,}")
        .replace("__STOP_TOTAL__", f"{stop_total:,}")
        .replace("__OMITTED__", f"{cube.get('omitted', 0):,}")
        .replace("__HOUR_COUNT__", f"{hours['max'] - hours['min'] + 1}")
        .replace("__HOUR_MIN__", str(hours["min"]))
        .replace("__HOUR_MAX__", str(hours["max"]))
        .replace("__SHIFTED__", f"{shifted:,}")
        .replace("__BOARDED__", f"{report.boardings:,}")
        .replace("__ALIGHTED__", f"{report.alightings:,}")
    )


MAP_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__ — synthetic passenger flows</title>
<link rel="stylesheet" href="__LEAFLET_CSS__" integrity="__LEAFLET_CSS_SRI__" crossorigin="">
<style>
__CSS__
  .app { grid-template-columns: 384px 1fr; }
  #map { position: absolute; inset: 0; background: var(--surface-sunken); }
  .leaflet-container { background: var(--surface-sunken); }
  .layers { display: flex; flex-direction: column; gap: var(--s2); }
  .map-wrap { position: relative; height: 100vh; }
  @media (max-width: 1024px) {
    .map-wrap { height: 62vh; }
    .stage { height: 62vh; }
  }
</style>
</head>
<body>
<div class="app">
  <main class="panel">
    <div>
      <p class="eyebrow">inMotion · synthetic passenger flows</p>
      <h1>__TITLE__</h1>
      <p class="eyebrow mono">seed __SEED__ · __FLEET_POLICY__ · __FINGERPRINT__</p>
    </div>

    <dl class="stats">
      <div class="stat"><dt>Boarded</dt><dd class="num">__BOARDED__</dd></div>
      <div class="stat"><dt>Alighted</dt><dd class="num">__ALIGHTED__</dd></div>
      <div class="stat"><dt>Left behind</dt><dd class="num">__DENIED__</dd></div>
      <div class="stat"><dt>Never boarded</dt><dd class="num">__STRANDED__</dd></div>
      <div class="stat"><dt>Peak load factor</dt><dd class="num">__PEAK__</dd></div>
      <div class="stat"><dt>Stops placed</dt><dd class="num">__PLACED__</dd></div>
    </dl>

__FILTERS__
__TIME__

    <section>
      <h2>Stop</h2>
      <div class="detail" id="detail">
        <p class="eyebrow">Select a stop on the map for its numbers.</p>
      </div>
    </section>

    <section>
      <h2>Legend</h2>
      <div class="legend">
        <div class="legend-item"><span class="swatch" style="background: var(--boarded)"></span>
          stop area — passengers boarded there</div>
        <div class="legend-item">
          <span class="swatch" style="background: var(--waiting)"></span>
          stop ring — riders left waiting
        </div>
        <div class="legend-item">
          <span class="swatch line" style="background: var(--boarded)"></span>
          OD relation — width is passengers carried
        </div>
        <div class="legend-item">
          <span class="swatch line" style="background: var(--ink-faint)"></span>
          network leg — the timetable's own geometry
        </div>
        <div class="legend-item">
          <span class="swatch" style="background: var(--stranded)"></span>
          never boarded by the end of the day
        </div>
      </div>
      <p class="eyebrow" style="margin-top: var(--s3)">
        __STOP_COUNT__ stops placed, __FLOW_COUNT__ OD relations drawn
        (the busiest of __MATRIX_CELLS__ matrix cells).
        <a href="__CUBE_FILE__">The same day as a space-time cube →</a>
      </p>
    </section>

    <section>
      <h2>How these flows were generated</h2>
      <div class="equation">
        <div class="eq-lead">
          T<sub>ij</sub> ∝ A<sub>i</sub> · A<sub>j</sub> · e<sup>−β·t<sub>ij</sub></sup>
        </div>
        <div>
          A<sub>i</sub> = w<sub>i</sub> · (1 + lines<sub>i</sub>)<sup>α</sup>,  α = __ALPHA__
        </div>
        <div>β = __BETA__ per minute, t<sub>ij</sub> = quickest direct ride</div>
        <div>Σ<sub>j</sub> T<sub>ij</sub> scaled to the day's target, drawn per rider</div>
        <div>boarded(i, t) ≤ capacity(t) − onboard(t)</div>
      </div>
      <ul class="terms">
        <li><b>A<sub>i</sub></b>
          <span>what a stop draws: its lines served, times any manual weight</span></li>
        <li><b>t<sub>ij</sub></b>
          <span>the shortest direct ride between the two stops, in minutes</span></li>
        <li><b>β</b>
          <span>how sharply a longer ride is avoided; higher means shorter trips</span></li>
        <li><b>α</b>
          <span>how much a well-served stop is preferred over a thin one</span></li>
        <li><b>capacity</b><span>the bus assigned to the trip: __FLEET__</span></li>
      </ul>
      <p class="eyebrow" style="margin-top: var(--s3)">
        Riders who do not fit stay at the stop and are picked up by the next bus for
        their destination; what is left when the day's last bus has gone is counted as
        never boarded. Every number is generated from the scenario above and
        reproduces from its seed.
      </p>
      <p class="callout" id="provenance"></p>
    </section>

    <footer>
      Stop names from the operators' own timetables; coordinates from
      <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors (ODbL).
      Basemap: __BASEMAP__ — __BASEMAP_ATTRIBUTION__.
    </footer>
  </main>

  <div class="map-wrap">
    <div id="map" role="application" aria-label="Passenger flows over the bus network"></div>
  </div>
</div>

<script src="__LEAFLET_JS__" integrity="__LEAFLET_JS_SRI__" crossorigin=""></script>
<script>
const DATA = __DATA__;
const TOKENS = DATA.tokens;
const state = {
  groups: {}, operators: {}, lines: {}, allLines: true,
  min: 1, from: DATA.hours.min, to: DATA.hours.max, text: ''
};
DATA.groups.forEach(function (group) { state[group] = true; });
DATA.operators.forEach(function (operator) { state[operator] = true; });
DATA.lines.forEach(function (line) { state.lines[line] = true; });

__FILTER_JS__
__SPARKLINE_JS__

const map = L.map('map', { zoomControl: true, preferCanvas: true });
let tileLayer = null;

function drawBasemap(name) {
  const chosen = DATA.basemaps.filter(function (entry) { return entry.name === name; })[0];
  if (tileLayer) { map.removeLayer(tileLayer); tileLayer = null; }
  document.getElementById('basemap-note').textContent = chosen.note || '';
  if (!chosen.available || !chosen.url) { return; }
  const options = { maxZoom: chosen.max_zoom, attribution: chosen.attribution };
  if (chosen.subdomains) { options.subdomains = chosen.subdomains; }
  tileLayer = L.tileLayer(chosen.url, options).addTo(map);
}

const legLayer = L.layerGroup().addTo(map);
const stopLayer = L.layerGroup().addTo(map);
const flowLayer = L.layerGroup().addTo(map);
const bounds = [];

DATA.legs.forEach(function (leg) {
  L.polyline([[leg[0][1], leg[0][0]], [leg[1][1], leg[1][0]]], {
    color: TOKENS['ink-faint'], weight: 1, opacity: 0.45, interactive: false
  }).addTo(legLayer);
});

let _max = null;
let _maxFlow = null;
function maxBoarded() {
  if (_max === null) {
    _max = DATA.stops.reduce(function (best, f) {
      return Math.max(best, f.properties.boarded);
    }, 1);
  }
  return _max;
}
function maxFlow() {
  if (_maxFlow === null) {
    _maxFlow = DATA.flows.reduce(function (best, f) {
      return Math.max(best, f.properties.passengers);
    }, 1);
  }
  return _maxFlow;
}
function radius(boarded) {
  return 4 + 10 * Math.sqrt(Math.max(boarded, 0) / maxBoarded());
}
function stopMatches(feature) {
  if (!state.text) { return true; }
  return feature.properties.name.toLowerCase().indexOf(state.text) >= 0;
}
function visibleStop(feature) {
  const props = feature.properties;
  return state[props.operator] !== false && stopMatches(feature);
}
function visibleFlow(feature, window) {
  const props = feature.properties;
  if (state[props.operator] === false) { return false; }
  if (state[props.service_group] === false) { return false; }
  if (!lineMatches(DATA, state, feature)) { return false; }
  return hourRiders(feature, window) >= state.min;
}

DATA.stops.forEach(function (feature) {
  const p = feature.properties;
  const point = [feature.geometry.coordinates[1], feature.geometry.coordinates[0]];
  bounds.push(point);
  const marker = L.circleMarker(point, {
    radius: radius(p.boarded),
    color: p.denied > 0 ? TOKENS.stranded : TOKENS.boarded,
    weight: p.denied > 0 ? 2 : 1,
    fillColor: TOKENS.boarded,
    fillOpacity: 0.55
  });
  marker.bindTooltip(p.name + ' — ' + p.boarded + ' boarded', { direction: 'top' });
  marker.on('click', function () { showStop(p); });
  marker.feature = feature;
  stopLayer.addLayer(marker);
});

DATA.flows.forEach(function (feature) {
  const p = feature.properties;
  const line = L.polyline(
    [
      [feature.geometry.coordinates[0][1], feature.geometry.coordinates[0][0]],
      [feature.geometry.coordinates[1][1], feature.geometry.coordinates[1][0]]
    ],
    { color: TOKENS.boarded, weight: 1, opacity: 0.5 }
  );
  line.bindTooltip(p.from + ' → ' + p.to + ' — ' + p.passengers + ' riders', { direction: 'top' });
  line.feature = feature;
  flowLayer.addLayer(line);
});

function showStop(props) {
  const detail = document.getElementById('detail');
  detail.textContent = '';
  const title = document.createElement('h3');
  title.textContent = props.name;
  detail.appendChild(title);
  const caption = document.createElement('p');
  caption.className = 'eyebrow';
  caption.textContent = props.operator + ' · ' + (props.lines || 'no lines') +
    ' · coords: ' + props.source;
  detail.appendChild(caption);
  const list = document.createElement('dl');
  [
    ['Boarded', props.boarded],
    ['Alighted', props.alighted],
    ['Left waiting', props.denied],
    ['Peak queue', props.peak_queue],
    ['Place group', props.cluster_id < 0 ? '—' : '#' + props.cluster_id]
  ].forEach(function (pair) {
    const term = document.createElement('dt');
    term.textContent = pair[0];
    const value = document.createElement('dd');
    value.className = 'num';
    value.textContent = pair[1];
    list.appendChild(term);
    list.appendChild(value);
  });
  detail.appendChild(list);
}

function apply() {
  const window = hourWindow(DATA, state);
  let stops = 0;
  let flows = 0;
  stopLayer.eachLayer(function (layer) {
    const on = visibleStop(layer.feature);
    layer.setStyle({ opacity: on ? 1 : 0, fillOpacity: on ? 0.55 : 0 });
    if (on) { stops += 1; }
  });
  flowLayer.eachLayer(function (layer) {
    const riders = hourRiders(layer.feature, window);
    const on = visibleFlow(layer.feature, window);
    layer.setStyle({
      opacity: on ? 0.35 + 0.5 * Math.sqrt(riders / Math.max(maxFlow(), 1)) : 0,
      weight: on ? 1 + 4 * Math.sqrt(riders / Math.max(maxFlow(), 1)) : 0
    });
    if (on) { flows += 1; }
  });
  document.getElementById('visible').textContent =
    stops + ' stops and ' + flows + ' OD relations in ' +
    windowLabel(state) + ' (at least ' + state.min + ' rider' + (state.min === 1 ? '' : 's') + ')';
  document.getElementById('window-label').textContent = windowLabel(state);
  sparkline(DATA, state, function (hour) {
    state.from = hour;
    state.to = hour;
    document.getElementById('hour-from').value = hour;
    document.getElementById('hour-to').value = hour;
    apply();
  });
  document.getElementById('toggle-all').textContent = allOn(DATA, state)
    ? 'Deselect all' : 'Select all';
}

const basemapSelect = document.getElementById('basemap');
DATA.basemaps.forEach(function (entry) {
  const option = document.createElement('option');
  option.value = entry.name;
  option.textContent = entry.available ? entry.name : entry.name + ' — ' + entry.note;
  option.disabled = !entry.available;
  option.selected = entry.name === DATA.basemap.name;
  basemapSelect.appendChild(option);
});
basemapSelect.addEventListener('change', function () { drawBasemap(basemapSelect.value); });
drawBasemap(DATA.basemap.name);

options('group-row', DATA.groups, state, apply);
options('operator-row', DATA.operators, state, apply);
lineRows(DATA, state, apply);
document.getElementById('line-search').addEventListener('input', function () {
  lineRows(DATA, state, apply);
});
document.getElementById('stop-search').addEventListener('input', function (event) {
  state.text = event.target.value.trim().toLowerCase();
  apply();
});
document.getElementById('min-passengers').addEventListener('input', function (event) {
  state.min = Number(event.target.value);
  document.getElementById('min-label').textContent = state.min;
  apply();
});
document.getElementById('hour-from').addEventListener('input', function (event) {
  state.from = Number(event.target.value);
  apply();
});
document.getElementById('hour-to').addEventListener('input', function (event) {
  state.to = Number(event.target.value);
  apply();
});
document.getElementById('toggle-all').addEventListener('click', function () {
  bulkToggle(DATA, state);
  lineRows(DATA, state, apply);
  apply();
});
document.getElementById('reset-view').addEventListener('click', function () {
  bulkToggle(DATA, state, true);
  state.min = 1;
  state.text = '';
  document.getElementById('min-passengers').value = 1;
  document.getElementById('min-label').textContent = '1';
  document.getElementById('stop-search').value = '';
  state.from = DATA.hours.min;
  state.to = DATA.hours.max;
  document.getElementById('hour-from').value = state.from;
  document.getElementById('hour-to').value = state.to;
  lineRows(DATA, state, apply);
  if (bounds.length) { map.fitBounds(bounds, { padding: [24, 24] }); }
  apply();
});

let playing = null;
let playCursor = DATA.hours.min;
document.getElementById('play').addEventListener('click', function (event) {
  if (playing) {
    clearInterval(playing);
    playing = null;
    event.target.textContent = 'Play the day';
    return;
  }
  event.target.textContent = 'Pause';
  playCursor = state.from;
  playing = setInterval(function () {
    playCursor += 1;
    if (playCursor > DATA.hours.max) { playCursor = DATA.hours.min; }
    state.from = playCursor;
    state.to = playCursor;
    document.getElementById('hour-from').value = playCursor;
    document.getElementById('hour-to').value = playCursor;
    apply();
  }, 700);
});

const provenance = document.getElementById('provenance');
if (provenance && DATA.provenance) {
  provenance.textContent = "What is real: the routes. " + DATA.provenance.trips +
    " trips, " + DATA.provenance.stops + " stops and every clock on this page come from " +
    "the operator timetables, and the run verified it. What is generated: the riders — " +
    "no operator recorded them, so the demand is a labelled scenario, not a measurement.";
}

if (bounds.length) {
  map.fitBounds(bounds, { padding: [24, 24] });
} else {
  map.setView([40.64, -8.65], 12);
}
apply();
</script>
</body>
</html>
"""


def render_map(
    stops: list[dict],
    flows: list[dict],
    legs: list[list[list[float]]],
    *,
    basemaps: list[dict],
    chosen: dict,
    hours: dict,
    lines: list[str],
    operators: list[str],
    groups: list[str],
    config: ScenarioConfig,
    report,
    matrix,
) -> str:
    """The map page: the day's flows on the ground, with filters and a clock."""
    payload = {
        "stops": stops,
        "flows": flows,
        "legs": legs,
        "lines": lines,
        "operators": operators,
        "groups": groups,
        "hours": hours,
        "basemaps": basemaps,
        "basemap": chosen,
        "tokens": TOKENS,
        "limit": len(flows),
        "provenance": {
            "trips": report.provenance.get("supply_trips", 0),
            "stops": report.provenance.get("supply_stops", 0),
        },
    }
    page = (
        MAP_TEMPLATE.replace("__CSS__", COMMON_CSS.replace("__TOKENS__", _css_tokens()))
        .replace("__FILTERS__", FILTER_MARKUP)
        .replace("__TIME__", TIME_MARKUP)
        .replace("__FILTER_JS__", FILTER_JS)
        .replace("__SPARKLINE_JS__", SPARKLINE_JS)
        .replace("__LEAFLET_CSS__", LEAFLET_CSS)
        .replace("__LEAFLET_CSS_SRI__", LEAFLET_CSS_SRI)
        .replace("__LEAFLET_JS__", LEAFLET_JS)
        .replace("__LEAFLET_JS_SRI__", LEAFLET_JS_SRI)
        .replace("__DATA__", json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
        .replace("__TITLE__", _escape(report.label))
        .replace("__STOP_COUNT__", f"{len(stops):,}")
        .replace("__FLOW_COUNT__", f"{len(flows):,}")
        .replace("__BASEMAP__", _escape(chosen["name"]))
        .replace("__BASEMAP_ATTRIBUTION__", chosen["attribution"] or "no basemap")
        .replace("__BOARDED__", f"{report.boardings:,}")
        .replace("__ALIGHTED__", f"{report.alightings:,}")
        .replace("__DENIED__", f"{report.denied:,}")
        .replace("__STRANDED__", f"{report.stranded:,}")
        .replace("__PEAK__", f"{report.peak_load_factor:.2f}")
        .replace("__PLACED__", f"{report.coords_resolved:,}/{report.coords_total:,}")
        .replace("__ALPHA__", f"{config.alpha:g}")
        .replace("__BETA__", f"{config.beta:g}")
        .replace("__FLEET_POLICY__", _escape(config.fleet_policy))
        .replace("__FLEET__", _escape(report.fleet or "none"))
        .replace("__SEED__", str(config.seed))
        .replace("__FINGERPRINT__", _escape(report.fingerprint))
        .replace("__MATRIX_CELLS__", f"{matrix.height:,}")
    )
    return page.replace("__CUBE_FILE__", "flows_cube.html")


REPLAY_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__ — route replay</title>
<link rel="stylesheet" href="__LEAFLET_CSS__" integrity="__LEAFLET_CSS_SRI__" crossorigin="">
<style>
__CSS__
  .app { grid-template-columns: 384px 1fr; }
  .map-wrap { position: relative; height: 100vh; }
  #map { position: absolute; inset: 0; background: var(--surface-sunken); }
  .leaflet-container { background: var(--surface-sunken); }
  .layers { display: flex; flex-direction: column; gap: var(--s2); }
  .clock {
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 34px; font-weight: 600; letter-spacing: -0.02em; line-height: 1.1;
  }
  .bus-icon {
    display: flex; align-items: center; gap: 6px;
    background: var(--surface-raised); border: 1px solid var(--boarded);
    border-radius: 999px; padding: 3px 9px; font-size: 12px; color: var(--ink);
    font-variant-numeric: tabular-nums; white-space: nowrap;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.45);
  }
  .bus-icon b { color: var(--boarded); font-weight: 700; }
  .people {
    display: flex; flex-wrap: wrap; align-content: flex-end; justify-content: center;
    gap: 2px; width: 30px; margin-bottom: 2px;
  }
  .person {
    width: 5px; height: 5px; border-radius: 50%;
    background: var(--waiting); box-shadow: 0 0 4px rgba(244, 185, 66, 0.6);
  }
  .person-count {
    font-size: 10.5px; color: var(--ink); background: var(--surface-raised);
    border: 1px solid var(--line); border-radius: 999px; padding: 0 5px;
    font-variant-numeric: tabular-nums;
  }
  .burst {
    font-size: 12px; font-weight: 700; font-variant-numeric: tabular-nums;
    text-shadow: 0 0 6px var(--surface); animation: rise 2s ease-out forwards;
  }
  .burst.up { color: var(--boarded); }
  .burst.down { color: var(--waiting); }
  .burst.left { color: var(--stranded); }
  @keyframes rise {
    from { opacity: 1; transform: translateY(0); }
    to { opacity: 0; transform: translateY(-26px); }
  }
  .route-line { stroke-linecap: round; }
  @media (max-width: 1024px) {
    .map-wrap { height: 62vh; }
  }
</style>
</head>
<body>
<div class="app">
  <main class="panel">
    <div>
      <p class="eyebrow">inMotion · one route, one day</p>
      <h1>__TITLE__</h1>
      <p class="eyebrow mono">seed __SEED__ · __FLEET_POLICY__ · __FINGERPRINT__</p>
    </div>

    <dl class="stats">
      <div class="stat"><dt>Route</dt><dd class="num" id="stat-route">—</dd></div>
      <div class="stat"><dt>Trips replayable</dt><dd class="num" id="stat-trips">—</dd></div>
      <div class="stat"><dt>Stops</dt><dd class="num" id="stat-stops">—</dd></div>
      <div class="stat"><dt>Boarded on this route</dt><dd class="num" id="stat-boarded">—</dd></div>
      <div class="stat"><dt>Buses on the road</dt><dd class="num" id="stat-live">0</dd></div>
      <div class="stat"><dt>People waiting</dt><dd class="num" id="stat-waiting">0</dd></div>
    </dl>

    <section>
      <h2>Route</h2>
      <div class="row">
        <select id="route" aria-label="Route"></select>
      </div>
      <p class="eyebrow" id="route-note"></p>
      <div class="row" style="margin-top: var(--s2)">
        <select id="basemap" aria-label="Basemap"></select>
      </div>
    </section>

    <section>
      <h2>The clock</h2>
      <div class="clock" id="clock">--:--</div>
      <div class="row" style="margin-top: var(--s2)">
        <button id="play" type="button">Play</button>
        <button id="rewind" type="button">Rewind</button>
        <label class="layer" for="speed">
          <select id="speed" aria-label="Speed">
            <option value="2">2 min/s</option>
            <option value="8" selected>8 min/s</option>
            <option value="30">30 min/s</option>
          </select>
        </label>
      </div>
      <input type="range" id="scrub" min="0" max="1440" value="0" style="margin-top: var(--s3)">
      <p class="eyebrow" id="progress">—</p>
    </section>

    <section>
      <h2>What you are seeing</h2>
      <div class="legend">
        <div class="legend-item"><span class="swatch" style="background: var(--boarded)"></span>
          a bus and how many riders are on it</div>
        <div class="legend-item">
          <span class="swatch" style="background: var(--waiting)"></span>
          one dot per waiting rider at a stop (up to 12, then the number)
        </div>
        <div class="legend-item">
          <span class="swatch line" style="background: var(--boarded)"></span>
          +N boarded when a bus calls (cyan)
        </div>
        <div class="legend-item">
          <span class="swatch line" style="background: var(--waiting)"></span>
          −N alighted (amber)
        </div>
        <div class="legend-item">
          <span class="swatch" style="background: var(--stranded)"></span>
          N left — did not fit on that bus (red)
        </div>
      </div>
      <p class="eyebrow" style="margin-top: var(--s3)">
        The buses move on the timetable, not on a guessed speed: each one is
        interpolated between the two stops it is between, and the boarding and
        alighting you see are the generated flows for that trip and stop. Trips
        that call at a stop OpenStreetMap could not place are left out, and so are
        the few whose timetables run backwards, because a bus that disappears for
        two stops — or jumps back up the route — is a lie about the geometry; the
        note under the route says how many that leaves.
      </p>
    </section>

    <footer>
      Stops and coordinates: the operators' timetables, placed from
      <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors (ODbL).
      Basemap: __BASEMAP__ — __BASEMAP_ATTRIBUTION__.
    </footer>
  </main>

  <div class="map-wrap">
    <div id="map" role="application" aria-label="A route's day, replayed"></div>
  </div>
</div>

<script src="__LEAFLET_JS__" integrity="__LEAFLET_JS_SRI__" crossorigin=""></script>
<script>
const DATA = __DATA__;
const TOKENS = DATA.tokens;
const state = { route: 0, clock: 0, playing: false, speed: 8, last: null };

const map = L.map('map', { zoomControl: true, preferCanvas: true });
let tileLayer = null;

function drawBasemap(name) {
  const chosen = DATA.basemaps.filter(function (entry) { return entry.name === name; })[0];
  if (tileLayer) { map.removeLayer(tileLayer); tileLayer = null; }
  if (!chosen.available || !chosen.url) { return; }
  const options = { maxZoom: chosen.max_zoom, attribution: chosen.attribution };
  if (chosen.subdomains) { options.subdomains = chosen.subdomains; }
  tileLayer = L.tileLayer(chosen.url, options).addTo(map);
}

const routeLayer = L.layerGroup().addTo(map);
const stopLayer = L.layerGroup().addTo(map);
const peopleLayer = L.layerGroup().addTo(map);
const busLayer = L.layerGroup().addTo(map);
const burstLayer = L.layerGroup().addTo(map);

function clockLabel(minutes) {
  const hour = Math.floor(minutes / 60);
  const minute = Math.floor(minutes % 60);
  return (hour < 10 ? '0' : '') + hour + ':' + (minute < 10 ? '0' : '') + minute;
}

function currentRoute() { return DATA.routes[state.route]; }

function stopOf(index) { return DATA.stops[index]; }

function drawRoute() {
  routeLayer.clearLayers();
  stopLayer.clearLayers();
  peopleLayer.clearLayers();
  busLayer.clearLayers();
  burstLayer.clearLayers();
  const route = currentRoute();
  const points = route.shape.map(function (index) {
    const stop = stopOf(index);
    return [stop.lat, stop.lon];
  });
  L.polyline(points, {
    color: TOKENS['ink-faint'], weight: 7, opacity: 0.55, className: 'route-line'
  }).addTo(routeLayer);
  L.polyline(points, { color: TOKENS.boarded, weight: 2, opacity: 0.55 }).addTo(routeLayer);
  if (!points.length) { return; }
  const bounds = L.latLngBounds(points);
  map.fitBounds(bounds, { padding: [40, 40] });
  if (map.getZoom() > 14) { map.setZoom(14); }

  route.stops.forEach(function (index) {
    const stop = stopOf(index);
    L.circleMarker([stop.lat, stop.lon], {
      radius: 4, color: TOKENS['ink-faint'], weight: 1,
      fillColor: TOKENS.surface, fillOpacity: 0.9
    }).bindTooltip(stop.name, { direction: 'top' }).addTo(stopLayer);
    stop.node = L.marker([stop.lat, stop.lon], {
      icon: L.divIcon({ className: '', html: '', iconSize: [36, 44], iconAnchor: [18, 40] })
    }).addTo(peopleLayer);
  });

  route.trips.forEach(function (trip) {
    trip.marker = L.marker([0, 0], {
      icon: L.divIcon({ className: '', html: '', iconSize: [92, 26], iconAnchor: [46, 13] }),
      zIndexOffset: 1000
    }).addTo(busLayer);
    trip.marker.setOpacity(0);
    trip.lastMinute = null;
  });

  document.getElementById('stat-route').textContent = route.line + ' · ' + route.direction;
  document.getElementById('stat-trips').textContent = route.trips.length + ' / ' + route.trips_all;
  document.getElementById('stat-stops').textContent = String(route.shape.length);
  document.getElementById('stat-boarded').textContent = String(route.boarded);
  const missing = route.trips_all - route.trips.length;
  document.getElementById('route-note').textContent = missing > 0
    ? missing + ' trips cannot be replayed: a stop with no coordinates, or a '
      + 'timetable whose times run backwards.'
    : 'Every trip on this route can be replayed.';
  const scrub = document.getElementById('scrub');
  scrub.min = String(route.day.from);
  scrub.max = String(route.day.to);
  scrub.value = String(route.day.from);
  state.clock = route.day.from;
}

function busIcon(onboard, capacity) {
  return '<div class="bus-icon"><b>' + onboard + '</b>/' + capacity + '</div>';
}

function peopleIcon(count) {
  if (count <= 0) { return '<div></div>'; }
  let dots = '';
  const shown = Math.min(count, 12);
  for (let index = 0; index < shown; index += 1) { dots += '<span class="person"></span>'; }
  return '<div class="people">' + dots + '</div><div class="person-count">' + count + '</div>';
}

function burstAt(stop, minutes, boarded, alighted, denied) {
  if (boarded > 0) {
    const up = L.marker([stop.lat, stop.lon], {
      icon: L.divIcon({
        className: '', html: '<div class="burst up">+' + boarded + '</div>',
        iconSize: [40, 18], iconAnchor: [20, 10]
      })
    }).addTo(burstLayer);
    setTimeout(function () { burstLayer.removeLayer(up); }, 2000);
  }
  if (alighted > 0) {
    const down = L.marker([stop.lat, stop.lon], {
      icon: L.divIcon({
        className: '', html: '<div class="burst down">−' + alighted + '</div>',
        iconSize: [40, 18], iconAnchor: [20, 30]
      })
    }).addTo(burstLayer);
    setTimeout(function () { burstLayer.removeLayer(down); }, 2000);
  }
  if (denied > 0) {
    const left = L.marker([stop.lat, stop.lon], {
      icon: L.divIcon({
        className: '', html: '<div class="burst left">' + denied + ' left</div>',
        iconSize: [60, 18], iconAnchor: [30, 46]
      })
    }).addTo(burstLayer);
    setTimeout(function () { burstLayer.removeLayer(left); }, 2400);
  }
}

function positionOf(trip, clock) {
  const events = trip.events;
  if (clock < events[0][1] || clock > events[events.length - 1][1] + 0.5) { return null; }
  let index = 0;
  for (let cursor = 0; cursor < events.length; cursor += 1) {
    if (events[cursor][1] <= clock) { index = cursor; }
  }
  const here = events[index];
  const next = events[index + 1];
  const from = stopOf(here[0]);
  let lat = from.lat;
  let lon = from.lon;
  let fraction = 0;
  if (next) {
    const to = stopOf(next[0]);
    const span = Math.max(next[1] - here[1], 0.001);
    fraction = Math.min(Math.max((clock - here[1]) / span, 0), 1);
    lat = from.lat + (to.lat - from.lat) * fraction;
    lon = from.lon + (to.lon - from.lon) * fraction;
  }
  return { lat: lat, lon: lon, event: here, index: index, fraction: fraction };
}

function drawFrame() {
  const route = currentRoute();
  const clock = state.clock;
  let live = 0;
  let waiting = 0;

  route.trips.forEach(function (trip) {
    const place = positionOf(trip, clock);
    if (!place) {
      if (trip.marker) { trip.marker.setOpacity(0); }
      trip.lastMinute = null;
      return;
    }
    live += 1;
    trip.marker.setLatLng([place.lat, place.lon]);
    trip.marker.setOpacity(1);
    trip.marker.setIcon(L.divIcon({
      className: '', html: busIcon(place.event[4], trip.capacity),
      iconSize: [92, 26], iconAnchor: [46, 13]
    }));
    if (trip.lastMinute !== place.event[1]) {
      if (trip.lastMinute !== null) {
        burstAt(
          stopOf(place.event[0]), place.event[1],
          place.event[2], place.event[3], place.event[6]
        );
      }
      trip.lastMinute = place.event[1];
    }
  });

  route.stops.forEach(function (index) {
    const stop = stopOf(index);
    let value = 0;
    let latest = null;
    route.trips.forEach(function (trip) {
      trip.events.forEach(function (event) {
        if (event[0] === index && event[1] <= clock) {
          if (!latest || event[1] > latest[1]) { latest = event; }
        }
      });
    });
    if (latest) { value = Math.max(latest[5] - latest[2], 0); }
    waiting += value;
    if (stop.node) {
      stop.node.setIcon(L.divIcon({
        className: '', html: peopleIcon(value), iconSize: [36, 44], iconAnchor: [18, 40]
      }));
    }
  });

  document.getElementById('clock').textContent = clockLabel(clock);
  document.getElementById('stat-live').textContent = String(live);
  document.getElementById('stat-waiting').textContent = String(waiting);
  document.getElementById('progress').textContent =
    clockLabel(clock) + ' of ' + clockLabel(route.day.to) + ' · ' + state.speed + ' min/s';
  document.getElementById('scrub').value = String(clock);
}

function tick(now) {
  if (state.playing) {
    if (state.last === null) { state.last = now; }
    const seconds = (now - state.last) / 1000;
    state.last = now;
    state.clock += seconds * state.speed;
    const route = currentRoute();
    if (state.clock > route.day.to + 1) {
      state.clock = route.day.from;
      route.trips.forEach(function (trip) { trip.lastMinute = null; });
    }
  } else {
    state.last = null;
  }
  drawFrame();
  requestAnimationFrame(tick);
}

const select = document.getElementById('route');
DATA.routes.forEach(function (route, index) {
  const option = document.createElement('option');
  option.value = String(index);
  option.textContent = route.label + ' · ' + route.trips.length + ' trips';
  select.appendChild(option);
});
select.addEventListener('change', function () {
  state.route = Number(select.value);
  currentRoute().trips.forEach(function (trip) { trip.lastMinute = null; });
  drawRoute();
});

const basemapSelect = document.getElementById('basemap');
DATA.basemaps.forEach(function (entry) {
  const option = document.createElement('option');
  option.value = entry.name;
  option.textContent = entry.available ? entry.name : entry.name + ' — ' + entry.note;
  option.disabled = !entry.available;
  option.selected = entry.name === DATA.basemap.name;
  basemapSelect.appendChild(option);
});
basemapSelect.addEventListener('change', function () { drawBasemap(basemapSelect.value); });

document.getElementById('play').addEventListener('click', function (event) {
  state.playing = !state.playing;
  event.target.textContent = state.playing ? 'Pause' : 'Play';
});
document.getElementById('rewind').addEventListener('click', function () {
  state.clock = currentRoute().day.from;
  currentRoute().trips.forEach(function (trip) { trip.lastMinute = null; });
  drawFrame();
});
document.getElementById('speed').addEventListener('change', function (event) {
  state.speed = Number(event.target.value);
});
document.getElementById('scrub').addEventListener('input', function (event) {
  state.clock = Number(event.target.value);
  currentRoute().trips.forEach(function (trip) { trip.lastMinute = null; });
  drawFrame();
});

DATA.stops.forEach(function (stop) { stop.node = null; });

drawBasemap(DATA.basemap.name);
drawRoute();
drawFrame();
requestAnimationFrame(tick);
</script>
</body>
</html>
"""


def render_replay(
    replay: dict,
    *,
    basemaps: list[dict],
    chosen: dict,
    config: ScenarioConfig,
    report,
) -> str:
    """The replay page: a route's day, its buses moving, its people waiting."""
    payload = {
        "routes": replay["routes"],
        "stops": replay["stops"],
        "omitted": replay["omitted"],
        "basemaps": basemaps,
        "basemap": chosen,
        "tokens": TOKENS,
    }
    return (
        REPLAY_TEMPLATE.replace("__CSS__", COMMON_CSS.replace("__TOKENS__", _css_tokens()))
        .replace("__LEAFLET_CSS__", LEAFLET_CSS)
        .replace("__LEAFLET_CSS_SRI__", LEAFLET_CSS_SRI)
        .replace("__LEAFLET_JS__", LEAFLET_JS)
        .replace("__LEAFLET_JS_SRI__", LEAFLET_JS_SRI)
        .replace("__DATA__", json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
        .replace("__TITLE__", _escape(report.label))
        .replace("__SEED__", str(config.seed))
        .replace("__FLEET_POLICY__", _escape(config.fleet_policy))
        .replace("__FINGERPRINT__", _escape(report.fingerprint))
        .replace("__BASEMAP__", _escape(chosen["name"]))
        .replace("__BASEMAP_ATTRIBUTION__", chosen["attribution"] or "no basemap")
    )

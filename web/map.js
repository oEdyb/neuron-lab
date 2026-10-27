import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const $ = (s) => document.querySelector(s);
const host = $('#viewport'), panel = $('#neuron-panel'), tip = $('#tooltip');
const scene = new THREE.Scene(), camera = new THREE.PerspectiveCamera(40, 1, .05, 500);
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: 'high-performance' });
renderer.setPixelRatio(Math.min(devicePixelRatio, 1.5));
host.append(renderer.domElement);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = .12;
controls.zoomToCursor = true;
controls.minDistance = 1;
controls.maxDistance = 160;
controls.maxPolarAngle = Math.PI * .95;
const layers = new Map(), details = new Map(), labels = [], pointer = new THREE.Vector2(), ray = new THREE.Raycaster();
const tmp = new THREE.Vector3(), plane = new THREE.Plane(new THREE.Vector3(0, 0, 1));
let manifest, selected, focused = null, tween, dragging = false, down, hoverKey = '', searchVersion = 0, sign = 0;
const spacing = 1.15;
const escape = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const tokens = (words, cls) => `<div class="tokens ${cls}">${words.map(w => `<span>${escape(w)}</span>`).join('')}</div>`;
function error(e) { $('#map-error').textContent = e.message; $('#map-error').hidden = false; }
async function fetchData(name) { const r = await fetch(`/api/view/${name}`); if (!r.ok) throw new Error('Map unavailable. Run uv run python build_view.py.'); return r; }
function detail(layer) { if (!details.has(layer)) details.set(layer, fetchData(`L${layer}.json`).then(r => r.json())); return details.get(layer); }
function move(position, target) { tween = { from: camera.position.clone(), to: position, start: controls.target.clone(), target, time: performance.now() }; }
function fit() {
  focused = null;
  const center = new THREE.Vector3(0, 0, (manifest.total_layers - 1) * spacing / 2);
  const direction = new THREE.Vector3(1.12, .68, 1.25).normalize();
  const right = new THREE.Vector3().crossVectors(camera.up, direction).normalize();
  const up = new THREE.Vector3().crossVectors(direction, right);
  const tangent = Math.tan(THREE.MathUtils.degToRad(camera.fov / 2));
  let distance = 0;
  for (const x of [-7.8, 7.8]) for (const y of [-5.5, 6.3]) for (const z of [-center.z, center.z]) {
    const corner = new THREE.Vector3(x, y, z), depth = corner.dot(direction);
    distance = Math.max(distance, depth + Math.abs(corner.dot(right)) / (tangent * camera.aspect * .9), depth + Math.abs(corner.dot(up)) / (tangent * .78));
  }
  move(center.clone().add(direction.multiplyScalar(distance)), center);
  paintFocus();
}
function face(layer) {
  if (!layers.has(layer)) return;
  focused = layer;
  const target = new THREE.Vector3(0, 0, layer * spacing);
  const distance = Math.max(6, 8 / camera.aspect) / Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)) * 1.12;
  move(target.clone().add(new THREE.Vector3(0, 0, distance)), target);
  paintFocus();
}
function paintFocus() {
  for (const [l, obj] of layers) {
    obj.points.visible = focused === null || l === focused;
    obj.points.material.uniforms.fade.value = 1;
    obj.sheet.visible = focused === null || l === focused;
    obj.edge.visible = focused === null || l === focused;
    obj.sheet.material.opacity = focused === null ? .025 : l === focused ? .055 : .003;
    obj.edge.material.opacity = focused === null || l === focused ? .33 : .035;
  }
  labels.forEach(({ button, layer }) => button.classList.toggle('active', focused === layer));
}
const vertex = `attribute float strength; varying float light; varying float core; uniform vec2 viewport; uniform vec2 stepSize; uniform float dotSize; uniform float gap; uniform float glow; uniform float pixelRatio; void main(){
light=clamp(log2(max(strength,1.0))/5.0,0.0,1.0);
vec4 p=modelViewMatrix*vec4(position,1.0); vec4 clip=projectionMatrix*p;
vec4 px=projectionMatrix*(p+modelViewMatrix*vec4(stepSize.x,0.,0.,0.));
vec4 py=projectionMatrix*(p+modelViewMatrix*vec4(0.,stepSize.y,0.,0.));
float space=min(length((px.xy/px.w-clip.xy/clip.w)*viewport*.5),length((py.xy/py.w-clip.xy/clip.w)*viewport*.5));
float diameter=max(.7,min(dotSize,space-gap));
float halo=max(diameter,min(diameter*glow,space-gap));
core=diameter/halo; gl_Position=clip; gl_PointSize=halo*pixelRatio;
}`;
const fragment = `varying float light; varying float core; uniform float fade; uniform float faint; void main(){float d=length(gl_PointCoord-.5)*2.;if(d>1.)discard;
float dot=1.-smoothstep(core*.65,core,d);
float halo=exp(-d*d*5.)*(1.-smoothstep(.7,1.,d))*light*.5;
float a=(dot*(faint+light*.76)+halo)*fade;
gl_FragColor=vec4(mix(vec3(.32,.55,.72),vec3(.70,.90,1.),light),a);}`;
function gridSize(count) {
  const [columns, rows] = manifest.grid;
  if (columns * rows === count) return [columns, rows];
  let width = Math.ceil(Math.sqrt(count * columns / rows));
  return [width, Math.ceil(count / width)];
}
function addLayer(layer, data) {
  const n = data.length / 3, positions = new Float32Array(n * 3), strength = new Float32Array(n), grid = new Map();
  const [columns, rows] = gridSize(n), stepX = 13.2 / (columns - 1), stepY = 8.8 / (rows - 1);
  const order = Array.from({length:n}, (_, i) => i).sort((a,b) => data[a*3+1]-data[b*3+1] || a-b);
  const cells = new Int32Array(n);
  for (let row = 0; row < rows; row++) {
    const band = order.slice(row*columns, (row+1)*columns).sort((a,b) => data[a*3]-data[b*3] || a-b);
    band.forEach((id, column) => { cells[id] = row*columns+column; });
  }
  for (let i = 0; i < n; i++) {
    const x = (cells[i] % columns) * stepX - 6.6, y = Math.floor(cells[i] / columns) * stepY - 4.4;
    positions.set([x, y, layer * spacing], i * 3); strength[i] = data[i * 3 + 2];
    const key = `${Math.floor(x * 4)},${Math.floor(y * 4)}`;
    if (!grid.has(key)) grid.set(key, []);
    grid.get(key).push(i);
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute('strength', new THREE.BufferAttribute(strength, 1));
  const material = new THREE.ShaderMaterial({ vertexShader: vertex, fragmentShader: fragment, uniforms: { fade: { value: 1 }, viewport: { value: new THREE.Vector2() }, stepSize: { value: new THREE.Vector2(stepX, stepY) }, dotSize: { value: manifest.dots.size }, gap: { value: manifest.dots.gap }, glow: { value: manifest.dots.glow }, faint: { value: manifest.dots.faint }, pixelRatio: { value: renderer.getPixelRatio() } }, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending });
  const points = new THREE.Points(geometry, material); scene.add(points);
  const sheet = new THREE.Mesh(new THREE.PlaneGeometry(14.4, 10.4), new THREE.MeshBasicMaterial({ color: 0x6eb4de, transparent: true, opacity: .025, side: THREE.DoubleSide, depthWrite: false, forceSinglePass: true }));
  sheet.position.z = layer * spacing; scene.add(sheet);
  const edge = new THREE.LineSegments(new THREE.EdgesGeometry(sheet.geometry), new THREE.LineBasicMaterial({ color: 0x46708b, transparent: true, opacity: .33, depthWrite: false }));
  edge.position.z = sheet.position.z; scene.add(edge);
  layers.set(layer, { points, positions, grid, sheet, edge });
}
const marker = new THREE.Mesh(new THREE.RingGeometry(.10, .14, 32), new THREE.MeshBasicMaterial({ color: 0xf2cb8e, side: THREE.DoubleSide, depthTest: false }));
marker.visible = false; marker.renderOrder = 10; scene.add(marker);
async function select(layer, n, fly = false) {
  if (!layers.has(layer) || !Number.isInteger(n) || n < 0 || n >= layers.get(layer).positions.length / 3) throw new Error(`L${layer} N${n} is not available.`);
  selected = { layer, n }; sign = 0; tip.hidden = true;
  const data = await detail(layer);
  if (selected?.layer !== layer || selected?.n !== n) return;
  const row = data.rows[n];
  marker.position.fromArray(layers.get(layer).positions, n * 3); marker.position.z += .02; marker.visible = true;
  panel.hidden = false;
  panel.innerHTML = `<header><span>L${layer} N${n}</span><button id="close-neuron" aria-label="Close neuron">×</button></header><h3>+ words</h3>${tokens(row.up, 'up')}<h3>− words</h3>${tokens(row.down, 'down')}<h3>I am a …</h3><div id="persona"></div><div class="metrics">KL + ${row.kl[0].toFixed(4)} / − ${row.kl[1].toFixed(4)}<br>Random p95 + ${data.p95[0].toFixed(4)} / − ${data.p95[1].toFixed(4)}</div><h3>Spread across layers</h3><div id="spread" class="spread" aria-label="Change size across layers"></div><div id="steer-sign"><button data-sign="0" aria-pressed="true">+${manifest.dose}</button><button data-sign="1" aria-pressed="false">−${manifest.dose}</button></div><button id="steer-neuron" class="primary" style="margin-top:8px">Add to steer</button>`;
  function direction() {
    $('#persona').innerHTML = tokens(row.persona[sign], sign ? 'down' : 'up');
    const values = row.spread[sign], max = Math.max(...values, .00001);
    $('#spread').innerHTML = values.map((v, i) => `<i title="L${i}: ${v}" style="height:${v / max * 100}%"></i>`).join('');
    for (const [l, obj] of layers) obj.edge.material.color.setRGB(.18 + .3 * values[l] / max, .32 + .4 * values[l] / max, .43 + .5 * values[l] / max);
  }
  direction();
  $('#close-neuron').onclick = clear;
  panel.querySelectorAll('[data-sign]').forEach(b => b.onclick = () => { sign = +b.dataset.sign; panel.querySelectorAll('[data-sign]').forEach(x => x.setAttribute('aria-pressed', x === b)); direction(); });
  $('#steer-neuron').onclick = () => {
    if ($('#model').value !== manifest.model) { error(new Error(`Load ${manifest.model} in chat before steering this map.`)); return; }
    window.dispatchEvent(new CustomEvent('neuron-steer', { detail: { layer, n, dose: manifest.dose * (sign ? -1 : 1) } }));
    $('#steer-neuron').textContent = 'Added to steer';
    if (document.body.dataset.view === 'map') setView('both');
  };
  if (fly) face(layer);
}
function clear() { selected = null; marker.visible = false; panel.hidden = true; for (const obj of layers.values()) obj.edge.material.color.setHex(0x46708b); }
function pick(event) {
  const rect = host.getBoundingClientRect(), x = event.clientX - rect.left, y = event.clientY - rect.top;
  pointer.set(x / rect.width * 2 - 1, 1 - y / rect.height * 2); ray.setFromCamera(pointer, camera);
  let best, distance = 9;
  for (const [layer, obj] of layers) {
    if (focused !== null && layer !== focused) continue;
    plane.constant = -layer * spacing;
    if (!ray.ray.intersectPlane(plane, tmp)) continue;
    const gx = Math.floor(tmp.x * 4), gy = Math.floor(tmp.y * 4);
    const radius = Math.min(6, Math.max(1, Math.ceil(camera.position.distanceTo(tmp) * .025)));
    for (let dx = -radius; dx <= radius; dx++) for (let dy = -radius; dy <= radius; dy++) {
      for (const n of obj.grid.get(`${gx + dx},${gy + dy}`) || []) {
        const p = new THREE.Vector3().fromArray(obj.positions, n * 3).project(camera);
        if (p.z < -1 || p.z > 1) continue;
        const d = Math.hypot((p.x + 1) * rect.width / 2 - x, (1 - p.y) * rect.height / 2 - y);
        if (d < distance) { distance = d; best = { layer, n }; }
      }
    }
  }
  return best;
}
let hoverTimer;
host.addEventListener('pointermove', e => {
  clearTimeout(hoverTimer);
  if (dragging || tween) { tip.hidden = true; return; }
  hoverTimer = setTimeout(async () => {
    const hit = pick(e), key = hit ? `${hit.layer}:${hit.n}` : '';
    hoverKey = key; host.style.cursor = hit ? 'pointer' : 'grab';
    if (!hit) { tip.hidden = true; return; }
    try {
      const data = await detail(hit.layer); if (hoverKey !== key || dragging) return;
      const row = data.rows[hit.n], rect = host.getBoundingClientRect();
      tip.innerHTML = `<strong>L${hit.layer} N${hit.n}</strong><div class="up">+ ${escape(row.up.slice(0, 5).join(' · '))}</div><div class="down">− ${escape(row.down.slice(0, 5).join(' · '))}</div><small>KL + ${row.kl[0].toFixed(4)} / − ${row.kl[1].toFixed(4)}</small>`;
      tip.hidden = false;
      const width = tip.offsetWidth, height = tip.offsetHeight;
      const limit = panel.hidden ? rect.width : panel.getBoundingClientRect().left - rect.left - 12;
      const left = e.clientX - rect.left + 18;
      tip.style.left = `${Math.max(12, Math.min(left + width > limit ? e.clientX - rect.left - width - 18 : left, limit - width))}px`;
      tip.style.top = `${Math.max(64, Math.min(e.clientY - rect.top + 18, rect.height - height - 36))}px`;
    } catch (e) { error(e); }
  }, 65);
});
host.addEventListener('pointerleave', () => { clearTimeout(hoverTimer); hoverKey = ''; tip.hidden = true; });
host.addEventListener('pointerdown', e => { down = [e.clientX, e.clientY]; dragging = true; tip.hidden = true; hoverKey = ''; tween = null; });
host.addEventListener('pointerup', e => { dragging = false; if (down && Math.hypot(e.clientX - down[0], e.clientY - down[1]) < 4) { const hit = pick(e); if (hit) select(hit.layer, hit.n).catch(error); } down = null; });
controls.addEventListener('start', () => { tween = null; tip.hidden = true; });
function setView(view) { document.body.dataset.view = view; document.querySelectorAll('nav [data-view]').forEach(b => b.setAttribute('aria-pressed', b.dataset.view === view)); }
document.querySelectorAll('nav [data-view]').forEach(b => b.onclick = () => setView(b.dataset.view));
$('#fit').onclick = fit; $('#shortcuts').onclick = () => $('#keys').showModal(); $('#close-keys').onclick = () => $('#keys').close();
const searchInput = $('#map-search input'), results = $('#map-results');
async function search() {
  const version = ++searchVersion, q = searchInput.value.trim();
  if (!q) { results.hidden = true; return; }
  const exact = q.match(/^L(\d+)\s+N(\d+)$/i);
  if (exact) { results.hidden = true; await select(+exact[1], +exact[2], true); return; }
  const response = await fetch(`/api/view/search?q=${encodeURIComponent(q)}`);
  const data = await response.json(); if (version !== searchVersion) return;
  results.innerHTML = ''; results.hidden = false;
  for (const hit of data.rows) {
    const b = document.createElement('button'); b.type = 'button'; b.innerHTML = `L${hit.layer} N${hit.n}<small>${escape(hit.words.join(' · '))}</small>`;
    b.onclick = () => { results.hidden = true; select(hit.layer, hit.n, true).catch(error); }; results.append(b);
  }
  if (!data.rows.length) results.innerHTML = '<p>No matching neurons</p>';
}
let searchTimer;
$('#map-search').onsubmit = e => { e.preventDefault(); clearTimeout(searchTimer); search().catch(error); };
searchInput.oninput = () => { clearTimeout(searchTimer); searchTimer = setTimeout(() => search().catch(error), 300); };
document.addEventListener('keydown', e => {
  if (/INPUT|TEXTAREA|SELECT/.test(e.target.tagName) || e.ctrlKey || e.metaKey || $('#keys').open) return;
  if (e.key === '/') { e.preventDefault(); searchInput.focus(); }
  if (e.key.toLowerCase() === 'f') fit();
  if (e.key === '1') face(selected?.layer ?? focused ?? 0);
  if (e.key === 'Escape') { clear(); results.hidden = true; }
  if (e.key === '[' || e.key === ']') { const available = [...layers.keys()].sort((a,b) => a-b), index = available.indexOf(focused ?? selected?.layer ?? 0); face(available[Math.max(0, Math.min(available.length - 1, index + (e.key === ']' ? 1 : -1)))]); }
});
new ResizeObserver(() => { const w = host.clientWidth, h = host.clientHeight; if (!w || !h) return; camera.aspect = w / h; camera.updateProjectionMatrix(); renderer.setSize(w, h); }).observe(host);
let frames = [];
renderer.setAnimationLoop(time => {
  if (document.body.dataset.view === 'chat') return;
  if (tween) {
    const t = Math.min(1, (time - tween.time) / (matchMedia('(prefers-reduced-motion: reduce)').matches ? 1 : 650)), ease = 1 - (1 - t) ** 3;
    camera.position.lerpVectors(tween.from, tween.to, ease); controls.target.lerpVectors(tween.start, tween.target, ease); if (t === 1) tween = null;
  }
  controls.update();
  for (const obj of layers.values()) obj.points.material.uniforms.viewport.value.set(host.clientWidth, host.clientHeight);
  renderer.render(scene, camera); frames.push(time); if (frames.length > 180) frames.shift();
});
async function init() {
  manifest = await (await fetchData('manifest.json')).json();
  await Promise.all(manifest.layers.map(async ({layer}) => { const buffer = await (await fetchData(`L${layer}.bin`)).arrayBuffer(); addLayer(layer, new Float32Array(buffer)); }));
  for (let layer = 0; layer < manifest.total_layers; layer++) {
    const button = document.createElement('button'); button.textContent = `L${layer}`; button.dataset.layer = layer; button.disabled = !layers.has(layer); button.title = layers.has(layer) ? `Layer ${layer} face-on` : `Layer ${layer} pending`; button.onclick = () => face(layer); $('#layer-labels').append(button);
    labels.push({ button, layer });
  }
  const count = manifest.layers.reduce((sum, l) => sum + l.count, 0);
  $('#map-count').textContent = `${manifest.layers.length} / ${manifest.total_layers} layers · ${count.toLocaleString()} neurons${manifest.layers.length < manifest.total_layers ? ' · remaining layers pending' : ''}`;
  camera.position.set(40, 25, 60); fit();
  window.neuronMap = { select, fit, face, project(layer,n) { const rect = host.getBoundingClientRect(), p = new THREE.Vector3().fromArray(layers.get(layer).positions,n*3).project(camera); return {x:rect.left+(p.x+1)*rect.width/2,y:rect.top+(1-p.y)*rect.height/2}; }, stats: () => ({ layers: layers.size, points: count, fps: frames.length > 1 ? (frames.length - 1) * 1000 / (frames.at(-1) - frames[0]) : 0, calls: renderer.info.render.calls, camera: camera.position.toArray(), target: controls.target.toArray(), focused, selected }), controls };
}
init().catch(error);

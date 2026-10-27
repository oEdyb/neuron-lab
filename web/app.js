const $ = (s) => document.querySelector(s);
const KEY = "neuron-lab";
const DEFAULTS = {
  pushes: [{ layer: 17, n: 5244, dose: 0.6, sign: 1 }],
  pairs: [{ want: "", opposite: "" }],
  layers: "3-23", where: "local", compare: true, sample: false, turns: [], results: [],
};
const S = restore();
let server = {}, busy = false;

function restore() {
  let saved = {};
  try { saved = JSON.parse(localStorage.getItem(KEY) || "{}") || {}; } catch { saved = {}; }
  const num = (x) => Number.isFinite(+x);
  const s = { ...DEFAULTS, ...saved };
  s.pushes = (Array.isArray(s.pushes) ? s.pushes : []).filter((p) => p && num(p.layer) && num(p.n) && num(p.dose))
    .map((p) => ({ layer: +p.layer, n: +p.n, sign: p.sign === -1 || +p.dose < 0 ? -1 : 1, dose: Math.min(1, Math.abs(+p.dose)) }))
    .map((p) => ({ ...p, dose: p.sign * p.dose }));
  s.pairs = (Array.isArray(s.pairs) && s.pairs.length ? s.pairs : DEFAULTS.pairs)
    .map((p) => ({ want: String(p?.want ?? ""), opposite: String(p?.opposite ?? "") }));
  s.results = (Array.isArray(s.results) ? s.results : []).filter((r) => r && Array.isArray(r.words) && num(r.real));
  s.turns = (Array.isArray(s.turns) ? s.turns : []).filter((t) => t && typeof t.you === "string");
  s.layers = typeof s.layers === "string" ? s.layers : DEFAULTS.layers;
  s.where = s.where === "modal" ? "modal" : "local";
  return s;
}

const save = () => localStorage.setItem(KEY, JSON.stringify(S));
const fmt = (d) => (d >= 0 ? "+" : "−") + Math.abs(d).toFixed(2);
const recipe = () => S.pushes.map((p) => `L${p.layer} N${p.n} ${fmt(p.dose)}`).join(", ");
const esc = (t) => String(t).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const rich = (t) => esc(t)
  .replace(/^#{1,6}\s*(.+)$/gm, "<b>$1</b>")
  .replace(/^(\s*)[*-]\s+/gm, "$1• ")
  .replace(/\*\*(.+?)\*\*/g, "<b>$1</b>");
const say = (msg) => { $("#status").textContent = msg; };

async function api(path, body) {
  const r = await fetch(path, body ? { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) } : {});
  let j;
  try { j = await r.json(); } catch { throw new Error(`Server error ${r.status}. Check the terminal running server.py.`); }
  if (!j.ok) throw new Error(`${j.error}. ${j.fix || ""}`.trim());
  return j;
}

function renderPushes() {
  const ul = $("#pushes");
  ul.innerHTML = "";
  S.pushes.forEach((p, i) => {
    const id = `L${p.layer} N${p.n}`;
    const li = document.createElement("li");
    li.className = "push " + (p.sign > 0 ? "up" : "down");
    const mag = Math.abs(p.dose);
    li.innerHTML = `<span class="id">${id}</span>
      <button type="button" class="sign" aria-label="${id} pushed ${p.sign > 0 ? "up" : "down"}; flip">${p.sign > 0 ? "+" : "−"}</button>
      <input type="range" min="0" max="1" step="0.05" value="${mag}" aria-label="Dose for ${id}">
      <span class="val">${mag.toFixed(2)}</span>
      <button type="button" class="x" aria-label="Remove ${id}">×</button>`;
    const range = li.querySelector("input");
    const paint = () => range.style.setProperty("--fill",
      `linear-gradient(to right, var(${p.sign > 0 ? "--up" : "--down"}) ${range.value * 100}%, var(--line) 0)`);
    paint();
    range.oninput = () => {
      p.dose = p.sign * +range.value;
      li.querySelector(".val").textContent = (+range.value).toFixed(2);
      paint(); save(); renderHeads();
    };
    li.querySelector(".sign").onclick = () => { p.sign = -p.sign; p.dose = p.sign * Math.abs(p.dose); save(); renderPushes(); };
    li.querySelector(".x").onclick = () => { S.pushes.splice(i, 1); save(); renderPushes(); };
    ul.append(li);
  });
  renderHeads();
}

function addPush(layer, n, dose = 0.6) {
  const sign = dose < 0 ? -1 : 1, d = sign * Math.min(1, Math.abs(dose));
  const hit = S.pushes.find((p) => p.layer === layer && p.n === n);
  if (hit) Object.assign(hit, { sign, dose: d }); else S.pushes.push({ layer, n, sign, dose: d });
  save(); renderPushes();
}

function renderPairs() {
  const box = $("#pairs");
  box.innerHTML = "";
  S.pairs.forEach((p, i) => {
    const d = document.createElement("div");
    d.className = "pair";
    d.innerHTML = `<input placeholder="${i ? "" : "I hate you. You are worthless."}" aria-label="Pair ${i + 1}: want">
      <input placeholder="${i ? "" : "I like you. You are valuable."}" aria-label="Pair ${i + 1}: opposite">
      ${S.pairs.length > 1 ? `<button type="button" class="x" aria-label="Remove pair ${i + 1}">×</button>` : ""}`;
    const [w, o] = d.querySelectorAll("input");
    w.value = p.want; o.value = p.opposite;
    w.oninput = () => { p.want = w.value; save(); };
    o.oninput = () => { p.opposite = o.value; save(); };
    const x = d.querySelector(".x");
    if (x) x.onclick = () => { S.pairs.splice(i, 1); save(); renderPairs(); };
    box.append(d);
  });
}

function renderResults() {
  const ol = $("#results");
  ol.innerHTML = "";
  const max = Math.max(...S.results.map((r) => r.real), 0.001);
  S.results.forEach((r) => {
    const li = document.createElement("li");
    const id = `L${r.layer} N${r.n}`, up = r.sign > 0;
    li.innerHTML = `<button type="button" class="res${r.beats_null ? "" : " weak"}" aria-label="Add ${id} ${up ? "up" : "down"}: ${esc(r.words.slice(0, 4).join(", "))}">
      <span class="id">${id} <span class="${up ? "up" : "down"}">${up ? "+" : "−"}</span></span>
      <span class="words">${esc(r.words.join(" "))}</span>
      <span class="bar"><i style="width:${Math.max(0, r.real / max) * 100}%"></i><b style="left:${Math.min(1, Math.max(0, r.null / max)) * 100}%"></b></span></button>`;
    li.firstElementChild.onclick = () => addPush(r.layer, r.n, 0.6 * r.sign);
    ol.append(li);
  });
}

function renderHeads() {
  $("#heads").innerHTML = `<div class="steered">${esc(recipe() || "no neurons")}</div><div class="normal">normal</div>`;
}

function turnHtml(t, i) {
  const normal = t.normal === null ? `<div class="reply normal off">—</div>` : `<div class="reply normal">${rich(t.normal || "")}</div>`;
  return `<div class="turn" data-i="${i}"><div class="you">${esc(t.you)}</div>
    <div class="cols"><div class="reply steered">${rich(t.steered || "")}</div>${normal}</div>
    ${t.error ? `<p class="error">${esc(t.error)}</p>` : ""}</div>`;
}

function renderLog() {
  $("main").classList.toggle("single", !S.compare);
  const log = $("#log");
  log.innerHTML = S.turns.map(turnHtml).join("");
  log.scrollTop = log.scrollHeight;
}

function renderTurn(i) {
  const el = $(`#log .turn[data-i="${i}"]`);
  if (el) el.outerHTML = turnHtml(S.turns[i], i);
  $("#log").scrollTop = $("#log").scrollHeight;
}

function history(side) {
  return S.turns.flatMap((t) => (t.error || typeof t[side] !== "string" ? [] : [
    { role: "user", content: t.you }, { role: "assistant", content: t[side] }]));
}

function setBusy(on) {
  busy = on;
  for (const s of ["#send", "#clear", "#compare", "#model"]) $(s).disabled = on;
}

async function send(e) {
  e?.preventDefault();
  const message = $("#msg").value.trim();
  if (!message || busy) return;
  setBusy(true); $("#msg").value = ""; grow();
  const body = { pushes: S.pushes.map(({ layer, n, dose }) => ({ layer, n, dose })), message, compare: S.compare,
    sample: S.sample, tokens: 300, history: { steered: history("steered"), normal: history("normal") } };
  const i = S.turns.length;
  const turn = { you: message, steered: "", normal: S.compare ? "" : null };
  S.turns.push(turn); renderLog();
  try {
    const r = await fetch("/api/chat", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
    if (!r.ok) {
      let j = {};
      try { j = await r.json(); } catch { j = { error: `Server error ${r.status}` }; }
      throw new Error(`${j.error}. ${j.fix || ""}`.trim());
    }
    const reader = r.body.getReader(), dec = new TextDecoder();
    let buf = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      const lines = buf.split("\n"); buf = lines.pop();
      for (const line of lines.filter(Boolean)) {
        const m = JSON.parse(line);
        if (m.ok === false) throw new Error(`${m.error || m.code}. ${m.fix || ""}`.trim());
        if (m.delta) turn[m.side] += m.delta;
        if (m.done) turn[m.side] = m.text;
        const el = $(`#log .turn[data-i="${i}"] .${m.side}`);
        if (el) el.innerHTML = rich(turn[m.side]);
        $("#log").scrollTop = $("#log").scrollHeight;
      }
    }
    say("Reply done");
  } catch (err) {
    turn.error = err.message;
    say(err.message);
  }
  renderTurn(i); save(); setBusy(false); $("#msg").focus();
}

async function runFind() {
  const pairs = S.pairs.filter((p) => p.want.trim() && p.opposite.trim());
  const err = $("#find-error");
  err.hidden = true;
  if (!pairs.length) { err.textContent = "Write a sentence and its opposite."; err.hidden = false; return; }
  const btn = $("#find"), t0 = Date.now();
  btn.disabled = true;
  const tick = setInterval(() => { btn.textContent = `${Math.round((Date.now() - t0) / 1000)}s`; }, 1000);
  try {
    const j = await api("/api/find", { pairs, layers: S.layers, where: S.where, model: server.model, top: 20 });
    S.results = j.rows; save(); renderResults(); say(`Found ${j.rows.length} neurons`);
  } catch (e) {
    err.textContent = e.message; err.hidden = false;
  }
  clearInterval(tick); btn.textContent = "Find"; btn.disabled = false;
}

const sh = (s) => `'${String(s).replace(/'/g, `'\\''`)}'`;

async function copyForAgent() {
  const cmd = ["cd ~/dev/neuron-lab && uv run python chat.py", "--model", server.model, "--push", sh(recipe()),
    "--tokens", "300"];
  if (S.turns.length) cmd.push("--say-json", sh(JSON.stringify(S.turns.map((t) => t.you))));
  if (S.sample) cmd.push("--sample");
  if (!S.compare) cmd.push("--no-normal");
  if (String(server.model).endsWith("4B")) cmd.push("--bf16");
  const text = [
    `neuron-lab session. Model: ${server.model}. Pushes (dose × median stream norm): ${recipe() || "none"}.`,
    `Reproduce in a terminal (greedy turns match exactly):\n${cmd.join(" ")}`,
    S.results.length ? `Last Find (pairs → ranked neurons; real > null beats random directions):\n${JSON.stringify(S.results.slice(0, 10).map(({ layer, n, sign, real, null: nl, words }) => ({ layer, n, sign, real, null: nl, words: words.slice(0, 5) })))}` : "",
    S.turns.length ? `Conversation (steered vs normal; normal null = Compare was off):\n${JSON.stringify(S.turns)}` : "",
    `API: GET ${location.origin}/api/help. Likely next: tune the doses, or find neurons for a new trait with find.py --pairs.`,
  ].filter(Boolean).join("\n\n");
  const b = $("#copy");
  try { await navigator.clipboard.writeText(text); b.textContent = "Copied"; } catch { b.textContent = "Copy blocked"; }
  setTimeout(() => (b.textContent = "Copy for agent"), 1400);
}

function grow() { const m = $("#msg"); m.style.height = "auto"; m.style.height = m.scrollHeight + "px"; }

function setWhere(v) {
  S.where = v; save();
  document.querySelectorAll("#where button").forEach((b) => b.setAttribute("aria-checked", b.dataset.v === v));
}

async function loadModel(model) {
  const sel = $("#model");
  sel.disabled = true;
  sel.options[sel.selectedIndex].textContent = "loading…";
  try { server = await api("/api/load", { model }); say(`Loaded ${model}`); } catch (e) {
    $("#find-error").textContent = e.message; $("#find-error").hidden = false;
    server = await api("/api/state").catch(() => server);
  }
  fillModels(); sel.disabled = false;
}

function fillModels() {
  $("#model").innerHTML = (server.models || []).map((m) =>
    `<option value="${m}" ${m === server.model ? "selected" : ""}>${m.replace("Qwen/", "")}</option>`).join("");
}

function addFromInput(e) {
  e.preventDefault();
  const input = $("#add-input");
  const m = input.value.match(/^\s*L?\s*(\d+)\s*[\s,/]\s*N?\s*(\d+)\s*([+\-−]?)\s*(\d*\.?\d+)?\s*$/i);
  if (!m) { input.setAttribute("aria-invalid", "true"); return; }
  input.removeAttribute("aria-invalid");
  const size = m[4] ? +m[4] : 0.6;
  addPush(+m[1], +m[2], /[-−]/.test(m[3]) ? -size : size);
  input.value = "";
}

async function init() {
  try { server = await api("/api/state"); } catch (e) { say(e.message); }
  fillModels();
  $("#model").onchange = (e) => loadModel(e.target.value);
  $("#add").onsubmit = addFromInput;
  $("#add-input").oninput = (e) => e.target.removeAttribute("aria-invalid");
  $("#add-pair").onclick = () => { S.pairs.push({ want: "", opposite: "" }); save(); renderPairs(); };
  $("#layers").oninput = (e) => { S.layers = e.target.value; save(); };
  $("#layers").value = S.layers;
  document.querySelectorAll("#where button").forEach((b) => (b.onclick = () => setWhere(b.dataset.v)));
  setWhere(S.where);
  $("#find").onclick = runFind;
  $("#copy").onclick = copyForAgent;
  $("#clear").onclick = () => { S.turns = []; save(); renderLog(); };
  $("#compare").checked = S.compare;
  $("#sample").checked = S.sample;
  $("#compare").onchange = (e) => { S.compare = e.target.checked; save(); renderLog(); };
  $("#sample").onchange = (e) => { S.sample = e.target.checked; save(); };
  $("#composer").onsubmit = send;
  $("#msg").oninput = grow;
  $("#msg").onkeydown = (e) => { if (e.key === "Enter" && !e.shiftKey && !e.isComposing) send(e); };
  renderPushes(); renderPairs(); renderResults(); renderLog();
  if (document.body.dataset.view === "chat") $("#msg").focus();
}

init();

window.addEventListener("neuron-steer", (e) => addPush(e.detail.layer, e.detail.n, e.detail.dose));

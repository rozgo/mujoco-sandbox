// Journal page: media wiring, replay viewer controls and training charts.
"use strict";

const $ = (id) => document.getElementById(id);

// ------------------------------------------------------------------ media
fetch("media.json").then((r) => r.json()).then((media) => {
  document.querySelectorAll("[data-media]").forEach((el) => {
    const src = media[el.dataset.media];
    if (src) el.src = src;
  });
});
if (window.hljs) hljs.highlightAll();

// ------------------------------------------------------------------ viewer
const canvas = $("viewer");
let api = null;
let policy = 0; // 0 robust learned, 1 scripted yardstick, 2 learned without disturbances
let level = 1;
const OUTCOME = ["timeout", "success", "collision"];
const POLICY = ["robust learned policy", "scripted yardstick", "learned, undisturbed"];

function episodesFor(p) {
  const list = [];
  for (let i = 0; i < api.count(); i++) if (api.policy(i) === p && Math.abs(api.level(i) - level) < 1e-3) list.push(i);
  return list;
}

function fillEpisodes() {
  const select = $("episode");
  const current = api.selected();
  const seed = api.seed(current);
  select.innerHTML = "";
  for (const i of episodesFor(policy)) {
    const o = document.createElement("option");
    o.value = i;
    o.textContent = `seed ${api.seed(i)} · target ${api.target(i)} · ${OUTCOME[api.outcome(i)]}`;
    select.appendChild(o);
  }
  const list = episodesFor(policy);
  const same = list.find((i) => api.seed(i) === seed);
  const pick = same !== undefined ? same : list[0];
  if (pick !== undefined) { api.select(pick); select.value = pick; }
}

function setPressed(selector, active) {
  document.querySelectorAll(selector).forEach((b) => b.setAttribute("aria-pressed", String(b === active)));
}

function resize() {
  if (!api) return;
  const rect = canvas.getBoundingClientRect();
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  api.resize(Math.round(rect.width * dpr), Math.round(rect.height * dpr));
  canvas.style.width = "100%";
  canvas.style.height = "auto";
}

let wired = false;
function wire() {
  if (wired) return;
  const c = (name, ret, args) => Module.cwrap(name, ret, args);
  api = {
    count: c("ni_episode_count", "number", []), seed: c("ni_seed", "number", ["number"]),
    policy: c("ni_policy", "number", ["number"]), target: c("ni_target", "number", ["number"]),
    outcome: c("ni_outcome", "number", ["number"]), selected: c("ni_selected", "number", []),
    select: c("ni_select", null, ["number"]), time: c("ni_time", "number", []),
    duration: c("ni_duration", "number", []), seek: c("ni_seek", null, ["number"]),
    play: c("ni_play", null, ["number"]), playing: c("ni_playing", "number", []),
    speed: c("ni_speed", null, ["number"]), camera: c("ni_camera", null, ["number"]),
    lateral: c("ni_lateral", "number", []), vertical: c("ni_vertical", "number", []),
    resize: c("ni_resize", null, ["number", "number"]),
    level: c("ni_level", "number", ["number"]), latency: c("ni_latency", "number", ["number"]),
    overlays: c("ni_overlays", null, ["number"]), overlayMask: c("ni_overlay_mask", "number", []),
  };
  // onRuntimeInitialized fires before the C main() loads the replays; wait for them.
  if (api.count() === 0) { setTimeout(() => wire(), 50); return; }
  document.title = "Surgical Thread Robot Journal";  // raylib sets the window title
  fillEpisodes();
  resize();
  window.addEventListener("resize", resize);

  document.querySelectorAll("[data-policy]").forEach((b) => (b.onclick = () => {
    policy = Number(b.dataset.policy);
    setPressed("[data-policy]", b);
    fillEpisodes();
  }));
  document.querySelectorAll("[data-level]").forEach((b) => (b.onclick = () => {
    level = Number(b.dataset.level);
    setPressed("[data-level]", b);
    fillEpisodes();
  }));
  document.querySelectorAll("[data-overlay]").forEach((b) => (b.onclick = () => {
    const bit = Number(b.dataset.overlay);
    const mask = api.overlayMask() ^ bit;
    api.overlays(mask);
    b.setAttribute("aria-pressed", String(!!(mask & bit)));
  }));
  $("episode").onchange = (e) => api.select(Number(e.target.value));
  const step = (d) => {
    const list = episodesFor(policy);
    const at = list.indexOf(api.selected());
    const next = list[(at + d + list.length) % list.length];
    api.select(next);
    $("episode").value = next;
  };
  $("prev").onclick = () => step(-1);
  $("next").onclick = () => step(1);
  $("play").onclick = (e) => {
    const on = api.playing() ? 0 : 1;
    api.play(on);
    e.target.textContent = on ? "Pause" : "Play";
    e.target.setAttribute("aria-pressed", String(!!on));
  };
  document.querySelectorAll("[data-speed]").forEach((b) => (b.onclick = () => {
    api.speed(Number(b.dataset.speed));
    setPressed("[data-speed]", b);
  }));
  document.querySelectorAll("[data-camera]").forEach((b) => (b.onclick = () => {
    api.camera(Number(b.dataset.camera));
    setPressed("[data-camera]", b);
  }));
  let scrubbing = false;
  $("scrub").addEventListener("input", (e) => { scrubbing = true; api.seek(api.duration() * e.target.value / 1000); });
  $("scrub").addEventListener("change", () => { scrubbing = false; });

  const tick = () => {
    const t = api.time(), d = api.duration();
    $("t").textContent = t.toFixed(2);
    $("lat").textContent = api.lateral().toFixed(1);
    $("vert").textContent = api.vertical().toFixed(1);
    const i = api.selected();
    $("target").textContent = String(api.target(i));
    $("level").textContent = api.level(i).toFixed(1);
    $("latency").textContent = Math.round(api.latency(i) * 1000);
    if ($("episode").value !== String(i)) $("episode").value = i;
    const done = t >= d - 1e-6;
    const status = $("status");
    status.textContent = done ? OUTCOME[api.outcome(i)].toUpperCase() : POLICY[policy];
    status.className = done && api.outcome(i) === 1 ? "status-success" : "status-moving";
    if (!scrubbing) $("scrub").value = d > 0 ? Math.round(1000 * t / d) : 0;
    requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
  wired = true;
}

window.Module = {
  canvas,
  locateFile: (path) => "viewer/" + path,
  onRuntimeInitialized: wire,
  print: (text) => console.log(text),
  printErr: (text) => console.warn(text),
};
const viewerScript = document.createElement("script");
viewerScript.src = "viewer/viewer.js";
viewerScript.async = true;
document.body.appendChild(viewerScript);

// ------------------------------------------------------------------ charts
const RUNS = [
  ["align_v1", "Run 1 · reward bug", "#b2456b"],
  ["align_v2", "Run 2 · learning rate 0.015", "#d38aaa"],
  ["align_v3", "Run 3 · learning rate 0.003", "#646da0"],
];

function steps(text) {
  const v = parseFloat(text);
  return text.endsWith("M") ? v * 1e6 : text.endsWith("K") ? v * 1e3 : text.endsWith("B") ? v * 1e9 : v;
}

function chart(svg, data, key, { log = false, min, max, unit = "", gate } = {}) {
  const W = 520, H = 300, L = 58, R = 12, T = 12, B = 38;
  const ns = "http://www.w3.org/2000/svg";
  const el = (tag, attrs, text) => {
    const e = document.createElementNS(ns, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (text !== undefined) e.textContent = text;
    svg.appendChild(e);
    return e;
  };
  const f = (v) => (log ? Math.log10(Math.max(v, 1e-12)) : v);
  const lo = f(min), hi = f(max), xmax = 100e6;
  const x = (s) => L + (W - L - R) * s / xmax;
  const y = (v) => T + (H - T - B) * (1 - (f(v) - lo) / (hi - lo));
  for (let s = 0; s <= xmax; s += 20e6) {
    el("line", { x1: x(s), x2: x(s), y1: T, y2: H - B, stroke: "#eee9f3" });
    el("text", { x: x(s), y: H - B + 18, "text-anchor": "middle", "font-size": 12, fill: "#5d5f7a" }, `${s / 1e6}M`);
  }
  const ticks = log ? Array.from({ length: hi - lo + 1 }, (_, k) => 10 ** (lo + k)) : [min, (min + max) / 2, max];
  for (const v of ticks) {
    el("line", { x1: L, x2: W - R, y1: y(v), y2: y(v), stroke: "#eee9f3" });
    const label = log ? (v >= 1000 ? `${v / 1000}k` : `${v}`) : `${Math.round(v * 100) / 100}`;
    el("text", { x: L - 6, y: y(v) + 4, "text-anchor": "end", "font-size": 12, fill: "#5d5f7a" }, label + unit);
  }
  el("text", { x: (L + W - R) / 2, y: H - 4, "text-anchor": "middle", "font-size": 12, fill: "#5d5f7a" }, "training steps");
  if (gate !== undefined) {
    el("line", { x1: L, x2: W - R, y1: y(gate), y2: y(gate), stroke: "#3c8f78", "stroke-dasharray": "5 4" });
  }
  for (const [name, , color] of RUNS) {
    const rows = (data[name]?.history || []).filter((r) => r[key] !== undefined);
    const pts = rows.map((r) => [x(steps(r.Steps)), y(Math.min(Math.max(parseFloat(r[key]), min), max))]);
    if (pts.length) el("polyline", { points: pts.map((p) => p.join(",")).join(" "), fill: "none", stroke: color, "stroke-width": 2.2 });
  }
}

fetch("data/training.json").then((r) => r.json()).then((data) => {
  chart($("chart-error"), data, "final_lateral_um", { log: true, min: 1, max: 1e5, unit: " µm", gate: 10 });
  chart($("chart-success"), data, "perf", { min: 0, max: 1 });
  chart($("chart-kl"), data, "kl", { log: true, min: 1e-3, max: 1e4 });
  chart($("chart-collision"), data, "collision", { min: 0, max: 1 });
  $("legend").innerHTML = RUNS.map(([, label, color]) => `<span><i class="swatch" style="background:${color}"></i>${label}</span>`).join("")
    + '<span><i class="swatch" style="background:#3c8f78"></i>10 µm tolerance</span>';
});

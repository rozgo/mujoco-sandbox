// Journal page: media wiring, replay viewer controls and training charts.
"use strict";

const $ = (id) => document.getElementById(id);
// Replaced at build time with a hash of the viewer files, so a rebuilt viewer is never served from cache.
const BUILD = "__VIEWER_BUILD__";

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
  locateFile: (path) => `viewer/${path}?v=${BUILD}`,
  onRuntimeInitialized: wire,
  print: (text) => console.log(text),
  printErr: (text) => console.warn(text),
};
const viewerScript = document.createElement("script");
viewerScript.src = `viewer/viewer.js?v=${BUILD}`;
viewerScript.async = true;
document.body.appendChild(viewerScript);

// ------------------------------------------------------------------ charts
const RUNS = [
  ["align_v1", "Run 1 · reward bug", "#b2456b"],
  ["align_v2", "Run 2 · learning rate 0.015", "#d38aaa"],
  ["align_v3", "Run 3 · learning rate 0.003", "#646da0"],
];

function steps(text) {
  if (typeof text === "number") return text;
  const v = parseFloat(text);
  return text.endsWith("M") ? v * 1e6 : text.endsWith("K") ? v * 1e3 : text.endsWith("B") ? v * 1e9 : v;
}

function chart(svg, data, key, { log = false, min, max, unit = "", gate, runs = RUNS, xmax = 100e6 } = {}) {
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
  const lo = f(min), hi = f(max);
  const x = (s) => L + (W - L - R) * s / xmax;
  const y = (v) => T + (H - T - B) * (1 - (f(v) - lo) / (hi - lo));
  for (let s = 0; s <= xmax; s += xmax / 5) {
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
  for (const [name, , color] of runs) {
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

// ------------------------------------------------------------------ robust training, robustness, compute
const ROBUST_RUNS = [
  ["align_v3", "Run 3 · no disturbances (reference)", "#c9c4e0"],
  ["align_v4", "Run 4 · disturbances, rate 0.003", "#b2456b"],
  ["align_v5", "Run 5 · rate 0.001, 100 M steps", "#e0a458"],
  ["align_v6", "Run 6 · rate 0.001, 300 M steps", "#646da0"],
];
const POLICIES = [
  ["robust", "Robust learned (run 6)", "#646da0"],
  ["run5", "Run 5 (100 M steps)", "#e0a458"],
  ["undisturbed", "Learned without disturbances (run 3)", "#d38aaa"],
  ["scripted", "Scripted yardstick", "#3c8f78"],
];
const svgEl = (svg, tag, attrs, text) => {
  const e = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  if (text !== undefined) e.textContent = text;
  svg.appendChild(e);
  return e;
};
const legend = (el, items, extra = "") =>
  (el.innerHTML = items.map(([, label, color]) => `<span><i class="swatch" style="background:${color}"></i>${label}</span>`).join("") + extra);

fetch("data/robust_training.json").then((r) => r.json()).then((data) => {
  const runs = ROBUST_RUNS.filter(([n]) => data[n]);
  const xmax = Math.max(100e6, ...runs.map(([n]) => steps(data[n].history.at(-1)?.Steps ?? 0)));
  const o = { runs, xmax: Math.ceil(xmax / 50e6) * 50e6 };
  chart($("chart-robust-error"), data, "final_lateral_um", { log: true, min: 1, max: 1e5, unit: " µm", gate: 10, ...o });
  chart($("chart-robust-kl"), data, "kl", { log: true, min: 1e-4, max: 1e2, ...o });
  chart($("chart-robust-success"), data, "perf", { min: 0, max: 1, ...o });
  chart($("chart-robust-collision"), data, "collision", { min: 0, max: 1, ...o });
  legend($("legend-robust"), runs, '<span><i class="swatch" style="background:#3c8f78"></i>10 µm tolerance</span>');
});

fetch("data/robustness.json").then((r) => r.json()).then((ev) => {
  const svg = $("chart-level");
  const W = 520, H = 300, L = 58, R = 12, T = 12, B = 38;
  const x = (v) => L + (W - L - R) * v, y = (v) => T + (H - T - B) * (1 - v);
  for (let v = 0; v <= 1.001; v += 0.25) {
    svgEl(svg, "line", { x1: x(v), x2: x(v), y1: T, y2: H - B, stroke: "#eee9f3" });
    svgEl(svg, "text", { x: x(v), y: H - B + 18, "text-anchor": "middle", "font-size": 12, fill: "#5d5f7a" }, v.toFixed(2));
    svgEl(svg, "line", { x1: L, x2: W - R, y1: y(v), y2: y(v), stroke: "#eee9f3" });
    svgEl(svg, "text", { x: L - 6, y: y(v) + 4, "text-anchor": "end", "font-size": 12, fill: "#5d5f7a" }, `${Math.round(v * 100)}%`);
  }
  svgEl(svg, "text", { x: (L + W - R) / 2, y: H - 4, "text-anchor": "middle", "font-size": 12, fill: "#5d5f7a" }, "disturbance level");
  const levels = Object.keys(ev.scripted || {}).sort();
  for (const [name, , color] of POLICIES) {
    if (!ev[name]) continue;
    const pts = levels.map((l) => [x(parseFloat(l)), y(ev[name][l].success)]);
    svgEl(svg, "polyline", { points: pts.map((p) => p.join(",")).join(" "), fill: "none", stroke: color, "stroke-width": 2.4 });
    for (const [px, py] of pts) svgEl(svg, "circle", { cx: px, cy: py, r: 3.6, fill: color });
  }
  legend($("legend-level"), POLICIES.filter(([n]) => ev[n]));
  const pct = (v) => `${(100 * v).toFixed(v === 1 || v === 0 ? 0 : 1)}%`;
  let html = `<tr><th>Policy</th>${levels.map((l) => `<th class="num">level ${parseFloat(l)}</th>`).join("")}</tr>`;
  for (const [name, label] of POLICIES) {
    if (!ev[name]) continue;
    html += `<tr><td>${label}</td>${levels.map((l) => {
      const e = ev[name][l];
      const coll = e.collision > 0 ? `<br><small>${pct(e.collision)} collisions</small>` : "";
      return `<td class="num">${pct(e.success)}${coll}</td>`;
    }).join("")}</tr>`;
  }
  $("robustness-table").innerHTML = html;
});

fetch("data/compute.json").then((r) => r.json()).then(({ compute, runs }) => {
  const f = (v, d = 0) => (v === null || v === undefined ? "–" : Number(v).toFixed(d));
  const mins = (s) => (s ? `${Math.round(s / 60)} min` : "–");
  let html = "<tr><th>Run</th><th>Change and outcome</th><th class=\"num\">Steps</th><th class=\"num\">Wall time</th>" +
    "<th class=\"num\">Steps/s</th><th class=\"num\">GPU use</th><th class=\"num\">GPU power</th>" +
    "<th class=\"num\">GPU memory</th><th class=\"num\">CPU threads busy</th><th class=\"num\">Host RAM</th></tr>";
  for (const name of Object.keys(compute)) {
    const c = compute[name], r = runs[name] || {}, s = c.sampled || {}, d = c.dashboard || {};
    const gpu = s.gpu_util_pct ? `${f(s.gpu_util_pct.mean)}% (max ${f(s.gpu_util_pct.max)}%)` : `${f(d.gpu_util_pct_mean)}% (max ${f(d.gpu_util_pct_max)}%)`;
    const power = s.gpu_power_w ? `${f(s.gpu_power_w.mean)} W` : "–";
    const mem = s.trainer_gpu_memory_mib && s.trainer_gpu_memory_mib.max > 0
      ? `${f(s.trainer_gpu_memory_mib.max / 1024, 2)} GB trainer`
      : `${f(d.device_memory_gb_max, 1)} GB device`;
    const threads = s.trainer_cpu_pct ? f(s.trainer_cpu_pct.mean / 100, 1) : "–";
    const ram = s.trainer_rss_mib ? `${f(s.trainer_rss_mib.max / 1024, 1)} GB` : `${f(d.host_memory_gb_max, 1)} GB`;
    html += `<tr><td><strong>${name.replace("align_v", "Run ")}</strong></td><td>${r.change || ""}<br><small>${r.outcome || ""}</small></td>` +
      `<td class="num">${f(c.steps / 1e6, 1)} M</td><td class="num">${mins(c.wall_s)}</td>` +
      `<td class="num">${f(c.steps_per_s / 1e3, 1)}k</td><td class="num">${gpu}</td><td class="num">${power}</td>` +
      `<td class="num">${mem}</td><td class="num">${threads}</td><td class="num">${ram}</td></tr>`;
  }
  $("compute-table").innerHTML = html;
});

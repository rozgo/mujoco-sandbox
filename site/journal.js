// Journal page: media wiring, replay viewer controls and training charts.
"use strict";

const $ = (id) => document.getElementById(id);
// Replaced at build time with hashes of the viewer files and of the page's data, so a rebuild is never
// served from cache.
const BUILD = "__VIEWER_BUILD__";
const DATA = "__DATA_BUILD__";
const load = (path) => fetch(`${path}?v=${DATA}`).then((r) => r.json());

// ------------------------------------------------------------------ media
load("media.json").then((media) => {
  document.querySelectorAll("[data-media]").forEach((el) => {
    const src = media[el.dataset.media];
    if (src) el.src = src;
  });
  document.querySelectorAll("[data-poster]").forEach((el) => {
    const src = media[el.dataset.poster];
    if (src) el.poster = src;
  });
});
if (window.hljs) hljs.highlightAll();

// ------------------------------------------------------------------ viewer
const canvas = $("viewer");
let api = null;
let mode = 0;   // 0 insertion (thread tube), 1 alignment
let policy = 0; // alignment: 0 robust learned, 1 scripted yardstick, 2 learned without disturbances
let tubePolicy = 0; // insertion: 0 learned policy (insert_v2), 1 scripted yardstick
let level = 1;
let alignSpeed = 1, tubeSpeed = 0;  // tube 0: automatic (0.02x for the needle's work, 0.2x between sites)
const OUTCOME = ["timeout", "success", "collision"];
const POLICY = ["robust learned policy", "scripted yardstick", "learned, undisturbed"];
const PHASE = ["ready", "reload", "move", "descend", "correct", "needle down", "insert", "release", "snap back", "lift",
  "learned approach"];
const TUBE_POLICY = ["learned policy insert_v2", "scripted yardstick"];

function episodesFor(p) {
  const list = [];
  for (let i = 0; i < api.count(); i++) {
    if (Math.abs(api.level(i) - level) > 1e-3) continue;
    if (api.policy(i) !== (mode === 0 ? tubePolicy : p)) continue;
    list.push(i);
  }
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
    o.textContent = mode === 0
      ? `seed ${api.seed(i)} · ${api.placed(i)} of 3 threads placed`
      : `seed ${api.seed(i)} · target ${api.target(i)} · ${OUTCOME[api.outcome(i)]}`;
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
    mode: c("ni_mode", null, ["number"]), modeGet: c("ni_mode_get", "number", []),
    phase: c("ni_tube_phase", "number", []), site: c("ni_tube_site", "number", []),
    depth: c("ni_tube_depth", "number", []), place: c("ni_tube_place", "number", []),
    force: c("ni_tube_force", "number", []), placed: c("ni_tube_placed", "number", ["number"]),
  };
  // onRuntimeInitialized fires before the C main() loads the replays; wait for them.
  if (api.count() === 0) { setTimeout(() => wire(), 50); return; }
  document.title = "Surgical Thread Robot Journal";  // raylib sets the window title
  mode = api.modeGet();
  document.querySelector(".stage").dataset.mode = String(mode);
  setPressed("[data-mode]", document.querySelector(`[data-mode="${mode}"]`));
  fillEpisodes();
  resize();
  window.addEventListener("resize", resize);

  document.querySelectorAll("[data-mode]").forEach((b) => (b.onclick = () => {
    mode = Number(b.dataset.mode);
    api.mode(mode);
    api.speed(mode === 0 ? tubeSpeed : alignSpeed);
    document.querySelector(".stage").dataset.mode = String(mode);
    setPressed("[data-mode]", b);
    setPressed("[data-camera]", document.querySelector('[data-camera="1"]'));
    fillEpisodes();
  }));
  document.querySelectorAll("[data-tspeed]").forEach((b) => (b.onclick = () => {
    tubeSpeed = Number(b.dataset.tspeed);
    api.speed(tubeSpeed);
    setPressed("[data-tspeed]", b);
  }));
  document.querySelectorAll("[data-tpolicy]").forEach((b) => (b.onclick = () => {
    tubePolicy = Number(b.dataset.tpolicy);
    setPressed("[data-tpolicy]", b);
    fillEpisodes();
  }));
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
    alignSpeed = Number(b.dataset.speed);
    api.speed(alignSpeed);
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
    $("t").textContent = t.toFixed(mode === 0 ? 3 : 2);
    const i = api.selected();
    $("level").textContent = api.level(i).toFixed(1);
    $("latency").textContent = Math.round(api.latency(i) * 1000);
    if ($("episode").value !== String(i)) $("episode").value = i;
    const done = t >= d - 1e-6;
    const status = $("status");
    if (mode === 0) {
      $("phase").textContent = PHASE[api.phase()] || "–";
      $("site").textContent = String(api.site());
      $("depth").textContent = (api.depth() * 1e3).toFixed(2);
      $("place").textContent = (api.place() * 1e6).toFixed(1);
      $("force").textContent = (api.force() * 1e3).toFixed(2);
      status.textContent = done ? `${api.placed(i)} OF 3 THREADS PLACED` : TUBE_POLICY[tubePolicy];
      status.className = done && api.placed(i) === 3 ? "status-success" : "status-moving";
    } else {
      $("lat").textContent = api.lateral().toFixed(1);
      $("vert").textContent = api.vertical().toFixed(1);
      $("target").textContent = String(api.target(i));
      status.textContent = done ? OUTCOME[api.outcome(i)].toUpperCase() : POLICY[policy];
      status.className = done && api.outcome(i) === 1 ? "status-success" : "status-moving";
    }
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
  const W = 520, H = 300, L = 58, R = 20, T = 12, B = 38;
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

load("data/training.json").then((data) => {
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

load("data/robust_training.json").then((data) => {
  const runs = ROBUST_RUNS.filter(([n]) => data[n]);
  const xmax = Math.max(100e6, ...runs.map(([n]) => steps(data[n].history.at(-1)?.Steps ?? 0)));
  const o = { runs, xmax: Math.ceil(xmax / 50e6) * 50e6 };
  chart($("chart-robust-error"), data, "final_lateral_um", { log: true, min: 1, max: 1e5, unit: " µm", gate: 10, ...o });
  chart($("chart-robust-kl"), data, "kl", { log: true, min: 1e-4, max: 1e2, ...o });
  chart($("chart-robust-success"), data, "perf", { min: 0, max: 1, ...o });
  chart($("chart-robust-collision"), data, "collision", { min: 0, max: 1, ...o });
  legend($("legend-robust"), runs, '<span><i class="swatch" style="background:#3c8f78"></i>10 µm tolerance</span>');
});

load("data/robustness.json").then((ev) => {
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

load("data/compute.json").then(({ compute, runs }) => {
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

// ------------------------------------------------------------------ parts glossary
load("data/glossary.json").then(({ parts }) => {
  const grid = $("glossary"), view = $("glossary-view");
  let at = 0;
  const show = (i) => {
    at = (i + parts.length) % parts.length;
    const p = parts[at];
    view.querySelector("img").src = p.media;
    view.querySelector("img").alt = p.title;
    view.querySelector("h3").textContent = p.title;
    view.querySelector(".role").textContent = p.role;
    view.querySelector(".specs").textContent = p.specs;
    if (!view.open) view.showModal();
  };
  parts.forEach((p, i) => {
    const card = document.createElement("button");
    card.className = "part";
    card.innerHTML = `<img loading="lazy" alt=""><span class="part-title"></span><span class="part-specs"></span>`;
    card.querySelector("img").src = p.media;
    card.querySelector("img").alt = p.title;
    card.querySelector(".part-title").textContent = p.title;
    card.querySelector(".part-specs").textContent = p.specs;
    card.onclick = () => show(i);
    grid.appendChild(card);
  });
  view.querySelector("[data-close]").onclick = () => view.close();
  view.querySelectorAll("[data-step]").forEach((b) => (b.onclick = () => show(at + Number(b.dataset.step))));
  view.addEventListener("click", (e) => { if (e.target === view) view.close(); });  // backdrop
  view.addEventListener("keydown", (e) => {
    if (e.key === "ArrowRight") show(at + 1);
    if (e.key === "ArrowLeft") show(at - 1);
  });
});

// ------------------------------------------------------------------ insertion baseline
const TUBE_LEVELS = [["0", "No disturbances", "#3c8f78"], ["1", "Nominal (level 1)", "#646da0"], ["2", "Stress (level 2)", "#b2456b"]];
load("data/tube_baseline.json").then(({ summary, runs }) => {
  const sites = [0, 5, 1];
  const svg = $("chart-placement");
  const W = 520, H = 300, L = 58, R = 12, T = 12, B = 38, ymax = 80;
  const y = (v) => T + (H - T - B) * (1 - Math.min(v, ymax) / ymax);
  const group = (W - L - R) / sites.length;
  for (let v = 0; v <= ymax; v += 20) {
    svgEl(svg, "line", { x1: L, x2: W - R, y1: y(v), y2: y(v), stroke: "#eee9f3" });
    svgEl(svg, "text", { x: L - 6, y: y(v) + 4, "text-anchor": "end", "font-size": 12, fill: "#5d5f7a" }, `${v} µm`);
  }
  const where = { 0: "first site, thread centred", 5: "thread settled off-axis", 1: "thread settled off-axis" };
  sites.forEach((site, g) => {
    const cx = L + group * (g + 0.5);
    svgEl(svg, "text", { x: cx, y: H - B + 18, "text-anchor": "middle", "font-size": 12, fill: "#23243a" }, `site ${site}`);
    svgEl(svg, "text", { x: cx, y: H - 4, "text-anchor": "middle", "font-size": 11, fill: "#5d5f7a" }, where[site]);
    TUBE_LEVELS.forEach(([lv, , color], k) => {
      const vals = runs.filter((r) => String(r.level) === lv).flatMap((r) => r.sites).filter((s) => s.site === site && s.passed)
        .map((s) => s.placement_lateral_um).sort((a, b) => a - b);
      const x = cx + (k - 1) * group * 0.26;
      vals.forEach((v, j) => svgEl(svg, "circle", { cx: x + ((j % 5) - 2) * 3, cy: y(v), r: 3.4, fill: color, "fill-opacity": 0.8 }));
      if (vals.length) {
        const med = vals.length % 2 ? vals[(vals.length - 1) / 2] : (vals[vals.length / 2 - 1] + vals[vals.length / 2]) / 2;
        svgEl(svg, "line", { x1: x - 11, x2: x + 11, y1: y(med), y2: y(med), stroke: color, "stroke-width": 3 });
      }
    });
  });
  legend($("legend-placement"), TUBE_LEVELS, '<span>bars: medians; dots: every placed thread</span>');
  const um = (v) => (v === null || v === undefined ? "–" : `${v.toFixed(0)} µm`);
  const siteMedian = (lv, site) => {
    const v = runs.filter((r) => String(r.level) === lv).flatMap((r) => r.sites).filter((s) => s.site === site && s.passed)
      .map((s) => s.placement_lateral_um).sort((a, b) => a - b);
    return v.length ? (v[Math.floor((v.length - 1) / 2)] + v[Math.ceil((v.length - 1) / 2)]) / 2 : null;
  };
  let html = '<tr><th>Level</th><th class="num">Runs</th><th class="num">Threads placed</th><th class="num">Median</th>' +
    '<th class="num">90th pct</th><th class="num">Max</th><th class="num">Site 0</th><th class="num">Site 5</th>' +
    '<th class="num">Site 1</th><th class="num">Depth</th><th class="num">Tissue contacts</th></tr>';
  for (const [lv, label] of TUBE_LEVELS) {
    const s = summary[`level_${lv}`];
    if (!s) continue;
    html += `<tr><td>${label}</td><td class="num">${s.runs}</td><td class="num">${s.threads_placed} / ${s.sites_attempted}</td>` +
      `<td class="num">${um(s.placement_um_median)}</td><td class="num">${um(s.placement_um_p90)}</td><td class="num">${um(s.placement_um_max)}</td>` +
      sites.map((site) => `<td class="num">${um(siteMedian(lv, site))}</td>`).join("") +
      `<td class="num">${s.end_depth_mm_range.map((v) => v.toFixed(2)).join("–")} mm</td><td class="num">${s.prohibited_contacts}</td></tr>`;
  }
  $("tube-table").innerHTML = html;
});

// ------------------------------------------------------------------ learned insertion
const INSERT_RUNS = [
  ["insert_v1", "Run 1 · stages free during the stroke (stopped)", "#d38aaa"],
  ["insert_v2", "Run 2 · stages held, stroke as the full cycle", "#646da0"],
];
const CONTROLLERS = [
  ["approved", "Approved cycle (aims the needle)", "#3c8f78"],
  ["compensating", "Scripted, compensating for the thread end", "#e0a458"],
  ["learned_deterministic", "Learned policy insert_v2", "#646da0"],
];

function frame(svg, { W = 520, H = 300, L = 58, R = 20, T = 12, B = 38 } = {}) {
  return { W, H, L, R, T, B, iw: W - L - R, ih: H - T - B };
}

function median(v) {
  const s = [...v].sort((a, b) => a - b);
  return s.length ? (s[Math.floor((s.length - 1) / 2)] + s[Math.ceil((s.length - 1) / 2)]) / 2 : null;
}

load("data/insert_results.json").then((res) => {
  // Learning curves (training rollouts).
  chart($("chart-insert-success"), res.runs, "perf", { min: 0, max: 1, runs: INSERT_RUNS, xmax: 300e6 });
  chart($("chart-insert-error"), res.runs, "placement_um", { log: true, min: 1, max: 1e5, unit: " µm", gate: 10,
    runs: INSERT_RUNS, xmax: 300e6 });
  legend($("legend-insert"), INSERT_RUNS, '<span><i class="swatch" style="background:#3c8f78"></i>10 µm tolerance</span>');

  // Checkpoint evaluations in the C environment against both yardsticks.
  {
    const svg = $("chart-insert-checkpoints"), f = frame(svg);
    const steps = { "65.5M": 65.5e6, "131M": 131e6, "196.6M": 196.6e6, "299.9M": 299.9e6 };
    const x = (s) => f.L + f.iw * s / 300e6, y = (v) => f.T + f.ih * (1 - v);
    for (let s = 0; s <= 300e6; s += 60e6) {
      svgEl(svg, "line", { x1: x(s), x2: x(s), y1: f.T, y2: f.H - f.B, stroke: "#eee9f3" });
      svgEl(svg, "text", { x: x(s), y: f.H - f.B + 18, "text-anchor": "middle", "font-size": 12, fill: "#5d5f7a" }, `${s / 1e6}M`);
    }
    for (let v = 0; v <= 1.001; v += 0.25) {
      svgEl(svg, "line", { x1: f.L, x2: f.W - f.R, y1: y(v), y2: y(v), stroke: "#eee9f3" });
      svgEl(svg, "text", { x: f.L - 6, y: y(v) + 4, "text-anchor": "end", "font-size": 12, fill: "#5d5f7a" }, `${Math.round(v * 100)}%`);
    }
    svgEl(svg, "text", { x: f.L + f.iw / 2, y: f.H - 4, "text-anchor": "middle", "font-size": 12, fill: "#5d5f7a" }, "training steps at the checkpoint");
    const c = res.c_environment, share = (v) => `${+(v * 100).toFixed(1)}%`;
    [["1", "#e0a458", "6 3", -5], ["2", "#e0a458", "2 3", -5]].forEach(([lv, color, dash, dy]) => {
      const v = c.compensating[lv].success;
      svgEl(svg, "line", { x1: f.L, x2: f.W - f.R, y1: y(v), y2: y(v), stroke: color, "stroke-width": 1.8, "stroke-dasharray": dash });
      svgEl(svg, "text", { x: f.L + 6, y: y(v) + dy, "font-size": 11, fill: "#a8742e" },
        `compensating, level ${lv}: ${share(v)}`);
    });
    const v = c.aiming["1"].success;
    svgEl(svg, "line", { x1: f.L, x2: f.W - f.R, y1: y(v), y2: y(v), stroke: "#3c8f78", "stroke-width": 1.8, "stroke-dasharray": "6 3" });
    svgEl(svg, "text", { x: f.W - f.R - 4, y: y(v) - 5, "text-anchor": "end", "font-size": 11, fill: "#2f7360" },
      `aiming the needle, level 1: ${share(v)}`);
    [["c_deterministic_level1", "#646da0", 1], ["c_deterministic_level2", "#af86ac", 2]].forEach(([key, color]) => {
      const pts = res.checkpoints.map((k) => [x(steps[k.checkpoint]), y(k[key])]);
      svgEl(svg, "polyline", { points: pts.map((p) => p.join(",")).join(" "), fill: "none", stroke: color, "stroke-width": 2.6 });
      pts.forEach(([px, py]) => svgEl(svg, "circle", { cx: px, cy: py, r: 4.2, fill: color, stroke: "#fff", "stroke-width": 1.5 }));
    });
    legend($("legend-insert-checkpoints"), [[, "Learned, level 1", "#646da0"], [, "Learned, level 2", "#af86ac"],
      [, "Compensating yardstick (dashed)", "#e0a458"], [, "Aiming yardstick (dashed)", "#3c8f78"]]);
  }

  // Full simulation: threads within 10 µm of 30, by level and controller.
  const full = res.full_simulation;
  {
    const svg = $("chart-insert-full"), f = frame(svg), ymax = 30;
    const y = (v) => f.T + f.ih * (1 - v / ymax);
    for (let v = 0; v <= ymax; v += 10) {
      svgEl(svg, "line", { x1: f.L, x2: f.W - f.R, y1: y(v), y2: y(v), stroke: "#eee9f3" });
      svgEl(svg, "text", { x: f.L - 6, y: y(v) + 4, "text-anchor": "end", "font-size": 12, fill: "#5d5f7a" }, `${v}`);
    }
    svgEl(svg, "text", { x: 14, y: f.T + f.ih / 2, "text-anchor": "middle", "font-size": 12, fill: "#5d5f7a",
      transform: `rotate(-90 14 ${f.T + f.ih / 2})` }, "threads within 10 µm, of 30");
    const levels = [["1", "Nominal (level 1)"], ["2", "Stress (level 2)"]];
    const group = f.iw / levels.length, bw = group * 0.2;
    levels.forEach(([lv, label], g) => {
      const cx = f.L + group * (g + 0.5);
      svgEl(svg, "text", { x: cx, y: f.H - f.B + 20, "text-anchor": "middle", "font-size": 12.5, fill: "#23243a" }, label);
      CONTROLLERS.forEach(([key, , color], k) => {
        const s = full.summary[key][`level_${lv}`];
        const v = s.within_10um, x0 = cx + (k - 1) * (bw + 8) - bw / 2;
        svgEl(svg, "rect", { x: x0, y: y(v), width: bw, height: y(0) - y(v), rx: 5, fill: color });
        svgEl(svg, "text", { x: x0 + bw / 2, y: y(v) - 6, "text-anchor": "middle", "font-size": 13, "font-weight": 650, fill: "#23243a" }, `${v}`);
        svgEl(svg, "text", { x: x0 + bw / 2, y: y(0) - 8, "text-anchor": "middle", "font-size": 10.5, fill: "#fff" },
          `${s.placement_um_median.toFixed(1)} µm`);
      });
    });
  }

  // Every placed thread in the full simulation, by controller and level.
  {
    const svg = $("chart-insert-dots"), f = frame(svg), ymax = 70;
    const y = (v) => f.T + f.ih * (1 - Math.min(v, ymax) / ymax);
    for (let v = 0; v <= ymax; v += 10) {
      svgEl(svg, "line", { x1: f.L, x2: f.W - f.R, y1: y(v), y2: y(v), stroke: v === 10 ? "#bfe0d4" : "#eee9f3",
        "stroke-width": v === 10 ? 2 : 1 });
      svgEl(svg, "text", { x: f.L - 6, y: y(v) + 4, "text-anchor": "end", "font-size": 12, fill: "#5d5f7a" }, `${v} µm`);
    }
    const levels = ["1", "2"], group = f.iw / levels.length;
    levels.forEach((lv, g) => {
      const cx = f.L + group * (g + 0.5);
      svgEl(svg, "text", { x: cx, y: f.H - f.B + 20, "text-anchor": "middle", "font-size": 12.5, fill: "#23243a" },
        lv === "1" ? "Nominal (level 1)" : "Stress (level 2)");
      CONTROLLERS.forEach(([key, , color], k) => {
        const vals = [...full.placements_um[key][lv]].sort((a, b) => a - b);
        const x = cx + (k - 1) * group * 0.28;
        vals.forEach((v, j) => svgEl(svg, "circle", { cx: x + ((j % 6) - 2.5) * 3.6, cy: y(v), r: 3.3, fill: color, "fill-opacity": 0.78 }));
        const m = median(vals);
        if (m !== null) svgEl(svg, "line", { x1: x - 14, x2: x + 14, y1: y(m), y2: y(m), stroke: "#23243a", "stroke-width": 2.5 });
      });
    });
  }
  legend($("legend-insert-full"), CONTROLLERS, '<span>bars and lines: medians; dots: every placed thread</span>');

  // Tables.
  const pct = (v) => `${(100 * v).toFixed(v === 0 || v === 1 ? 0 : 1)}%`;
  const um = (v) => (v === null || v === undefined ? "–" : `${v.toFixed(1)} µm`);
  {
    const rows = [["approved", "Approved cycle (aims the needle)"], ["compensating", "Scripted, compensating"],
      ["learned_sampled", "Learned, with its training noise"], ["learned_deterministic", "Learned, deterministic"]];
    let html = '<tr><th>Controller</th><th class="num">Level</th><th class="num">Placed</th><th class="num">Within 10 µm</th>' +
      '<th class="num">Median</th><th class="num">90th pct</th><th class="num">Max</th></tr>';
    for (const [key, label] of rows) {
      for (const lv of ["1", "2"]) {
        const s = full.summary[key][`level_${lv}`];
        html += `<tr${key.startsWith("learned_d") ? ' class="hl"' : ""}><td>${lv === "1" ? label : ""}</td><td class="num">${lv}</td>` +
          `<td class="num">${s.threads_placed} / ${s.sites_attempted}</td><td class="num">${s.within_10um} / 30</td>` +
          `<td class="num">${um(s.placement_um_median)}</td><td class="num">${um(s.placement_um_p90)}</td><td class="num">${um(s.placement_um_max)}</td></tr>`;
      }
    }
    $("insert-full-table").innerHTML = html;
  }
  {
    const c = res.c_environment;
    const rows = [["aiming", "Scripted, aims the needle"], ["compensating", "Scripted, compensating"],
      ["learned_sampled", "Learned, with its training noise"], ["learned_deterministic", "Learned, deterministic"]];
    let html = '<tr><th>Controller</th>' + ["0", "1", "2"].map((lv) => `<th class="num">level ${lv}</th>`).join("") +
      '<th class="num">Touched a placed thread</th></tr>';
    for (const [key, label] of rows) {
      const touch = Math.max(...["0", "1", "2"].map((lv) => c[key][lv].thread_touch));
      html += `<tr${key === "learned_deterministic" ? ' class="hl"' : ""}><td>${label}</td>` + ["0", "1", "2"].map((lv) =>
        `<td class="num">${pct(c[key][lv].success)}<br><small>median ${um(c[key][lv].placement_um_median)}</small></td>`).join("") +
        `<td class="num">up to ${pct(touch)}</td></tr>`;
    }
    $("insert-c-table").innerHTML = html;
  }
  {
    const f = (v, d = 0) => (v === null || v === undefined ? "–" : Number(v).toFixed(d));
    let html = '<tr><th>Run</th><th>Change and outcome</th><th class="num">Steps</th><th class="num">Wall time</th>' +
      '<th class="num">Steps/s</th><th class="num">GPU use</th><th class="num">GPU power</th><th class="num">CPU threads busy</th></tr>';
    for (const [name, label] of INSERT_RUNS) {
      const c = res.compute[name], r = res.runs[name], s = c.sampled || {};
      html += `<tr><td><strong>${label.split(" · ")[0]}</strong></td><td>${r.change}<br><small>${r.outcome}</small></td>` +
        `<td class="num">${f(c.steps / 1e6, 1)} M</td><td class="num">${Math.round(c.wall_s / 60)} min</td>` +
        `<td class="num">${f(c.steps_per_s / 1e3, 1)}k</td>` +
        `<td class="num">${s.gpu_util_pct ? `${f(s.gpu_util_pct.mean)}%` : "–"}</td>` +
        `<td class="num">${s.gpu_power_w ? `${f(s.gpu_power_w.mean)} W` : "–"}</td>` +
        `<td class="num">${s.trainer_cpu_pct ? f(s.trainer_cpu_pct.mean / 100, 1) : "–"}</td></tr>`;
    }
    $("insert-compute-table").innerHTML = html;
  }

  // Error budget: the thread end's distance from the target at three moments, median and 90th percentile.
  {
    const svg = $("chart-budget"), f = frame(svg, { W: 640, H: 320, L: 64, R: 150, T: 20, B: 52 }), b = res.error_budget;
    const ymax = 25, y = (v) => f.T + f.ih * (1 - v / ymax);
    const stages = [["aim", "stroke starts", "the policy's aim"], ["after_stroke", "held at depth", "after the stroke"],
      ["placed", "placed", "after the release"]];
    const x = (k) => f.L + f.iw * (k + 0.5) / stages.length;
    for (let v = 0; v <= ymax; v += 5) {
      svgEl(svg, "line", { x1: f.L, x2: f.W - f.R, y1: y(v), y2: y(v), stroke: v === 10 ? "#bfe0d4" : "#eee9f3",
        "stroke-width": v === 10 ? 2 : 1 });
      svgEl(svg, "text", { x: f.L - 8, y: y(v) + 4, "text-anchor": "end", "font-size": 12, fill: "#5d5f7a" }, `${v} µm`);
    }
    svgEl(svg, "text", { x: (x(0) + x(1)) / 2, y: y(10) + 14, "text-anchor": "middle", "font-size": 11, fill: "#2f7360" },
      "10 µm tolerance");
    const band = stages.map(([k], i) => [x(i), y(b[k].p90)]).concat(stages.map(([k], i) => [x(i), y(b[k].median)]).reverse());
    svgEl(svg, "polygon", { points: band.map((p) => p.join(",")).join(" "), fill: "#646da0", "fill-opacity": 0.1 });
    stages.forEach(([, title, sub], i) => {
      svgEl(svg, "text", { x: x(i), y: f.H - f.B + 20, "text-anchor": "middle", "font-size": 13, "font-weight": 650, fill: "#23243a" }, title);
      svgEl(svg, "text", { x: x(i), y: f.H - f.B + 37, "text-anchor": "middle", "font-size": 11, fill: "#5d5f7a" }, sub);
    });
    [["p90", "#b2456b", "90th percentile"], ["median", "#646da0", "median"]].forEach(([key, color, label]) => {
      const pts = stages.map(([k], i) => [x(i), y(b[k][key])]);
      svgEl(svg, "polyline", { points: pts.map((p) => p.join(",")).join(" "), fill: "none", stroke: color, "stroke-width": 3 });
      pts.forEach(([px, py], i) => {
        svgEl(svg, "circle", { cx: px, cy: py, r: 5.5, fill: color, stroke: "#fff", "stroke-width": 2 });
        svgEl(svg, "text", { x: px, y: py - 11, "text-anchor": "middle", "font-size": 13, "font-weight": 650, fill: color },
          `${b[stages[i][0]][key].toFixed(1)}`);
      });
      svgEl(svg, "text", { x: f.W - f.R + 10, y: pts[2][1] + 4, "font-size": 12.5, "font-weight": 650, fill: color }, label);
    });
  }
});

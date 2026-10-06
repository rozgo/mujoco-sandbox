# Project website

October 6, 2026. A journal page with an interactive replay at the top, published
on GitHub Pages at <https://rozgo.github.io/mujoco-sandbox/surgical/>. Source:
[`site/`](../../site/); build: [`scripts/build_site.py`](../../scripts/build_site.py);
publish: [`scripts/publish_pages.sh`](../../scripts/publish_pages.sh).

- **Replay viewer:** raylib 6.0 compiled to WebAssembly with Emscripten 6.0.9.
  It replays recorded MuJoCo states; it does not simulate. *Insertion* (format
  `NIT2`): the final learned policy (insert_v2, deterministic) or the scripted
  yardstick placing threads at sites 0, 5 and 1 on the first evaluation seeds
  at each disturbance level, frames every 2 ms during the needle's work and
  every 8 ms between. *Alignment*: one of three policies (robust learned,
  scripted yardstick, learned without disturbances) at each disturbance level,
  on the first 30 of the 200 evaluation seeds. The camera opens on the whole
  workcell; zoom eases toward a clamped target from page-side wheel and pinch
  input.
- **Disturbance overlays:** replays carry every disturbance component the core
  recorded. The viewer draws them in the scene, enlarged so they read at any
  zoom: force arrows along each slide axis at 1 mN per 7 screen pixels (amber
  noise, cyan friction, lilac vibration), the table acceleration at 1 mm/s² per
  15 pixels with a trace, and micrometre offsets (the moving true target's path,
  the target estimate, the late, noisy tip measurement) magnified about their
  reference by a 1-2-5 factor, up to ×2000, chosen so 10 µm spans about 80
  pixels; the legend states both scales. The alignment Micro view keeps a
  true-scale 10 µm tolerance cylinder and a 50 µm ruler. Values
  sit in camera-facing panels in the scene with leader lines; panels have fixed
  sizes, avoid each other on screen and shrink with the scene beyond the
  workcell framing. The Micro camera (0.5 mm) shows micrometre quantities at
  true scale. Recorded forces are 50 Hz samples, so table vibration above 25 Hz
  is aliased in the display (not in the simulation). Labels use DejaVu Sans
  (bundled from matplotlib, subset, with its licence).
- **Rendering:** GLSL ES 3.00 shaders with Cook-Torrance GGX, a 4096² shadow map
  with depth packed into two 8-bit channels, an analytic studio environment for
  diffuse and specular ambient, and ACES tone mapping, in the site palette
  `#D38AAA #AF86AC #8981AF #7378A8 #646DA0`. The approach follows the AlienWars
  Map Lab shaders. The prebuilt raylib 6 web library uses its WebGL1-compatible
  code paths inside the WebGL 2 context; rendering is correct.
- **Data:** the exporter tessellates the MuJoCo scene per body and material
  (16-bit index chunks) and writes body poses from MuJoCo forward kinematics of
  recorded joint positions.
- **Journal:** fifteen chapters from [the process account](PROCESS.md), with the
  exact observation, action and reward specification, a network diagram read
  from PufferLib's code, training charts from the recorded run histories, code
  excerpts, figures and videos; the disturbance layer, robust training, success
  against disturbance level, and the compute each run used. Repository PNGs are
  converted to WebP at build time because Pages cannot serve Git LFS files.

```sh
git clone --depth 1 https://github.com/emscripten-core/emsdk.git .local/emsdk
./.local/emsdk/emsdk install 6.0.9 && ./.local/emsdk/emsdk activate 6.0.9
# raylib 6.0 webassembly and macOS archives from the 6.0 GitHub release, unpacked into build/site_deps/
uv run --locked python scripts/build_site.py
python3 -m http.server 8790 --bind 127.0.0.1 --directory build/site
```

Checked in Chrome: the viewer loads, lists episodes per policy and level,
plays, pauses, seeks, switches camera and overlay layers; readouts update;
charts, code highlighting, figures and videos load. The page requests the
viewer files with a hash of the build, so a rebuild is never served from cache.
Native screenshots: `build/site_native/viewer --screenshot OUT.png --camera 0-3
--episode N --time T [--zoom F] [--overlays MASK]`.

Publishing (with the user's go-ahead, given October 6): build from a committed
source, then `scripts/publish_pages.sh` copies `build/site` into the `surgical/`
folder of the `gh-pages` branch as plain files (Pages serves Git LFS pointers,
not their content) and keeps any other folders there. Pages serves that branch.

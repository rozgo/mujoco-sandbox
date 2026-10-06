# Project website pilot

October 6, 2026. A journal page with an interactive replay at the top, built
locally and not yet published. Source: [`site/`](../../site/); build:
[`scripts/build_site.py`](../../scripts/build_site.py).

- **Replay viewer:** raylib 6.0 compiled to WebAssembly with Emscripten 6.0.9.
  It replays recorded MuJoCo states; it does not simulate. The page selects one
  of three policies (robust learned, scripted yardstick, learned without
  disturbances) at disturbance level 0, 0.5 or 1.0, on the first 30 of the 200
  evaluation seeds each. Display interpolates between recorded 50 Hz states;
  error readouts use recorded states.
- **Disturbance overlays:** replays (format `NIR2`) carry every disturbance
  component the core recorded at each policy step. The viewer draws them in the
  scene: force arrows along each slide axis at 5 cm per 0.1 N (amber noise, cyan
  friction, lilac vibration), the table acceleration with a 0.3 s trace, and the
  moving true target with its path, a true-scale 10 µm tolerance cylinder, a
  50 µm ruler, the target estimate and the late, noisy tip measurement. Values
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

Publishing needs the user's go-ahead: push `main`, choose the Pages source, and
commit the built artifacts outside Git LFS (or deploy them from CI).

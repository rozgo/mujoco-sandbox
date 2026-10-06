# Project website pilot

October 6, 2026. A journal page with an interactive replay at the top, built
locally and not yet published. Source: [`site/`](../../site/); build:
[`scripts/build_site.py`](../../scripts/build_site.py).

- **Replay viewer:** raylib 6.0 compiled to WebAssembly with Emscripten 6.0.9.
  It replays recorded MuJoCo states of the 200 evaluation episodes, for the
  learned policy and the scripted reference; it does not simulate. Display
  interpolates between recorded 50 Hz states; error readouts use recorded states.
- **Rendering:** GLSL ES 3.00 shaders with Cook-Torrance GGX, a 4096² shadow map
  with depth packed into two 8-bit channels, an analytic studio environment for
  diffuse and specular ambient, and ACES tone mapping, in the site palette
  `#D38AAA #AF86AC #8981AF #7378A8 #646DA0`. The approach follows the AlienWars
  Map Lab shaders. The prebuilt raylib 6 web library uses its WebGL1-compatible
  code paths inside the WebGL 2 context; rendering is correct.
- **Data:** the exporter tessellates the MuJoCo scene per body and material
  (16-bit index chunks) and writes body poses from MuJoCo forward kinematics of
  recorded joint positions.
- **Journal:** twelve chapters from [the process account](PROCESS.md), with the
  exact observation, action and reward specification, a network diagram read
  from PufferLib's code, training charts from the recorded run histories, code
  excerpts, figures and videos. Repository PNGs are converted to WebP at build
  time because Pages cannot serve Git LFS files.

```sh
git clone --depth 1 https://github.com/emscripten-core/emsdk.git .local/emsdk
./.local/emsdk/emsdk install 6.0.9 && ./.local/emsdk/emsdk activate 6.0.9
# raylib 6.0 webassembly and macOS archives from the 6.0 GitHub release, unpacked into build/site_deps/
uv run --locked python scripts/build_site.py
python3 -m http.server 8790 --bind 127.0.0.1 --directory build/site
```

Checked in Chrome: the viewer loads, lists 200 episodes per policy, plays,
pauses, seeks and switches camera; readouts update; charts, code highlighting,
figures and videos load. Native screenshots: `build/site_native/viewer --screenshot`.

Publishing needs the user's go-ahead: push `main`, choose the Pages source, and
commit the built artifacts outside Git LFS (or deploy them from CI).

# Sixlegs

A physically modeled hexapod with two independent Kinova Gen3 arms and Robotiq 2F-85 grippers in MuJoCo.

**Current milestone: static scene for review, before movement.** The floating robot, contact geometry, actuator limits, source clutter, hollow mug, target block, barrier, destination and six cameras are ready. Walking, picking, carrying and releasing are pending scene review.

![Scene preview](previews/preview.png)

## Run on macOS

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/) and Git LFS. Tested locally on Apple Silicon macOS with Python 3.12.12 and MuJoCo 3.12.0. No ROS, Conda or source-built MuJoCo required.

```sh
# On a fresh clone, retrieve the mesh and preview binaries:
git lfs install
git lfs pull

uv sync --locked
uv run sixlegs render
uv run sixlegs inspect
uv run pytest -q

# Interactive frozen preview; rotate/zoom or select cameras in the viewer UI:
uv run sixlegs view
```

The `view` command automatically launches through `mjpython` on macOS: [MuJoCo requires it for passive viewers on macOS](https://mujoco.readthedocs.io/en/stable/python.html#passive-viewer). It also supplies the uv-managed Python library directory to the launcher, fixing the `libpython3.12.dylib` lookup failure reproduced on this machine ([upstream issue](https://github.com/google-deepmind/mujoco/issues/1923)). No global shell changes are needed. Rendering PNGs works with the ordinary `uv run` command on this Mac. Leave `MUJOCO_GL` unset on macOS; EGL/OSMesa are Linux backends. The interactive viewer requires a logged-in graphical session. On Linux, the same `uv run sixlegs view` command runs directly; offscreen rendering may need an appropriate installed backend. Intel macOS and Linux have not been tested here.

The viewer deliberately does not step physics. Its third-person view is selected initially. The five-second stability test advances actual physics separately without a task controller.

## Files

- [Original prompt](docs/INITIAL_PROMPT.md)
- [Masses, limits, contact design, layout and asset attribution](docs/DESIGN.md)
- [Time log](docs/TIME_LOG.md)
- `src/sixlegs/scene.py`: reproducible MJCF assembly, including complete preview keyframe
- `src/sixlegs/cli.py`: static viewer, six camera renders, inspection
- `assets/menagerie/`: pinned models, original licenses and integrity manifest
- `tests/test_scene.py`: vendor integrity, independent controls, contact/clearance checks and physics hold
- `previews/`: saved PNGs for all cameras plus the contact sheet
- `build/scene.xml`: generated on load, ignored by Git; uses local absolute mesh paths

`uv.lock` pins the environment. Git LFS tracks mesh, image and video binaries. Virtual environments, caches, generated XML/build artifacts, and ad hoc `outputs/` are ignored. This is a local repository; no remote is configured.

To restore vendored assets from their pinned upstream revision, run `uv run python scripts/fetch_assets.py`. Normal runs use the checked-in assets and need no asset download.

## Validation at this checkpoint

Five tests verify asset hashes, 34 independent actuator channels, initial six-foot contact without self-intersection, 150 sampled leg poses in a documented swing envelope, and five seconds of bounded-torque physics without warnings or lost objects. Six camera views render successfully on this Mac. These checks validate scene assembly and static support; task completion remains untested until a controller exists.

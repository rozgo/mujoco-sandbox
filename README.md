# Sixlegs

**New scene: six RC rovers communicating over simulated LoRa, with optional real Reticulum stacks.** [Watch the complete rover demo](previews/rovers/reticulum_degraded.mp4) · [Run it and read the measured comparison](docs/rovers/README.md)

```sh
uv sync --locked --extra reticulum
uv run --extra reticulum rover-comms view --reticulum --case degraded
```

A physically simulated hexapod with two independent Kinova Gen3 arms and Robotiq 2F-85 grippers. It walks from an offset start, grasps a mug and a block, carries both around a barrier, then places and releases them on a second table.

**The full transfer runs now.** The approved scene is preserved. The floating base moves through leg contact forces; the objects are held by finger contact. The nominal task takes about 113 simulated seconds.

[Watch the full task with all six views, real time](previews/transfer_all_views.mp4) · [Watch the shorter 2× overview](previews/transfer.mp4) · [Original static preview](previews/preview.png)

![Physical transfer checkpoints](previews/transfer_checkpoints.png)

## Run on macOS

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/) and Git LFS. Tested on Apple Silicon macOS with Python 3.12.12 and MuJoCo 3.12.0. No ROS, Conda or source-built MuJoCo required.

```sh
# On a fresh clone, retrieve the mesh and preview binaries:
git lfs install
git lfs pull
uv sync --locked

# Live physical demo, starts moving automatically:
uv run sixlegs view

# Optional: twice real-time, starting in the head camera:
uv run sixlegs view --speed 2 --camera head

# Frozen scene inspection:
uv run sixlegs view --static

# Headless complete task; writes trajectory and success report into outputs/:
uv run sixlegs run

# Tests include the complete physical transfer and rover missions:
uv run pytest -q

# Static camera PNGs and model properties:
uv run sixlegs render
uv run sixlegs inspect

# Record a new run (bundled FFmpeg, no separate installation needed):
uv run sixlegs record --speed 2

# Render the entire task with all six views, at real time:
uv run sixlegs record --trajectory outputs/transfer.npz --layout all --speed 1 --output previews/transfer_all_views.mp4

# Render the shorter overview without repeating simulation:
uv run sixlegs record --trajectory outputs/transfer.npz --speed 2
```

Viewer controls:

| Key | Action |
| --- | --- |
| **1** | Full scene |
| **2** | Head camera |
| **3** | Left wrist camera (block) |
| **4** | Right wrist camera (mug) |
| **5** | Overhead |
| **6** | Following third-person view; mouse orbit/zoom |
| **Space** | Pause / resume |
| **R** | Restart the complete task |

The all-view video shows the full scene, a following close-up, overhead, head, and both wrist cameras simultaneously in 1920 × 1080. It covers the entire run and holds the final placement for four seconds.

The on-screen overlay shows the active phase. The viewer holds the final successful state until closed or restarted. Camera images are rendered from the scene: wrists show their own fingers and the objects. Control uses simulator state and known task coordinates, not image-based object detection. This is a deterministic demonstration for this scene, not a general-purpose navigation or vision policy.

The launcher automatically uses `mjpython` because [MuJoCo requires it for passive viewers on macOS](https://mujoco.readthedocs.io/en/stable/python.html#passive-viewer). It supplies the uv-managed Python library directory to fix the `libpython3.12.dylib` lookup failure reproduced here ([upstream issue](https://github.com/google-deepmind/mujoco/issues/1923)). No global shell changes are needed. Leave `MUJOCO_GL` unset on macOS; the native viewer needs a logged-in graphical session. Linux uses the same CLI without the macOS trampoline; Intel macOS and Linux have not been tested here.

## Implementation and evidence

- [Original prompt](docs/INITIAL_PROMPT.md)
- [Masses, joint and torque limits, controller and contact design](docs/DESIGN.md)
- [Validation results and limitations](docs/VALIDATION.md)
- [Machine-readable nominal success report](docs/TRANSFER_REPORT.json)
- [Time log](docs/TIME_LOG.md)
- `src/sixlegs/scene.py`: reproducible MJCF assembly and preview keyframe
- `src/sixlegs/control.py`: five-foot-support gait, leg IK, independent arm IK
- `src/sixlegs/task.py`: approach / grasp / carry / place state machine and success checks
- `src/sixlegs/simulation.py`: shared physical stepping for viewer, CLI and tests
- `src/sixlegs/recording.py`: multi-camera video of a saved dynamic trajectory
- `assets/menagerie/`: pinned models, original licenses and integrity manifest
- `build/scene.xml`: generated locally; ignored, with absolute local mesh paths

`uv.lock` pins the environment. Git LFS tracks mesh, image and video binaries. Virtual environments, caches, generated build artifacts and `outputs/` are ignored. The repository is local; no remote is configured.

To restore the vendored assets at the pinned upstream revision, run `uv run python scripts/fetch_assets.py`. Normal runs use the checked-in assets and need no asset download.

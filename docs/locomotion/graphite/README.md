# Graphite presentation

An opt-in native MuJoCo look on `feature/graphite-presentation`: graphite surfaces,
neutral robot materials, restrained overlays, and teal/orange task accents. No
generative model, retraining or physics changes are needed.

[Watch the complete 2 min 04 s showcase at 1×](COMPLETE_VIDEO.md)
· [Watch the ten-second preview](../../../previews/locomotion/graphite/moving_graphite_v1.mp4)
· [Static design preview](../../../previews/locomotion/graphite/style_preview_v1.png).

The visible Go2/Unitree names are part of the source mesh geometry. Hiding the white
lettering alone exposes the underlying recessed letter shapes. This theme instead
uses thin curved side covers and small forward covers in the observer's `MjvScene`.
They follow the measured body transform and conceal the wordmarks. No physical
geometry is added, removed or repositioned. Vendor assets and attribution remain
intact; the robot still uses the pinned Menagerie Go2 model and existing body variants.

Lighting and materials use the standard MuJoCo renderer. A following spotlight
keeps its orientation, 4096-pixel shadows and 8× multisampling improve edge clarity,
and neutral fill preserves leg visibility against the darker floor. The theme does
not claim ray-traced metal or physical calibration of the display materials.

## Live viewers

From repository root, choose the relevant frozen policy:

```sh
uv tool run --from uv==0.12.12 uv run --project experiments/adaptive_locomotion --locked adaptive-dog demo --theme graphite
uv tool run --from uv==0.12.12 uv run --project experiments/adaptive_locomotion --locked adaptive-balance view --checkpoint assets/locomotion/checkpoints/standing/consolidate_120s_seed12.pt --theme graphite
uv tool run --from uv==0.12.12 uv run --project experiments/adaptive_locomotion --locked adaptive-moving view --checkpoint assets/locomotion/checkpoints/moving/round2_60s_seed22.pt --theme graphite
```

The classic appearance remains the default, keeping previous reproduction commands
unchanged. The three viewer commands use their original control and physics paths;
only presentation hooks differ. Cameras are observer output, not policy inputs.

## Preview and replay

```sh
uv tool run --from uv==0.12.12 uv run --project experiments/adaptive_locomotion --locked python experiments/adaptive_locomotion/scripts/record_graphite.py --preview-only --output previews/locomotion/graphite/new_preview.png
uv tool run --from uv==0.12.12 uv run --project experiments/adaptive_locomotion --locked python experiments/adaptive_locomotion/scripts/record_graphite.py --output previews/locomotion/graphite/new_replay.mp4
```

Replay needs the original cached `combined_2` model and trace from the moving-support
capture host. `--trace-root` may point to that checkout. It verifies the original
binary model and trajectory hashes before changing presentation. On headless Linux,
prefix with `MUJOCO_GL=egl`. Raw cache paths are not required for the live viewers.

The new film uses an uninterrupted ten-second accepted trajectory at 1× with
synchronized third-person, overhead and head views. The simpler overlay reports
actual simulation time, measured deck-relative drift and per-foot contact state.
It does not rerun learning or select a different trial for appearance.

Two focused tests compare themed and original flat/moving models for 500 physics
steps each: physical parameters, every pose, velocity and sensor reading agree
exactly. Observer decorations do not modify data state. Previous policy validation
and failed gates remain in the original reports.

The Mac suite passes **80 tests**, with **25 CUDA-only skips** and the two existing
upstream warnings. Both new invariance tests also pass on Linux/NVIDIA. All three
native Mac viewer commands completed four-second smoke checks. Lint/format checks
pass across 105 Python files. The movie rendered from the verified cached trace in
**5.564 seconds**, with zero new simulation or learning for capture. All **250
frames** decode at **1920×1080 / 25 fps / 10 seconds**; opening, intermediate,
camera and ending samples were inspected. A darker head view still shows the deck,
ground markings and changing horizon. Original accepted media/checkpoint hashes
and Git LFS integrity were verified.

MuJoCo documents [display materials](https://mujoco.readthedocs.io/en/stable/XMLreference.html#asset-material)
and the distinction between [model changes and simulation state](https://mujoco.readthedocs.io/en/stable/programming/simulation.html#model-changes).

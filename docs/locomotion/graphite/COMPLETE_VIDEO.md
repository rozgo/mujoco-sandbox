# Adaptive dog — complete learned-policy showcase

[Watch the final film](../../../previews/locomotion/graphite/adaptive_dog_complete_v2.mp4).

**2 min 04 s, 1920×1080, 25 fps.** All physical footage plays at **1×**, labeled
throughout; the clock reports actual simulation time. The final six seconds are
an explicit results card. No new training, rollout or generative styling was
needed. The film re-renders the accepted physical trajectories in the native
graphite theme, then combines the chapters. Previous films remain unchanged. The earlier
[65-second 2× edit](../../../previews/locomotion/graphite/adaptive_dog_complete_v1.mp4)
is archived separately; normal speed is now the preferred version.

## What is included

| Video time | Learned behavior |
| --- | --- |
| 0:00–0:12 | Healthy walking, with observer, overhead and head cameras |
| 0:12–0:24 | All four lower-leg removals; entire calf and foot absent |
| 0:24–0:36 | All four entire-leg removals; hip, thigh and calf absent |
| 0:36–0:48 | Walk, stop to balance, then resume walking; three cameras |
| 0:48–0:58 | Ordinary pads, slopes and missing foot supports |
| 0:58–1:08 | High/extreme pads, steep slopes and large steps |
| 1:08–1:18 | Damaged-body standing |
| 1:18–1:38 | All eight accepted terrain-review trials, including measured misses |
| 1:38–1:48 | Translation, yaw, vertical motion and rocking |
| 1:48–1:58 | Combined platform motion, with three synchronized cameras |
| 1:58–2:04 | Results card |

The 39 recorded trials comprise nine walking cases, 17 original standing cases,
eight reviewed terrain cases and five trained platform cases. The standing section
therefore retains every trial in the accepted v3 film. A repeated terrain condition
may show a different previously accepted trial; this is documented in the manifest.
The moving section shows the trained policy; its original three-way comparison
remains available separately.

## Which policy is shown

This is a progression through **three successive accepted checkpoints**. Each
stage uses one shared actor across its cases; footage from the earlier stages is
not relabeled as a rollout of the newest weights.

| Stage | Frozen checkpoint | Original physics capture |
| --- | --- | --- |
| Walking | `assets/locomotion/checkpoints/adaptive_dog_v1.pt` | CPU MuJoCo/mjbatch |
| Standing and reviewed terrain | `assets/locomotion/checkpoints/standing/consolidate_120s_seed12.pt` | MuJoCo Warp; reviewed terrain uses CPU MuJoCo/mjbatch |
| Moving supports | `assets/locomotion/checkpoints/moving/round2_60s_seed22.pt` | MuJoCo Warp |

Recorded poses, velocities, controls where saved, and contact forces drive replay
and annotations. Every source trace and physical model was hash-checked before
applying materials or observer decorations. The 50 Hz records are sampled at
indices 1, 3, …, ending at each exact trial endpoint: each displayed frame advances
40 ms of simulated time, producing 1× motion at 25 fps. No interpolation or pose
correction is applied. Cameras are observer output, not policy inputs.

Orange spheres mark the actual limb cuts. Branding is concealed by cosmetic
observer panels, with vendor geometry and attribution preserved. Dark surfaces,
teal task geometry, lighting and shadows follow the [graphite theme](README.md).

Original quantitative failures remain failures. The selected standing sequences
include drift, tilt or support violations, visibly labeled as missed strict
targets. Walking retains its documented gait/contact limitations. Moving training
covers the healthy body; damaged moving-platform transfer remains unresolved.
This compilation is a demonstration of recorded behavior, not a new robustness
measurement or a claim of arbitrary-damage recovery.

## Watch and reproduce

From the repository root on macOS:

```sh
open previews/locomotion/graphite/adaptive_dog_complete_v2.mp4
```

The four finished chapter files are versioned in Git LFS so either machine can
rebuild the complete film without copying private host configuration or raw caches:

```sh
git lfs pull
uv tool run --from uv==0.12.12 uv run --project experiments/adaptive_locomotion --locked python experiments/adaptive_locomotion/scripts/record_adaptive_journey.py --part assemble --speed 1 --output previews/locomotion/graphite/adaptive_dog_complete_reproduction.mp4
uv tool run --from uv==0.12.12 uv run --project experiments/adaptive_locomotion --locked python experiments/adaptive_locomotion/scripts/qa_adaptive_journey.py previews/locomotion/graphite/adaptive_dog_complete_reproduction.mp4
```

To re-render a chapter from the original ignored capture cache, use the same script
with `--part walking`, `standing`, `terrain`, or `moving`, `--speed 1`, and set `--trace-root`
to the original capture checkout. `--preview-only` emits one inspection frame per
chapter. Walking and reviewed terrain rendered on the Mac; standing and moving
rendered with EGL on the NVIDIA host. Compilation and QA run on the Mac. Headless
Linux rendering needs `MUJOCO_GL=egl`; macOS does not.

The normal-speed edition re-renders the same recorded states at 25 fps; it does
not stretch the earlier encoded movie or duplicate its frames. The recorder
defaults to `--speed 1`, saving `v2` chapter files and the `v2` complete film.
Use `--speed 2` for the archived `v1` recipe. Assembly rejects mismatched chapter
speeds. Timings are recorded in the movie manifest and time log.

All **3,100 encoded frames** decode. Dimensions, rate, duration, case coverage,
three-checkpoint provenance and playback setting pass automated checks. All 36
opening/midpoint/ending chapter samples were inspected, including transitions,
head-camera visibility, damage markers, support contacts and the final card.
The head views are deliberately dark but show the ground markings/deck and horizon.
Lint and formatting checks pass. No dynamics tests were repeated for this replay
and layout-only change; the theme's existing physical-invariance checks still apply.

[Full provenance](../../../previews/locomotion/graphite/adaptive_dog_complete_v2.json)
· [Encoded-video QA](../../../previews/locomotion/graphite/adaptive_dog_complete_v2.qa.json)
· [Walking release](../OFFICIAL_V1.md)
· [Standing results](../standing/RESULTS_VIDEO_V3.md)
· [Moving results](../moving/README.md).

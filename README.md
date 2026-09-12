# MuJoCo Sandbox

Robotics simulation demos exploring learned locomotion, manipulation, amphibious attachments, rover communication, and drone delivery with learned wind forecasts.

Explore the demos below. Click a screenshot to watch the video, or open its guide for run instructions, physical assumptions, and measured results.

## Learning the wind

A quadrotor carries a suspended parcel through crosswinds, lowers it onto a platform, and releases it. Watch a learned wind forecast and a frozen-field forecast drive the same predictive controller in a matched delivery comparison.

[![Side-by-side drone deliveries in strong crosswinds, with payload cameras and tracking error](previews/wind/aggressive/frame-crosswinds.png)](previews/wind/aggressive/comparison.mp4)

**[Watch stronger-wind demo](previews/wind/aggressive/comparison.mp4)** · [Run & technical details](docs/wind/aggressive/README.md) · [Original video](previews/wind/comparison.mp4) · [Original experiment](docs/wind/README.md)

*Featured video: stronger crosswinds and faster delivery, with the physical flight at real time. The original remains the default preset.*

## Amphibious quadruped

A quadruped with leg-mounted floats walks into a basin, extends its front legs, floats across, and walks out. The concept demo combines contact-driven walking with modeled buoyancy, drag, and water thrust.

[![Quadruped floating without ground contact, shown from following, side, and overview cameras](previews/amphibious/frame-floating.png)](previews/amphibious/crossing.mp4)

**[Watch the crossing](previews/amphibious/crossing.mp4)** · [Run & technical details](docs/amphibious/README.md)

*Three synchronized views. Walking plays at 8×; flotation and its transitions play at 2×.*

## Communicating rovers

Six RC rovers survey an inspection yard and share discoveries over simulated LoRa. The featured run uses real Reticulum stacks over the simulated channel, showing how delivered messages change a rover's decisions during radio outages.

[![Six-rover inspection dashboard with front cameras, radio links, and each rover's local knowledge](previews/rovers/reticulum_degraded.png)](previews/rovers/reticulum_degraded.mp4)

**[Watch the rover demo](previews/rovers/reticulum_degraded.mp4)** · [Run & technical details](docs/rovers/README.md)

*The complete 150-second mission at 2× playback, with an overview and all six front cameras.*

## Sixlegs manipulation

The original project: a hexapod with two independent Kinova Gen3 arms and Robotiq grippers. It grasps a mug and a block, carries both around a barrier, and places and releases them on a second table using physical foot and finger contacts.

[![Sixlegs lifting two objects, carrying them around a barrier, and releasing them at the destination](previews/transfer_checkpoints.png)](previews/transfer_all_views.mp4)

**[Watch the full transfer](previews/transfer_all_views.mp4)** · [Shorter 2× overview](previews/transfer.mp4) · [Run & technical details](docs/hexapod/README.md)

*The full transfer plays at real time with six synchronized views, including head and wrist cameras.*

## Run locally

Tested on Apple Silicon macOS. Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and Git LFS, then run from the repository root:

```sh
git lfs install
git lfs pull
uv sync --locked
```

Choose a demo; the `--extra` flags install its optional dependencies:

```sh
# Learning the wind — featured stronger-wind version
uv run --locked --extra wind wind-demo view --profile aggressive

# Amphibious quadruped — real-time motion
uv run --locked amphibious view --speed 1

# Communicating rovers — real Reticulum over simulated LoRa
uv run --locked --extra reticulum rover-comms view --reticulum --case degraded

# Sixlegs manipulation
uv run --locked sixlegs view
```

Viewer controls, recording commands, setup notes, and validation live in each demo's guide above.

## Documentation

[Development workflow](docs/DEVELOPMENT.md) · [Simulation guidelines (AGENTS.md)](AGENTS.md) · [Hexapod validation](docs/VALIDATION.md) · [Wind results](docs/wind/RESULTS.md) · [Project time log](docs/TIME_LOG.md)

## Adaptive walking — one policy, nine body configurations

**One learned policy, nine physical bodies.** A Go2-derived quadruped walks intact, with any one lower leg removed, or with any one entire leg removed. Every panel uses identical neural-network weights. PPO learns joint commands through torque-limited MuJoCo dynamics, with healthy-motion guidance, visible-step rewards and a rear-support preference.

[![One shared policy walking across nine healthy and missing-leg conditions](previews/locomotion/adaptive_dog_v1.png)](previews/locomotion/adaptive_dog_v1.mp4)

**[Watch the official 4K video](previews/locomotion/adaptive_dog_v1.mp4)** · [Run guide and experiment history](docs/locomotion/README.md) · [Frozen weights, validation and known limits](docs/locomotion/OFFICIAL_V1.md)

The user accepted this gait as the official baseline. The video preserves the approved twelve-second motion at **1×**, with orange damage markers and contact shadows. Its checkpoint has **32 min 03 s of training ancestry**; the latest two refinement trials used **2 min 59 s** on CPU MuJoCo/mjbatch with Apple GPU learning.

The final frozen-policy audit completes **288/288 tasks**, plus **72/72** checks at half the physics timestep. **84 tests** pass; the original gait-selection and contact limitations below remain recorded.

This release covers single-leg removals on flat ground, with known missing-joint inputs and scripted velocity commands. It does not establish arbitrary-damage recovery or learned parkour. The accepted baseline retains two documented limitations: FR rear timing falls below the original alternation target, and one recorded contact reaches **8.889 mm penetration against an 8 mm target**. Original reports, earlier checkpoints and videos remain available.

Run the frozen policy on Mac or Linux in its isolated uv project:

```sh
git lfs pull
cd experiments/adaptive_locomotion
uv tool run --from uv==0.12.12 uv sync --locked
uv tool run --from uv==0.12.12 uv run --locked adaptive-dog demo
```

Use `--case healthy`, `--case lower_fr`, or another case shown by `demo --help`. The default is `whole_fr`; runtime physics is **0.5 ms**, control **20 ms**. The `adaptive-dog-v1` Git tag preserves this baseline for subsequent experiments.

### Standing on uneven supports

The standing extension uses one policy to walk, hold position and walk again,
with the same actor balancing on pads, slopes, steps and missing supports.
It was fine-tuned with MuJoCo Warp and PPO on an RTX 4090 for **8 min 57 s**
beyond the existing walker. Rays and body-state feedback guide the controller;
RGB cameras provide the video views.

[![Shared policy balancing on uneven supports](previews/locomotion/standing/adaptive_standing_v2.png)](previews/locomotion/standing/adaptive_standing_v2.mp4)

**[Watch standing and terrain challenges](previews/locomotion/standing/adaptive_standing_v2.mp4)** · [Run guide, results and limits](docs/locomotion/standing/README.md)

The new checkpoint passes all **36 walking-retention trials**. Healthy standing
and transition tests pass **48/80** on CPU and Warp, including both front-foot
missing-support cases. The video also retains failures on harder terrain.
This extension preserves the accepted walking checkpoint above.

# MuJoCo Sandbox

Robotics simulation demos exploring manipulation, amphibious locomotion, rover communication, and drone delivery with learned wind forecasts.

Explore the demos below. Click a screenshot to watch the video, or open its guide for run instructions, physical assumptions, and measured results.

## Experimental adaptive locomotion

A shared learned controller handles shortened calves, weakened motors and low steps after 329 seconds of training. This experimental branch compares reactive control, a sensor-history estimator and a longer missing-calf fine-tune; the failures are retained alongside the successes.

[![Learned Go2 control over physical low steps with a shortened calf](previews/locomotion/frame-steps.png)](previews/locomotion/reactive_5m30s.mp4)

**[Watch the RL pilot](previews/locomotion/reactive_5m30s.mp4)** · [Results and isolated uv setup](docs/locomotion/README.md) · [History comparison](previews/locomotion/history_vs_reactive.mp4) · [Extra-training comparison](previews/locomotion/extra_training.mp4)

**[Healthy dog with improved gait balance](previews/locomotion/healthy_walk_balanced.mp4)** · [Known rewards and validation](docs/locomotion/BALANCED_GAIT.md) · [Gait comparison](previews/locomotion/healthy_gait_balance.mp4) · [Earlier stride experiment](docs/locomotion/LONGER_STRIDE.md)

*Real-time motion with three synchronized cameras. Learned joint control follows scripted velocity commands; general parkour and arbitrary-damage recovery remain open.*

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

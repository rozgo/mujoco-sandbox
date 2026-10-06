# Alignment learning environment

October 6, 2026. The first learning task is robot-only needle alignment, as the
brief sequences it: no thread, 1 ms robot clock. Training uses PufferLib 5.0 at
the pinned revision `6ffa5b10dbbbe4d1e8288367c7d9d3acd3bad4a2` on the RTX 4090.

## Task, fixed before training

- **Start:** gantry at rest at a random pose over the phantom (±40 mm, ±30 mm),
  needle tip 3–8 mm above the dome; one of six targets chosen at random.
- **Action:** three continuous commands in [−1, 1], cubed and scaled to needle-tip
  velocity (40 mm/s lateral, 20 mm/s vertical at full scale) at 50 Hz. The cube
  gives 40 µm/s resolution at 0.1. The programmed servo of [first motion](MOTION.md)
  tracks the integrated reference through force-limited actuators.
- **Observation (16):** tip error to the goal at two scales (5 mm and 100 µm),
  tip velocity, previous action, offset to the nearest vessel centerline,
  height above the dome and remaining time. These are exact simulator states,
  not camera measurements.
- **Success:** tip within 10 µm laterally and vertically of the point 1 mm above
  the target, slower than 0.2 mm/s, for 0.3 s. Deadline 5 s.
- **Failure:** any robot contact with the environment ends the episode.
- **Reward:** change in log-distance potential, a small action-change penalty,
  −0.05 per step low over a vessel, +2 for success and −2 for collision.
- **Evaluation:** 200 predetermined seeds (1,000,000–1,000,199), one episode
  each, never used for training or tuning. Training seeds start elsewhere.

Before any training, the scripted reference exposed two task-design problems:
descending early cut through the dome, and a 4 s deadline at 20 mm/s could not
cover the longest traverses. The reference now keeps 3 mm of clearance until it
is within 0.5 mm laterally, and the limits became 40 mm/s and 5 s.

## Implementation

One C core, [surgical_core.h](../../src/sixlegs/neural_insertion/native/surgical_core.h),
holds physics stepping, servo, observations and rewards. It is compiled into
the [Puffer adapter](../../experiments/neural_insertion/puffer/surgical_align/surgical_align.h)
for training and into a local library for tests and evaluation. The model is
the physics-only scene, with vessel and dome data embedded. One policy step
takes 0.155 ms on one Apple M3 Max core.

## Baselines on the evaluation seeds

[Manifest](ALIGN_BASELINES.json).

| Policy | Success | Collision | Mean time to success | Worst final lateral error |
| --- | ---: | ---: | ---: | ---: |
| Scripted reference (programmed) | 100% | 0% | 2.06 s | 3.4 µm |
| Zero action | 0% | 0% | — | — |
| Uniform random | 0% | 5% | — | — |

The adapter builds and runs inside Puffer's own CPU evaluation runtime on the
Mac (20 untrained episodes: all timeouts, as expected).

```sh
uv run --locked python -m sixlegs.neural_insertion.align_env baselines \
  --output outputs/neural_insertion/align/baselines_recheck_v1
experiments/neural_insertion/puffer/install.sh <pufferlib-checkout>         # CUDA trainer, Linux
experiments/neural_insertion/puffer/install.sh <pufferlib-checkout> --cpu   # CPU evaluation binary
```

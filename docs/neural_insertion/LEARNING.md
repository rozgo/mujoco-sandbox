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

## Training setup

Run `align_v1` started 2026-10-06 01:08:47 UTC on the RTX 4090 (driver 595.91,
CUDA 13.0, 32 CPU cores) from repository commit `35b67e2`, synchronized from the
Mac by a Git bundle. PufferLib 5.0 at the pinned revision, its default
recurrent PufferNet (128 hidden, 2 MinGRU layers, 100.9 K parameters) and its
default optimizer settings, except γ = 0.995, λ = 0.95 and entropy 5e−4.
2048 worlds on 28 CPU threads; MuJoCo physics stays on the CPU and only
the network trains on the GPU. 100 M steps; throughput about 60 K steps/s.

Getting the trainer to build took four fixes, all in our installer or
adapter; the pinned PufferLib source is unchanged. The native trainer needs two
bookkeeping fields the CPU build does not. nvcc rejects a versioned `.so` as an
input. The macOS CPU build needs a bash 3.2 shim for one expansion and a
framework-style rpath. The model file also differed between machines by about
1e−18 m in vessel coordinates; rounding to the picometre made both byte-identical.

The first run, `align_v1`, diverged after about 18 M steps (KL ≈ 13,500, clip
fraction 1.0, episode return −46,557). The action-change penalty used the raw
Gaussian samples instead of the clipped actions the robot receives; once the
action means drifted past ±1 the penalty grew without bound. We stopped it at
20.8 M steps, kept its log, fixed the penalty with a regression test and
restarted unchanged otherwise as `align_v2`. Baseline outcomes are identical
under the fix because they never leave ±1; seven random-policy returns differ
by at most 1.2e−7 from float rounding.

Evaluation runs a checkpoint with Puffer's own CPU network code and
deterministic mean actions on the shared core, over the 200 predetermined seeds
([evaluator](../../experiments/neural_insertion/puffer/align_eval.c)).

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

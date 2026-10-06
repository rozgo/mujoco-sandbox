# Alignment learning

October 6, 2026. **A PufferLib 5.0 policy trained on the RTX 4090 aligns the
needle on all 200 predetermined evaluation episodes, with no collisions, in
1.40 s on average, faster than the scripted reference.** This is the first
learning task as the brief sequences it: robot-only needle alignment, with no
thread and no insertion yet, at the 1 ms robot clock. Training uses PufferLib 5.0 at
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
about 22 M steps, kept its log, fixed the penalty with a regression test and
restarted unchanged otherwise as `align_v2`. Baseline outcomes are identical
under the fix because they never leave ±1; seven random-policy returns differ
by at most 1.2e−7 from float rounding.

`align_v2` learned well for 12.7 M steps: no collisions and a final error
falling to 0.48 mm. Then, as the action noise shrank, the per-update KL rose
from 0.16 to about 300 by 15.9 M steps and the policy collapsed. The clip
fraction near 0.5 throughout already indicated oversized updates at PufferLib's
default learning rate of 0.015. We stopped it at about 21 M steps (log kept)
and started `align_v3` with one change: learning rate 0.003.

Evaluation runs a checkpoint with Puffer's own CPU network code and
deterministic mean actions on the shared core, over the 200 predetermined seeds
([evaluator](../../experiments/neural_insertion/puffer/align_eval.c)).

## Results

`align_v3` trained 99.9 M steps in 29 minutes (about 59 K steps/s). Training
rollouts first succeeded at about 52 M steps and reached 100% by 69 M. Both
saved checkpoints were evaluated with deterministic actions on the 200
predetermined seeds. [Manifest](ALIGN_RESULTS.json), with the metric histories
of all three runs.

| Policy | Success | Collisions | Mean time | Worst final error | Low steps over vessels |
| --- | ---: | ---: | ---: | --- | ---: |
| **Learned, final (99.9 M steps)** | **100%** | **0%** | **1.40 s** | 8.6 µm lateral, 4.4 µm vertical | 0.25 |
| Learned, 65.5 M steps | 100% | 0% | 1.98 s | 10.0 µm lateral, 7.4 µm vertical | 0.39 |
| Scripted reference | 100% | 0% | 2.06 s | 3.4 µm lateral, — | 1.68 |

The learned policy is 32% faster and passes low over vessels less often, but it
settles closer to the tolerance than the scripted controller: an episode ends
as soon as the 0.3 s hold is met. The checkpoint is
[`align_v3_policy.bin`](../../assets/neural_insertion/align_v3_policy.bin)
(SHA-256 `5e6ee612…2a`), with Puffer's saved
[run configuration](../../assets/neural_insertion/align_v3_run.ini).

[Video](../../previews/neural_insertion/align_v3/learned_alignment.mp4): four
evaluation episodes chosen before evaluation (the first seeds covering four
targets), real time, then a results card.

![Learned policy holding at target 0](../../previews/neural_insertion/align_v3/seed1000009_01.13s.png)

```sh
uv run --locked neural-insertion view --task learned \
  --weights assets/neural_insertion/align_v3_policy.bin
uv run --locked python -m sixlegs.neural_insertion.align_policy \
  assets/neural_insertion/align_v3_policy.bin --output outputs/neural_insertion/align/recheck_v3
```

Both need the pinned PufferLib checkout under `build/` (run the installer with
`--cpu` once) and the evaluator from `puffer/build_eval.sh`.

## Robust alignment under the disturbance layer (brief v2)

The disturbance layer (documented in `native/surgical_core.h` and the journal)
scales force noise, extra friction, table vibration, tip sensing noise and
latency, target estimate errors and tissue motion by one level from 0 to 1.
Training draws each episode's level uniformly in [0, 1]; evaluation uses the
same 200 seeds at fixed levels 0, 0.25, 0.5, 0.75 and 1. At level 0 the core
reproduces the undisturbed results exactly.
[Manifest](ROBUST_ALIGN_RESULTS.json) with learning curves, compute and every
evaluation.

| Run | Change | Outcome |
| --- | --- | --- |
| `align_v4` | disturbances on, rate 0.003 | about 50 µm mean error by 35 M steps, then KL 0.1 to 14; stopped at 48.5 M |
| `align_v5` | rate 0.001 | stable, no collisions; still improving when the schedule reached zero at 99.9 M |
| `align_v6` | 300 M step budget | running |

Success on the 200 evaluation seeds (deterministic actions; collisions 0% for
every policy and level):

| Policy | Level 0 | 0.25 | 0.5 | 0.75 | 1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `align_v5`, 99.9 M steps | 62.5% | 50.5% | 26.5% | 3.5% | 0.5% |
| `align_v3`, trained without disturbances | 100% | 100% | 93.0% | 22.0% | 1.5% |
| Scripted yardstick | 100% | 100% | 85.5% | 11.0% | 1.0% |

`align_v5` is not yet robust: it trails the undisturbed policy at every level,
and its successes take about 4.4 s of the 5 s allowed (1.4 s for `align_v3` at
level 0). Its 65.5 M checkpoint scored 31.5%, 20.0% and 0% at levels 0, 0.5 and
1, so it was still improving. The checkpoint is kept as
[`align_v5_policy.bin`](../../assets/neural_insertion/align_v5_policy.bin) with
its [run configuration](../../assets/neural_insertion/align_v5_run.ini).

## What this does and does not show

It shows that the robot, servo and environment support learning a precise,
collision-free alignment from scratch, reproducibly on held-out seeds. It does
not yet show: three independent training seeds (the brief's gate), sensing
(observations are exact simulator state), tissue motion, or any part of the
thread mission. The phantom is a rigid solid, so nothing can enter it yet.

Next, in order: a tissue insertion model (puncture, insertion resistance,
retention), then a programmed pick, insert and release cycle with the thread on
the needle, measured against gates as the approach was, then learning on that
cycle with a curriculum. Thread simulation costs about 350 s per simulated
second, so learning with the thread needs a cheaper thread model first.

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

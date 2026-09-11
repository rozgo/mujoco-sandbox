# Experimental GPU physics backend

**Historical first integration.** The maintained path now has explicit optimizer
minibatches and concurrent body batches: see [matched CPU/Warp training](WARP_TRAINING.md)
for the newer three-seed **3.23× training speedup** and gait validation. The
measurements below preserve the original scheduling, GPU load and pilot results.

Started 2026-09-11 17:32:50 UTC (first retained clock checkpoint). The user asked
for a separate MuJoCo Warp path to accelerate physical simulation on the RTX 4090.
Work stays on `experiment/adaptive-dog-gpu-comparison`; official v1 stays frozen.

## Measured GPU physics results

The optional backend runs real MuJoCo Warp physics on NVIDIA, independently of
the learner device. Each of the nine physical topologies has its own GPU batch.
At 4096 worlds, the full nine-body family reaches **4.59× CPU physics throughput**
and **3.57× with the existing NumPy environment bridge included**. At 512 worlds,
that family is slower than CPU; use the larger batch for this experiment.

Median 20 ms control intervals per second, three repeats, same desktop:

| Bodies | Worlds | CPU physics | Warp physics | CPU environment | Warp environment |
| --- | ---: | ---: | ---: | ---: | ---: |
| Healthy | 512 | 16,416 | 99,850 | 15,406 | 70,127 |
| All nine | 512 | 16,996 | 11,458 | 14,769 | 10,520 |
| Healthy | 4096 | 18,140 | 320,291 | 15,891 | 115,905 |
| All nine | 4096 | 18,008 | 82,740 | 15,476 | 55,194 |

Physics uses ten 2 ms integration steps per control interval, so integration
steps/s are ten times these values. These standing-command measurements exclude
policy inference, PPO updates and healthy-motion reference matching. The
environment column includes transfers, observations, rewards and episode resets.
They are not complete training rates. Full repeats, versions, topology counts,
source commits and before/after load samples are in the
[archived reports](warp_backend/index.json).

Warm-cache Warp setup took 2.32–4.70 seconds across the four configurations;
the nine-body 4096-world setup took 4.70 seconds. First-time compilation is a
separate cost: the initial failed compatibility check took 134.84 seconds.
Measurements used a Ryzen 5950X / RTX 4090 with another GPU workload left running.
These are observed shared-machine results, not exclusive-hardware benchmarks.

## Plan declared before measurement

- Keep `mjbatch` CPU physics as the default and native Mac evaluation/viewing path.
- Add optional `--physics-backend warp`, independently of the PyTorch learner
  device. Pin current releases MuJoCo Warp 3.13.0 and Warp 1.17.0 in the uv extra.
- Preserve all nine real body models, their contact geometry, sensors, joint and
  torque limits, solver settings, and 2 ms training / 20 ms control clocks. Batch
  each topology separately; do not replace missing limbs with invisible parts.
- First implement a compatibility bridge: physics and CUDA graphs on GPU, existing
  NumPy observations/rewards/reference matching on CPU. Disclose host transfers;
  this is not an entirely GPU-resident RL loop.
- Validate deterministic reset and short fixed-action trajectories against CPU,
  contact sensors and actuator limits, finite state and contact/constraint capacity.
  Predeclared short-step tolerances: max qpos error 0.002, qvel error 0.1 over ten
  2 ms steps from identical initialized states and controls. Long contact-rich
  trajectories may diverge; judge those by task and per-leg gait metrics.
- Measure cold setup/compilation separately from synchronized warm stepping.
  Compare 512 and 4096 environments, healthy and all nine bodies, CPU and Warp,
  three 3-second repeats per configuration. No physics-only speed claim may be
  presented as end-to-end RL throughput.
- If correctness passes and the bridge is usable, run one bounded 90-second PPO
  continuation with the previous frozen parent, motion bank, seed and recipe,
  followed by eight CPU evaluation trials per body. Preserve the final checkpoint
  and all failed gates. Do not tune or promote from this pilot. If bridge overhead
  dominates, document it and stop before expensive training.
- Leave unrelated GPU workloads running. Record load and limit interpretation.

## Sources

[MuJoCo Warp documentation](https://mujoco.readthedocs.io/en/stable/mjwarp/index.html)
explains GPU batching, graph capture, capacity checks and supported features.
[Official source](https://github.com/google-deepmind/mujoco_warp) and the installed
3.13.0 API are the implementation references. Release versions were checked on
[PyPI](https://pypi.org/project/mujoco-warp/) before pinning the optional stack.

## Compatibility investigation

The first NVIDIA check failed before stepping: Warp 1.17's occupancy query loaded
one CCD kernel variant, then a 64-thread launch changed its symbol identity; the
next query raised a missing-symbol/metadata error. The original failed test log
is retained under ignored outputs (134.84 seconds including cold compilation).
The next attempt kept the CCD block width at 256, matching Warp's default module
load width. This is a GPU launch configuration adjustment, not a geometry,
collision, contact-count or solver relaxation. Latest dependency pins are retained.

## Commands (on the experiment branch)

From `experiments/adaptive_locomotion` on an NVIDIA host:

```sh
uv sync --locked --extra warp
uv run --locked --extra warp pytest -q tests/test_warp_backend.py
uv run --locked --extra warp python scripts/run_physics_matrix.py

# Bounded same-parent learning pilot, after the physical checks pass.
uv run --locked --extra warp python scripts/benchmark_learner.py \
  --label warp_bvh_4096 --device cuda --physics-backend warp --num-envs 4096
```

General training supports the same independent flags: `adaptive-dog train
--physics-backend warp --device cuda ...`. Omit the physics flag for the original
CPU path. Warp requires NVIDIA CUDA; Mac viewers/evaluation retain ordinary CPU
MuJoCo. No Warp import or GPU requirement is imposed on default training.

The bridge uses real per-world reset masks and per-world actuator parameters.
Only explicit reset/forward transfers state to the device; ordinary steps upload
actuator controls and advance the GPU state. Copies back to NumPy are measured as
bridge overhead. Contact/constraint/sensor-matching overflow flags raise errors;
nonfinite state and out-of-envelope velocity stop the run. CPU evaluation remains
the authoritative task check after any GPU training.

The 256-thread workaround passed all nine topology/reset/physics tests, but its
first healthy-512 benchmark reached only **902 physics control intervals/s**,
versus **16,416/s** on CPU. Its completed reports are retained, and the in-progress
nine-body attempt was stopped before continuing the expensive matrix.
A narrower version-scoped compatibility hook now sets each generated CCD
kernel's *module default* to its already-requested 64-thread launch width. This
avoids changing its launch bounds. It uses Warp's public `set_module_options`
API, with one explicit private MJWarp builder hook guarded to the pinned versions;
review/remove it on upgrades. No solver or collision mathematics are patched.
The revised attempt reran all GPU checks before new throughput claims.

## Profile-driven sensor acceleration

The narrower CCD fix also passes all nine physical/reset tests (**10 tests,
95.31 seconds with cached/remaining compilation**), but healthy-512 throughput
stays near **925 physics control intervals/s**. Thus the slow result was not
explained by CCD launch width. The incomplete larger matrix was again stopped.

A three-step, 512-world CUDA-event profile attributes **186.340 / 190.284 ms
(97.9%)** of measured kernel time to `_ray`. MJWarp's default rangefinder path
scans mesh triangles; its public `rays(..., rc=...)` API also supports BVH queries.
The bridge now builds a camera-free render context solely for that spatial index,
refits it from current geometry during each sensor call, and routes only this
bridge's rangefinders through the accelerated public query. All six geom groups,
body exclusions, ray origins/directions and contact sensors remain enabled.
A second version-scoped internal dispatch hook is required because the upstream
rangefinder call does not supply a render context. No sensor values are faked.

Before claiming equivalence/speed, test **24 randomized poses per body** (seeds
9222–9224, eight worlds each, nine bodies), including body orientation and joint
variation, against CPU range distances with **0.1 mm absolute tolerance**. Test
positions span ±10 m, beyond the walking course. BVH plane bounds are finite in
MJWarp (this floor's bound extends ±24 m); this bridge is intended for the bounded
course, not arbitrary unbounded plane-ray queries. Physics plane collisions
remain unchanged. The new attempt uses `bvh_` result labels; prior results stay
available.

## First accelerated results and bounded learning decision

The BVH path passes **19 optional-backend checks**, including all 216 randomized
sensor poses (0.1 mm tolerance), in 20.77 seconds with existing compiled kernels.
Healthy-512 rises to **99,850 physics control intervals/s** and **70,127/s with
the environment bridge**, versus CPU **16,416 / 15,406**. The unchanged nine-body
split at 512 worlds reaches **11,458 / 10,520**, below CPU **16,996 / 14,769**:
48-world injury batches underfill the GPU. These are standing-command benchmarks,
not PPO update rates.

Before inspecting the 4096-world results, declare the next learning attempt:
if the larger nine-body benchmark shows a useful gain, run **one 90-second
continuation at 4096 environments** (`warp_bvh_4096`), with the same frozen parent,
healthy reference bank, seed 2, rewards, timestep and optimizer settings. This is
an explicit scaling experiment: batch size/update count differ from the earlier
512-world learner comparison, so neither speed nor learning quality isolates
hardware alone. No small-batch GPU continuation is needed if it is already slower.
Evaluate the final checkpoint in CPU MuJoCo with eight trials per body, seed 9217.
Keep failures and do not replace official v1. The training budget stays 90 seconds.

## Completed learning pilot

The single predeclared run used **90.916 seconds** of training, **5.823 seconds**
of setup, **24 PPO updates**, and **2,359,296 transitions**: **25,950 transitions/s**
including the training loop's deadline handling. Process wall time was 96.815
seconds. The nominal 90-second limit is checked at rollout boundaries; the small
overrun is retained in the reported actual time. This is a continuation from the
archived v1 parent, not training from scratch. Total selected ancestry is
1924.074 seconds; the pilot adds only 90.916 seconds.

The learner and actor both ran in CUDA; rigid-body physics ran in MuJoCo Warp.
Rewards, observation assembly and the frozen healthy-motion sequence search still
ran on CPU. End-to-end throughput is about **2.18×** the earlier desktop
CUDA-learner/CPU-physics run, but world count changed **512 → 4096**, so this is a
scaling result and does not isolate the backend alone. One seed and a different
number of optimizer updates cannot establish better learning quality.

GPU telemetry peaked at **13,432 MiB total VRAM**, including the other application,
leaving about 10.6 GiB of the card available. Sampled utilization ranged 14–90%.
There was no out-of-memory failure or need to stop the user's workload. Less
competition would improve repeatability; extra memory was not required.

Source was clean commit `5d20223`. Initial parameter tensors match the previous
three runs, and the same hashed healthy-motion bank, seed, rewards, optimizer,
torque limits and 2 ms training step were used. The checkpoint was transferred
through Git LFS with tensor and byte hashes checked; only private parent/source
path metadata was normalized. The original training report remains intact.
See [training](gpu_comparison/warp_bvh_4096_training.json),
[load telemetry](gpu_comparison/warp_bvh_4096_benchmark.json), and
[checkpoint provenance](gpu_comparison/warp_bvh_4096_checkpoint.json).

### CPU evaluation of the final checkpoint

Eight predetermined trials per body, seed 9217, twelve simulated seconds, 0.5 ms
physics and support checks at every substep: **65/72 full tasks pass**. Every
condition passes 8/8 except whole-FR, which passes 1/8. All 72 stay upright and
have valid support; the seven failures miss the compound goal condition, not the
contact rule. Completion requires at least 5 m progress followed by one continuous
second within the ±1 m lane by the twelve-second deadline. Whole-FR ends at
5.19–5.44 m across its trials; crossing 5 m alone is insufficient. The existing
report does not separate insufficient dwell from lateral goal tolerance.

All eight other gait checks pass: healthy gait, mean swing peak, visible-swing
fraction, visible steps in every trial/foot, stride/stance, speed, vertical motion
and airborne fraction. Healthy mean stride is **32.75 cm**, speed **0.583 m/s**,
height standard deviation **8.22 mm**, and left/right duty gap **2.53 percentage
points**. Task completion still fails the aggregate acceptance check, so the
evaluation command correctly exits 1. No extra training, checkpoint search or
promotion followed. The archived final checkpoint and all failures are retained.
Full [task/gait evaluation](gpu_comparison/warp_bvh_4096_evaluation.json) and
[comparison gates](gpu_comparison/warp_bvh_4096_comparison.json).

## Watch the experimental policy

[Nine-case video](../../previews/locomotion/warp_bvh_4096.mp4): GPU-trained weights
executed in ordinary CPU MuJoCo, one shared policy, fixed seed 9143, 0.5 ms physics,
twelve seconds at 1×. This video is CPU validation of learned weights, not a
screen recording of CUDA physics. The separate frozen-policy report above is the
GPU-physics mission evidence. Main and the user-approved video remain unchanged.

From `experiments/adaptive_locomotion`, including on macOS:

```sh
uv run --locked adaptive-dog view \
  --checkpoint ../../assets/locomotion/checkpoints/learner_warp_bvh_4096_90s_seed2.pt \
  --case whole_fr --presentation damage --timestep 0.0005
```

The exact command passed a five-second native macOS viewer smoke test. The Warp
extra is not required to evaluate, view or record its trained checkpoint on Mac.

The encoded video is 3840×2160, 25 fps, 300 frames; every frame decoded, and the
opening/middle/ending images were inspected. Whole-FR is explicitly marked
**INCOMPLETE** at the end; all nine runs and the eight successes remain visible.
Captured states, timestamps, checkpoint and model hashes were verified. All
original torque caps hold; maximum penetration sampled at 20 ms is **7.772 mm**,
below the unchanged 8 mm target. Capture took **9.589 seconds**, rendering/export
**40.589 seconds**, excluding context setup. [Video QA](warp_backend/video_qa.json).

The useful next optimization is moving observation/reward/reference computation
onto the GPU and reducing synchronization between the nine batches. This pilot
establishes a tested optional physics path; it does not yet deliver an entirely
GPU-resident environment or a superior replacement policy.

## Correctness and boundaries

- Mac package plus mjbatch suite: **86 passed**, **18 CUDA-only checks skipped**;
  the 18 CUDA checks passed on the 4090 (19 tests there including the shared
  invalid-backend check). Two upstream Warp/Python 3.14 deprecation warnings
  remain. Ruff passed.
- All nine body types passed short CPU/GPU trajectory tolerance, explicit reset
  isolation, graph-capture non-advancement, support sensor, actual torque-limit,
  motor-strength and finite-state checks. All 216 randomized range-sensor poses
  matched CPU within 0.1 mm.
- Before training, unchanged official v1 completed **8/8 full GPU-physics trials**:
  healthy and whole-FR, four each, seed 9225, twelve simulated seconds at 0.5 ms.
  Allowed support was checked every integration step. This is a physical backend
  mission check, separate from CPU evaluation of the newly trained policy.
  [Full mission report](warp_backend/frozen_policy_trials.json).
- The two internal dispatch hooks are intentionally tied to MuJoCo Warp 3.13.0
  and Warp 1.17.0. Review them when upgrading; the version guard fails explicitly.
  Ray acceleration uses a finite BVH plane bound (±24 m here); sensor agreement
  was tested within ±10 m. This supports the bounded walking course.
- MuJoCo Warp reports no multicontact support for capsule–cylinder pairs, using
  at most one contact for those pairs. Original collision geometry was retained;
  the backends should not be described as numerically identical.
- This pilot does not replace official v1, resolve its recorded phase/penetration
  limitations, or establish obstacle-course or real-robot performance.

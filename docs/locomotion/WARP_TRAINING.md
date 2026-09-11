# Matching CPU learning with an optimized Warp backend

Goal started **2026-09-11 18:27:44 UTC**. User brief: keep the same network and
reward with Warp as far as possible, optimize its implementation, reproduce CPU
learning behavior, and compare training speed and learning attainable within a
short time budget. Deliver a maintained training path, not a one-off speed demo.
The approved v1 weights and original videos remain frozen while this backend is
validated on the existing branch.

**Result: the 4096-world matched-update comparison passes the declared backend
acceptance checks.** Same network, rewards, initial weights, experience and
optimizer updates; **3.23× pooled training speedup** on the desktop, with
**205/216 Warp vs 200/216 CPU** task completions. This validates a maintained
backend for continuing adaptive walking, not complete training from scratch.

## Contract declared before new training

- Same 128×128 ELU actor/critic, observation channels, action distribution, PPO
  loss, reward terms/weights, reference bank, nine physical body models, contacts,
  torque limits and physics/control clocks. No reward tuning or gait scripting.
- Make optimizer minibatch size independent of world count. The current CPU
  recipe uses 3072 samples per minibatch, four epochs, 24 rollout steps and
  learning rate 0.0001. Default historical behavior remains available; the
  maintained recipe sets these values explicitly.
- Add an iteration cap and record actual optimizer steps, so paired runs can
  compare equal experience and update count instead of just equal wall time.
- Optimize scheduling without changing the physics: separate streams for the
  independent body batches, then synchronize before reading observations. Verify
  serial/concurrent equivalence including partial resets before timing or learning.
- First matched learning: 512 worlds, 48 PPO rounds, seeds 2/3/4, CPU physics
  versus Warp, both actor/learner CUDA on the same desktop. Same archived parent
  and healthy bank as the earlier runs. Hard limit 180 seconds per run; a run
  that cannot finish 48 rounds is reported as incomplete, not silently compared.
- Evaluate final checkpoints only on CPU physics, seed 9237, eight trials/body,
  0.5 ms / 20 ms, 12 seconds. Retain every trial. Compare support-valid completion
  and existing healthy/foot-lift/stride/stance/body-motion metrics across seeds.
- Engineering acceptance: no numerical/support failures; Warp pooled task
  completion within five percentage points of matched CPU; no body more than
  two of 24 trials worse; retain the existing healthy and visible-step checks.
  This is a small development comparison, not a generalization guarantee.
- Once matched behavior is demonstrated, compare 4096 worlds with 3072-sample
  minibatches on both backends using a bounded equal-time run, and evaluate both
  final checkpoints. Keep this scaling comparison distinct from the first test.
- Preserve failed checks and iterations. If differences appear, diagnose
  numerical/contact/update behavior before changing optimization settings.

## Implementation sources

[Warp concurrency documentation](https://nvidia.github.io/warp/latest/user_guide/execution_and_performance/concurrency.html)
describes stream ordering and synchronization. Implementation uses the installed
Warp 1.17.0 API and keeps the pinned MuJoCo Warp 3.13.0 compatibility hooks from
[the initial backend integration](WARP_BACKEND.md).

## Time and machine conditions

Initial GPU check: 3809 MiB total VRAM, 7% utilization. This is lower competing
load than the earlier backend pilot. Fresh CPU/Warp pairs must run under the same
current machine conditions; do not attribute gains over old loaded measurements
solely to new code. Setup, compilation, learning, evaluation and rendering are
reported separately.

## First matched result and diagnosis

The 512-world pairs completed identical 48 rounds / 768 optimizer steps /
589,824 transitions. CPU took 44.314, 43.427, 43.940 seconds; Warp took 25.770,
26.306, 24.934 seconds: **1.710× pooled speedup**. All seeds and results are
retained. CPU completes **198/216**, Warp **183/216**; all 432 trials remain
upright with valid support. The gap is lower-FR (24/24 vs 14/24) and whole-FR
(6/24 vs 1/24). The predeclared completion and per-seed gait gates fail. This
is not accepted as matched behavior.

A frozen-policy cross-check (seed 9241, four trials per case, both front-right
removals, both engines, both 2 ms and 0.5 ms) shows very similar distances on
the two engines for the same weights. For example, the Warp seed-4 lower-FR
policy reaches 5.4045 m on CPU and 5.3998 m on Warp at 2 ms, with whole-FR
5.3426 vs 5.3431 m. Its weakness persists on its own training engine. This
supports diagnosing drift during continuation rather than a runtime-only
transfer failure; it does not prove every numerical difference is harmless.
The unchanged archived parent scores 71/72 and official v1 72/72 on seed 9237.

### Next decision, before more training

Use **4096 worlds on both engines**, **six PPO rounds**, and the same
**3072-sample minibatches, four epochs, 589,824 transitions and 768 optimizer
steps** as the first matched test. Keep the same parent, seeds 2/3/4, learning
rate 0.0001, network, rewards, reference bank and clocks. This trades sequential
rollout depth for a broader simultaneous population, a possible way to reduce
gradient variability while exploiting GPU batching. It is a hypothesis to test,
not an established explanation. Apply the same acceptance gates and preserve
the failed smaller-batch results. Labels start with `scaled_`.

## Numerical scheduling checks

Each body owns its state, model parameters, CUDA stream and pinned host buffers.
Controls, physics and copies are ordered within that stream. All streams finish
before CPU observation/reward code reads the buffers. Overflow is copied with
the other fields rather than triggering an additional per-body device sync.
The CPU reward functions and deployed network implementation are unchanged.

The first scheduling test compared a continuing 0.6-second trajectory with a
2e-5 absolute reward threshold and failed at 4.24e-5. A three-copy diagnostic
found that even serial/serial repeats vary: maximum reward difference 2.56e-4,
qpos 1.19e-6 and qvel 5.39e-5. Serial/concurrent trajectories can diverge further
through contact events (qpos 0.000322, qvel 0.255 after 0.6 seconds). These are
not bitwise-deterministic trajectories, and the failed first test is retained
in the run log and [diagnostic](warp_training/schedule_diagnostic.json).

The delivered test checks **960 one-control-interval samples from matched fresh
states**, with observations within 2e-5, torque within 2e-4 Nm, reward within
1e-4 absolute + 1e-4 relative tolerance, and identical termination flags. Partial
resets must preserve every unreset world's qpos exactly. This passed along with
the original nine CPU/Warp physical checks, nine range-sensor pose checks and
three optimizer-scheduling tests: **23 GPU-host tests passed**. Long-run quality
is judged separately by the multi-seed physical task evaluations above.

Mac package/mjbatch suite: **89 passed, 19 CUDA checks skipped**. The existing
Warp/Python 3.14 deprecation and capsule–cylinder multicontact warnings remain
documented; no collision shapes or torque/contact limits were changed. The new
`adaptive-dog learn` command passed an actual one-round CPU smoke run (0.120 s
new training), and explicit minibatches reproduced legacy updates exactly in
the separate short scheduling test.

## Matched learning result: 4096 worlds

Every run uses 589,824 transitions, six PPO rounds and 768 Adam steps. Physics is
CPU MuJoCo/mjbatch or MuJoCo Warp; actor inference and learner updates use CUDA
in both cases. Runs alternate backend order across seeds on the same Ryzen 5950X
and RTX 4090. These are learning-loop times with compiled kernels; initialization
and compilation are excluded and reported separately in each run's JSON.

| Seed | CPU training | Warp training | Speedup | CPU tasks | Warp tasks |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2 | 46.740 s | 15.286 s | 3.06× | 67/72 | 70/72 |
| 3 | 45.269 s | 13.542 s | 3.34× | 65/72 | 71/72 |
| 4 | 44.763 s | 13.493 s | 3.32× | 68/72 | 64/72 |
| Pooled | 136.772 s | 42.321 s | **3.23×** | **200/216** | **205/216** |

Pooled throughput is **12,937 → 41,811 transitions/s**. Setup with cached
kernels takes about 2.48 s for CPU and 4.68 s for Warp. Including setup, the
short six-round processes average 48.17 s vs 18.85 s (**2.56×**). This does not
include a fresh installation's first kernel compilation. Peak total sampled
VRAM is 4,468 MiB for the CPU-physics runs and 6,808 MiB for Warp, including the
unchanged competing workload. Timings are not exclusive-machine measurements.

All 432 trials stay upright with allowed support. Both backends complete 24/24
trials for every body except the entire front-right removal (CPU 8/24, Warp
13/24). Every seed passes the existing healthy-gait, visible-step, stride,
stance, speed and body-motion retention comparison. All configuration checks
match. See [machine-readable acceptance](warp_training/scaled_summary.json),
[all paired gait checks](warp_training/scaled_pairs.json), and per-run reports.

This is backend equivalence within the predeclared engineering margins, not a
claim that Warp improves the policy. Seed 4 still performs worse with Warp, and
both continuations can lose quality relative to the unchanged parent. The parent
has 71/72 completions and official v1 72/72 on the same development seed. The
larger population improves the pooled match relative to the failed 512-world
comparison; it does not isolate the cause or establish arbitrary-damage recovery.

## Equal-time result and longer-update control

One predeclared seed-2 pair used the same 4096 worlds, parent, rewards and
3072-sample minibatches with a 90-second budget. All actual updates completed
four epochs; final checkpoints only were evaluated. Setup remains separate.

| Backend | Actual training | Samples used | PPO rounds / Adam steps | Task completion |
| --- | ---: | ---: | ---: | ---: |
| CPU physics / CUDA learner | 89.219 s | 1,081,344 | 11 / 1,408 | 70/72 |
| Warp physics / CUDA learner | 90.406 s | 3,932,160 | 40 / 5,120 | 59/72 |

Warp provides **3.64× more experience**, or **3.59× measured throughput**, but
the final policy regresses. All trials remain upright with allowed support;
lower-FR completes 3/8 vs CPU 8/8 and whole-FR 0/8 vs CPU 6/8. This is not a
better-policy result, and an unrestricted 90-second continuation is not promoted.
The fixed-update result above remains a separate comparison.
[Full equal-time report](warp_training/time_summary.json).

Before additional learning, the decision is to run **one CPU control with exactly
40 PPO rounds / 5,120 Adam steps / 3,932,160 transitions**, matching the completed
Warp run. Same seed and recipe, with a 330-second hard budget; no network, reward,
learning-rate or checkpoint changes. This tests whether the longer continuation
also regresses on CPU instead of attributing the equal-time difference to Warp.
The extra allowance is for this bounded diagnostic; ordinary `learn` invocations
retain their five-minute requested limit. Results will be retained regardless of
outcome.

**Control result:** CPU completes the same 40 rounds in **301.874 s**, versus
Warp's **90.406 s**: **3.34× faster** with Warp. CPU scores **60/72**, Warp
**59/72**; all trials remain upright with valid support and every healthy,
visible-step, stride, stance, speed and body-motion comparison passes. Initial
parameter hashes, learning settings and training/environment/network source
hashes match. See [the complete control report](warp_training/depth_summary.json).

This supports a shared longer-continuation regression rather than interpreting
the equal-time result as a general Warp disadvantage. Per-body outcomes still
differ: lower-FR is CPU 0/8 vs Warp 3/8, whole-FR CPU 4/8 vs Warp 0/8. One
follow-up seed does not establish per-body equivalence at this depth, and neither
long-run policy is promoted. For backend comparisons use matched round/sample/
optimizer counts; evaluate before adopting further fine-tuning. The maintained
backend is ready for that workflow, while improving the continuation recipe is
a separate learning problem.

## Maintained training command

From `experiments/adaptive_locomotion`, on an NVIDIA host:

```sh
uv tool run --from uv==0.12.12 uv sync --locked --extra warp
uv tool run --from uv==0.12.12 uv run --locked --extra warp adaptive-dog learn \
  --physics-backend warp --device cuda --num-envs 4096 \
  --minibatch-size 3072 --max-iterations 6 --seconds 90 \
  --output ../../outputs/locomotion/my_warp_round
```

`learn` fixes the accepted adaptive-walking reward recipe, defaults to frozen v1
as its parent, preserves previous run directories, and records checkpoint ancestry
separately from new training time. It is the final continuation stage. Remove
`--max-iterations` for a bounded time-budget run; use a new output directory each
time. The existing time-budget guard can discard a rollout that finishes after
the deadline and stop updates between minibatches. Actual optimizer-step counts
are recorded; a final PPO round can be partial. An in-progress rollout or
minibatch can exceed the deadline slightly. Each requested round is capped at
five minutes. No automatically selected intermediate checkpoint replaces the final.
The actual maintained command passed one-round smoke tests on Mac CPU (0.120 s
training) and NVIDIA Warp/CUDA (0.946 s training, 16 optimizer steps); these tiny
runs validate the command path and are excluded from the performance comparison.

For a matched CPU-physics run on the same desktop, change only
`--physics-backend mjbatch` and the output directory, retaining `--device cuda`
and all counts. On a Mac, omit `--extra warp`, use `--physics-backend mjbatch`
and `--device auto`; that is a portability path, not the same-machine benchmark.
Default world counts are 512 for CPU and 4096 for Warp; **set the same explicit
world count for a comparison**.

The archived benchmark uses the earlier parent
`limb_rear_overlap_iter100_seed2.pt` to stay comparable with preceding hardware
runs. Reproduce its six runs with `python scripts/train_warp_match.py --prefix
scaled --num-envs 4096 --iterations 6` on a clean source checkout with empty run
directories. Curate with `scripts/curate_warp_training.py`, evaluate with
`scripts/evaluate_warp_match.py --prefix scaled` and all six labels, then run
`scripts/summarize_warp_match.py --prefix scaled`. The curator verifies the exact
initial tensor identity and retains every final checkpoint.

## Implementation and remaining limits

- The 128×128 ELU actor and critic, observation/action mapping, PPO loss and
  reward formulas are shared source. GPU scheduling changes no gait targets.
- Nine independent streams overlap the nine physically different body batches.
  Missing parts remain absent from the model; no dummy hidden leg is introduced.
- Minibatch size is independent of environment count. This prevents increasing
  GPU batching from silently making each gradient update eight times larger.
- NumPy observation/reward computation and healthy-reference matching still run
  on CPU. This is GPU physics plus CUDA actor/learner, not a fully GPU-resident
  environment. Those host operations and copies limit further throughput.
- Contact reductions are not bitwise deterministic. The pinned compatibility
  hooks, finite ±24 m BVH sensor floor and capsule–cylinder multicontact
  limitation remain documented in [backend integration](WARP_BACKEND.md).
- One pretrained policy per run handles all nine body configurations. These
  results cover single removals on flat ground; no new parkour, online fault
  diagnosis or full curriculum training claim is made.

All fifteen comparison checkpoints and their failed/successful evaluations are
retained. They used **869.284 s (14 min 29 s) of new training in total**; the
separate CLI smoke runs add 1.066 s. Checkpoint ancestry is recorded independently
and must not be confused with this task's additional compute. Individual
matched runs take 13–47 seconds; equal-time runs about 90 seconds; the final CPU
control took 302 seconds. Elapsed implementation/testing/media time is recorded
in [the project time log](../TIME_LOG.md).

## Video and native viewing

**[Watch the matched CPU/Warp comparison](../../previews/locomotion/warp_scaled_comparison.mp4)**

[![CPU and Warp adaptive-walking comparison](../../previews/locomotion/warp_scaled_comparison.png)](../../previews/locomotion/warp_scaled_comparison.mp4)

The predetermined seed-2 pair is shown at 1× in three twelve-second chapters,
covering every body. One policy per column; identical physical models and initial
conditions per row. Both columns execute in CPU MuJoCo at 0.5 ms for independent
validation; the heading identifies their **training** backend. Lane commands are
scripted, joint targets are learned, and orange spheres are visual damage markers.
Both whole-FR demonstration trials miss the timed goal; the final captions retain
those failures. No training or checkpoint selection was done for the video.
Completion requires at least 5 m progress and one continuous second inside the
±1 m lane, while upright with allowed support, before the twelve-second deadline.
Crossing 5 m in the final second alone is insufficient.

All **900 frames** decode at **1920×1440 / 25 fps / 36 s**. Chapter openings,
midpoints, endings and transitions were inspected. Capturing the eighteen live
trajectories took **19.776 s**; rendering took **61.641 s**, excluding renderer
setup. State/model/checkpoint hashes and original torque caps hold. All trials
remain upright with allowed support. The Warp lower-FR trajectory has **8.090 mm**
maximum 20 ms sampled penetration, exceeding the existing 8 mm target; video
physical QA therefore remains failed on that check. This known small numerical
overrun is retained, separate from the learning-comparison acceptance. It is
smaller than frozen v1's documented 8.889 mm but is not relabeled as a pass.

From the isolated package directory, view the same Warp-trained weights live:

```sh
uv tool run --from uv==0.12.12 uv run --locked adaptive-dog view \
  --checkpoint ../../assets/locomotion/checkpoints/warp_training/scaled_warp_seed2.pt \
  --case whole_fr --timestep 0.0005
```

The actual native Mac launch passed with `--seconds 5`. Reproduce the video with
`python scripts/record_warp_match.py --prefix scaled`; it preserves an existing
MP4 unless `--replay` explicitly rerenders its saved hashed trajectories.
`python scripts/qa_warp_video.py --prefix scaled` decodes the entire result and
returns a nonzero exit status for the retained penetration failure.

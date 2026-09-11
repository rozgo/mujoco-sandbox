# Matching CPU learning with an optimized Warp backend

Goal started **2026-09-11 18:27:44 UTC**. User brief: keep the same network and
reward with Warp as far as possible, optimize its implementation, reproduce CPU
learning behavior, and compare training speed and learning attainable within a
short time budget. Deliver a maintained training path, not a one-off speed demo.
The approved v1 weights and original videos remain frozen while this backend is
validated on the existing branch.

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

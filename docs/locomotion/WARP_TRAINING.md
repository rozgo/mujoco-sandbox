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

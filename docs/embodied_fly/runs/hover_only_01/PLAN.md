# Hover-first PPO on the accepted plant

User-approved order: hover; straight flight and turns; stand; land; walk; takeoff.
One command-conditioned actor continues across stages. This stage uses hover in
every world. Old ground skills/checkpoints stay preserved; no ground retention
claim or hidden ground rehearsal is made here.

Observed implementation start: 2026-09-13 20:34:53 UTC.

## Fixed first pilot

- Parent: `position_ppo_timing_02.pt`, SHA256
  `1b926a9e1489d1792177519b3b7de6d56cc2723882077daf7ff7a1bf1ad0fd1e`.
- Explicit frozen-weight migration: accepted instantaneous wing-force model,
  1,000 Hz physics, 500 Hz actor (two physics ticks/action). Same physical body,
  actuator limits and other options. No wing averaging or PID in the actor.
- Add two ideal current horizontal target-error observations in anatomical axes.
  397 -> 399 observations; 14 -> 16 sensor-extension values. The existing encoder
  gains 256 zero-initialized weights; old actor outputs remain unchanged for
  matching observation prefixes and neural state. Measured graph edges/routing
  and actor architecture otherwise stay unchanged. These are not vision inputs.
- Start fresh critic and Adam states for the changed physics, reward and input
  dimensions. This is continuation of learned actor weights, not scratch learning.
- 64 hover worlds, 16 native MuJoCo CPU physics threads; RTX 4090 neural training.
  Full MaleCNS graph inference/backprop still limits batching; no Warp claim.
- Ten-minute requested training budget, measured including rollout collection,
  optimization and episode bookkeeping. Setup, evaluation and rendering separate.
- PPO: horizon 512 (1.024 s/world), recurrent sequence 128 (256 ms), activation
  recomputation, two epochs, actor learning rate 3e-6, critic 1e-4, four critic-only
  warmup rollouts. Independent critic epochs survive actor KL stop. Gamma .999,
  GAE lambda .995, target KL .03, entropy zero, initial/minimum action noise .003.
  Seed 120101. All 78 actuator outputs execute from the actor; no masking/helper.
- Reward: physical hover costs with 1 mm altitude/horizontal-position scales,
  low velocity and tilt costs, small measured joint-effort cost. Rates integrate
  at 2 ms; a failure incurs one -1 then resets. No wing stillness, phase target,
  action imitation or reference force in the reward.
- Begin at the approved airborne cold-wing state. Widen only after 64 complete
  episodes of at least five seconds, no failures and mean reward rate >1.5.
  Each promotion adds .1; half the worlds remain nominal. At full widening,
  other worlds have initial altitude +/-0.2 mm and per-axis velocity +/-1 cm/s.

## Evaluation and video

Capture the migrated actor before training, then the trained actor, on the same
predeclared ten-second starts: nominal; 0.2 mm lower with -1 cm/s vertical speed;
0.2 mm lateral offset with +0.5 cm/s horizontal speed. No resets or action noise.
Run the accepted PID as a fourth independent world in the same compiled model.
PID scratch state is a read-only mirror of that world for computing wing actions.

The requested video shows PID and PPO side by side at 1x with fixed cameras,
identical nominal starts and the same live error scales. Keep startup and failures.
Report complete-window errors, failure times and warnings. The ambitious target
remains the accepted reference gate: after one second, height band <0.2 mm,
peak position error <0.5 mm, vertical-speed RMS <15 mm/s; upright >.99 and no
forbidden support. Survival alone is not success. No takeoff/landing claim.

Decide any continuation from the measured pilot. Do not silently change reward,
optimizer settings, plant or world count inside a continuation. Save every result.

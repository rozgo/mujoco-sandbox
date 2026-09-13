# Fixed PPO flight experiment — September 13, 2026

User approved a coherent 5–10 minute PPO phase after discussion of the failed
motor iterations. Work started at 2026-09-13 17:55:29 UTC. Choose **600 seconds**
of training, including critic warmup, rollout collection and optimization.
Do not change the recipe during the run or promote a selected successful start.

## Invariants and starting point

- Parent: `position_sustain_retention_01.pt`, SHA-256
  `80175c618d2e214700fa16b39d4ffe27442d1fb04daae044d4939d9fb499bc71`.
- Same `wing_position` physical contract, 78 output channels, 397 observations,
  166,700-cell fixed measured graph and existing trainable motor architecture.
- 5,000 Hz native CPU MuJoCo/mjbatch physics, 500 Hz actions; RTX 4090 neural work.
- One actor for every world and command. Utility remains disabled/frozen.
- No teacher actions, imitation loss, reference force targets, output masks,
  externally supplied wing clock or changed mechanics. The reference is an
  independent feasibility benchmark, not a deployed controller.

## Recipe fixed before execution

32 worlds, 16 CPU physics threads: 8 stand, 8 walk, 16 hover. Commands remain
fixed within each episode in this hover-focused run. Continuous ground-command
learning is not silently substituted for the preserved parent's behavior.

- Rollout: 512 actions = 1.024 seconds/world, 16,384 transitions.
- Recurrent gradient sequence: 128 actions = 256 ms, about three reference
  wingbeats. Recompute the identical actor activations during backward passes
  to fit memory. Verify outputs and gradients against ordinary execution.
- Discount .999 (1.999 s e-fold); GAE lambda .995 (about .333 s trace e-fold).
  These are not hard horizons; the critic bootstraps future returns.
- Two critic-only warmup rollouts, included in the 600-second budget. Then two
  PPO epochs per rollout; actor Adam LR 1e-6, critic LR 3e-4, clip .2, target KL
  .03, gradient norm limit 1.0. Independent pre-tanh initial/minimum noise .003;
  entropy weight zero. Start a fresh critic and optimizer from the imitation
  parent; do not reuse the critic from a different reward definition.
- Five-second physical episodes. All hover starts target 1.8–2.2 cm, with cold
  wings. Half remain the inherited starts; half gradually acquire altitude
  offsets up to .2 cm, vertical speeds up to 5 cm/s, wing-angle offsets up to
  .08 rad and wing speeds up to 5 rad/s. Widening depends only on collected
  experience, reaching full scale after 10 simulated seconds per world.
- Ground: existing physical stand/walk rewards, unchanged. Hover: alive rate 2;
  smooth costs for commanded height (weight 2, scale .3 cm), vertical speed
  (weight 1, scale 5 cm/s), horizontal speed (weight 1, scale .5 cm/s), tilt
  (weight 1), angular speed (weight .1, scale 1 rad/s), and existing forbidden
  support cost 2. Smooth cost is `sqrt(1+x*x)-1`, retaining differences outside
  the old exponential reward bands. Rates multiply the actual .002 s interval.
- One -1 terminal penalty, then physical/neural episode reset, for altitude
  below .5 cm or upright below .5. Timeouts bootstrap their terminal state.
- Seed 99503. No reference-driven curriculum selection or runtime assistance.

This changes timing, task allocation and hover reward together. It tests a
coherent PPO recipe, not a causal ablation attributing improvement to one knob.

## Evaluation and decision

1. Exact existing five-second, three-command review: seed 97013, neural activity
   capture, unchanged physical/accuracy/support/posture gates. Show every case
   in a 1x damped-camera video, including failures; inspect, decode and open it.
2. Three predetermined additional starts, seeds 98103/98113/98123, deterministic
   actor, ten seconds each, all commands. Run parent and candidate on matched
   starts. No teacher, critic, action noise or live reset in evaluation.
3. Report survival and first failure for every attempted case. Measure height
   span, vertical-speed RMS, altitude/root error and ground tracking. Never
   report a short pre-fall error window as full-duration success.

A useful continuation requires retained ground stability/posture and better
hover survival across starts, together with lower nominal bobbing. Aim for
all three extra hover starts surviving ten seconds, settled height span below
5 mm and settled vertical-speed RMS below 4 cm/s; report any failed gate.
The existing root accuracy gate is not relaxed. Retained ground does not mean
the parent's yaw and stand-to-walk limitations have been solved.

No automatic second pilot. Decide on further training from these results.
Save training/setup/evaluation/render/elapsed clocks separately, task-specific
experience, critic quality, checkpoints, physical/graph hashes and failures.

## Reproduction

```sh
uv run --project experiments/embodied_fly --locked python -m embodied_fly.ppo \
  --motor-all --hover-physical --checkpoint-activations --preset wing_position \
  --resume assets/embodied_fly/diagnostics/position_sustain_retention_01.pt \
  --graph outputs/fly_survival/malecns \
  --output outputs/embodied_fly/position_ppo_timing_01 \
  --seconds 600 --worlds 32 --threads 16 --episode-seconds 5 \
  --horizon 512 --sequence 128 --epochs 2 --lr .000001 \
  --noise .003 --minimum-noise .003 --entropy 0 --target-kl .03 \
  --gamma .999 --gae-lambda .995 --critic-warmup-rollouts 2 --seed 99503
```

On the GPU checkout, use the existing checksummed graph cache in the adjacent
fly-survival checkout. No graph download or package upgrade is needed.

# Goal-directed velocity reward: implemented and preference-tested

This is a reward correction and audit, not a new trained policy. The user asks
for clear objectives without overfitting many extra terms. Preserve the same
plant, actor, critic architecture, target paths and prior checkpoints.

One change: the two translational velocity scores now compare measured motion
with target motion plus a bounded correction toward the target. At a hold,
target velocity is zero. Far from the target, the correction asks for up to
5 mm/s toward it; within 2.5 mm, the correction decreases with error using a
0.5-second approach time. The same 3D rule applies in all directions.

This is a desired motion used only to calculate reward. It sends no motor
commands or body forces and adds no deployed controller. The actor still has
to learn how wing actions produce that motion. Analytic target velocity is
computed at the same post-step time as position rewards and timeout bootstrap.
The live physics and measured wing/body velocities are not filtered.

All existing weights stay fixed: alive 0.5, height 2, horizontal position 1,
vertical velocity 1, horizontal velocity 1, upright 0.5, angular velocity 0.1,
actuator effort at most-0.002 per simulated second. Position scale stays 1 mm;
velocity-error scales stay 20 mm/s vertical and 5 mm/s horizontal. Normal scores
are multiplied by 2 ms. Physical failure remains-1 once, with all ordinary terms
zeroed. The legacy reward remains the default reproduction path; this variant
requires `--flight-tracking-reward --round-trip` and an explicit critic reset
when switching from an older reward. Critic design/optimizer recipe are unchanged.

## The specific incentive corrected

At the **same 8 mm horizontal position error**, compare only the sum of the two
velocity reward rates. Every other reward term is identical in this example.

| Motion | Old velocity score | New velocity score |
|---|---:|---:|
| Toward target at 5 mm/s | 1.7071 | **2.0000** |
| Stopped | **2.0000** | 1.7071 |
| Away at 5 mm/s | 1.7071 | 1.4472 |

Near a stationary target the preferred speed becomes zero, so the same rule
rewards slowing and settling. During a moving request it also includes the
current target velocity. There are no axis-specific reward rules, new waypoint
bonuses, wing-frequency targets, imitation losses or oscillator targets.

## Audit result

- All 9 combinations of 1.2 / 1.5 / 1.8 mm excursion and 0.85 / 1 / 1.15 timing pass across
  six route directions and five unsuccessful trajectory types:270 comparisons.
- Ideal tracking scores above staying at origin, drifting, oscillating, stopping
  at the first target, and falling. Minimum total margin is 2.3181 reward points.
  These are kinematic scoring fixtures, not physical simulations or learned results.
- Actual saved PID flight scores above both recorded learned checkpoints on
  all 7 cases. Mean new return is 69.9924 for PID,32.2340 for the preferred parent,
  and 31.9759 for the preceding failed PPO child. The smallest pairwise PID lead
  is 37.0435 points. Physical states and their capture hashes are preserved.
- Physical capture scoring excludes actuator effort because it was not recorded.
  Its magnitude is bounded by 0.024 over 12 s; the ranking gate uses a conservative
  0.048 margin. The last 2 ms sample is omitted because its post-step angular
  velocity is not stored. These audit returns are not claimed to be exact
  original training episode returns.
- The old reward also ranks complete successful tracking above these unsuccessful
  whole trajectories. The precise correction is the local velocity preference
  shown above. Reward mismatch was a weakness, not proof of the sole cause of
  failed PPO. Passing this audit does not establish learnability or eliminate
  every possible reward exploit.

Focused tests initially pass 14/14 in 8.29 s. They cover analytic velocities with
timing jitter, bounded/smooth correction, consistent behavior under rotations,
recovery/stop/escape ordering, unchanged non-velocity terms and physical state,
correct failure handling and legacy reward reproduction. Full-suite results are
recorded in validation.json after completion.

No actor/critic updates and no new physical trajectory collection in this audit.
The test suite does exercise physical stepping for regression checks. No new
training video is due; the existing physical reference and learned captures are
used as evidence. The next bounded PPO comparison should change only the reward
variant, keeping other experimental settings explicit and unchanged.

Reproduce from the repository root, with preserved ignored capture folders:

```sh
uv run --project experiments/embodied_fly --locked python -m embodied_fly.flight_reward_audit --output outputs/embodied_fly/flight_reward_audit_repeat
```

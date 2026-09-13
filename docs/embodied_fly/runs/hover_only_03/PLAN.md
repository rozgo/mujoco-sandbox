# Correct the early-termination incentive

Pilot 02 fails all three deterministic starts at 0.794, 0.796 and 0.804 s.
Its lower nominal position error includes time on the floor and is not improved
hover. Keep its weights and failure video; resume pilot 01, which remained
airborne in all three sampled starts. This is an explicit rollback in training
ancestry, not a second deployed policy.

The first reward introduced unbounded horizontal-position cost while retaining
a -1 failure penalty. In pilot 01, completed five-second episodes averaged
-62.548 return; failed episodes averaged -6.171 over 1.194 s. These unequal
windows do not prove causation, but the equation permits ending a bad flight
to cost less than continuing it. This incentive must be corrected before spending
more time on the same objective.

## Fixed five-minute correction pilot

- Parent: `hover_only_01/actor.pt`, SHA256
  `6d5681aa96c2b6bc154d63b21a53d0236180bb607f6b1f317b11a2543d3f21a9`.
- Keep its actor and actor Adam/exploration; reset the critic and critic Adam
  because return semantics changed. Four critic-only warmup rollouts included.
- 300 requested seconds, 64 hover worlds, 16 native physics threads, RTX 4090.
  Same 1 kHz plant, 500 Hz policy, body/actuator limits, 399 inputs, measured graph,
  512/128 rollout/sequence, two epochs, rates 3e-6 / 1e-4, exploration floor .003,
  KL .03, gamma .999, GAE .995, no entropy/imitation. Seed 120103.
- Opt in with `--bounded-hover-reward --reset-critic`; historical defaults and
  saved old rewards remain unchanged. Each tracking score is
  `1 / sqrt(1 + normalized_error**2)`: height weight 2, horizontal position 1,
  vertical velocity 1, horizontal velocity 1, upright .5 and angular speed .1.
  Add .5 alive rate and subtract at most .002 normalized joint-effort rate.
  Scales stay 1 mm position, 5 cm/s vertical and .5 cm/s horizontal velocity.
- A valid airborne state earns a rate in [.498, 6.1], even far from target;
  good tracking earns more. Physical failure gets exactly -1, then resets.
  Forbidden support >.1 bodyweights now explicitly terminates, matching review.
  This repairs the cheap-termination incentive; it guarantees neither learning
  success nor real-world safety.
- Curriculum advancement still requires 64 successful >=5 s episodes, now with
  mean reward rate >5.5 on the new scale. No live force, PID, wing phase/frequency
  target, reference trajectory or motion override is introduced.

Evaluate the same three ten-second starts with the identical PID world and make
a third 1x comparison. Report physical errors and failures separately from new
reward values, which cannot be compared numerically to the old objective.
The second pilot's compute remains in total cost but outside this actor's ancestry.

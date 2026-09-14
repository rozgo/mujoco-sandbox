# Minimal goal-directed reward correction — declared before trajectory audit

Started 2026-09-14 02:18:17 UTC. User asks to focus on rewards that express the
objectives without overreacting or overfitting. This stage implements and audits
one reward correction. No new actor training, critic redesign or physics change.

Keep every existing reward weight, position scale, speed-error scale, failure
condition and actuator-effort term. Replace the preferred zero translational
velocity with:

`desired velocity = current target velocity + bounded((target-position)/0.5 s)`

The correction has at most 5 mm/s magnitude, using one 3D vector rule. The 0.5 s
approach time makes correction diminish near the target; its speed cap uses the
existing horizontal velocity-score scale. Neither value is tuned to the saved
policy or one axis. Analytic target velocity includes timing jitter and becomes
zero on holds. Measured body velocity remains instantaneous, unfiltered.

This desired velocity exists only in the reward calculation. It does not write
actuator commands, applied forces, body poses, actor observations or graph state.
Same actor, critic design, target schedule and plant. No teacher actions. No new
completion bonus, wing-frequency target, phase template or route-specific term:
reaching intermediate positions and returning home use the same dense position
and motion scores. Existing independent completion gates stay unchanged.

The reward remains bounded and a valid airborne transition remains positive.
Failure gets exactly -1 once and ends its episode; critic reset is required when
switching this explicit reward variant. Legacy reward remains reproducible.

Before training, audit with unchanged constants:
- Actual saved PID and learned physical trajectories from round_trip_ppo_01.
- Kinematic scoring fixtures: correct tracking, no movement, drift, oscillation,
  stopping at the first waypoint and falling. These fixtures are not simulated
  flight and cannot establish physical feasibility or learning success.
- Correct tracking must beat unsuccessful fixtures across six directions and
  declared distance/timing variations. At equal error, recovery must beat
  remaining still and moving away. Holds must prefer zero velocity.
- Keep all raw scores, margins and failures. Do not tune coefficients to make a
  chosen learned checkpoint win. Passing these checks establishes reward
  preferences only; a later bounded PPO run must demonstrate learned improvement.

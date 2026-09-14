# Coordinated hover after vertical recovery

Effort start: **2026-09-14 20:40:58 UTC**. The continuing goal authorizes further
learning. Run 13 was progress: vertical control improved, while exact physical
replay identified a reward tradeoff that worsened combined hover.

## Objective audit before training

Score the saved run 11 midpoint (run 13's parent capture), both run 12 snapshots,
and both run 13 snapshots under a coordinated inverse-quadratic velocity reward:
`3 / (1 + ||mean_velocity / 0.5 cm/s||²)`. Keep the existing causal 100 ms
window, alive/angular/upright terms and terminal penalty. This retains the
5 mm/s vertical precision; run 08's vector experiment instead used 20 mm/s.

Replay recorded actions through the original physical plant and require exact
body positions. The candidate must rank the retained run 11 flight above each
observed sideways regression, on every paired case. Synthetic zero-motion,
axis-rotated drift, recovery and early-failure outcomes must have the intended
return ordering, with both undiscounted return and the existing two-second
discount horizon. Synthetic examples test scoring, not physically demonstrated
capabilities. Do not train if the audit fails.

## Bounded training, conditional on the audit

- Parent: preserved run 13 final, SHA256
  `2d65d1ad151d055f255dec195171b71c46773b15d750fc812437635719f38962`.
  Run 11 midpoint remains preferred overall.
- Same full MaleCNS actor, body, observations, actions, instantaneous wing force
  model, 1 kHz physics and 500 Hz control. Only its existing wing readout trains.
- 32 worlds: 16 normal starts and 16 recovery starts. Collect a new bank from
  this exact parent at 2/4/6 seconds; retain full physical, neural and raw reward
  histories. Training episodes 0–7, withheld episodes 8–9. Bank scoring explicitly
  uses the new objective; scoring changes no physical or neural history.
- Restore actor/critic weights and optimizer state. One new-reward rollout fits
  the critic before actor updates. Fixed exploration 0.0015, KL limit 0.005,
  existing light imitation, 108 rollouts (approximately ten minutes). No gusts,
  teacher physical control or new assistance. Measure time rather than assume it.
- CPU MuJoCo/mjbatch with 16 physics threads; RTX 4090 for neural work.

## Evaluation and decision

Evaluate all four ten-second cold starts halfway and at the end. Evaluate six
withheld recovery histories for parent/midpoint/final. Preserve failed cases and
startup/settled windows. Record and open a PID / parent / candidate comparison.

Promote only on combined improvement: total RMS below 12.76 mm/s, horizontal RMS
below 10 mm/s, climb below 50 mm, all cold/recovery cases completing, and the new
parent's vertical/startup improvement retained. Report any tradeoffs explicitly.
If a correctly ordered objective still fails physically, diagnose phase-conditioned
control authority and neural representation before another unchanged training
block. The broader learning goal remains active.

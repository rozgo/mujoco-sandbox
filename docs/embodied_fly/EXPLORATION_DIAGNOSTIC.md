# Frozen exploration diagnostic and conditional PPO continuation

Approved September 14, 2026. Work started 18:33:16 UTC. Preserve run 05 as the
best checkpoint. First isolate action sampling from learning: same frozen
weights and physical contract, zero/half/current exploration, 32 worlds per
condition. Eight starts (0, 1, 2, 3, 6, 7, 8, 9), four independent noise streams
per start, common Gaussian draws across the nonzero conditions. The repeats
are noise replicates, not separately trained policies. No optimizer or teacher
acts during this comparison.

Use the exact PPO observation timing (physics-step return without an extra
forward pass), 500 Hz control, 1 kHz physics, ten-second episodes and the
unchanged hover failure criteria. Measure survival first; velocity RMS and
climb on truncated failures must not masquerade as better complete flight.
Record all worlds and the first replicate of the four standard video starts.

If current noise materially disrupts flight and half noise reduces that damage,
continue run 05 with only the fixed exploration amplitude halved, original
rewards/imitation/optimizer state and 32 worlds. Match the previous 108-rollout
experience budget (approximately ten minutes, actual duration measured), with
midpoint and final deterministic evaluation. Promote only a combined physical
improvement. If noise is not responsible, inspect accepted PPO update effects
before choosing the next training change. Correlated noise is a possible later
experiment, not part of the initial test or a new force filter.

Run provenance, measured results and the subsequent decision will be appended.

## Frozen result and training decision

Source `168d5f7`, checkpoint run 05. No noise: **32/32** survive ten seconds;
half noise: **30/32**; original noise: **18/32**. Mean airborne durations are
10.000, 9.421 and 6.630 seconds respectively. This isolates a material sampling
effect with fixed weights. The lower average climb among original-noise
survivors is not a fair improvement claim: 14 failed flights are excluded.
Collection took 50.349, 50.883 and 50.780 seconds separately from setup/IO.

Proceed with run 11 from run 05, fixed tanh-latent standard deviation **.0015**
instead of **.003** on all 78 actions. Preserve imitation weight 1, original
separate-axis reward, Adam/critic state, recurrent core, physical model and
sampling RNG. The altered distribution is used consistently in collection,
log probabilities and analytic KL. This also changes the absolute action
change allowed by the same KL bound; it is a distribution change, not an
extra filter on controls or physics. Match 108 rollouts / 1,769,472 transitions,
with midpoint after 54. Measure actual wall time, keep all outcomes.

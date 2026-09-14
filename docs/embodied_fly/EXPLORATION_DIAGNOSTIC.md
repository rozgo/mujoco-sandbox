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

# Smaller-update refinement from the retained hover midpoint

User-approved effort started **2026-09-14 19:28:23 UTC**. Resume run11's selected
midpoint (`69061012aa237832fcbd42be2b39bc9c2e6530932865b82d4df5190af9f56077`).
Change the PPO policy-change cap **.02 -> .005**. The analytic pre-update stop
remains75% of the cap, now .00375. Apply the same cap to the sampled early stop
and post-step backtracking. Keep actor LR3e-6 and PPO clipping .2.

Retain the current .0015 exploration standard deviation, 32 worlds,16 CPU
physics threads, RTX4090 neural work, fixed MaleCNS graph, wing-readout training
scope, Adam/critic state, original reward and light imitation. Same 1kHz physics,
500Hz control and reviewed FlyBody/wing_motion_agile_v5 contract. No new reward,
physical assistance or task curriculum in this first refinement.

Use108 rollouts /1,769,472 transitions, approximately ten minutes with actual
wall time recorded. Capture checkpoints after27/54/108 rollouts, approximately
2.5/5/10 minutes. All evaluation time is excluded from training time. Reuse the
parent capture only after matching its exact checkpoint SHA, including midpoint
snapshots; never substitute run11's later final capture.

Assess all four ten-second starts. The immediate target is total RMS below
12.76mm/s, horizontal RMS below10mm/s and climb below50mm. Measure the first
two seconds, minimum startup altitude and subsequent2–10s motion separately,
so a larger initial dip cannot silently qualify as improved hover. Preserve
every snapshot and choose by physical performance, including inconvenient
startup outcomes. These are development starts and one training seed.

Decision after this bounded run: if it improves regulated hover enough, propose
mixed gentle velocity changes/braking as the next stage of the same policy.
If it plateaus, the next bounded experiment is recovery from gentle climbing
and drifting states with matching physical/neural histories. If both approaches
fail, pause further training to examine information/control through the learned
interfaces. End the report with the recommended next action and its evidence.

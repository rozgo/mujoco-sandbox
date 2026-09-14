# Improve overall flight: one vector-velocity objective

The previous goal was incorrectly marked complete after the trial work ended.
The user corrected this: success requires improved physical flight, not completed
training blocks. The restored goal began September 14 at **15:50:06 UTC** and
remains active until the physical requirements are demonstrated.

Run 05 remains the best overall parent, SHA256
`3106cb363aa75dab78322181ad9f46c229a8a29ca8f1f0599bc18e8d59763a32`.
Run 06's unchanged continuation regressed. Run 07 reduced climb but worsened
sideways and total velocity error. Offline scoring of the captured states shows
the separate-axis reward actually prefers run 07; its objective does not rank
these flights according to the intended total-velocity criterion.

Replace the two separate velocity rewards with one isotropic term:

`velocity rate = 3 / (1 + (vx² + vy² + vz²) / (2 cm/s)²)`

Use the same causal 100 ms average. The scale is the retained horizontal reward
scale, now applied equally to all three components; the maximum velocity rate
stays 3 and the maximum total live rate stays 5. Same alive, upright and angular
rates, failure penalty and failure thresholds. No action penalty or height target.
This removes the separate credits for controlling individual axes. It does not
guarantee optimization success; evaluate physical flight and retain failures.
Rotation-invariance, reward-ordering, recovery-versus-failure and causal-reset
tests cover the change. Saved-trace scoring is recorded separately from exact
on-policy returns; it uses pre-step kinematics and recorded upright values.

Resume run 05's actor Adam, critic/Adam, input calibration and sampling RNG.
One critic-only rollout adapts to the revised reward; no input recalibration.
Same 32 worlds, 16 CPU MuJoCo/mjbatch physics threads, RTX 4090 neural computation,
1,000 Hz physics, 500 Hz actor, four internal neural updates. Same FlyBody,
wing_motion_agile_v5 contract, fixed MaleCNS graph, 391 observations, 78 actions,
four zero velocity commands and 105,222 trainable existing wing-readout parameters.
Upstream weights remain frozen; full graph runs in every live actor step.
Same horizon 512, sequence 64, two actor epochs, four critic epochs, proposed
LR 3e-6, fixed .003 exploration, .02 analytic KL ceiling and light PID-label anchor.

First run a short discarded smoke for the explicit objective transition. Then
start again from run 05 for 600 measured training seconds, finishing the current
rollout. Setup, audit, evaluation, checkpoint IO and rendering are separate.
Evaluate the same four starts at halfway and completion. Preserve both snapshots.
The goal requires four ten-second flights, lower total velocity RMS than run 05,
climb at most 50 mm in every case, and horizontal RMS at most 10 mm/s in every
case. A single-axis gain does not pass. These repeated starts are development
cases, not a broad robustness benchmark. Keep the best jointly controlled
checkpoint and continue correction if neither snapshot meets the goal.

Render, fully decode, inspect and open a matched 1x PID / parent / candidate
comparison; archive all measured changes, times, parameter hashes and outcomes.

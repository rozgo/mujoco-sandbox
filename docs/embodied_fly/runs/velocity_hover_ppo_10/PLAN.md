# Does the PID imitation anchor restrict useful PPO corrections?

User approved this controlled ablation on 2026-09-14; work started at
17:54:14 UTC. Start from retained run 05, SHA256
`3106cb363aa75dab78322181ad9f46c229a8a29ca8f1f0599bc18e8d59763a32`.
Use completed run 06 as the matched imitation-enabled continuation. Both start
from the same actor, optimizer, critic and sampling RNG. Its 108 rollouts are
the fixed experience budget for this comparison: 1,769,472 transitions, normally
about ten minutes. Measure actual wall time rather than infer it from the budget.

The sole learning-objective change is **imitation weight 1 -> 0**. Still sample
and score the original teacher replay batch, preserving RNG draws, but do not
backpropagate its loss. Log those presentations as diagnostic, not supervision.
Retain parent Adam moments deliberately, as in run 06; do not silently reset
optimizer history. Keep the original separate-axis reward from run 05/06,
including horizontal scale 2 cm/s and vertical weight 2. No gusts, new reward,
phase alignment, PID feedback penalty or supplied neural clock in this trial.

Unchanged: 32 worlds, 16 CPU MuJoCo/mjbatch threads, RTX 4090 neural work,
1 kHz physics, 500 Hz actor, four neural substeps, same FlyBody/v5 physical
contract, fixed measured MaleCNS graph, 391 observations, 78 controls, existing
105,222-parameter wing readout, frozen upstream actor. Horizon 512, sequence 64,
two actor epochs, four critic epochs, actor LR 3e-6, critic LR 3e-4, KL .02,
fixed .003 exploration and the same gamma/lambda. Seed 151101 as before.

The explicit rollout stop changes budget accounting only. Midpoint evaluation
occurs after rollout 54, as in run 06; final after 108. Validate first-rollout
rewards and recurrent replay against run 06 to confirm initial collection is
matched before the objective change can affect actions. Check upstream weights,
actual zero supervised presentations and accepted KL. A discarded three-rollout
smoke verifies execution; its weights never initialize the full trial.

Primary comparison: total/vertical/horizontal velocity RMS, climb, survival,
body bobbing and wing motion on the four existing ten-second starts. Use the
same measurement windows and retain midpoint/final snapshots. Compare against
both run 05 and the imitation-enabled run 06. A single paired training seed
does not establish a general phase-mismatch cause. Preserve the best checkpoint
and do not promote survival or a single-axis improvement as solved hover.

Render, fully decode, inspect and open the comparison. Record failed outcomes
and measured costs. The broader flight-improvement criteria remain unchanged:
four ten-second flights, lower total RMS than run 05, climb at most 50 mm and
horizontal RMS at most 10 mm/s in each case.

The initial strict bitwise check did not pass: first-rollout vertical reward
differs from run 06 by 0.000138 with imitation off and 0.000108 in a separate
imitation-on smoke. The teacher diagnostic loss differs by about 3e-11.
Repeating the original setting also varies, so this is a matched-settings
comparison, not a bitwise replay. Preserve the mismatch in verification.json;
functional replay, exact unchanged upstream weights and reward checks pass.
Do not treat a small single-seed result difference as proof of phase causality.
Both three-rollout smokes are discarded before full training.

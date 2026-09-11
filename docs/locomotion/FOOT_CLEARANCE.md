# Clear the healthy foot after rear-leg loss

Started 2026-09-11 03:40:11 UTC. The user likes the temporal-reference gait but
sees the surviving rear foot drag when the other rear leg is damaged. Improve
that foot's lift and replacement while retaining the rest of the approved motion.

Use `limb_healthy_sequence_90s_seed2.pt` as the parent, preserving its video and
the earlier globally selected ground-support policy. The existing recordings
confirm only 3.6–5.9 mm speed-weighted clearance for the surviving rear foot in
the four rear-removal cases. Healthy rear feet show 14–29 mm in that recording.

## Small training change

Add a weight-1 cost for moving an intact foot with less than 3 cm ground
clearance. The squared normalized height deficit is multiplied by `tanh(2*speed)`
using measured world-frame horizontal foot speed, then averaged over intact
legs. A planted, stationary foot has zero cost. Contact is deliberately not a
gate: a sliding foot must not be exempt. Apply the same rule to every intact
foot on damaged bodies, excluding stumps and missing legs; healthy gait keeps its
existing objective. This version is explicitly flat-ground only.

This adapts the speed-weighted clearance idea in
[Isaac Lab's Spot rewards](https://github.com/isaac-sim/IsaacLab/blob/b0542fe2d45bf91c4e1d9ef6952b9c709c80b4e8/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/spot/mdp/rewards.py),
with local one-sided normalization and damage masks. No physical or runtime
controller changes, foot trajectory, or externally applied lift.

First trial: 90 seconds, seed 2, 512 environments, the same nine-body mixture,
CPU MuJoCo/mjbatch and MPS learning. Keep temporal healthy-reference reward/loss
2/0.25 and no-support weight 0.5. Add a soft 0.5 reference loss to the approved
parent on damaged states and lower the learning rate to 0.0001 to limit drift.
Healthy teacher/retention remain unchanged. No training extension is assumed.

## Evaluation declared before training

Development seed 9153, eight trials per body; final seed 20260921, 32 per body;
video seed 9143. Inspect iterations 50, 100 and final when available. Evaluate
all nine equally, with allowed ground support checked every 2 ms physics step.

Require all 72 development tasks and existing healthy gait gates. Across the
four rear-removal cases, require the intact rear foot's speed-weighted clearance
to improve by at least 5 mm on average and 3 mm in each case. Ground-drag travel
(foot bottom below 5 mm while horizontal speed exceeds 0.2 m/s) must decrease
25% on average, with no individual target case worse. These metrics cover 1–12 s
at 20 ms, distinct from the reward's midpoint height/tanh weighting.

Retain each body's mean intact-leg stride and stance within 10% of its parent,
speed at least 90% of min(0.55 m/s, parent), mean planted slip within 15%, and
each body's vertical-velocity RMS within 15% plus 0.01 m/s. Average removal
airborne fraction may rise no more than one percentage point, allowing a small
spring in the step. Require completed cycles on every intact leg. Among eligible
candidates choose the largest mean rear-foot clearance gain. These gates address
the user's new foot-drag request; they do not retroactively claim the earlier
global healthy-style targets were achieved.

## First result and bounded second trial

The first run used 89.374 seconds. Iteration 100 passes all task, healthy gait,
stride/stance, speed and body-motion retention gates, but its mean rear-foot
clearance gain is below 1 mm. Mean per-case dragging decreases about 23%.
The final policy cuts dragging more, but also loses retention gates. None meets
the declared clearance criteria; all three inspected candidates are preserved.

Before training again, declare one 120-second trial from the first run's
iteration 100. Increase clearance weight 1 → 3 and the reference loss to the
user-approved temporal parent 0.5 → 2; retain learning rate 0.0001 and all other
settings. This tests a stronger lift incentive while resisting unrelated gait
changes. Selection still compares against the original user-approved temporal
parent, with identical seeds and gates; inspect iterations 50, 100 and final.

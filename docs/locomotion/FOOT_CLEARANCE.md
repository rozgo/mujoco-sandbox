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

## Scope correction after the second trial

The stronger final policy passes 72/72 tasks and the clearance/drag gates:
rear clearance gains 7.3–9.7 mm, with dragging reduced 54–78%. Healthy gait,
stride/stance and vertical-motion gates pass, but some speeds are too low and
average airborne time increases, particularly in front-damage cases.

Declare one final 120-second continuation from this stronger final checkpoint.
Apply clearance only to the intact rear foot when another rear leg has missing
joints (`--clearance-scope surviving_rear`). The rule covers both sides and both
removal types. Front-damage cases and front feet retain their existing task and
reference objectives. Weight 1 on this one selected foot equals its per-foot
coefficient in the previous weight-3 average over three intact feet. Reference
weight stays 2 and learning rate 0.0001; all other parameters and evaluation
gates remain fixed. This is a scope correction for the user's specific request,
not a relaxed acceptance threshold. Stop training after this continuation.

## Final numerical result and visual rejection

The rear-scoped iteration 50 passed all predeclared development and fresh final
gates: 288/288 tasks, 72/72 half-timestep checks, and CPU/MPS action difference
8.35e-7. Mean surviving-rear speed-weighted clearance increased 4.65 → 12.64 mm,
and mean per-case near-ground travel fell 67.1%. Healthy gait was retained.
The three new attempts used 328.148 seconds and 6,291,456 transitions. Selected
ancestry was 1412.865 seconds; later unused updates remain counted in total cost.

Preserved videos: [nine cases](../../previews/locomotion/limb_foot_clearance.mp4)
and [four rear cases before/after](../../previews/locomotion/limb_foot_clearance_comparison.mp4).
Both contain 300 decoded frames at 25 fps, twelve seconds, 1×. Native Mac
viewing, 78 tests and Ruff passed. Raw reports and hashes accompany the media.

**User rejected the visual result at 2026-09-11 04:06:15 UTC.** The feet still
look like they drag, and the request now explicitly covers ALL intact feet.
The archived `selected` filename reflects numerical selection only, not visual
approval. The low heights (rear p95 only 16–26 mm) explain why passing those
gates did not satisfy the intended visible gait. Preserve this attempt and use
completed swing height/duration/landing measures for the next experiment.

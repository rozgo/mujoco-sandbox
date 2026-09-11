# Visible steps for every intact foot

User follow-up started 2026-09-11 04:06:15 UTC: all intact feet must visibly lift,
swing and land, instead of dragging. The previous rear-clearance result passed
its numerical gates but was rejected visually. Preserve it and the earlier
user-liked temporal-reference video.

## Change and bounded experiment

New optional flat-ground `visible_step_weight` reward, including healthy bodies.
Per-foot contact hysteresis detects real liftoff/landing. Landing quality is the
product of clipped peak height / 6 cm, air time / 0.12 s, and forward replacement
/ 18 cm. Each landing contributes 8 * (quality - 0.5). Dense low-moving-foot cost
is 2 * clipped height deficit squared * tanh(2 * horizontal world speed). Average
over intact feet only. No commanded trajectories, actor clocks, extra forces,
geometry changes or rewards for hovering. Stumps remain valid support surfaces
but are excluded from intact-foot targets. Existing no-support cost stays on.

First round: 180 s from archived rear-clearance candidate, seed 2, 512 envs,
CPU MuJoCo/mjbatch, MPS learner, learning rate 0.0002, step weight 2. Old clearance
weight zero. Remove damaged-parent action imitation and damaged healthy-action
imitation; retain temporal healthy-motion reward 2. Healthy retention reward
0.5/loss 0.5 (previous loss 2). Other settings unchanged. Inspect iterations
100, 200 and final. One further <=180 s trial is reasonable only if measured
all-foot swing quality improves without collapse. Record its change before it
runs. This is extra compute, not a claim that total training takes three minutes.

## Criteria declared before training

Development seed 9157, 8 trials per each of 9 physical bodies. Final seed 20260922,
32 per body, only after development selection. Demo seed 9143, 12 seconds, 1×.
Use actual forces and positions, after 1 s, sampled at 20 ms. Offline contact
measurement bridges one-sample dropouts; incomplete opening/end swings excluded.

For EACH of the 28 intact body/foot combinations: mean completed-swing peak at
least 4 cm; at least 80% of completed swings peak >=3 cm, clear 1 cm for >=60 ms,
last >=80 ms, and advance >=8 cm. Every trial/foot must complete >=8 such steps.
These gates prevent hiding a dragging leg behind a pooled average. Record stance,
stride, force, slip and clearance metrics for all legs, including allowed stumps.

Require all 72 full tasks with allowed support at every 2 ms substep. Retain
healthy stride 28–35 cm, forward speed within 10% of 0.6092 m/s, duty gap <=0.12
and height standard deviation <=0.011 m. Each body's mean intact stride/stance
>=90% of parent, speed >=90% of min(0.55,parent). Each body's vertical-velocity
RMS <=1.2*parent+0.01 m/s; mean damaged airborne fraction <=parent+0.015.
Select the eligible checkpoint with highest worst-foot visible-swing fraction.
No eligible checkpoint means experimental preview only, not a solved claim.
Actual video review remains necessary even when numerical gates pass.

## First result and predeclared support refinement

First round used 179.748 seconds, 3,575,808 transitions. Final checkpoint meets
all per-foot swing gates: the worst mean peak is 5.04 cm, every foot/trial
completes >=8 qualifying steps, and all 72 tasks pass. Healthy gait and damaged
stride/stance retention pass. Speed fails in whole-FR (0.483 m/s); body vertical
motion and airborne retention fail, especially front-right removals. The largest
mean swing peak reaches 13.9 cm. Earlier checkpoints and all failures are retained.

One 180-second continuation from this final checkpoint: keep visible-step weight
2; increase no-support weight 0.5 → 3; enable existing body-motion cost at weight
1 (0.8*v_z² + 0.2*(abs roll rate + abs pitch rate)); lower learning rate to
0.0001. All other settings unchanged, no new imitation loss. These costs apply
without choosing gait pairs or phases. They address excess body flight while
leaving the successful foot-swing objective intact. Same parent comparison,
seeds, gates and checkpoint inspection times. Stop training after this round.

## Development selection

Support refinement iteration 100 and final pass every development gate. Iteration
200 fails vertical-motion retention. The predeclared worst-foot score selects
the final checkpoint: all 28 intact body/foot combinations score 100% qualifying
swings, minimum per-foot mean peak 5.48 cm. All 72 tasks complete; healthy gait,
stride/stance, speed, vertical motion and airborne retention pass. Freeze this
choice before final seed 20260922. Both rounds and all inspected candidates are
archived; no further training planned.

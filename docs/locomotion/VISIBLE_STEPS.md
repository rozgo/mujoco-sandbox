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

## Video integrity caught a timestep issue

The final policy passed all final seed 20260922 gait/task gates and 72/72 1 ms
checks, but its 2 ms video failed the existing 8 mm contact-penetration limit:
9.39 mm on lower-FR's intact RR foot. The earlier eligible iteration 100 also
fails this limit (9.25 mm), so do not substitute it or increase the limit. All
actual torques remain at or below the original caps. Preserve the 2 ms video as
an archived failed integrity check, not the final delivery.

A numerical-resolution probe of the same final policy, same demo seed and same
lower-FR task gives 7.96 mm at 1 ms and 7.62 mm at 0.5 ms. Keep the original
contact solref/solimp, all limits, masses, controller frequency and weights.
Declare 0.5 ms physics for the final delivery, explicitly recorded in manifests
and viewer arguments. Old commands retain their 2 ms default and old videos are
unchanged. Recheck BOTH parent and candidate at 0.5 ms on development seed 9157;
only if the original gates pass, use NEW final seed 20260923 (32/body), with
0.25 ms half-timestep checks. No retraining or changed acceptance thresholds.

## Watching and reproducing

[Verified nine-case video](../../previews/locomotion/limb_visible_steps_verified.mp4),
3840×2160, 25 fps, twelve seconds at 1×. Every panel uses the same frozen actor.
Orange markers show removed limb locations. Cameras are observer output; lane
commands use ideal localization. All motion comes from torque-limited MuJoCo
joint dynamics. The visible-step reward and motion references are training only.

From the repository root, run the native Mac viewer:

```sh
cd experiments/adaptive_locomotion
uv tool run --from uv==0.12.12 uv run --locked adaptive-dog view \
  --checkpoint ../../assets/locomotion/checkpoints/limb_visible_steps_selected_seed2.pt \
  --case whole_rl --presentation damage --timestep 0.0005
```

Use `healthy`, `lower_fl`, `lower_fr`, `lower_rl`, `lower_rr`, `whole_fl`,
`whole_fr`, `whole_rl` or `whole_rr` for the case. The timestep argument matters:
training and older demos use 2 ms; the final video uses 0.5 ms physics with the
same 20 ms control period. No new training for this numerical-resolution change.

The actor remains one reactive 128×128 policy with joint state, IMU, previous
action, command, missing-joint mask and range observations. No phase input,
contact-force input, foot-height input, hard symmetry or hand-drawn foot paths.
This is a flat-ground, fixed-morphology experiment, not arbitrary damage recovery
or parkour. Brief flight phases still occur in some front-removal gaits; the
claim is visible foot swings with measured gait retention, not perfect dog motion.

## Final verified result

Fresh seed **20260923**, matched 0.5 ms physics for parent and candidate:
**288/288** support-valid completions; every one of the 28 intact body/foot
combinations passes every lift gate. Each individual trial/foot completes at
least **19** qualifying steps. All measured completed swings meet the visible
swing definition; per-foot mean peaks range **5.54–10.84 cm**. The equally
weighted mean swing peak rises **2.41 → 7.26 cm**. This is swing peak, not
speed-weighted clearance or a prescribed foot path.

Average damaged-body airborne fraction falls **4.38% → 2.57%**. All stride, stance,
speed and vertical-motion retention gates pass. Healthy stride **32.27 → 32.21 cm**,
speed **0.612 → 0.589 m/s**, duty gap **4.23 → 1.45 percentage points**, body-height
SD **6.38 → 6.71 mm**. Some front-removal flight remains; individual gaits are
still asymmetric compensations. **72/72** additional checks pass at 0.25 ms.

**81 tests**, Ruff, native Mac viewing at 0.5 ms and CPU/MPS action agreement
(max absolute difference **9.54e-7**) pass. All 300 video frames decode; opening,
middle and end were inspected. Nine recorded tasks pass. Maximum sampled
penetration **7.62 mm < 8 mm**, measured from recorded 20 ms states; actual
recorded torque never exceeds the original caps (some reach saturation). Allowed
support is checked at every physics substep. Capture **9.985 s**, render/export
**45.549 s**, excluding context setup. Old 2 ms media and failed integrity checks
remain archived; use the `verified` video linked above.

New training: **179.748 + 179.395 = 359.143 s (5 min 59 s)** and **7,225,344
transitions**, CPU MuJoCo/mjbatch with MPS learning. Full selected ancestry is
**1772.007 s (29 min 32 s)** and **35,340,288 transitions**, including earlier
healthy/damage/clearance learning. No new NVIDIA training and no retraining for
the smaller runtime timestep. All six inspected new checkpoints are retained.

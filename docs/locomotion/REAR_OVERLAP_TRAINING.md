# Minimal rear swing overlap refinement

Started **2026-09-11 04:39:57 UTC** after explicit approval for the proposed
single-change fine-tune. Preserve the user-liked `limb_visible_steps_selected_seed2.pt`
and its verified video. That policy already lifts every intact foot, but both
front-right-removal cases swing the rear feet almost together.

## One change, declared before training

Add a weight-1 cost when neither intact rear foot carries >1 N of contact force,
while a front limb has missing joints and commanded horizontal speed >0.15 m/s.
The missing-joint mask handles both front sides and both removal types equally.
A missing rear foot disables the pair cost; double rear stance remains free.
No prescribed phase, hard symmetry, actor input or physical change. This extends
the support objective to the rear pair while retaining all visible-step rewards.

One **90-second** round from the selected visible-step policy, training seed 2,
512 environments, the same nine-body mixture and all other hyperparameters from
`limb_visible_steps_support_180s_seed2`. Learning rate 0.0001, visible-step weight
2, no-support weight 3, body-motion weight 1, temporal healthy-motion reward 2,
healthy retention reward/loss 0.5/0.5. CPU MuJoCo/mjbatch at 2 ms with MPS learning;
evaluate and record at the already validated 0.5 ms timestep, 20 ms control.

## Frozen selection and final checks

Development seed **9159**, 8 trials/body. Inspect iterations 50, 100 and final.
Compare every candidate to the current selected policy at the same 0.5 ms
physics. Preserve the existing all-foot visible swing, healthy gait, support,
stride/stance, speed and body-motion gates (`evaluate_visible_steps.compare`).

Additionally, BOTH lower-FR and whole-FR must halve the fraction of time both
rear feet exceed 1 cm clearance and achieve mean phase separation >=0.25 cycle
(where separation is min(phase,1-phase)). BOTH FL cases must retain separation
>=0.30 and overlap within +1 percentage point. Every front-damage trial must
provide >=8 complete rear phase samples. This checks timing rather than equal
average strides. Phase is measured from physical upward 1 cm crossings; force
is only the training signal. Select the eligible checkpoint with greatest mean
FR phase separation. No eligible checkpoint means no promotion.

Final seed **20260924**, 32/body, only after selection. Additional 0.25 ms
half-timestep support checks, CPU/MPS agreement, native Mac viewer, and a new
1× video using fixed demo seed **9143**. All nine cases and old/new versions
remain available. Require the existing 8 mm sampled-penetration and original
torque-cap checks for media. Record all attempts and elapsed/training/render time.

## First result and bounded weight refinement

The first round used **89.546 seconds**. Iteration 100 and final retain every
preexisting task/lift/gait gate. Final rear overlap falls from 7.0% / 7.8% to
3.6% / 3.6%, but phase separation remains only 11.7% / 7.2% of a cycle. This is
less simultaneous swing without the requested clear alternation. No promotion.
Iteration 50 also has a task failure; all three candidates are retained.

Before a second run, declare one **90-second** continuation with only the same
cost's weight increased **1 → 3**. Resume iteration 100, which retains a 5.0 cm
minimum mean swing peak and slightly better phase separation than final, giving
more lift margin. Keep all other settings, seeds and gates unchanged. This uses
the user's standing allowance to extend short RL experiments when the measured
trend supports it. Total new training remains about three minutes. Stop after
this bounded continuation if it still cannot produce alternating rear steps.

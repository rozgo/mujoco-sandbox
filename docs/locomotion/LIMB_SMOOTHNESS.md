# Smoothing the missing-limb gaits

Follow-up began 2026-09-10 23:37:39 UTC. The user observed excessive
oscillation in the front-right lower-leg and whole-leg removal panels.
Preserve the previous policy and video, and keep one deployed policy.

Initial diagnostic, original demonstration seed 9143, seconds 1–12:
front-right active-action step RMS was 0.648 (lower) and 0.636 (whole),
versus 0.300 and 0.362 in the matching left cases. Dominant joint motion
was near 8 Hz in both right cases. This is rapid joint correction, even
though roll/pitch angle amplitudes are smaller than in the left cases.

Refinement adds soft costs for active-joint action changes and roll/pitch
angular velocity on missing-joint bodies. It changes neither servo physics
nor runtime action filtering, and imposes no mirrored gait or phase pattern.
Training keeps the healthy reference and a frozen reference on the other
removal cases, with a weaker reference on the two front-right cases so the
learner can change their oscillatory behavior.

Predeclared selection: development seed 9145, eight initial conditions per
body. Evaluate all nine bodies and preserve the existing healthy gait gates.
Prefer reduced active-action step RMS and joint oscillation above 6 Hz in
both front-right cases, without losing valid completion or more than 10%
forward speed. Select checkpoints using development results only; then
compare on fresh seed 20260918, 32 initial conditions per body. Record seed
9143 again for the matched video. Report failures and any unmet targets.
Start with 150 seconds of new training; extend only if the measured trend
justifies it. Record all attempted training separately from selected ancestry.

## User steering: general behavior, not side-specific tuning

The user requested general symmetry-encouraging rewards during the first
150-second diagnostic run. Preserve that run and count its cost, but do not
select it. Restart from the original selected policy, remove side-specific
reference weighting, and add soft bilateral policy consistency for all bodies.
The reflected observation includes joint validity, so a missing left limb maps
to a missing right limb; this does not constrain two different limbs within
one damaged body to move alike. Inference remains one unmodified MLP call.

The mirror-loss concept follows [RSL-RL's symmetry extension](https://github.com/leggedrobotics/rsl_rl/blob/main/rsl_rl/extensions/symmetry.py),
which references Mittal et al., ICRA 2024. This repository implements its own
observation mapping and active-joint MSE; it does not install another trainer.
The reflection is a soft inductive bias: vendor visual meshes and range hits
are approximately bilateral, not a proof of exact physical equivalence.

Use the same smoothness weights, reference strength, and mirror-loss weight
for all missing-limb bodies. Compare all four left/right pairs, healthy gait
retention, completion, speed and motion quality. Do not optimize a hand-picked
side. The previously declared final seed and demonstration seed remain fixed.

General selection gate, fixed before inspecting the bilateral run: all nine
bodies must complete all eight development trials with valid support, and all
healthy gait gates must pass. For each removal, require at least 90% of the
smaller of the parent's speed and the 0.55 m/s command; reducing excessive
speed toward the command is acceptable. Rank eligible candidates by the mean
across all eight removals of their relative active-command step RMS and their
absolute joint-motion RMS above 6 Hz. The two components have equal weight.
All cases contribute equally; there is no front-right selection bonus.

Motion metrics use seconds 1–12 and every trial, including failed trials.
Joint spectra use a demeaned Hann window with Parseval normalization.
Report absolute high-frequency RMS, not just its fraction of total motion.
Angular-rate RMS here is sqrt(mean(wx² + wy²)); earlier stride reports average
across axes and therefore use a different normalization.

## General refinement extension

The first bilateral run's iteration 200 passed all 72 development trials and
healthy gait gates, with lower mean command jitter. Its high-frequency joint
motion did not improve consistently. Extend by 150 seconds from that candidate,
with the same balanced curriculum and mirror weight. Increase the missing-body
action-change weight from 0.05 to 0.10 and angular-rate weight from 0.15 to 0.30;
add 2.5e-6 times the summed squared finite-difference joint acceleration, measured
at the 50 Hz controller. Only existing joints contribute. The frozen damage
reference is this passing candidate, weighted 0.5 identically on all removals.
Selection gates and seeds are unchanged. This is a soft learned-motion cost,
not a physical damping change or runtime filter.

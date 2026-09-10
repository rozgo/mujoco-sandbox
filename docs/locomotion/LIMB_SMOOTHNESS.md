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

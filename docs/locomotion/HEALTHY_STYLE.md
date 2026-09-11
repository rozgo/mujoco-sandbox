# Compensate while retaining the healthy walk

Started 2026-09-11 01:17:07 UTC. The user found the damaged dog's steps too short
and jumpy, then clarified: keep this simple, rewarding compensation while looking
as much as possible like the healthy walk. The proposed stance timer was removed
before training; no training time was spent on it.

## Objective

Keep the locomotion task and valid support, and use the trained healthy walker
as the principal style reference on surviving joints. The frozen healthy policy
suggests an action from the current feedback. Missing joint channels are encoded
as nominal pose, zero velocity/previous action, and intact validity bits **for
the teacher only**, avoiding the healthy network's untrained validity channels.
The learner receives the real observation and missing-joint mask.

Add a bounded similarity reward (weight 1) and a soft actor imitation loss
(weight 2), computed only over surviving joints on removed-limb bodies. Absent
joints never contribute. The teacher is training-only: one reactive actor still
controls all nine bodies at deployment. This does not copy a clock, impose a
footfall order or force the damaged robot to match an impossible four-leg gait.
It is a state-dependent action prior, not a recorded-motion discriminator or a
guarantee of healthy-looking motion. Actual stride, stance and body motion must
improve in evaluation.

Remove the damaged-policy teacher and the extra damaged action-rate,
angular-rate, joint-acceleration and bilateral imitation terms. Keep the base
task's ordinary effort/smoothness/stability costs, the 0.5 no-flight cost and
strict support checks. Existing intact-only stride/balance/posture and healthy
retention remain to protect the normal dog. No new stance-duration target.

First trial: 150 seconds from `limb_ground_support_selected_seed2.pt`, seed 2,
512 environments (128 healthy, 48 for each removal), CPU MuJoCo/mjbatch with MPS
learning. Healthy teacher `healthy_balanced_405s_seed2.pt`, healthy-only bonus 0.5
and loss 2, damaged healthy-style bonus 1 and loss 2, learning rate 0.0003. All
physical parameters and actor inputs remain unchanged.

## Evaluation fixed before training

Development seed 9149, eight trials per body; final seed 20260920, 32 per body;
demonstration seed 9143. Inspect iteration 50, iteration 100 and the final policy.
All nine bodies are evaluated equally on the existing 12-second, 5 m flat-ground
mission with scripted 0.55 m/s lane commands.

Require all 72 development completions with allowed support, existing healthy
gait gates, and each damaged body's speed at least 90% of the smaller of 0.55 m/s
and its parent speed. Relative to the parent, target at least 20% less difference
from the measured healthy teacher in both mean stance and stride. Aggregate per
intact remaining leg within body, then equally across eight removals. Require
completed cycles for every intact leg; unused stumps are reported separately.
No body's stride may fall more than 10%; mean planted-foot slip may not increase
more than 15%; mean vertical velocity RMS and airborne fraction may not rise.
Also report a fixed-state-bank comparison to the healthy teacher, without using
it to replace the physical gait measurements.

Contact/stride diagnostics use completed intervals after the first second and
the existing one-sample contact debounce. Motion samples are 20 ms; prohibited
support remains checked every physics substep. Preserve failed candidates and
prior videos. If the strong prior prevents compensation, keep that result and
adjust the reference strength in a separately recorded short trial.

Results will be added after the experiment.

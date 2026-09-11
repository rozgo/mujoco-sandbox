# Healthy motion sequences after limb loss

Started 2026-09-11 01:48:49 UTC after the user reviewed the unsuccessful
snapshot-reference video and approved continuing the planned experiment.

The reference now compares each leg's actual joint state **0.12 seconds ago**
with the current state and command. This matches a short causal motion segment
instead of an isolated pose. Each leg can choose its own healthy reference phase.
The reward is a soft similarity target, not a guarantee of the desired gait.
The existing actor still receives 66 inputs, with no history or gait clock added.

One 90-second trial restarts from `limb_ground_support_selected_seed2.pt`,
using seed 2 and 512 environments, CPU MuJoCo/mjbatch physics and MPS learning.
Reference reward/loss weights remain 2 and 0.25. Healthy retention and ordinary
task/support costs remain; no new stance-duration or footfall-order reward.
The reference is four physically simulated healthy walks at 0.30/0.45/0.60/0.75
m/s, sampled after the first second. Missing joints are excluded from both
current and historical distance. Small masked matrix products calculate the
distance, checked against a brute-force implementation in both modes.

Selection and gait gates remain those declared in [HEALTHY_STYLE.md](HEALTHY_STYLE.md):
development seed 9149, eight trials per body, inspect iteration 50, iteration 100
and the final checkpoint when available. Require all 72 support-valid completions,
retained healthy gait, at least 20% smaller stance and stride differences from
the healthy teacher, and retained speed, slip and body-motion gates. The final
holdout seed 20260920 remains unused until a candidate passes selection;
the demonstration seed is 9143. Previous policies and videos are preserved.

This is training-time reference matching for a single reactive actor on nine
fixed body variants, not online retraining or arbitrary injury recovery.

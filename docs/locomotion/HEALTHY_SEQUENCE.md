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

## First result and bounded continuation

The first run used 89.685 seconds and 1,609,728 transitions. Its final checkpoint
passed all 72 task trials and retained healthy gait, but did not meet the gait
improvement gates. Average stance rose 0.194 → 0.201 s and stride 0.176 → 0.178 m.
The gaps to healthy stance and stride shrank only 9.5% and 2.0%, below the required
20%. Two rear-left removal cases lost more than 10% stride, and average airborne
fraction increased. Earlier inspected checkpoints also failed task completion.

Before further training, a single additional 90-second continuation from the
final temporal checkpoint was declared under the user's standing allowance for
short extensions. Later development checkpoints showed improving stance, stride
and task completion, while the healthy dog remained retained.
This uses the same source, rewards, seed, environment mix and learning rate,
with no new heuristic. Evaluate iteration 50, iteration 100 and final using the
same gates against the original ground-support parent. Stop this experiment
after the extension even if the gates still fail; preserve every inspected
candidate, and leave the final holdout unused unless a candidate qualifies.

## Outcome and review video

**No temporal checkpoint met all predeclared gait gates.** The selected policy
remains `limb_ground_support_selected_seed2.pt`. The first run's final checkpoint
is preserved as the most promising temporal **visual candidate**, without a
selected-policy alias. The user liked the video during this follow-up; that
visual feedback is recorded separately from the numerical acceptance result.

[Watch all nine conditions](../../previews/locomotion/limb_healthy_sequence_preview.mp4) ·
[Previous selected video](../../previews/locomotion/limb_ground_support.mp4) ·
[All inspected checkpoints, hashes and gates](HEALTHY_SEQUENCE_SELECTION.json).

| Development measure | Ground-support parent | Temporal preview | Extension final |
| --- | ---: | ---: | ---: |
| Support-valid task completions | 72/72 | 72/72 | 59/72 |
| Healthy gait gates | pass | pass | pass |
| Mean intact-leg stance, eight removals | 0.194 s | 0.201 s | 0.195 s |
| Mean intact-leg stride, eight removals | 0.176 m | 0.178 m | 0.150 m |

In the preview candidate, the differences from healthy stance and stride shrink
9.5% and 2.0%, below the required 20%. Rear-left lower/whole removals lose more
than 10% stride. Average airborne fraction rises **3.35% → 4.18%**, although
vertical-velocity RMS improves **0.263 → 0.240 m/s** and forward speed remains
**0.573 → 0.564 m/s**. Those tradeoffs prevent a claim of general gait improvement.
The extension's iteration 100 also completes all 72 tasks but fails the gait
criteria; its final checkpoint regresses to 59/72. All failed outcomes are kept.

Training cost is **89.685 + 89.218 = 178.903 seconds (2 min 59 s)** and
**3,330,048 transitions**. Including the two earlier unsuccessful style trials,
this line of experiments used **477.875 seconds (7 min 58 s)**. Preview policy
ancestry is **1197.326 seconds (19 min 57 s)** and **23,973,888 transitions**;
the unsuccessful extension is counted in experiment cost, not preview ancestry.
Both temporal runs started from clean source commits recorded in their reports.
Reference collection/setup is separate from training (about one second per run).
No new NVIDIA training was used.

A fixed bank of 1,080 parent-policy states confirms that surviving actions move
closer to the healthy reference: mean per-removal temporal-target RMSE
**0.708 → 0.605**, and direct healthy-policy RMSE **0.458 → 0.431**. This does not
establish better physical gait. The short reference window appears insufficient
to enforce a complete healthy stride; that is an interpretation of these local
results, not a general limitation of motion imitation.
[Reference-action audit](HEALTHY_SEQUENCE_REFERENCE_AUDIT.json).

No final holdout or half-timestep acceptance was run for these unselected policies.
Final seed 20260920 remains unused. This experiment stops at the declared
extension limit. A future full-stride reference would be a separate experiment.

## Verification

The new video uses the same checkpoint in all nine panels, seed 9143, real-time
playback, the existing orange damage markers and contact shadows. It is labeled
**EXPERIMENT / TEMPORAL HEALTHY REFERENCE / NOT SELECTED**. All nine captured
missions complete with allowed support checked at every 2 ms physics step.

The MP4 is 3840×2160, 25 fps, twelve seconds. All 300 frames decoded; encoded
opening, middle and ending views were inspected. Trajectory/model/checkpoint
hashes match. Maximum sampled penetration is **4.19 mm**, and peak recorded
torque reaches **82.4%** of the unchanged cap. Capture took **3.929 seconds**,
rendering/export **40.699 seconds**, excluding renderer setup.
[Video QA](HEALTHY_SEQUENCE_PREVIEW_QA.json).

**74 tests pass** (36 locomotion, 38 mjbatch), Ruff passes, and native Mac viewing
passes a five-second smoke test. CPU/MPS maximum action difference is **7.16e-7**
on the fixed state bank. [Backend audit](HEALTHY_SEQUENCE_BACKEND.json).

## Reproduce

From the repository root:

```sh
cd experiments/adaptive_locomotion
uv tool run --from uv==0.12.12 uv run --locked adaptive-dog view \
  --checkpoint ../../assets/locomotion/checkpoints/limb_healthy_sequence_90s_seed2.pt \
  --case whole_fr --presentation damage
```

Training from the package directory (use a fresh output path):

```sh
uv tool run --from uv==0.12.12 uv run --locked adaptive-dog train \
  --output ../../outputs/locomotion/my_healthy_sequence --seconds 90 \
  --seed 2 --num-envs 512 --mode blind --terrain flat --device mps \
  --threads 16 --epochs 4 --allowance 1600 \
  --resume ../../assets/locomotion/checkpoints/limb_ground_support_selected_seed2.pt \
  --extension-reason 'Reproduce the bounded temporal-reference experiment' \
  --reward-profile walk --support-weight 2 --stride-weight 1 --balance-weight 5 \
  --body-motion-weight 0 --damage-flight-weight 0.5 \
  --reference ../../assets/locomotion/checkpoints/healthy_balanced_405s_seed2.pt \
  --reference-reward-weight 0.5 --reference-loss-weight 2 \
  --damage-healthy-reward-weight 2 --damage-healthy-loss-weight 0.25 \
  --healthy-style-source motion_sequence --learning-rate 0.0003 \
  --limb-stage consolidate
uv tool run --from uv==0.12.12 uv run --locked python scripts/evaluate_healthy_style.py \
  ../../outputs/locomotion/my_healthy_sequence
```

For the continuation, change the resume checkpoint to
`../../assets/locomotion/checkpoints/limb_healthy_sequence_90s_seed2.pt` and use a
new output directory; all other training parameters are unchanged. Wall-clock
budgets can produce different update counts on another machine.

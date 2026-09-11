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

## Review outcome

**No new policy was selected.** The latest snapshot-reference policy completes
72/72 development trials, but mean intact-leg stance **0.194 → 0.178 seconds**
and stride **0.176 → 0.162 m** moved away from the healthy teacher. Healthy gait
was retained. The earlier direct-policy imitation trial also failed the task
gates. [All candidates and hashes](HEALTHY_STYLE_SELECTION.json).

[Watch the experimental nine-case preview](../../previews/locomotion/limb_healthy_style_preview.mp4) ·
[Previous selected version](../../previews/locomotion/limb_ground_support.mp4).
The new video is labeled **EXPERIMENT / HEALTHY STYLE / NOT SELECTED** and shows
the final snapshot-motion checkpoint, not a claimed improvement. It uses the same
seed 9143, cameras, 1× playback, damage markers and shadows as the previous video.
The current selected policy remains `limb_ground_support_selected_seed2.pt`.

The user requested a video before further training. A temporal-reference draft
was saved locally and removed from the delivered implementation; it was **not
trained**. Final holdout seed 20260920 remains unused. No final acceptance or
half-timestep certification of this unselected policy is claimed. All task
support checks in development and capture run at every 2 ms physics step.

Total new training: **298.972 seconds (4 min 59 s)** and **4,534,272 transitions**,
including both unsuccessful experiments. Preview policy ancestry is **1257.296
seconds (20 min 57 s)**; the first failed trial is not in its ancestry but is
included in total experiment cost. All six inspected checkpoints are retained.

From the repository root, to inspect the experimental trial live:

```sh
cd experiments/adaptive_locomotion
uv tool run --from uv==0.12.12 uv run --locked adaptive-dog view \
  --checkpoint ../../assets/locomotion/checkpoints/limb_healthy_motion_150s_seed2.pt \
  --case whole_fr --presentation damage
```

Reproduce the clearly labeled preview from that package directory:

```sh
uv tool run --from uv==0.12.12 uv run --locked adaptive-dog grid \
  --checkpoint ../../assets/locomotion/checkpoints/limb_healthy_motion_150s_seed2.pt \
  --family limb_loss --seed 9143 --presentation damage \
  --label 'EXPERIMENT / HEALTHY STYLE / NOT SELECTED' \
  --output ../../outputs/locomotion/healthy_style_preview.mp4
```


## First result and phase-independent reference refinement

The 149.317-second direct-policy trial failed the task gates. It increased
support duration by slowing down, and mean removal stride fell to 0.115 m. Healthy
walking survived, but damaged compensation was lost. All three inspected
checkpoints are retained as rejected candidates, with no final holdout tuning.

A second 150-second trial restarts from the valid ground-support parent and keeps
the same simple objective. Use actual healthy-walk samples as the reference:
four six-second healthy rollouts at 0.30/0.45/0.60/0.75 m/s, seed 2, keeping 248
frames after the first second. Require valid support at every physics step.
Each surviving leg independently matches its joint position/velocity and command
to the nearest healthy sample. Normalize by healthy feature standard deviation,
with floors 0.1 rad for position, 0.05 in the scaled-velocity observation (1 rad/s),
and 0.2 m/s for command. Missing channels are excluded from the distance.

The main style reward is `2 * mean(exp(-matched_motion_distance))` over remaining
legs, using actual physical joint state. A softer 0.25 imitation loss nudges
surviving controls toward the matched sample. Existing healthy-only retention
remains; the damaged policy does not have to copy a global four-leg phase.
Nearest-phase matching is used only to make training targets and rewards; the
saved actor remains an independent 66-input reactive network. This is a small
reference-library prior, not adversarial motion imitation or online adaptation.
The stance and stride targets remain evaluation criteria, not new reward terms.
The second run uses the same development/final/demo seeds and acceptance gates.

## Preview verification

All nine captured trials complete with allowed support. The new 3840×2160,
25 fps, twelve-second MP4 plays at 1×. All 300 frames decoded and opening, middle
and ending frames were inspected. Checkpoint/model/trajectory hashes were checked;
maximum sampled penetration was 5.16 mm and peak recorded torque 93.3% of cap.
Capture took 3.804 seconds and rendering/export 39.951 seconds. The native Mac
viewer passed a five-second smoke test; the delivered source passes 71 tests and
Ruff. [Preview QA report](HEALTHY_STYLE_PREVIEW_QA.json). These checks verify the
preview and physical task, not achievement of the requested gait appearance.

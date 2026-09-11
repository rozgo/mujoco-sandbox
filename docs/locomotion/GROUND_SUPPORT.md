# Minimum ground-support experiment

**Step 1 helped: 40.8% less sampled airborne time, 288/288 valid completions,
healthy gait retained after 119.5 seconds of additional training.** Stop here for
review; phase guidance and further reward changes have not been added.

[Watch all nine bodies](../../previews/locomotion/limb_ground_support.mp4) ·
[Previous version](../../previews/locomotion/limb_damage_visible.mp4)

## Single change and predeclared evaluation

Request: try only the first, smallest correction before adding phase guidance.
Started 2026-09-11 00:57:18 UTC.

Diagnosis: the healthy stride reward includes a no-flight cost, but complete-loss
training gates that entire reward off on damaged bodies. Add only
`damage_flight_weight=0.5`: subtract 0.5 per 20 ms control sample when a body with
missing joints is commanded to move and no allowed foot or distal stump carries
more than 1 N. Healthy rewards, actor observations, other reward weights,
teachers, joint limits and physical servos remain unchanged. This is a soft
training penalty, with no prescribed phase or foot trajectory. It is not a hard
no-flight constraint. Force sampling differs from the healthy stride tracker's
1/5 N contact hysteresis; both use a 1 N support floor.

One 120-second continuation from `limb_bilateral_selected_seed2.pt`, seed 2,
512 environments: healthy 128, each of eight complete-removal bodies 48.
Keep the selected parent's configuration, including frozen healthy teacher
`healthy_balanced_405s_seed2.pt` and damaged teacher
`limb_bilateral_iter200_seed2.pt`; reference tensors are unchanged. CPU MuJoCo
batch stepping, MPS learner. Retain checkpoints at iterations 50 and 100 plus the
last update (if present), and preserve all outcomes.

Before tuning, fix development seed 9147 (8 trials per body), final seed 20260919
(32 per body), demonstration seed 9143. Each trial is the existing 12-second,
5 m lane task at a 0.55 m/s command. Selection requires all 72 valid development
completions, existing healthy gait gates, and each damaged body's speed at least
90% of the smaller of 0.55 m/s and its parent's speed. Among eligible candidates,
prefer the lowest equally weighted airborne fraction across the eight removals.
Accept a candidate only if average airborne fraction falls by at least 25%,
average unsupported-force fraction falls, and no body's airborne fraction rises
by more than 2 percentage points. Otherwise keep the existing policy and report
the first intervention as insufficient before proceeding to another reward idea.

Metrics use the fixed 1–12 s window, sampled every 20 ms. Unsupported force means
no allowed tip exceeds 1 N; geometric flight means all allowed terminal sphere
bottoms are more than 1 mm above the flat ground. Weakly loaded or barely separated
contacts can count in the first metric but not the second. Prohibited support is
still checked every 2 ms. No inference about unsampled airborne intervals or
natural canine gait is implied.

## Result

The final checkpoint (iteration 167) was the only inspected candidate to meet the
predeclared 25% airborne reduction gate; iterations 50 and 100 reduced airborne
time by 19.9% and 23.6% and passed all 72 task trials, but were not selected.
[Selection and checkpoint hashes](GROUND_SUPPORT_SELECTION.json),
[full training configuration and progress](runs/limb_ground_support_120s_seed2.json).

Fresh seed 20260919: **288/288 valid completions for both policies**, including
32 healthy and 256 removal trials. Mean removal airborne fraction fell
**5.64% → 3.34%** (40.8% relative reduction), and the no-support-force fraction fell
**11.92% → 6.91%** (42.0%). Mean speed was **0.568 → 0.573 m/s**. Mean vertical
velocity RMS fell 9.2%, and roll/pitch rate RMS fell 14.3%. These are equally
weighted eight-body means, excluding healthy trials. [Paired summaries](GROUND_SUPPORT_SUMMARY.json),
[reference trials](GROUND_SUPPORT_REFERENCE.json), [new trials](GROUND_SUPPORT_VALIDATION.json).

| Body | Airborne before → after | No support force before → after | Height SD before → after | After speed |
| --- | ---: | ---: | ---: | ---: |
| healthy | 0.00% → 0.00% | 0.00% → 0.00% | 8.22 → 7.50 mm | 0.601 m/s |
| lower_fl | 4.27% → 3.89% | 10.19% → 7.65% | 21.71 → 20.82 mm | 0.561 m/s |
| lower_fr | 18.40% → 10.13% | 21.01% → 15.26% | 14.85 → 15.30 mm | 0.580 m/s |
| lower_rl | 0.01% → 0.00% | 9.79% → 0.03% | 9.31 → 8.06 mm | 0.589 m/s |
| lower_rr | 0.00% → 0.00% | 9.89% → 4.33% | 14.67 → 12.74 mm | 0.582 m/s |
| whole_fl | 4.23% → 3.99% | 12.19% → 12.31% | 18.13 → 19.72 mm | 0.547 m/s |
| whole_fr | 18.19% → 8.69% | 20.89% → 13.82% | 11.64 → 10.09 mm | 0.559 m/s |
| whole_rl | 0.00% → 0.00% | 3.79% → 0.00% | 11.12 → 7.46 mm | 0.593 m/s |
| whole_rr | 0.00% → 0.00% | 7.58% → 1.85% | 11.91 → 11.12 mm | 0.576 m/s |

This is a modest improvement, not a natural canine limp or a strict no-flight
gait. Front removal cases still have airborne intervals; whole-FL has slightly
higher height variation and no-support-force fraction. Lower-FR height variation
also rises slightly. The healthy policy retains its stride, speed, timing and
bounce gates. Only this reward correction was added; one shared actor still
controls all nine bodies with no phase clock, runtime teacher or action filter.
One training seed and one continuation do not isolate the penalty's causal
contribution from additional PPO updates. No terrain or arbitrary-damage
robustness claim is made.

## Verification and artifacts

**68 tests passed** (30 project, 38 mjbatch). The new regression test exercises all
nine bodies with healthy stride gating and confirms reward-only changes with
identical actions, observations and physical states. The existing loaded-stump
and loaded-knee tests still pass. All **72 half-timestep trials** passed at 1 ms.
[Physics report](GROUND_SUPPORT_PHYSICS.json). CPU/MPS maximum action difference
was **9.54e-7** on 1,080 saved reference states; the native Mac viewer passed a
five-second smoke test. [Backend report](GROUND_SUPPORT_BACKEND.json).

Recorded nine new complete trajectories at seed 9143 using the selected weights.
The **3840×2160, 25 fps, 12-second** video plays at **1×**, with orange damage
markers and shadows. All 300 encoded frames decoded, and opening/middle/ending
frames were inspected. Recorded model/checkpoint/trajectory hashes were verified.
Maximum sampled penetration was **4.06 mm**, peak recorded torque **87.3%** of its
original cap. All demonstration trials pass the every-substep support check.
[Video QA](GROUND_SUPPORT_DELIVERY.json). Contact dots display force peaks within
a 20 ms control window, whereas reported no-support fractions use the endpoint
sample; they can differ on landing/takeoff frames. Rendering applies no physics
changes.

The single new run used **119.483 seconds**, **2,052,096 transitions**, CPU
MuJoCo/mjbatch stepping with an MPS learner. Including its parent, the selected
policy has **1107.641 seconds (18 min 28 s)** and **22,364,160 transitions** of
training ancestry. No new NVIDIA training was needed. Checkpoints at iterations
50, 100 and 167 are retained with their development outcomes. Curation changes
only the parent path to be repository-relative; learned tensors are verified
unchanged. Source commit: `5ac072b01e763879fe821b1bcf5bec9c68cd6dee`, clean at
training start. Earlier policies and videos remain available.

## Run

From the repository root:

```sh
cd experiments/adaptive_locomotion
uv tool run --from uv==0.12.12 uv run --locked adaptive-dog view \
  --checkpoint ../../assets/locomotion/checkpoints/limb_ground_support_selected_seed2.pt \
  --case whole_fr --presentation damage
```

Reproduce the training continuation from the same package directory (the 24-step
rollout horizon is the train function's default):

```sh
uv tool run --from uv==0.12.12 uv run --locked adaptive-dog train \
  --output ../../outputs/locomotion/ground_support_repeat \
  --seconds 120 --seed 2 --num-envs 512 --mode blind --device mps \
  --resume ../../assets/locomotion/checkpoints/limb_bilateral_selected_seed2.pt \
  --threads 16 --epochs 4 --allowance 1200 \
  --extension-reason 'One two-minute continuation to restore ground support' \
  --reward-profile walk --support-weight 2 --stride-weight 1 --balance-weight 5 \
  --body-motion-weight 0.5 --damage-action-rate-weight 0.1 \
  --damage-angular-rate-weight 0.3 --damage-joint-accel-weight 0.0000025 \
  --damage-flight-weight 0.5 --symmetry-weight 1 --front-reference-scale 1 \
  --reference ../../assets/locomotion/checkpoints/healthy_balanced_405s_seed2.pt \
  --reference-reward-weight 0.5 --reference-loss-weight 2 \
  --single-reference ../../assets/locomotion/checkpoints/limb_bilateral_iter200_seed2.pt \
  --single-reference-weight 0.5 --learning-rate 0.0003 --limb-stage consolidate
```

Wall-clock stopping means iteration count can vary on another machine. Final
validation and video reproduction:

```sh
uv tool run --from uv==0.12.12 uv run --locked python -m adaptive_locomotion.limb_eval \
  --checkpoint ../../assets/locomotion/checkpoints/limb_ground_support_selected_seed2.pt \
  --output ../../outputs/locomotion/ground_support_final.json --trials 32 --seed 20260919
uv tool run --from uv==0.12.12 uv run --locked adaptive-dog grid \
  --checkpoint ../../assets/locomotion/checkpoints/limb_ground_support_selected_seed2.pt \
  --family limb_loss --seed 9143 --presentation damage \
  --output ../../outputs/locomotion/ground_support_repeat.mp4
```

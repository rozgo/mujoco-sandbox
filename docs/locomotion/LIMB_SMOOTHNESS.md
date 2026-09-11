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

## Selected result and honest limits

[Watch all nine bodies with the refined policy](../../previews/locomotion/limb_bilateral_smooth.mp4).
The [previous video](../../previews/locomotion/limb_loss_all_cases.mp4) and policy
remain available. Every panel runs the same reactive MLP, with no runtime
mirroring, policy selection, online training, action filter or changed physics.

The selected second-general-round iteration 100 passed all **288/288** fresh
trials at seed 20260918 and **72/72** half-timestep trials. The original policy
also completed all 288 trials at this fresh seed; this follow-up improves motion
quality, not the completion rate on this split. All tests check permitted support
every physics substep and preserve the original torque caps.

Across the eight removal conditions, the mean per-body relative reduction is
**32.2%** for active-command step RMS, **15.0%** for absolute joint motion above
6 Hz, and **53.6%** for roll/pitch angular-rate RMS. High-frequency motion is
not uniformly lower: lower-RR increases 8.8%, whole-RL increases 2.5%. This is
a general improvement with remaining imperfections, not perfectly symmetric
gaits or arbitrary-damage robustness. Whole-RL now follows the 0.55 m/s speed
command more closely instead of overshooting to approximately 0.79 m/s.

| Body | Fresh valid trials | Action-step RMS, old → new | Joint >6 Hz RMS (rad), old → new | Speed (m/s), old → new |
| --- | --- | --- | --- | --- |
| healthy | 32/32 | 0.209 → 0.195 | 0.0094 → 0.0094 | 0.635 → 0.608 |
| lower_fl | 32/32 | 0.301 → 0.241 | 0.0137 → 0.0136 | 0.540 → 0.554 |
| lower_fr | 32/32 | 0.644 → 0.275 | 0.0331 → 0.0221 | 0.605 → 0.576 |
| lower_rl | 32/32 | 0.412 → 0.282 | 0.0185 → 0.0166 | 0.571 → 0.615 |
| lower_rr | 32/32 | 0.286 → 0.238 | 0.0140 → 0.0153 | 0.619 → 0.557 |
| whole_fl | 32/32 | 0.365 → 0.264 | 0.0234 → 0.0154 | 0.605 → 0.580 |
| whole_fr | 32/32 | 0.645 → 0.300 | 0.0268 → 0.0180 | 0.480 → 0.577 |
| whole_rl | 32/32 | 0.409 → 0.262 | 0.0174 → 0.0179 | 0.793 → 0.544 |
| whole_rr | 32/32 | 0.300 → 0.256 | 0.0227 → 0.0180 | 0.618 → 0.545 |

Healthy stride is **32.30 cm** at **0.608 m/s**, with 8.23 mm height variation.
Its left/right duty-factor gap is 8.30 percentage points, versus 5.23 previously;
it remains within the predeclared 12-point limit. Healthy retention means the
limits passed, not that every individual healthy gait measure improved.

The fixed-state mirror audit compares both policies on the same 1,080
observations from original-policy rollouts. Mean per-case mirror RMSE falls
from **0.619 to 0.128 (79.3%)**. This measures the policy's response to reflected
states, including reflected absent joints; it does not claim matched footfall
phases between separate walking episodes. CPU/MPS maximum action difference
is **8.35e-7**. Native macOS viewing and **67 tests** passed.

Three new rounds cost **448.991 seconds (7 min 29 s)** and **7,876,608 control
transitions**, including the first side-focused round abandoned after user
steering and all unselected later updates. Both general rounds together cost
299.363 seconds. The selected policy adds **223.475 seconds** and **3,686,400
transitions** to the original; total ancestry is **988.158 seconds (16 min 28 s)**
and **20,312,064 transitions**. CPU MuJoCo/mjbatch physics with an MPS learner
on the Mac; no new NVIDIA training.

Reports: [selection and checkpoint hashes](LIMB_BILATERAL_SELECTION.json),
[fresh reference](LIMB_BILATERAL_REFERENCE.json),
[fresh selected](LIMB_BILATERAL_VALIDATION.json),
[half timestep](LIMB_BILATERAL_PHYSICS.json),
[fixed-state mirror audit](LIMB_BILATERAL_SYMMETRY.json),
and [media/physics checks](LIMB_BILATERAL_DELIVERY.json). All evaluated candidates
and their training logs are retained under `assets/locomotion/checkpoints` and
`docs/locomotion/runs`. Curated checkpoints preserve tensors exactly and rewrite
only parent metadata to repository-relative asset paths; the selection manifest
maps original run hashes to curated hashes. Original training configurations
retain their original output paths and reference hashes for provenance.

## Watch and reproduce

From the repository root:

```sh
cd experiments/adaptive_locomotion
uv tool run --from uv==0.12.12 uv run --locked adaptive-dog view \
  --checkpoint ../../assets/locomotion/checkpoints/limb_bilateral_selected_seed2.pt \
  --case whole_fr
uv tool run --from uv==0.12.12 uv run --locked adaptive-dog grid \
  --family limb_loss --seed 9143 \
  --checkpoint ../../assets/locomotion/checkpoints/limb_bilateral_selected_seed2.pt \
  --output ../../previews/locomotion/limb_bilateral_smooth.mp4
uv tool run --from uv==0.12.12 uv run --locked python -m adaptive_locomotion.limb_eval \
  --checkpoint ../../assets/locomotion/checkpoints/limb_bilateral_selected_seed2.pt \
  --seed 20260918 --trials 32 --output ../../outputs/locomotion/bilateral_recheck.json
uv tool run --from uv==0.12.12 uv run --locked python -m adaptive_locomotion.symmetry_audit \
  --reference ../../assets/locomotion/checkpoints/limb_selected_seed2.pt \
  --checkpoint ../../assets/locomotion/checkpoints/limb_bilateral_selected_seed2.pt \
  --output ../../outputs/locomotion/bilateral_symmetry_recheck.json
```

Training configurations are in the three `runs/limb_*150s_seed2.json` reports.
For the selected general extension, use `adaptive-dog train` with
`--limb-stage consolidate --mode blind --reward-profile walk`, the preceding
`limb_bilateral_iter200_seed2.pt` as resume and single-reference, the original
healthy reference, and these weights: stride 1, balance 5, body-motion 0.5,
damage-action-rate 0.1, damage-angular-rate 0.3, damage-joint-accel 0.0000025,
symmetry 1, healthy-reference reward 0.5/loss 2, single-reference 0.5.
Use learning rate 0.0003, seed 2, 512 environments, 16 threads, 150 seconds,
allowance 1300 and a recorded extension reason. Select using the declared
development gates; wall-clock-bounded runs need not end at the same iteration
on different machines. Physical masses, limits and support definitions are
unchanged from [complete limb loss](LIMB_LOSS.md).

The new video is twelve seconds at 25 fps, 3840×2160, synchronized at 1×.
All nine demonstration runs pass. Capture took 3.890 seconds and rendering
39.245 seconds (excluding renderer setup). All 300 encoded frames decode;
opening, middle and ending frames were inspected. Maximum sampled penetration
is 4.953 mm and peak recorded torque is 91.8% of its original limit.

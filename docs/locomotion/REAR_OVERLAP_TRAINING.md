# Minimal rear swing overlap refinement

**Status update, 2026-09-11:** the user subsequently accepted the final stronger checkpoint and its video as the official v1 baseline, with the recorded limitations retained. See [the release record](OFFICIAL_V1.md). The original automatic selection results below remain unchanged.

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

## Second result: stop without promotion

The stronger round used **89.568 seconds**. Its final checkpoint passes **72/72**
development tasks and all prior lift, healthy-gait, stride/stance, speed and
body-motion retention gates. Every intact foot retains visible swings; the
lowest per-foot mean swing peak is **5.13 cm**. However, neither FR case passes
the predeclared >=0.25-cycle separation threshold:

| Case | Parent rear overlap | Candidate rear overlap | Parent phase separation | Candidate phase separation |
| --- | ---: | ---: | ---: | ---: |
| Lower FL | 0% | 0% | 41.9% | 44.3% |
| Lower FR | 7.0% | 0.27% | 10.8% | 17.6% |
| Whole FL | 0% | 0% | 40.0% | 43.2% |
| Whole FR | 7.80% | 1.48% | 6.7% | 11.0% |

These are means across eight trials per body, seed 9159 at 0.5 ms. Separation is
the shorter interval around the gait cycle: 0% means together, 50% means halfway
apart. The cost reduces simultaneous rear swing but does not create clear
alternation. Earlier checkpoints also have task failures. All six candidates and
reports remain archived; **no policy is promoted** and final seed 20260924 remains
unused. There is no new final acceptance or half-timestep claim.

Total new training was **179.114 seconds (2 min 59 s)** and **3,624,960 transitions**
on CPU MuJoCo/mjbatch with MPS learning. The final experimental checkpoint's
ancestry is **1922.725 seconds (32 min 03 s)** and **38,400,000 transitions**; the
unused portion of the first round is counted only in experiment cost. The
approved `limb_visible_steps_selected_seed2.pt` remains selected.

The user subsequently requested evaluation of symmetry, explicitly without
authorizing its use yet. See the [frozen-policy symmetry probe](SYMMETRY_OPTION.md).
No symmetry training or runtime mirroring was added to the deployed controller.

## Experimental media and implementation checks

[Watch the unselected overlap-only experiment](../../previews/locomotion/limb_rear_overlap.mp4).
The video is **not a symmetry-trained policy**. Nine physical bodies use the same
final stronger checkpoint, at 1×, 3840×2160, 25 fps for twelve seconds. All nine
demonstration tasks complete with allowed support. All 300 encoded frames decode;
opening, middle and ending frames were inspected. Capture took **9.421 seconds**,
render/export **40.088 seconds**, excluding context setup. Existing approved media
is preserved.

The preview **fails** the unchanged 8 mm penetration gate: lower-FR reaches
**8.889 mm** at the 20 ms recorded samples. Other cases are below 7 mm, and actual
torques remain within their original caps. The failed check is retained in
[video QA](REAR_OVERLAP_VIDEO_QA.json); no threshold was relaxed and no additional
tuning was performed on this rejected policy. This preview is archived evidence,
not an accepted replacement video.

The full relevant suite passed **84 tests**, including the new reward/phase
checks. Ruff and the native Mac viewer smoke test pass. Maximum CPU/MPS action
difference is **1.67e-6** in the [backend audit](REAR_OVERLAP_BACKEND_AUDIT.json).

To inspect the unselected experiment from `experiments/adaptive_locomotion`:

```sh
uv tool run --from uv==0.12.12 uv run --locked adaptive-dog view \
  --checkpoint ../../assets/locomotion/checkpoints/limb_rear_overlap_strong_90s_seed2.pt \
  --case whole_fr --presentation damage --timestep 0.0005
```

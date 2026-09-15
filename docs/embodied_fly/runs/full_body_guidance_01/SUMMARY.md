# Small model-guided updates improve flight; large updates break neural feedback

The first shared-decoder policy-improvement trial produces modest physical gains,
but no candidate meets every predeclared promotion criterion. The preferred
parent and its manifest remain unchanged. All twelve tested candidates and their
outcomes are retained. [Watch the comparison](../../../../previews/embodied_fly/full_body_guidance_01_comparison_v1.mp4).

| Complete ten-second flights | Velocity RMS, mm/s | Horizontal RMS, mm/s | Net climb, mm |
| --- | ---: | ---: | ---: |
| Parent, 4/4 | 12.757 | 9.186 | 60.720 |
| Analytical-guided, 4/4 | 12.127 | 9.335 | 39.948 |
| Learned-model-guided, 4/4 | 12.547 | 8.873 | 54.807 |

These are the lowest-velocity-error complete candidates for each method after
physical line search. Analytical guidance improves velocity error **4.935%** and
climb **34.21%**; learned-model guidance improves them **1.643%** and **9.739%**.
The analytical candidate misses the declared 5% velocity improvement gate by a
small margin. The rule is not rounded or relaxed after seeing the result.
This is one training seed and four familiar development starts, not evidence of
general robustness. The learned residual does not beat analytical guidance here.

Velocity uses the same rolling-velocity metric over 0.2–10 s as the parent,
averaged across starts. Climb is final vertical displacement over the complete
flight. Low errors from prematurely ended episodes never satisfy promotion.

## What learned

- One shared **815 motor inputs → generic preprocessing → 384 hidden units →
  78 outputs** decoder; 657,964 trainable parameters. All 78 output rows change.
  The encoder, MaleCNS graph/cell parameters, utility machinery and world model
  stay frozen. There is no wing-specific branch, output mask or sensor bypass.
- Actual parent execution supplies eight ten-second histories: **80 simulated
  seconds / 40,000 transitions**. Five episodes train and three validate; starts
  8/9 are withheld from this fit and used for physical checks. All eight survive.
  Raw motor-cell history reconstructs the acting commands exactly during capture.
- 980 training windows, 200 ms each, sampled from those five histories. Repeated
  windows are not extra physical experience. The 588 validation windows are
  separate; sixteen fixed windows provide each inexpensive model-cost check.
- Each arm uses the same batches and initialization, **100 Adam updates**, batch
  16, learning rate 3e-6. The model supplies differentiable physical consequences;
  this is model-based policy improvement, **not PPO or new PID imitation**.
- The objective penalizes mean velocity over two 100 ms intervals, angular
  motion, tilt, insufficient sustained support, floor proximity and changes to
  the parent's complete action vector. There is no reference wing angle or rapid
  motion penalty. A training-only parameter interpolation limits maximum action
  change on a fixed calibration bank to 0.015; it is not a runtime limiter.
- CUDA graphs match eager loss and gradients. Full optimization takes **8.381 s**
  for learned residual guidance and **8.229 s** for analytical guidance. These are
  incremental decoder updates to an already trained actor, not training a fly
  from scratch. The full MaleCNS graph executes during experience collection and
  physical evaluation; fixed motor histories make this local fit inexpensive.

## What failed, and the diagnostic

All full midpoint/final updates fail from the cold starts. Learned-model 50/100
updates fail at approximately 0.505/0.334 s; analytical 50/100 at 1.984/0.544 s.
Their predicted validation costs nevertheless decrease substantially.

The diagnostic isolates a large mismatch in the **commands being predicted**.
For the learned final candidate at cold start 0, after 200 ms:

| Prediction/evidence | Vertical velocity, mm/s |
| --- | ---: |
| Model given new decoder outputs on frozen parent neural history | +4.237 |
| Same model given the commands actually executed by the changed actor | −35.728 |
| Actual MuJoCo flight | −35.541 |

The model's full XYZ velocity error given live commands is 2.174 mm/s. Predicted
and actually executed command sequences differ by up to 0.01848 normalized units.
The analytical arm exhibits the same issue: +6.992 mm/s predicted using cached
neural history versus −9.424 mm/s actually measured; using executed commands
predicts −9.795 mm/s vertically. This is evidence of the local training
approximation failing during startup, not proof that the predictor is accurate
in every circumstance.

Changing the decoder changes physical feedback, which changes subsequent neural
activity and wing timing. Holding that future neural history fixed makes a large
weight update look better than it behaves in the actual recurrent loop. The
calibration-bank action limit does not bound deviations after feedback changes.

## Physical backtracking

No additional optimizer updates: interpolate every shared-decoder tensor between
parent and trained candidate, then execute the complete actor in MuJoCo. First
test 10%, 3%, and 1% of each full update. A 3% learned-model update preserves
flight and reduces drift; its 10% update fails. Analytical 10% preserves flight.
Two evidence-led refinements test learned-model 5% and analytical 20%: the former
fails, while the latter gives the best overall complete result above.

The video shows the 100-update startup failures, then PID/parent/analytical-20%/
learned-3% for all four complete starts at 1x. The display selection is explicit;
all midpoint, final and step-size results remain in the JSON reports.

## Costs and evidence

| Work | Wall seconds |
| --- | ---: |
| Experience setup / live capture / serialization | 8.295 / 38.946 / 12.735 |
| Learned arm: bank / graph setup / optimization / full invocation | 2.909 / 4.695 / 8.381 / 18.134 |
| Analytical arm: bank / graph setup / optimization / full invocation | 2.861 / 4.489 / 8.229 / 17.706 |
| Four original candidate evaluations | 38.388 |
| Six initial backtracking candidates | 235.352 |
| Learned 5% / analytical 20% refinement evaluations | 11.515 / 46.735 |

Physical integration is MuJoCo/mjbatch on CPU, eight collection worlds or four
evaluation worlds. Neural inference, model derivatives and decoder optimization
use RTX 4090 CUDA. Physics remains 1 kHz, control 500 Hz, agile-v5. Peak allocated
CUDA memory during decoder fitting is about 624 MiB (learned) / 618 MiB (analytical).

The suite passes **289 tests**, 45 upstream warnings, in **177.45 s**. Focused
objective, gradient and decoder checks pass 9/9. Archive checks verify unchanged
upstream actor tensors, unchanged physical contract and no wing-branch tensors in
all twelve candidates. The active parent checkpoint and selection manifest hashes
remain unchanged. [Verification](verification.json).

[Experience](experience.json), [learned training](learned_training.json),
[analytical training](analytical_training.json), [initial physical evaluations](initial_evaluation.json),
[initial step sizes](backtrack_initial.json), [learned refinement](backtrack_learned_refinement.json),
[analytical refinement](backtrack_analytical_refinement.json), [feedback diagnostic](diagnosis.json).
All candidate weights are under `assets/embodied_fly/diagnostics/full_body_guidance_01/`.
The [predeclared plan](PLAN.md) records the original budgets and promotion rule.

## Next action

Use short on-policy improvement cycles: collect fresh full-brain/body histories,
propose a small full-decoder update, execute that proposal with actual MaleCNS
feedback, and refresh histories after an accepted step. Test both the proposed
and opposite direction during the physical check, so a model-gradient direction
must improve the real recurrent loop before more fitting. Preserve analytical
guidance as the comparator. Do not spend a longer optimizer budget on the same
frozen future neural history: this trial directly shows why that can fail.

Judge the cumulative candidate by the same four complete flights and combined
velocity/climb criteria. The current smaller candidates demonstrate a local path
to improvement; they do not yet establish reliable model-based hover learning.

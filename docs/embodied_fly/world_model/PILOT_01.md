# A tested dynamics predictor, but no JEPA policy-guidance approval

The prediction-only pilot is complete. **Do not train the fly policy against
this JEPA checkpoint.** Its physical forecasts lose to both comparison models,
and its responses to corrective commands are too small or incorrectly directed.
This result concerns one compact model and training recipe, not JEPA in general.

The useful result is the analytical comparator: it predicts every significant
tested velocity correction with the correct sign, and the largest normalized
effect RMSE among the cold/warm, horizon and velocity-axis groups is below 10%.
This supports a bounded experiment with analytical model guidance for the
existing MaleCNS controller. No policy training occurred during this pilot.

![Prediction errors and effects of all tested wing corrections](../../../previews/embodied_fly/world_model_01_comparison.png)

## What was trained

A fresh **165,406-parameter** action-conditioned physical world model:

- Input: 25 observations at 500 Hz. Each contains the full physical joint pose
  except absolute x/y, joint velocities, actuator activations and height; a
  presence flag distinguishes padded cold-start history. 288 input values.
- Three causal temporal-convolution layers, width 64 and dilations 1/2/4,
  encode physical history. The receptive field contains no future observations.
- An action MLP maps all 78 normalized actuator commands to 64 features. A
  width-64 GRU predicts 100 successive latent states, one per 2 ms action.
  No wing samples or commands are averaged or dropped.
- Stage one: 1,000 updates of future-embedding prediction plus SIGReg, weight
  0.02, 32 random projections, 17 frequencies. Batch size 128.
- Stage two: freeze the latent model and fit a 95 → 128 → 128 → 30 readout for
  2,000 updates. It predicts physical deltas relative to the initial state,
  with constant-velocity position extrapolation as its reference. Outputs are
  position, velocity, rotation matrix, angular velocity and six wing angles/speeds.
- Checkpoints are selected on fixed validation batches, never on intervention
  outcomes: latent update 900, physical-readout update 300. Training continued
  to the specified budgets; those later steps remain counted in the timing.

This borrows the two-stage latent-prediction/physical-readout pattern from
[SkyJEPA](https://arxiv.org/html/2606.23444). The physical readout and 500 Hz fly
schema are our adaptation, not a reproduction of the paper's rotor model.

The acting fly is unchanged: sensors → encoder → frozen MaleCNS core → existing
motor decoder → bounded wing commands. The world model neither replaces that
path nor sends live controls. Supplied future action sequences are explicit
conditioning for a prediction test, not future information given to the actor.

## Data and physical identity

26 source trajectories from the accepted continuous velocity PID, preserved
run-11/run-13 students, saved run-13 recovery flights, and the pre-failure portion
of run 14. Source files, reports and the physical fingerprint are recorded in
[data.json](pilot_01/data.json).

| Split | Trajectories | State samples | Recorded duration at 500 Hz |
| --- | ---: | ---: | ---: |
| Training | 7 | 188,000 | 376.000 s |
| Validation | 5 | 116,800 | 233.600 s |
| Test | 14 | 100,996 | 201.992 s |

Whole-episode partition: 0/2/3/4/5 train, 1/6/7 validation, 8/9 test, across
all source families. The six explicit restarted recovery captures and two
failed-policy captures are test-only; established drift/recovery states also
occur in training students' continuous flights. These episodes were previously
used in policy development, so this is held-out world-model fitting, not a new
blind robotics benchmark. Overlapping windows are reused learning examples,
not additional independently simulated experience.

The canonical plant remains `wing_motion_agile_v5`, 1 kHz physics, 500 Hz
commands, zero-mass/non-colliding wings and instantaneous wing-state forces.
The actual compiled integrator is MuJoCo Euler. While building the comparator,
a test exposed an incorrect assumption that actuator damping was implicit;
fixing that made the isolated analytical wing integration match MuJoCo to the
test's 1e-10 rad / 1e-9 rad/s tolerances. The live plant was never changed.

## Forecast results

336 predetermined windows from all 14 test trajectories. Each forecast uses
only its initial history and the explicit future action sequence. Errors are
vector RMS velocity errors on matched valid time intervals, in mm/s.

| Horizon | JEPA pilot | Constant velocity | Analytical predictor |
| --- | ---: | ---: | ---: |
| 50 ms | 6.562 | 3.535 | **1.258** |
| 100 ms | 7.466 | 6.038 | **3.294** |
| 150 ms | 8.459 | 7.515 | **5.146** |
| 200 ms | 9.699 | 8.503 | **6.674** |

The analytical predictor uses the known wing actuation/force equations and
initial-pose locked body inertia. It approximates subsequent non-wing
articulation and hard joint stops and omits contact; it does not query future
MuJoCo states. Its absolute forecast errors grow with the horizon. Predictions
of *changes caused by small corrections* are substantially more accurate.

## The decisive test: action consequences

36 predefined held-out initial conditions: PID, run-11 and run-13 recordings,
episodes 8/9, at 0/0.2/0.6/2/2.016/4 seconds. Each initializes **16 physical
worlds**: unchanged control, six positive/negative wing-target bias pairs,
positive/negative collective sweep scaling, and an unchanged duplicate.

Bias magnitude is 0.003 normalized action units; sweep scaling is ±3%.
All 78 commanded values retain their original bounds, non-wing commands stay
identical, and no values clip. Each variant runs the full MuJoCo plant for
200 ms with 1 kHz force recording. Future commands remain fixed in this test;
no PID or policy reacts to the altered branch.

- **576 complete physical continuations**, including **504 changed-action
  continuations**. No failures; no post-failure samples entered the comparison.
- All 36 replay checks pass. Maximum position discrepancy against the source
  recording is **1.1921e-7 cm**; duplicate-control metric discrepancy is zero.
- On effects of at least 0.2 mm/s, the analytical predictor achieves **100%
  sign agreement in every velocity axis, phase and horizon group**. Normalized
  effect RMSE ranges from below 0.1% to 9.6%.
- JEPA's normalized velocity-effect errors are approximately 98.5–100.5%:
  predictions barely respond to corrections that measurably change the body.
  That is evidence of inadequate action sensitivity in this checkpoint.
- On mean-lift changes of at least 0.001 body weights, JEPA gets the sign right
  but misses effect magnitude substantially: normalized error 83.7–92.3%.
  The analytical predictor gets the sign right and reproduces these lift
  effects closely. JEPA lift is reconstructed from predicted 500 Hz wing states;
  truth and analytical forces are recorded at all 1 kHz ticks.

Every raw group, magnitude, informative-sample count and failure exclusion is
in [evaluation.json](pilot_01/evaluation.json). The predefined forecast,
velocity-effect and support-effect gates all fail for JEPA; the replay gate
passes. No acceptance criterion was relaxed after seeing the result.

## Measured cost and verification

| Work | Wall time |
| --- | ---: |
| Prepare previously recorded data | 9.614 s |
| Training setup | 0.755 s |
| Latent-model training, RTX 4090 | 6.100 s |
| Physical-readout training, RTX 4090 | 7.205 s |
| Validation during training | 0.122 s |
| Complete training command, including IO/overhead | 15.072 s |
| Evaluation setup | 0.295 s |
| Held-out forecasts and analytical comparison | 3.290 s |
| Physical interventions, forecasts, capture and scoring | 14.257 s |
| Complete evaluation command | 17.867 s |

**13.305 s of model training**; this is not the fly's policy-training time.
Peak PyTorch CUDA allocation: 1,002,816,512 bytes, approximately 0.93 GiB.
128,000 latent-stage and 256,000 readout-stage window presentations reuse the
recorded dataset. New intervention physics comprises 57,600 action transitions
and 115,200 physics steps. MuJoCo/mjbatch runs those worlds on 16 CPU threads;
GPU inference/training uses PyTorch CUDA, not MuJoCo Warp.

Implementation began 21:34:29 UTC, September 14; archived evidence was created
at 22:00:49 UTC. Source: `a637a13` for data/training, `fb72690` for evaluation.
The full fly suite passes **280 tests**, with 45 upstream warnings, in 177.62 s;
the focused causal/physical tests pass 7/7 in 2.86 s. The plot was visually
inspected. [Hashes and verification](pilot_01/verification.json),
[training log](pilot_01/training.json).

Weights and all intervention traces are preserved with Git LFS under
`assets/embodied_fly/world_models/pilot_01/`. Each combined NPZ key begins with
the case name from the evaluation report, followed by `__` and the array name.
The retained fly policy and `PREFERRED_HOVER.json` remain unchanged.

## Reproduce

On the existing GPU checkout, with the source captures named in data.json:

```sh
MUJOCO_GL=egl uv run --project experiments/embodied_fly --locked python \
  -m embodied_fly.world_data --output outputs/embodied_fly/world_data_new
uv run --project experiments/embodied_fly --locked python \
  -m embodied_fly.world_train --dataset outputs/embodied_fly/world_data_new \
  --output outputs/embodied_fly/world_model_new
MUJOCO_GL=egl uv run --project experiments/embodied_fly --locked python \
  -m embodied_fly.world_evaluate --dataset outputs/embodied_fly/world_data_new \
  --checkpoint outputs/embodied_fly/world_model_new/prober.pt \
  --output outputs/embodied_fly/world_evaluation_new
```

Use unique output directories. The checkpoint binds its exact prepared manifest;
re-preparing data creates new provenance and requires a corresponding model.
macOS can run the CPU data/physics paths without `MUJOCO_GL=egl`; training and
evaluation expose `--device cpu`. The measured run used CUDA.

## Next action

**Recommend a short analytical-model-guided controller experiment, rather than
extended PPO or longer unchanged JEPA training.** Port the validated wing/body
response to differentiable tensor operations, verify gradients against these
physical intervention effects, then use it to teach corrections to the fly's
own wing pattern through its existing decoder. This targets predicted lift
and velocity outcomes without matching the PID's oscillator phase.

Start from the retained policy with its own matching physical and full neural
histories. Keep the same body, MaleCNS graph and deployed action path. Require
preserved cold-start support and reduced combined horizontal/vertical motion
on the existing complete-flight checks before extending training. The pilot
does not establish that a learned policy will improve; it establishes a much
better-tested local source of guidance for that next experiment. If JEPA is
revisited, collect controlled interventions in training episodes first and
require useful action sensitivity before any policy updates.

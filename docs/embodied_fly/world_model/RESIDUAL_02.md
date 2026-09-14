# Learned acceleration residuals pass the fly prediction pilot

The two-minute follow-up passes the predeclared prediction and intervention
checks. A probe now learns acceleration corrections **inside a differentiable
physical integrator**, instead of predicting direct future-state deltas.
The fly actor, MaleCNS graph, body and live force law remain unchanged.

The strongest improvement is on the held-out controlled-command continuations:
at 200 ms, vector velocity RMSE drops from **9.470 to 3.811 mm/s (59.8%)** versus
the analytical model, across all 504 changed-action branches. This measures
absolute future-velocity accuracy, distinct from accuracy of the *difference*
between changed and unchanged branches.

On 336 ordinary held-out flight windows, learned improvement over the analytical
model is **modest: 1.6%**, using the mean velocity RMSE across four horizons.
The 200 ms improvement is 5.5%; 50/100 ms forecasts are slightly worse. Most of
the improvement over the first direct-state pilot comes from retaining known
physics. This single run does not establish statistical significance or better
flight control. The learned residual is useful enough for a bounded policy-
guidance experiment, not a reason to discard the analytical comparator.

![Learning and held-out results](../../../previews/embodied_fly/world_residual_02_comparison.png)

## What actually learned

- Fresh 158,342-parameter world model. A causal temporal history encoder and
  action-conditioned GRU predict latent states at 500 Hz, 200 ms ahead. Inputs,
  normalization and episode boundaries follow the original pilot. No future
  observations enter prediction; candidate future actions are explicit inputs.
- Latent stage: 60.003606 s, **14,006 updates**, batch 128. Future-embedding
  prediction plus 0.02 SIGReg. Best validation checkpoint: update 10,484.
- Freeze the history encoder, action encoder and recurrent predictor. A
  **64 → 128 → 128 → 6** prober receives predicted latents only. It supplies
  world-frame linear and body-frame angular acceleration residuals, bounded by
  tanh with scales 100 cm/s² and 20 rad/s². The last layer starts at zero,
  recovering the analytical model before learning. No initial-state MLP bypass.
- Known capped wing PD, independent wing inertia/damping, instantaneous wing
  forces, body drag, gravity and rotational dynamics integrate at **1 kHz**.
  The learned acceleration correction is held for the two physical ticks of
  each 500 Hz command. Every tick stays differentiable. Physical state enters
  the integrator; the probe cannot directly overwrite predicted position.
- Probe stage: 60.045289 s, **1,128 updates**, batch 128. It minimizes equally
  weighted normalized position, velocity, rotation-matrix and angular-velocity
  mean squared errors; adds 0.1 times paired velocity/angular-effect error and
  1e-4 normalized residual regularization. No PPO rewards or actor updates.
- Each probe batch contains 64 ordinary flight windows and 32 matched pairs of
  unchanged/changed command continuations. Fixed bank: 1,024 ordinary windows
  plus 1,104 physical intervention windows. Wings and frozen latent predictions
  are cached; gradients still traverse every body-integration tick to the probe.
- AdamW, 1e-3 → 1e-4 cosine learning-rate schedule, 1e-5 weight decay, gradient
  norm cap 1. Validation chooses the physical checkpoint, including zero as an
  eligible baseline. Best probe: update 282, after 15.050 s; the complete
  60-second budget was used and the final checkpoint is also retained.

The selected probe reduces the equal-weight validation average from 5.556 to
3.613 mm/s. This is uneven: ordinary-flight validation worsens from 4.640 to
5.251 mm/s, while intervention validation improves from 6.471 to 1.974 mm/s.
Later probe updates do not improve the selection metric. The saved learning
curve shows those later updates instead of presenting only the best point.

## Data and controls

The previous episode split is preserved: training 0/2/3/4/5, validation 1/6/7,
test 8/9. These are held out from world-model fitting, not previously unseen
episodes across the entire fly project.

- Existing training recordings: **376 s**, 188,000 state samples at 500 Hz.
- New training: **69 matched starts × 16 branches × 0.2 s = 220.8 s**.
  Combined available training experience: **596.8 s**, about ten minutes.
  Histories and repeated window presentations are not additional experience.
- New validation: 29 matched starts × 16 branches = 92.8 s. Original validation
  and test captures are preserved; no test intervention enters training.
- Six individual wing-coordinate ±bias pairs, collective sweep gain ±changes,
  unchanged and duplicate controls. Biases 0.0015/0.003/0.006 normalized units;
  sweep gain changes 1.5/3/6%. No action clipping in retained groups.
- Of 100 collected groups, two were excluded because one branch failed in each:
  run-11 episodes 0 and 1 at 0.6 s. Their exclusions remain in the manifest;
  post-failure states were not fitted. All 98 retained groups replay their
  source, with maximum position error 9.537e-7 cm; duplicates agree exactly.
- New physics uses 16 MuJoCo/mjbatch CPU worlds and 16 threads. Learning and
  differentiable prediction use PyTorch CUDA on RTX 4090; this is not Warp.

## Held-out evidence

Same 336 ordinary forecast windows and four horizons as the original pilot.
Units: vector velocity RMSE, mm/s. Columns share physical starts and commands.

| Horizon | Physics + learned residual | Analytical | Constant velocity | Previous direct-state probe |
| --- | ---: | ---: | ---: | ---: |
| 50 ms | 1.339 | **1.257** | 3.535 | 6.562 |
| 100 ms | 3.371 | **3.294** | 6.038 | 7.466 |
| 150 ms | **5.102** | 5.146 | 7.515 | 8.459 |
| 200 ms | **6.304** | 6.674 | 8.503 | 9.699 |

For the unchanged 36 matched intervention starts (576 total continuations,
504 changed-action branches), every replay passes and no branch fails.

- Significant velocity changes: **100% sign agreement** in every cold/warm,
  horizon and XYZ group; largest normalized effect RMSE **4.995%**. Thresholds
  remain 0.2 mm/s, at least 80% sign agreement and less than 50% normalized RMSE.
- Support effects: 100% sign agreement; maximum normalized error 4.25e-6.
  **Support accuracy comes from the known wing/force equations**, not a newly
  learned lift model. The residual corrects body acceleration, not wing lift.
- All four predeclared gates pass: replay, forecasts, velocity effects, support
  effects. Forecast acceptance permits a small short-horizon regression if all
  horizons beat constant velocity and mean error is no worse than analytical.
  The rule was not relaxed after seeing this run.

## Measured costs and verification

| Work | Time |
| --- | ---: |
| New intervention collection / compression | 43.431 s |
| Latent optimizer steps | 60.004 s |
| Probe optimizer steps | 60.045 s |
| **Combined actual optimization** | **120.049 s** |
| Resumed probe invocation, including its setup and validation | 75.058 s |
| Probe bank preparation | 1.173 s |
| CUDA graph warmup/capture and gradient comparison | 3.350 s |
| Held-out evaluation, including physical interventions | 31.818 s |

The original invocation stopped during CUDA graph capture, **before any probe
optimizer updates**, because autograd retained a default-stream reference.
Warmup and capture now use the same dedicated stream. The preserved latent
checkpoint was resumed without repeating latent training. The first process's
latent validation/setup timings were not persisted; they are not guessed or
included as training. The 75.058 s figure is the resumed invocation only.

CUDA graph replay matches eager loss and gradients. No weights change during
graph setup; the latent parameters remain byte-identical through probe training.
Peak allocated CUDA memory in the resumed process is **1.64 GiB**. This compact
world-model fit does not execute the full 166,700-cell MaleCNS graph.

Zero residual agrees with the independent analytical implementation over full
200 ms trajectories at double-precision test tolerances. Action and residual
gradients agree with finite differences. The fly suite passes **283 tests**,
45 upstream warnings, in **174.87 s**. Two archived cold/warm cases additionally
verify Mac CPU predictions against the saved CUDA predictions. The chart was
visually inspected. No new actor video was generated because no actor changed.

[Training log](residual_02/training.json), [data and exclusions](residual_02/data.json),
[complete evaluation](residual_02/evaluation.json), [hashes and verification](residual_02/verification.json).
Weights and intervention traces are in `assets/embodied_fly/world_models/residual_02/`.
Data source commit: `eff6273`; latent training: `8692a53`; resumed probe/evaluation:
`7d95af8`. Existing run-11 policy and preferred-policy manifest are unchanged.

## Reproduce and next step

Use the existing uv environment and saved source captures listed in the original
pilot. Fresh output directories preserve all earlier runs:

```sh
MUJOCO_GL=egl uv run --project experiments/embodied_fly --locked python \
  -m embodied_fly.world_residual_data --dataset outputs/embodied_fly/world_data_01 \
  --output outputs/embodied_fly/residual_data_new
uv run --project experiments/embodied_fly --locked python \
  -m embodied_fly.world_residual_train --dataset outputs/embodied_fly/world_data_01 \
  --interventions outputs/embodied_fly/residual_data_new \
  --output outputs/embodied_fly/residual_train_new
MUJOCO_GL=egl uv run --project experiments/embodied_fly --locked python \
  -m embodied_fly.world_residual_evaluate --dataset outputs/embodied_fly/world_data_01 \
  --checkpoint outputs/embodied_fly/residual_train_new/prober.pt \
  --output outputs/embodied_fly/residual_evaluation_new
```

Mac: omit `MUJOCO_GL=egl` and use `--device cpu` for tensor paths. Timed training
will execute a hardware-dependent number of updates; saved weights reproduce
the measured checkpoint. This remains an adaptation of latent prediction with
physical probing, not a reproduction of another robot's dynamics model.

**Next: a short model-guided update of the existing MaleCNS wing decoder.** Use
the student's own physical and neural histories, differentiate small bounded
wing corrections through this predictor, and train for reduced body velocity
while retaining support. Keep the analytical-only guidance as a comparator;
the learned benefit is strongest around interventions and modest on ordinary
flight. Avoid PID phase-angle matching. Validate the resulting actor in complete
ten-second MuJoCo hovers before extending training. The deployed path stays
sensors → encoder → frozen MaleCNS → decoder → wings; the world model is training
machinery, not an external deployed controller. No such policy update has yet
been run in this follow-up.

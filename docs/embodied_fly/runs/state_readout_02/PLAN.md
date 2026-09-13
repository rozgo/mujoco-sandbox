# State readout 02 — declared before execution

Keep state_hover_retention_02 as the parent. Preserve all earlier checkpoints and films. This stage changes only the six existing wing-output rows (1,542 parameters). Same 397 inputs, 78 outputs, frozen upstream brain and other outputs, canonical wing_motion body, and ordinary motor-only deployment.

Extend the preceding readout fit with corrective labels on two real unassisted hover histories: state_readout_01_evaluation and sweep_probe_01_evaluation. Labels use the existing measured-state hover reference after offline mj_forward. No integration, pose helper or live reference control is added. Keep each history only until its first height below 0.5 cm or upright below 0.5, capped at two seconds. The actual source audit retains 116 and 822 frames respectively and excludes post-fall history from fitting.

Retain the previous parent stand/walk histories and successful state-hover reference. First four seconds fit and last second validate for these original histories. For correction histories, hold out every fifth 20 ms block; this keeps near-failure examples in training. These are temporally related fitting checks, not independent episodes. Total: 6,758 training frames and 1,680 descriptive validation frames. Five causal neural replay histories, 12,500 processed observation frames including excluded history (four internal graph updates per frame); zero newly collected physical transitions.

Base per-frame weights: 8 stand, 8 walk, 1 successful reference, 1 each correction. Weight the first 100 ms of airborne histories by four. Ridge grid unchanged: [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]. Select by weighted per-history held-out action MSE. Retain every candidate metric. This changes the training data and startup weighting together; do not attribute an outcome to either separately.

Evaluate the exported checkpoint unassisted on all three commands for five seconds each, new seed 72005. Keep existing strict task and posture gates. Capture neural view and complete failures; encode, decode, inspect and automatically open the full video. Preserve the parent if the combined behavior is worse.

```sh
MUJOCO_GL=egl uv run --project experiments/embodied_fly --locked python -m embodied_fly.motor_state_fit \
 --resume assets/embodied_fly/diagnostics/state_hover_retention_02.pt \
 --graph ../fly-survival/outputs/fly_survival/malecns \
 --ground outputs/embodied_fly/state_hover_retention_02_evaluation \
 --hover outputs/embodied_fly/state_hover_reference_02 \
 --correction-capture outputs/embodied_fly/state_readout_01_evaluation \
 --correction-capture outputs/embodied_fly/sweep_probe_01_evaluation \
 --startup-weight 4 --output outputs/embodied_fly/state_readout_02
```

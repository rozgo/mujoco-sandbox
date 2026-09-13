# State readout 01 — declared before execution

Parent remains state_hover_retention_02. Fit only the six existing final wing-output rows and biases (1,542 parameters). Frozen neural core, sensory encoder, hidden decoder, other output rows and physical model. One motor-only397-input/78-output actor for every command.

Replay three complete, previously captured5s histories through the parent onGPU: parent standing, parent walking, and the successful measured-state hover reference. All are the same canonical wing_motion body at500Hz actions. Features are the actual256 hidden decoder activations after the graph update; labels are parent ground outputs and recorded state-hover wing commands. Teacher actions are targets only. No new physics or live teacher execution occurs in fitting.

Fit first4s of each history (6,000total frames), use last1s (1,500frames) for descriptive fit selection. These windows are temporally related and do not constitute independent validation. Solve weighted ridge regression for changes to existing output logits, with task weights8:8:1 and regularization values[1e-5,1e-4,1e-3,1e-2,1e-1]. Select by weighted final-window action MSE. Preserve all candidate fit metrics. Changes are baked into the existing linear output layer, with no runtime helper.

Then evaluate the exported actor unassisted for5s each stand/walk/hover, seed72003, with the existing strict gates, neural activity capture, and full video. Retain failed outcomes. Fit quality alone cannot establish motor improvement.

```sh
MUJOCO_GL=egl uv run --project experiments/embodied_fly --locked python -m embodied_fly.motor_state_fit \
 --resume assets/embodied_fly/diagnostics/state_hover_retention_02.pt \
 --graph ../fly-survival/outputs/fly_survival/malecns \
 --ground outputs/embodied_fly/state_hover_retention_02_evaluation \
 --hover outputs/embodied_fly/state_hover_reference_02 \
 --output outputs/embodied_fly/state_readout_01
```

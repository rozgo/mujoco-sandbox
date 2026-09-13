# Nonlinear motor readout 01 — declared before execution

The previous linear motor-cell fit corrected the sign of some recorded wing-limit mistakes but failed walking and hover in a live test. Test a nonlinear readout of those same neural features. Retention02 stays the preserved parent; no candidate is promoted from fitting accuracy.

One feedforward wing decoder inside the same actor: existing 815 normalized motor-cell activities → 128 tanh units → six wing-logit corrections. The final layer starts at zero, preserving the original actor function. Training-set-only feature centering/scaling (standard deviation floor 0.05, input clip ±10) is stored in the checkpoint. This adds 105,222 trainable parameters and no recurrent memory. Every original actor weight remains frozen. No raw-sensor bypass, runtime teacher, action mask, oscillator, pose helper or root-force control is introduced.

Reuse the hash-verified motor_cell_readout01 cache from the unchanged canonical wing_motion body: parent ground histories, successful state-hover reference, and two actual pre-fall correction histories. Same 6,758 fitting frames, 1,680 temporally related validation frames, 8:8:1:1:1 history weights and fourfold first-100ms air weight. No new neural replay or integrated physical experience during this fit. Earlier feature-cache collection is accounted for separately.

Fit only the new decoder using AdamW, learning rate 0.0003, weight decay 0.0001, gradient-norm cap 1, batch size 2,048 sampled uniformly with per-frame loss weights. Minimize bounded action MSE, a deliberate change from the previous ridge/logit objective as well as a nonlinear architecture change. Fit at most 120 seconds or 18,000 updates, whichever occurs first; include setup and export as separate clocks. Seed 72008. Select lowest weighted per-history validation action MSE, including the zero-output initial candidate. These validation frames belong to the same histories and are not independent episodes.

Then run the ordinary actor loader in a fresh, unassisted five-second stand/walk/hover evaluation, seed 72009, with the same physical fingerprint and unchanged strict posture/support/flight gates. Save all failures and the full 15-second 1× video. Decode, inspect and open the completed film. One checkpoint commands all 78 actuators throughout.

```sh
uv run --project experiments/embodied_fly --locked python -m embodied_fly.motor_nonlinear_fit \
 --resume assets/embodied_fly/diagnostics/state_hover_retention_02.pt \
 --cache outputs/embodied_fly/motor_cell_readout_01 \
 --output outputs/embodied_fly/nonlinear_motor_readout_01 \
 --seconds 120 --max-updates 18000 --hidden 128 --batch-size 2048 \
 --learning-rate 0.0003 --seed 72008
```

On the GPU host, set `MUJOCO_GL=egl` for physical evaluation. Keep the native macOS rendering backend on Mac.

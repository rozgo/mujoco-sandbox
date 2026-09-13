# Frozen-actor exploration diagnosis

User authorized one diagnostic and at most two further approximately three-minute
PPO pilots on September 13, 2026. Preserve `position_sustain_retention_01` as the
development baseline; neither previous outcome PPO checkpoint is promoted.

Run the parent with no optimization, teacher, critic or live resets. Compare
pre-tanh Gaussian standard deviations 0, 0.003 and 0.01 on all 78 outputs. The
0.01 setting matches the previous PPO exploration floor. Use common random
samples and identical complete physical initial states within each comparison.
Three predetermined development seeds: 98103, 98113, 98123. Evaluate stand, walk
and hover for five seconds each, with nine parallel worlds per seed. Preserve
full traces and parameter hashes before/after. This is a diagnostic, not an
independent acceptance test; matched means comparisons, not a guaranteed outcome.

If noise alone destabilizes the successful controller, address exploration
first. Otherwise investigate update/retention behavior. Keep the physical body,
force model, actor architecture and commands fixed. Declare each subsequent
pilot before running it. A failed refinement must not replace the parent.

```sh
uv run --project experiments/embodied_fly --locked python -m embodied_fly.motor_exploration \
  --resume assets/embodied_fly/diagnostics/position_sustain_retention_01.pt \
  --graph outputs/fly_survival/malecns \
  --output outputs/embodied_fly/position_exploration_01
```
